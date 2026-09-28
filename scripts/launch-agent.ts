import { execFile } from 'node:child_process';
import { lstat, mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { homedir, platform } from 'node:os';
import { join, resolve } from 'node:path';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

export const LAUNCH_AGENT_LABEL = 'com.gaylon.birds-llm-lab';
export const LAUNCH_AGENT_FILENAME = `${LAUNCH_AGENT_LABEL}.plist`;
export const LAUNCH_AGENT_VERSION = 1;

export interface LaunchAgentPaths {
  projectDirectory: string;
  homeDirectory: string;
  nodePath: string;
  tsxCliPath: string;
  serverPath: string;
  distIndexPath: string;
  plistPath: string;
  logPath: string;
}

function xml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

export function resolveLaunchAgentPaths(
  projectDirectory = process.cwd(),
  homeDirectory = homedir(),
): LaunchAgentPaths {
  const project = resolve(projectDirectory);
  const home = resolve(homeDirectory);
  return {
    projectDirectory: project,
    homeDirectory: home,
    nodePath: join(project, 'node_modules', 'node', 'bin', 'node'),
    tsxCliPath: join(project, 'node_modules', 'tsx', 'dist', 'cli.mjs'),
    serverPath: join(project, 'server', 'index.ts'),
    distIndexPath: join(project, 'dist', 'index.html'),
    plistPath: join(home, 'Library', 'LaunchAgents', LAUNCH_AGENT_FILENAME),
    logPath: join(home, 'Library', 'Logs', 'birds-llm-lab.log'),
  };
}

export function renderLaunchAgentPlist(paths: LaunchAgentPaths): string {
  return `<?xml version="1.0" encoding="UTF-8"?>
<!-- birds-llm-lab managed v${LAUNCH_AGENT_VERSION} -->
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LAUNCH_AGENT_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${xml(paths.nodePath)}</string>
    <string>${xml(paths.tsxCliPath)}</string>
    <string>${xml(paths.serverPath)}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${xml(paths.projectDirectory)}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>NODE_ENV</key>
    <string>production</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>5</integer>
  <key>StandardOutPath</key>
  <string>${xml(paths.logPath)}</string>
  <key>StandardErrorPath</key>
  <string>${xml(paths.logPath)}</string>
</dict>
</plist>
`;
}

export function plistLabel(contents: string): string | null {
  const match = contents.match(/<key>Label<\/key>\s*<string>([^<]+)<\/string>/);
  return match?.[1] ?? null;
}

async function requireFile(path: string, description: string): Promise<void> {
  const details = await stat(path).catch(() => null);
  if (!details?.isFile()) throw new Error(`${description} is missing at ${path}.`);
}

export async function validateExistingLaunchAgent(
  plistPath: string,
  uid = process.getuid?.(),
): Promise<void> {
  const details = await lstat(plistPath);
  if (!details.isFile() || details.isSymbolicLink()) {
    throw new Error(`Refusing to replace non-regular LaunchAgent path: ${plistPath}`);
  }
  if (uid !== undefined && details.uid !== uid) {
    throw new Error(`Refusing to replace a LaunchAgent owned by uid ${details.uid}.`);
  }
  const existingLabel = plistLabel(await readFile(plistPath, 'utf8'));
  if (existingLabel !== LAUNCH_AGENT_LABEL) {
    throw new Error(
      `Refusing to replace LaunchAgent labeled ${existingLabel ?? '(missing label)'}.`,
    );
  }
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false;
    throw error;
  }
}

function requireMacOs(): void {
  if (platform() !== 'darwin')
    throw new Error('LaunchAgent management is available only on macOS.');
}

function currentUid(): number {
  const uid = process.getuid?.();
  if (uid === undefined) throw new Error('The current user id is unavailable.');
  return uid;
}

function serviceTarget(): string {
  return `gui/${currentUid()}/${LAUNCH_AGENT_LABEL}`;
}

async function bootstrapLaunchAgent(domain: string, plistPath: string): Promise<void> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      await execFileAsync('/bin/launchctl', ['bootstrap', domain, plistPath]);
      return;
    } catch (error) {
      lastError = error;
      if (attempt < 2) {
        await new Promise((resolveDelay) => setTimeout(resolveDelay, 250 * (attempt + 1)));
      }
    }
  }
  throw lastError;
}

export async function installLaunchAgent(paths = resolveLaunchAgentPaths()): Promise<void> {
  requireMacOs();
  await Promise.all([
    requireFile(paths.nodePath, 'Project-local Node 22 runtime'),
    requireFile(paths.tsxCliPath, 'tsx service runner'),
    requireFile(paths.serverPath, 'Loopback service entrypoint'),
    requireFile(paths.distIndexPath, 'Built app'),
  ]);
  await mkdir(join(paths.homeDirectory, 'Library', 'LaunchAgents'), { recursive: true });
  await mkdir(join(paths.homeDirectory, 'Library', 'Logs'), { recursive: true });
  if (await exists(paths.plistPath)) await validateExistingLaunchAgent(paths.plistPath);

  const temporaryPath = `${paths.plistPath}.${process.pid}.tmp`;
  await writeFile(temporaryPath, renderLaunchAgentPlist(paths), { flag: 'wx', mode: 0o600 });
  try {
    await execFileAsync('/usr/bin/plutil', ['-lint', temporaryPath]);
    await rename(temporaryPath, paths.plistPath);
  } catch (error) {
    await rm(temporaryPath, { force: true });
    throw error;
  }

  const domain = `gui/${currentUid()}`;
  await execFileAsync('/bin/launchctl', ['bootout', serviceTarget()]).catch(() => undefined);
  await bootstrapLaunchAgent(domain, paths.plistPath);
}

export async function launchAgentStatus(): Promise<string> {
  requireMacOs();
  try {
    const { stdout } = await execFileAsync('/bin/launchctl', ['print', serviceTarget()], {
      encoding: 'utf8',
    });
    return stdout;
  } catch {
    throw new Error(`${LAUNCH_AGENT_LABEL} is not loaded.`);
  }
}

export async function restartLaunchAgent(paths = resolveLaunchAgentPaths()): Promise<void> {
  requireMacOs();
  await validateExistingLaunchAgent(paths.plistPath);
  try {
    await execFileAsync('/bin/launchctl', ['kickstart', '-k', serviceTarget()]);
  } catch {
    const domain = `gui/${currentUid()}`;
    await bootstrapLaunchAgent(domain, paths.plistPath);
  }
}

export async function uninstallLaunchAgent(paths = resolveLaunchAgentPaths()): Promise<void> {
  requireMacOs();
  if (!(await exists(paths.plistPath))) return;
  await validateExistingLaunchAgent(paths.plistPath);
  await execFileAsync('/bin/launchctl', ['bootout', serviceTarget()]).catch(() => undefined);
  await rm(paths.plistPath);
}
