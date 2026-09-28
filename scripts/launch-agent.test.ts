import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { afterEach, describe, expect, it } from 'vitest';
import {
  LAUNCH_AGENT_LABEL,
  plistLabel,
  renderLaunchAgentPlist,
  resolveLaunchAgentPaths,
  validateExistingLaunchAgent,
} from './launch-agent';

const execFileAsync = promisify(execFile);
const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true, force: true })),
  );
});

describe('LaunchAgent plist', () => {
  it('renders a fixed-label, absolute-path, loopback service definition', () => {
    const paths = resolveLaunchAgentPaths('/Applications/Birds LLM Lab', '/Users/example');
    expect(renderLaunchAgentPlist(paths)).toMatchInlineSnapshot(`
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>
      <!-- birds-llm-lab managed v1 -->
      <plist version=\"1.0\">
      <dict>
        <key>Label</key>
        <string>com.gaylon.birds-llm-lab</string>
        <key>ProgramArguments</key>
        <array>
          <string>/Applications/Birds LLM Lab/node_modules/node/bin/node</string>
          <string>/Applications/Birds LLM Lab/node_modules/tsx/dist/cli.mjs</string>
          <string>/Applications/Birds LLM Lab/server/index.ts</string>
        </array>
        <key>WorkingDirectory</key>
        <string>/Applications/Birds LLM Lab</string>
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
        <string>/Users/example/Library/Logs/birds-llm-lab.log</string>
        <key>StandardErrorPath</key>
        <string>/Users/example/Library/Logs/birds-llm-lab.log</string>
      </dict>
      </plist>
      "
    `);
  });

  it('passes macOS plist validation', async () => {
    if (process.platform !== 'darwin') return;
    const directory = await mkdtemp(join(tmpdir(), 'birds-llm-plist-'));
    temporaryDirectories.push(directory);
    const path = join(directory, 'service.plist');
    await writeFile(
      path,
      renderLaunchAgentPlist(resolveLaunchAgentPaths('/project', '/Users/example')),
    );
    const { stdout } = await execFileAsync('/usr/bin/plutil', ['-lint', path], {
      encoding: 'utf8',
    });
    expect(stdout).toContain('OK');
  });

  it('refuses to replace a differently labeled plist', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'birds-llm-plist-'));
    temporaryDirectories.push(directory);
    const path = join(directory, 'service.plist');
    const foreign = renderLaunchAgentPlist(
      resolveLaunchAgentPaths('/project', '/Users/example'),
    ).replace(LAUNCH_AGENT_LABEL, 'example.foreign.service');
    await writeFile(path, foreign);
    expect(plistLabel(await readFile(path, 'utf8'))).toBe('example.foreign.service');
    await expect(validateExistingLaunchAgent(path)).rejects.toThrow(
      'Refusing to replace LaunchAgent labeled',
    );
  });
});
