import {
  installLaunchAgent,
  launchAgentStatus,
  restartLaunchAgent,
  uninstallLaunchAgent,
} from './launch-agent.js';

async function main(): Promise<void> {
  const command = process.argv[2];
  if (command === 'install') {
    await installLaunchAgent();
    console.log('Birds LLM Lab LaunchAgent installed and started.');
    return;
  }
  if (command === 'status') {
    const status = await launchAgentStatus();
    const state = status.match(/\bstate = ([^\n]+)/)?.[1]?.trim() ?? 'loaded';
    console.log(`Birds LLM Lab LaunchAgent: ${state}`);
    return;
  }
  if (command === 'restart') {
    await restartLaunchAgent();
    console.log('Birds LLM Lab LaunchAgent restarted.');
    return;
  }
  if (command === 'uninstall') {
    await uninstallLaunchAgent();
    console.log('Birds LLM Lab LaunchAgent uninstalled.');
    return;
  }
  throw new Error('Expected one of: install, status, restart, uninstall.');
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : 'LaunchAgent command failed.');
  process.exitCode = 1;
});
