import type { LiveTrainingCommand, LiveTrainingReply } from './worker-protocol';

/** The browser entry point for M5.2; training code stays off the main thread. */
export function createLiveTrainingWorker(): Worker {
  return new Worker(new URL('./live-training.worker.ts', import.meta.url), { type: 'module' });
}

export function sendLiveTrainingCommand(worker: Worker, command: LiveTrainingCommand): void {
  worker.postMessage(command);
}

export function onLiveTrainingReply(
  worker: Worker,
  listener: (reply: LiveTrainingReply) => void,
): () => void {
  const handle = (event: MessageEvent<LiveTrainingReply>) => listener(event.data);
  worker.addEventListener('message', handle);
  return () => worker.removeEventListener('message', handle);
}
