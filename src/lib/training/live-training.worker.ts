import { LiveTrainingWorkerRuntime } from './worker-runtime';
import type { LiveTrainingReply } from './worker-protocol';

const worker = self as unknown as {
  onmessage: ((event: MessageEvent<unknown>) => void) | null;
  postMessage: (reply: LiveTrainingReply, transfer?: Transferable[]) => void;
};

const runtime = new LiveTrainingWorkerRuntime((reply, transfer) =>
  worker.postMessage(reply, transfer),
);
worker.onmessage = (event) => runtime.handle(event.data);
