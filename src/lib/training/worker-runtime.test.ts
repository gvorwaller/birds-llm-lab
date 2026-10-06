import { describe, expect, it } from 'vitest';
import type { ModelConfig } from '../data/schemas';
import { decodeCheckpointWeights } from '../model/checkpoint';
import { trainingSample, selectedWeightSnapshot, validationLoss } from './live-metrics';
import { smallBirdNamePreset } from './live-preset';
import { LiveTrainingWorkerRuntime } from './worker-runtime';
import {
  LIVE_TRAINING_PROTOCOL_VERSION as VERSION,
  type LiveTrainingCommand,
  type LiveTrainingConfig,
  type LiveTrainingReply,
} from './worker-protocol';

const model: ModelConfig = {
  formatVersion: 1,
  vocabSize: 259,
  contextLength: 4,
  dModel: 4,
  nLayers: 2,
  nHeads: 2,
  dHead: 2,
  dMlp: 6,
  tiedEmbeddings: true,
  useBias: true,
  layerNormEpsilon: 1e-5,
  gelu: 'tanh-approximation',
};
const config: LiveTrainingConfig = {
  model,
  sequences: [
    [256, 97, 98, 99, 257],
    [256, 98, 99, 100, 257],
    [256, 99, 100, 101, 257],
  ],
  seed: 123,
  batchSize: 2,
  totalSteps: 8,
  displayEvery: 2,
  baseLearningRate: 0.001,
  warmupSteps: 1,
  optimizer: {
    beta1: 0.9,
    beta2: 0.95,
    epsilon: 1e-8,
    weightDecay: 0.1,
    gradientClipNorm: 1,
  },
};

class Harness {
  readonly replies: LiveTrainingReply[] = [];
  readonly runtime = new LiveTrainingWorkerRuntime((reply) => {
    this.replies.push(reply);
    for (const notify of this.listeners) notify(reply);
  });
  private readonly listeners = new Set<(reply: LiveTrainingReply) => void>();

  async send(
    command: LiveTrainingCommand,
    matches: (reply: LiveTrainingReply) => boolean,
  ): Promise<LiveTrainingReply> {
    const waiting = new Promise<LiveTrainingReply>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.listeners.delete(listener);
        reject(new Error(`Timed out waiting for ${command.type}.`));
      }, 2_000);
      const listener = (reply: LiveTrainingReply) => {
        if (!matches(reply)) return;
        clearTimeout(timeout);
        this.listeners.delete(listener);
        resolve(reply);
      };
      this.listeners.add(listener);
    });
    this.runtime.handle(command);
    return waiting;
  }

  command(type: Exclude<LiveTrainingCommand['type'], 'start'>, runId = 'run'): LiveTrainingCommand {
    return { version: VERSION, runId, type };
  }

  dispose(): void {
    this.runtime.dispose();
  }
}

describe('live training worker protocol', () => {
  it('steps while paused, limits progress/checkpoint cadence, and reports typed errors', async () => {
    const harness = new Harness();
    try {
      const start = await harness.send(
        { version: VERSION, runId: 'run', type: 'start', config, startPaused: true },
        (reply) => reply.type === 'status' && reply.state === 'paused',
      );
      expect(start).toMatchObject({ step: 0, totalSteps: 8 });
      await harness.send(
        harness.command('step'),
        (reply) => reply.type === 'status' && reply.state === 'paused' && reply.step === 1,
      );
      expect(harness.replies.filter((reply) => reply.type === 'progress')).toHaveLength(0);
      const first = await harness.send(
        harness.command('checkpoint'),
        (reply) => reply.type === 'checkpoint',
      );
      if (first.type !== 'checkpoint') throw new Error('Expected checkpoint reply.');
      expect(first.checkpoint.step).toBe(1);
      expect(first.checkpoint.weightBytes.byteLength).toBeGreaterThan(0);
      await harness.send(
        harness.command('step'),
        (reply) => reply.type === 'status' && reply.state === 'paused' && reply.step === 2,
      );
      const progress = harness.replies.filter((reply) => reply.type === 'progress');
      expect(progress).toHaveLength(1);
      expect(Object.keys(progress[0]).sort()).toEqual([
        'adamStep',
        'gradientNorm',
        'learningRate',
        'predictionCount',
        'runId',
        'sampleTokenIds',
        'selectedWeight',
        'step',
        'totalSteps',
        'trainLoss',
        'type',
        'validationLoss',
        'version',
      ]);
      const cadence = await harness.send(
        harness.command('checkpoint'),
        (reply) => reply.type === 'error',
      );
      expect(cadence).toMatchObject({ code: 'checkpoint-cadence', recoverable: true });
      const invalid = await harness.send(
        harness.command('resume', 'other-run'),
        (reply) => reply.type === 'error' && reply.runId === 'other-run',
      );
      expect(invalid).toMatchObject({ code: 'invalid-state' });
      harness.runtime.handle({ version: 2, runId: 'run', type: 'pause' });
      expect(harness.replies.at(-1)).toMatchObject({
        type: 'error',
        code: 'unsupported-version',
      });
      const oversized = await harness.send(
        {
          version: VERSION,
          runId: 'oversized',
          type: 'start',
          config: { ...config, selectedWeightName: 'token_embedding.weight' },
          startPaused: true,
        },
        (reply) => reply.type === 'error' && reply.runId === 'oversized',
      );
      expect(oversized).toMatchObject({ code: 'invalid-config' });
      const invalidScalar = await harness.send(
        {
          version: VERSION,
          runId: 'invalid-scalar',
          type: 'start',
          config: { ...config, selectedScalar: { name: 'blocks.0.attn.q.weight', index: -1 } },
          startPaused: true,
        },
        (reply) => reply.type === 'error' && reply.runId === 'invalid-scalar',
      );
      expect(invalidScalar).toMatchObject({ code: 'invalid-config' });
    } finally {
      harness.dispose();
    }
  });

  it('resumes an exact checkpoint without changing subsequent weights or optimizer state', async () => {
    const first = new Harness();
    const resumed = new Harness();
    const baseline = new Harness();
    try {
      for (const harness of [first, baseline]) {
        await harness.send(
          { version: VERSION, runId: 'run', type: 'start', config, startPaused: true },
          (reply) => reply.type === 'status' && reply.state === 'paused',
        );
      }
      for (let step = 1; step <= 4; step += 1) {
        await first.send(
          first.command('step'),
          (reply) => reply.type === 'status' && reply.state === 'paused' && reply.step === step,
        );
      }
      const saved = await first.send(
        first.command('checkpoint'),
        (reply) => reply.type === 'checkpoint',
      );
      if (saved.type !== 'checkpoint') throw new Error('Expected saved checkpoint.');
      await resumed.send(
        {
          version: VERSION,
          runId: 'run',
          type: 'start',
          config,
          startPaused: true,
          resumeFrom: saved.checkpoint,
        },
        (reply) => reply.type === 'status' && reply.state === 'paused' && reply.step === 4,
      );
      for (let step = 1; step <= 6; step += 1) {
        await baseline.send(
          baseline.command('step'),
          (reply) => reply.type === 'status' && reply.state === 'paused' && reply.step === step,
        );
      }
      for (let step = 5; step <= 6; step += 1) {
        await resumed.send(
          resumed.command('step'),
          (reply) => reply.type === 'status' && reply.state === 'paused' && reply.step === step,
        );
      }
      const baselineSnapshot = await baseline.send(
        baseline.command('checkpoint'),
        (reply) => reply.type === 'checkpoint',
      );
      const resumedSnapshot = await resumed.send(
        resumed.command('checkpoint'),
        (reply) => reply.type === 'checkpoint',
      );
      if (baselineSnapshot.type !== 'checkpoint' || resumedSnapshot.type !== 'checkpoint') {
        throw new Error('Expected checkpoints after six steps.');
      }
      expect(resumedSnapshot.checkpoint).toEqual(baselineSnapshot.checkpoint);
      const mismatch = await resumed.send(
        {
          version: VERSION,
          runId: 'new-run',
          type: 'start',
          config: { ...config, seed: 124 },
          startPaused: true,
          resumeFrom: saved.checkpoint,
        },
        (reply) => reply.type === 'error' && reply.runId === 'new-run',
      );
      expect(mismatch).toMatchObject({ code: 'invalid-config' });
      const wrongCursor = await resumed.send(
        {
          version: VERSION,
          runId: 'cursor-run',
          type: 'start',
          config,
          startPaused: true,
          resumeFrom: { ...saved.checkpoint, epoch: saved.checkpoint.epoch + 1 },
        },
        (reply) => reply.type === 'error' && reply.runId === 'cursor-run',
      );
      expect(wrongCursor).toMatchObject({ code: 'invalid-config' });
    } finally {
      first.dispose();
      resumed.dispose();
      baseline.dispose();
    }
  });

  it('cancels a running stream within one additional batch', async () => {
    const harness = new Harness();
    try {
      await harness.send(
        { version: VERSION, runId: 'run', type: 'start', config: { ...config, displayEvery: 1 } },
        (reply) => reply.type === 'progress' && reply.step === 2,
      );
      const cancelled = await harness.send(
        harness.command('cancel'),
        (reply) => reply.type === 'status' && reply.state === 'cancelled',
      );
      if (cancelled.type !== 'status') throw new Error('Expected cancelled status.');
      expect(cancelled.step).toBeLessThanOrEqual(3);
      const stepAtCancel = cancelled.step;
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(harness.replies.filter((reply) => reply.type === 'progress').at(-1)?.step).toBe(
        stepAtCancel,
      );
    } finally {
      harness.dispose();
    }
  });

  it('reports validation, seeded sample, and exact selected weights only at display steps', async () => {
    const harness = new Harness();
    const preset = { ...smallBirdNamePreset(2), totalSteps: 6, warmupSteps: 1, sampleNewTokens: 5 };
    try {
      await harness.send(
        { version: VERSION, runId: 'run', type: 'start', config: preset, startPaused: true },
        (reply) => reply.type === 'status' && reply.state === 'paused',
      );
      for (const step of [1, 2]) {
        await harness.send(
          harness.command('step'),
          (reply) => reply.type === 'status' && reply.state === 'paused' && reply.step === step,
        );
      }
      const reports = harness.replies.filter((reply) => reply.type === 'progress');
      expect(reports).toHaveLength(1);
      const inspected = harness.replies.filter((reply) => reply.type === 'adam-step');
      expect(inspected).toHaveLength(2);
      const report = reports[0];
      if (report.type !== 'progress') throw new Error('Expected progress.');
      const saved = await harness.send(
        harness.command('checkpoint'),
        (reply) => reply.type === 'checkpoint',
      );
      if (saved.type !== 'checkpoint') throw new Error('Expected checkpoint.');
      const registry = decodeCheckpointWeights(
        preset.model,
        saved.checkpoint.weightIndex,
        saved.checkpoint.weightBytes,
      );
      expect(report.validationLoss).toBe(
        validationLoss(
          preset.validationSequences ?? [],
          registry,
          preset.model,
          preset.batchSize,
          preset.seed,
        ),
      );
      expect(report.sampleTokenIds).toEqual(
        trainingSample(preset.samplePromptIds ?? [], 5, registry, preset.model, preset.seed, 2),
      );
      expect(report.selectedWeight).toEqual(
        selectedWeightSnapshot(registry, preset.selectedWeightName ?? ''),
      );
      expect(report.selectedWeight?.histogramCounts.reduce((sum, count) => sum + count, 0)).toBe(
        report.selectedWeight?.values.length,
      );
      const trace = inspected[1].type === 'adam-step' ? inspected[1].trace : null;
      if (!trace) throw new Error('Expected Adam scalar trace.');
      expect(trace.step).toBe(2);
      expect(report.adamStep).toEqual(trace);
      expect(trace.valueAfter).toBe(registry.get(trace.name).value.data[trace.index]);
      expect(trace.firstMoment).toBe(
        saved.checkpoint.optimizer.moments.find((moment) => moment.name === trace.name)?.first[
          trace.index
        ],
      );
    } finally {
      harness.dispose();
    }
  });

  it('keeps final weights identical when display cadence drops visual reports', async () => {
    const fast = new Harness();
    const slow = new Harness();
    const base = { ...smallBirdNamePreset(), totalSteps: 4, warmupSteps: 1, sampleNewTokens: 4 };
    try {
      for (const [harness, displayEvery] of [
        [fast, 1],
        [slow, 3],
      ] as const) {
        await harness.send(
          {
            version: VERSION,
            runId: 'run',
            type: 'start',
            config: { ...base, displayEvery },
            startPaused: true,
          },
          (reply) => reply.type === 'status' && reply.state === 'paused',
        );
        for (const step of [1, 2, 3, 4]) {
          await harness.send(
            harness.command('step'),
            (reply) =>
              reply.type === 'status' &&
              reply.step === step &&
              reply.state === (step === 4 ? 'completed' : 'paused'),
          );
        }
      }
      const fastSnapshot = await fast.send(
        fast.command('checkpoint'),
        (reply) => reply.type === 'checkpoint',
      );
      const slowSnapshot = await slow.send(
        slow.command('checkpoint'),
        (reply) => reply.type === 'checkpoint',
      );
      if (fastSnapshot.type !== 'checkpoint' || slowSnapshot.type !== 'checkpoint') {
        throw new Error('Expected final checkpoints.');
      }
      expect(fastSnapshot.checkpoint.weightBytes).toEqual(slowSnapshot.checkpoint.weightBytes);
      expect(fastSnapshot.checkpoint.optimizer).toEqual(slowSnapshot.checkpoint.optimizer);
      expect(fast.replies.filter((reply) => reply.type === 'progress')).toHaveLength(4);
      expect(slow.replies.filter((reply) => reply.type === 'progress')).toHaveLength(2);
    } finally {
      fast.dispose();
      slow.dispose();
    }
  });
});
