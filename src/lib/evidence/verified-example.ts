import type { InspectionModel } from '../data/inspection';
import { generateReplay, type GenerationReplay } from '../generation/replay';
import { decodeText, encodeText } from '../tokenizer/bpe';
import { KESTREL_EXAMPLE } from './saved-example';

export interface VerifiedExampleReplay {
  readonly replay: GenerationReplay;
  readonly focusProbability: number;
  readonly focusRank: number;
}

/** Replays the saved example and refuses to show it with a different model or output. */
export function replayKestrelExample(inspection: InspectionModel): VerifiedExampleReplay {
  if (
    inspection.checkpointId !== KESTREL_EXAMPLE.checkpointId ||
    inspection.config.trainingStep !== KESTREL_EXAMPLE.trainingStep ||
    inspection.config.corpusSha256 !== KESTREL_EXAMPLE.corpusSha256 ||
    inspection.config.tokenizerSha256 !== KESTREL_EXAMPLE.tokenizerSha256
  ) {
    throw new Error('Select the verified ready-v1 checkpoint for this example.');
  }
  const promptIds = encodeText(KESTREL_EXAMPLE.prompt, inspection.tokenizer, { bos: true });
  const replay = generateReplay(
    promptIds,
    KESTREL_EXAMPLE.seed,
    KESTREL_EXAMPLE.controls,
    KESTREL_EXAMPLE.count,
    inspection.registry,
    inspection.config.model,
  );
  if (
    replay.generatedIds.length !== KESTREL_EXAMPLE.generatedIds.length ||
    replay.generatedIds.some((id, index) => id !== KESTREL_EXAMPLE.generatedIds[index]) ||
    decodeText(replay.generatedIds, inspection.tokenizer) !== KESTREL_EXAMPLE.continuation ||
    replay.generatedIds[KESTREL_EXAMPLE.focusStep] !== KESTREL_EXAMPLE.focusTokenId ||
    decodeText([KESTREL_EXAMPLE.focusTokenId], inspection.tokenizer) !==
      KESTREL_EXAMPLE.focusTokenText
  ) {
    throw new Error('The checkpoint did not reproduce the saved token sequence.');
  }
  const focus = replay.steps[KESTREL_EXAMPLE.focusStep];
  const probability = focus.distribution.probabilities[KESTREL_EXAMPLE.focusTokenId];
  if (Math.abs(probability - KESTREL_EXAMPLE.focusProbability) > 1e-12) {
    throw new Error('The checkpoint did not reproduce the saved token probability.');
  }
  return {
    replay,
    focusProbability: probability,
    focusRank: focus.distribution.rankOrder.indexOf(KESTREL_EXAMPLE.focusTokenId) + 1,
  };
}
