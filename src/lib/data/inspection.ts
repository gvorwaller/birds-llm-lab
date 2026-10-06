import { getActiveCheckpointForInspection } from './api';
import { parseCheckpointConfig, parseTokenizerArtifact } from './schemas';
import { decodeCheckpointWeights } from '../model/checkpoint';
import type { ParameterRegistry } from '../model/parameters';
import type { CheckpointConfig, TokenizerArtifact, WeightIndexArtifact } from './schemas';

export interface InspectionModel {
  readonly checkpointId: string;
  readonly config: CheckpointConfig;
  readonly tokenizer: TokenizerArtifact;
  readonly weightIndex: WeightIndexArtifact;
  readonly registry: ParameterRegistry;
}

function decodeBase64(value: string): Uint8Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

export async function loadInspectionModel(): Promise<InspectionModel> {
  const bundle = await getActiveCheckpointForInspection();
  const config = parseCheckpointConfig(bundle.config);
  const tokenizer = parseTokenizerArtifact(bundle.tokenizer);
  const registry = decodeCheckpointWeights(
    config.model,
    bundle.weightIndex,
    decodeBase64(bundle.weightsBase64),
  );
  return {
    checkpointId: bundle.checkpointId,
    config,
    tokenizer,
    weightIndex: bundle.weightIndex,
    registry,
  };
}
