import { stat } from 'node:fs/promises';
import { join } from 'node:path';

export const GPT2_MODEL_ID = 'Xenova/gpt2';
const MODEL_FILES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'onnx/model.onnx',
] as const;

export interface Gpt2Token {
  id: number;
  text: string;
}

export interface Gpt2Prediction extends Gpt2Token {
  probability: number;
}

export interface Gpt2Analysis {
  model: string;
  prompt: string;
  tokens: Gpt2Token[];
  predictions: Gpt2Prediction[];
  continuation: string;
  generatedTokens: Gpt2Token[];
}

export function gpt2CacheDirectory(projectDirectory: string): string {
  return join(projectDirectory, '.cache', 'gpt2');
}

export async function gpt2CacheStatus(projectDirectory: string): Promise<{
  installed: boolean;
  model: string;
  bytes: number;
}> {
  const root = join(gpt2CacheDirectory(projectDirectory), GPT2_MODEL_ID);
  try {
    const files = await Promise.all(MODEL_FILES.map((name) => stat(join(root, name))));
    if (files.some((file) => !file.isFile() || file.size === 0)) throw new Error('incomplete');
    return {
      installed: true,
      model: GPT2_MODEL_ID,
      bytes: files.reduce((total, file) => total + file.size, 0),
    };
  } catch {
    return { installed: false, model: GPT2_MODEL_ID, bytes: 0 };
  }
}

async function loadModel(projectDirectory: string) {
  const { env, AutoTokenizer, AutoModelForCausalLM } = await import('@huggingface/transformers');
  env.cacheDir = gpt2CacheDirectory(projectDirectory);
  env.allowRemoteModels = false;
  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(GPT2_MODEL_ID),
    AutoModelForCausalLM.from_pretrained(GPT2_MODEL_ID, { dtype: 'fp32' }),
  ]);
  return { tokenizer, model };
}

function topPredictions(
  logits: Float32Array,
  count: number,
): { id: number; probability: number }[] {
  const maximum = Math.max(...logits);
  const weights = new Float64Array(logits.length);
  let denominator = 0;
  for (let id = 0; id < logits.length; id += 1) {
    const weight = Math.exp(logits[id] - maximum);
    weights[id] = weight;
    denominator += weight;
  }
  return Array.from({ length: logits.length }, (_, id) => id)
    .sort((left, right) => weights[right] - weights[left] || left - right)
    .slice(0, count)
    .map((id) => ({ id, probability: weights[id] / denominator }));
}

export class Gpt2Companion {
  private loading: ReturnType<typeof loadModel> | null = null;
  private busy = false;

  constructor(private readonly projectDirectory: string) {}

  status() {
    return gpt2CacheStatus(this.projectDirectory);
  }

  async analyze(prompt: string): Promise<Gpt2Analysis> {
    if (prompt.length < 1 || prompt.length > 300 || !prompt.trim()) {
      throw new Error('Prompt must contain 1 to 300 characters of text.');
    }
    if (!(await this.status()).installed) {
      throw new Error('GPT-2 is not installed in the local cache.');
    }
    if (this.busy) throw new Error('GPT-2 is already analyzing another prompt.');
    this.busy = true;
    try {
      this.loading ??= loadModel(this.projectDirectory).catch((error: unknown) => {
        this.loading = null;
        throw error;
      });
      const { tokenizer, model } = await this.loading;
      const input = await tokenizer(prompt);
      const ids = Array.from(input.input_ids.data, Number);
      if (ids.length > 128) throw new Error('Prompt exceeds the 128-token comparison limit.');
      const output = await model(input);
      const vocabularySize = output.logits.dims[2];
      const offset = (ids.length - 1) * vocabularySize;
      const logits = output.logits.data.slice(offset, offset + vocabularySize) as Float32Array;
      const predicted = topPredictions(logits, 10);
      const generated = await model.generate({ ...input, max_new_tokens: 20, do_sample: false });
      if (!('data' in generated))
        throw new Error('GPT-2 returned an unexpected generation format.');
      const generatedIds = Array.from(generated.data, Number).slice(ids.length);
      const token = (id: number): Gpt2Token => ({ id, text: tokenizer.decode([id]) });
      return {
        model: GPT2_MODEL_ID,
        prompt,
        tokens: ids.map(token),
        predictions: predicted.map(({ id, probability }) => ({ ...token(id), probability })),
        continuation: tokenizer.decode(generatedIds, { skip_special_tokens: true }),
        generatedTokens: generatedIds.map(token),
      };
    } finally {
      this.busy = false;
    }
  }
}
