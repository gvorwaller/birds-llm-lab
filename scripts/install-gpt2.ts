import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { AutoModelForCausalLM, AutoTokenizer, env } from '@huggingface/transformers';
import { gpt2CacheDirectory, gpt2CacheStatus, GPT2_MODEL_ID } from '../server/gpt2/companion.js';

const projectDirectory = resolve(fileURLToPath(new URL('..', import.meta.url)));
env.cacheDir = gpt2CacheDirectory(projectDirectory);
env.allowRemoteModels = true;

console.log(`Installing ${GPT2_MODEL_ID} into ${env.cacheDir}`);
await AutoTokenizer.from_pretrained(GPT2_MODEL_ID);
await AutoModelForCausalLM.from_pretrained(GPT2_MODEL_ID, { dtype: 'fp32' });

// Prove that the same model can be opened from disk with network access disabled.
env.allowRemoteModels = false;
const tokenizer = await AutoTokenizer.from_pretrained(GPT2_MODEL_ID);
const model = await AutoModelForCausalLM.from_pretrained(GPT2_MODEL_ID, { dtype: 'fp32' });
const output = await model(await tokenizer('The osprey dives'));
if (output.logits.dims[2] !== 50_257) {
  throw new Error(`Unexpected GPT-2 vocabulary size: ${output.logits.dims[2]}`);
}
const status = await gpt2CacheStatus(projectDirectory);
if (!status.installed) throw new Error('GPT-2 cache is incomplete.');
console.log(`GPT-2 ready offline: ${(status.bytes / 1024 / 1024).toFixed(1)} MiB cached.`);
