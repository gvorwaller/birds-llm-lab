import type { ModelConfig } from '../data/schemas';
import { SeededRandom } from '../math/rng';
import { elementCount, tensor, type Tensor } from '../math/tensor';
import { validatedModelConfig } from './config';

export type ParameterKind =
  'embedding' | 'weight' | 'bias' | 'layer-norm-scale' | 'layer-norm-offset';

export interface ParameterSpec {
  readonly name: string;
  readonly shape: readonly number[];
  readonly kind: ParameterKind;
  readonly decay: boolean;
}

export interface Parameter extends ParameterSpec {
  readonly value: Tensor;
  readonly gradient: Tensor;
}

function spec(name: string, shape: readonly number[], kind: ParameterKind): ParameterSpec {
  return { name, shape: [...shape], kind, decay: kind === 'embedding' || kind === 'weight' };
}

export function parameterSpecs(configInput: ModelConfig): ParameterSpec[] {
  const config = validatedModelConfig(configInput);
  const specs: ParameterSpec[] = [
    spec('token_embedding.weight', [config.vocabSize, config.dModel], 'embedding'),
    spec('position_embedding.weight', [config.contextLength, config.dModel], 'embedding'),
  ];

  for (let layer = 0; layer < config.nLayers; layer += 1) {
    const prefix = `blocks.${layer}`;
    specs.push(
      spec(`${prefix}.ln1.scale`, [config.dModel], 'layer-norm-scale'),
      spec(`${prefix}.ln1.offset`, [config.dModel], 'layer-norm-offset'),
    );
    for (const projection of ['q', 'k', 'v', 'out'] as const) {
      specs.push(
        spec(`${prefix}.attn.${projection}.weight`, [config.dModel, config.dModel], 'weight'),
      );
      if (config.useBias) {
        specs.push(spec(`${prefix}.attn.${projection}.bias`, [config.dModel], 'bias'));
      }
    }
    specs.push(
      spec(`${prefix}.ln2.scale`, [config.dModel], 'layer-norm-scale'),
      spec(`${prefix}.ln2.offset`, [config.dModel], 'layer-norm-offset'),
      spec(`${prefix}.mlp.in.weight`, [config.dModel, config.dMlp], 'weight'),
    );
    if (config.useBias) specs.push(spec(`${prefix}.mlp.in.bias`, [config.dMlp], 'bias'));
    specs.push(spec(`${prefix}.mlp.out.weight`, [config.dMlp, config.dModel], 'weight'));
    if (config.useBias) specs.push(spec(`${prefix}.mlp.out.bias`, [config.dModel], 'bias'));
  }

  specs.push(
    spec('final_ln.scale', [config.dModel], 'layer-norm-scale'),
    spec('final_ln.offset', [config.dModel], 'layer-norm-offset'),
  );
  if (!config.tiedEmbeddings) {
    specs.push(spec('unembedding.weight', [config.vocabSize, config.dModel], 'weight'));
    if (config.useBias) specs.push(spec('unembedding.bias', [config.vocabSize], 'bias'));
  }
  return specs;
}

function initialValues(specification: ParameterSpec, random: SeededRandom | null): Float32Array {
  const values = new Float32Array(elementCount(specification.shape));
  if (specification.kind === 'layer-norm-scale') {
    values.fill(1);
  } else if (
    random !== null &&
    (specification.kind === 'embedding' || specification.kind === 'weight')
  ) {
    for (let index = 0; index < values.length; index += 1) values[index] = random.normal() * 0.02;
  }
  return values;
}

export class ParameterRegistry implements Iterable<Parameter> {
  private readonly ordered: Parameter[];
  private readonly named: Map<string, Parameter>;

  constructor(specifications: readonly ParameterSpec[], random: SeededRandom | null) {
    this.ordered = [];
    this.named = new Map();
    for (const specification of specifications) {
      if (this.named.has(specification.name)) {
        throw new Error(`Duplicate parameter name: ${specification.name}.`);
      }
      const shape = [...specification.shape];
      const parameter: Parameter = {
        ...specification,
        shape,
        value: tensor(initialValues(specification, random), shape, `${specification.name} value`),
        gradient: tensor(
          new Float32Array(elementCount(shape)),
          shape,
          `${specification.name} gradient`,
        ),
      };
      this.ordered.push(parameter);
      this.named.set(parameter.name, parameter);
    }
  }

  get size(): number {
    return this.ordered.length;
  }

  get elementCount(): number {
    return this.ordered.reduce((total, parameter) => total + parameter.value.data.length, 0);
  }

  entries(): readonly Parameter[] {
    return this.ordered;
  }

  get(name: string): Parameter {
    const parameter = this.named.get(name);
    if (!parameter) throw new Error(`Unknown parameter: ${name}.`);
    return parameter;
  }

  setValues(name: string, values: Float32Array | readonly number[]): void {
    const parameter = this.get(name);
    const validated = tensor(values, parameter.shape, `${name} checkpoint value`);
    parameter.value.data.set(validated.data);
  }

  zeroGradients(): void {
    for (const parameter of this.ordered) parameter.gradient.data.fill(0);
  }

  [Symbol.iterator](): Iterator<Parameter> {
    return this.ordered[Symbol.iterator]();
  }
}

export function initializeParameters(config: ModelConfig, seed: number): ParameterRegistry {
  return new ParameterRegistry(parameterSpecs(config), new SeededRandom(seed));
}

export function emptyParameters(config: ModelConfig): ParameterRegistry {
  return new ParameterRegistry(parameterSpecs(config), null);
}
