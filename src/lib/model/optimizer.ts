import { assertFinite } from '../math/tensor';
import type { ParameterRegistry } from './parameters';

export interface AdamWHyperparameters {
  readonly beta1: number;
  readonly beta2: number;
  readonly epsilon: number;
  readonly weightDecay: number;
  readonly gradientClipNorm: number;
}

export interface LearningRateSchedule {
  readonly baseLearningRate: number;
  readonly warmupSteps: number;
  readonly totalSteps: number;
}

export interface OptimizerStepResult {
  readonly step: number;
  readonly learningRate: number;
  readonly gradientNorm: number;
  readonly clipScale: number;
  readonly scalarTrace: AdamWScalarTrace | null;
}

export interface AdamWScalarSelection {
  readonly name: string;
  readonly index: number;
}

export interface AdamWScalarTrace extends AdamWScalarSelection {
  readonly step: number;
  readonly valueBefore: number;
  readonly rawGradient: number;
  readonly clipScale: number;
  readonly clippedGradient: number;
  readonly firstMomentBefore: number;
  readonly secondMomentBefore: number;
  readonly firstMoment: number;
  readonly secondMoment: number;
  readonly firstBiasCorrection: number;
  readonly secondBiasCorrection: number;
  readonly firstEstimate: number;
  readonly secondEstimate: number;
  readonly adaptiveTerm: number;
  readonly decayTerm: number;
  readonly learningRate: number;
  readonly valueAfter: number;
}

export interface AdamWMomentSnapshot {
  readonly name: string;
  readonly first: Float32Array;
  readonly second: Float32Array;
}

export interface AdamWSnapshot {
  readonly step: number;
  readonly moments: AdamWMomentSnapshot[];
}

interface AdamWMoments {
  readonly first: Float32Array;
  readonly second: Float32Array;
}

function validateHyperparameters(hyperparameters: AdamWHyperparameters): void {
  if (!(hyperparameters.beta1 > 0 && hyperparameters.beta1 < 1)) {
    throw new Error('AdamW beta1 must be strictly between 0 and 1.');
  }
  if (!(hyperparameters.beta2 > 0 && hyperparameters.beta2 < 1)) {
    throw new Error('AdamW beta2 must be strictly between 0 and 1.');
  }
  if (!(hyperparameters.epsilon > 0) || !Number.isFinite(hyperparameters.epsilon)) {
    throw new Error('AdamW epsilon must be positive and finite.');
  }
  if (hyperparameters.weightDecay < 0 || !Number.isFinite(hyperparameters.weightDecay)) {
    throw new Error('AdamW weight decay must be non-negative and finite.');
  }
  if (
    !(hyperparameters.gradientClipNorm > 0) ||
    !Number.isFinite(hyperparameters.gradientClipNorm)
  ) {
    throw new Error('Gradient clip norm must be positive and finite.');
  }
}

export function learningRateAtStep(step: number, schedule: LearningRateSchedule): number {
  if (!Number.isInteger(step) || step < 0 || step >= schedule.totalSteps) {
    throw new Error(`Schedule step must be an integer from 0 to ${schedule.totalSteps - 1}.`);
  }
  if (!(schedule.baseLearningRate > 0) || !Number.isFinite(schedule.baseLearningRate)) {
    throw new Error('Base learning rate must be positive and finite.');
  }
  if (
    !Number.isInteger(schedule.warmupSteps) ||
    schedule.warmupSteps < 0 ||
    schedule.warmupSteps >= schedule.totalSteps
  ) {
    throw new Error('Warmup steps must be an integer smaller than total steps.');
  }
  if (!Number.isInteger(schedule.totalSteps) || schedule.totalSteps < 1) {
    throw new Error('Total steps must be a positive integer.');
  }
  if (schedule.totalSteps === 1) return schedule.baseLearningRate;
  if (step === schedule.totalSteps - 1) return 0;
  if (schedule.warmupSteps > 0 && step < schedule.warmupSteps) {
    return (schedule.baseLearningRate * (step + 1)) / schedule.warmupSteps;
  }
  const decayStart = schedule.warmupSteps;
  const decaySpan = schedule.totalSteps - 1 - decayStart;
  const progress = decaySpan === 0 ? 1 : (step - decayStart) / decaySpan;
  return schedule.baseLearningRate * 0.5 * (1 + Math.cos(Math.PI * progress));
}

export class AdamWOptimizer {
  private readonly moments = new Map<string, AdamWMoments>();
  private completedSteps = 0;

  constructor(readonly hyperparameters: AdamWHyperparameters) {
    validateHyperparameters(hyperparameters);
  }

  get stepCount(): number {
    return this.completedSteps;
  }

  step(
    registry: ParameterRegistry,
    learningRate: number,
    inspect?: AdamWScalarSelection,
  ): OptimizerStepResult {
    if (learningRate < 0 || !Number.isFinite(learningRate)) {
      throw new Error('Optimizer learning rate must be non-negative and finite.');
    }
    let squaredNorm = 0;
    for (const parameter of registry) {
      assertFinite(parameter.gradient.data, `${parameter.name} gradient`);
      for (const gradient of parameter.gradient.data) squaredNorm += gradient * gradient;
    }
    const gradientNorm = Math.sqrt(squaredNorm);
    const clipScale =
      gradientNorm > this.hyperparameters.gradientClipNorm
        ? this.hyperparameters.gradientClipNorm / gradientNorm
        : 1;
    const updateNumber = this.completedSteps + 1;
    const firstCorrection = 1 - this.hyperparameters.beta1 ** updateNumber;
    const secondCorrection = 1 - this.hyperparameters.beta2 ** updateNumber;
    if (inspect) {
      const parameter = registry.get(inspect.name);
      if (
        !Number.isSafeInteger(inspect.index) ||
        inspect.index < 0 ||
        inspect.index >= parameter.value.data.length
      ) {
        throw new Error(`AdamW scalar index is outside ${inspect.name}.`);
      }
    }
    let scalarTrace: AdamWScalarTrace | null = null;

    for (const parameter of registry) {
      let state = this.moments.get(parameter.name);
      if (!state) {
        state = {
          first: new Float32Array(parameter.value.data.length),
          second: new Float32Array(parameter.value.data.length),
        };
        this.moments.set(parameter.name, state);
      } else if (state.first.length !== parameter.value.data.length) {
        throw new Error(`AdamW state shape changed for ${parameter.name}.`);
      }
      for (let index = 0; index < parameter.value.data.length; index += 1) {
        const selected = inspect?.name === parameter.name && inspect.index === index;
        const valueBefore = selected ? parameter.value.data[index] : 0;
        const rawGradient = parameter.gradient.data[index];
        const firstMomentBefore = selected ? state.first[index] : 0;
        const secondMomentBefore = selected ? state.second[index] : 0;
        const gradient = rawGradient * clipScale;
        state.first[index] =
          this.hyperparameters.beta1 * state.first[index] +
          (1 - this.hyperparameters.beta1) * gradient;
        state.second[index] =
          this.hyperparameters.beta2 * state.second[index] +
          (1 - this.hyperparameters.beta2) * gradient * gradient;
        const firstEstimate = state.first[index] / firstCorrection;
        const secondEstimate = state.second[index] / secondCorrection;
        const adaptive = firstEstimate / (Math.sqrt(secondEstimate) + this.hyperparameters.epsilon);
        const decay = parameter.decay
          ? this.hyperparameters.weightDecay * parameter.value.data[index]
          : 0;
        parameter.value.data[index] -= learningRate * (adaptive + decay);
        if (selected) {
          scalarTrace = {
            name: parameter.name,
            index,
            step: updateNumber,
            valueBefore,
            rawGradient,
            clipScale,
            clippedGradient: gradient,
            firstMomentBefore,
            secondMomentBefore,
            firstMoment: state.first[index],
            secondMoment: state.second[index],
            firstBiasCorrection: firstCorrection,
            secondBiasCorrection: secondCorrection,
            firstEstimate,
            secondEstimate,
            adaptiveTerm: adaptive,
            decayTerm: decay,
            learningRate,
            valueAfter: parameter.value.data[index],
          };
        }
      }
      assertFinite(parameter.value.data, `${parameter.name} updated value`);
    }
    this.completedSteps = updateNumber;
    return { step: updateNumber, learningRate, gradientNorm, clipScale, scalarTrace };
  }

  stepWithSchedule(
    registry: ParameterRegistry,
    schedule: LearningRateSchedule,
  ): OptimizerStepResult {
    const learningRate = learningRateAtStep(this.completedSteps, schedule);
    return this.step(registry, learningRate);
  }

  restore(snapshot: AdamWSnapshot, registry: ParameterRegistry): void {
    if (!Number.isInteger(snapshot.step) || snapshot.step < 0) {
      throw new Error('AdamW snapshot step must be a non-negative integer.');
    }
    const expected = new Map(registry.entries().map((parameter) => [parameter.name, parameter]));
    const restored = new Map<string, AdamWMoments>();
    for (const moment of snapshot.moments) {
      const parameter = expected.get(moment.name);
      if (!parameter) throw new Error(`AdamW snapshot contains unknown parameter ${moment.name}.`);
      if (restored.has(moment.name)) {
        throw new Error(`AdamW snapshot repeats parameter ${moment.name}.`);
      }
      if (
        moment.first.length !== parameter.value.data.length ||
        moment.second.length !== parameter.value.data.length
      ) {
        throw new Error(`AdamW snapshot shape mismatch for ${moment.name}.`);
      }
      assertFinite(moment.first, `${moment.name} first moment`);
      assertFinite(moment.second, `${moment.name} second moment`);
      restored.set(moment.name, {
        first: moment.first.slice(),
        second: moment.second.slice(),
      });
    }
    if (snapshot.step > 0) {
      for (const name of expected.keys()) {
        if (!restored.has(name)) throw new Error(`AdamW snapshot is missing parameter ${name}.`);
      }
    } else if (restored.size > 0) {
      throw new Error('AdamW step-zero snapshot must not contain moments.');
    }
    this.moments.clear();
    for (const [name, moments] of restored) this.moments.set(name, moments);
    this.completedSteps = snapshot.step;
  }

  snapshot(): AdamWSnapshot {
    return {
      step: this.completedSteps,
      moments: [...this.moments.entries()].map(([name, state]) => ({
        name,
        first: state.first.slice(),
        second: state.second.slice(),
      })),
    };
  }
}
