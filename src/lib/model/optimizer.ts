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
}

export interface AdamWMomentSnapshot {
  readonly name: string;
  readonly first: Float32Array;
  readonly second: Float32Array;
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

  step(registry: ParameterRegistry, learningRate: number): OptimizerStepResult {
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
        const gradient = parameter.gradient.data[index] * clipScale;
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
      }
      assertFinite(parameter.value.data, `${parameter.name} updated value`);
    }
    this.completedSteps = updateNumber;
    return { step: updateNumber, learningRate, gradientNorm, clipScale };
  }

  stepWithSchedule(
    registry: ParameterRegistry,
    schedule: LearningRateSchedule,
  ): OptimizerStepResult {
    const learningRate = learningRateAtStep(this.completedSteps, schedule);
    return this.step(registry, learningRate);
  }

  snapshot(): { step: number; moments: AdamWMomentSnapshot[] } {
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
