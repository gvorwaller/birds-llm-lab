import {
  runTinyOverfit,
  TINY_OVERFIT_LEARNING_RATE,
  TINY_OVERFIT_STEPS,
} from '../src/lib/training/learning-gates';

const first = runTinyOverfit();
const second = runTinyOverfit();
if (JSON.stringify(first) !== JSON.stringify(second)) {
  throw new Error('Tiny overfit gate is not deterministic for a fixed seed.');
}
if (first.lossReduction < 0.8) {
  throw new Error(
    `Tiny overfit loss fell only ${(first.lossReduction * 100).toFixed(2)}%; required at least 80%.`,
  );
}
if (!first.predictions.every((token, index) => token === first.targets[index])) {
  throw new Error(
    `Tiny overfit greedy predictions ${first.predictions.join(',')} do not match ${first.targets.join(',')}.`,
  );
}

console.log(
  JSON.stringify(
    {
      status: 'passed',
      learningRate: TINY_OVERFIT_LEARNING_RATE,
      steps: TINY_OVERFIT_STEPS,
      initialLoss: first.initialLoss,
      finalLoss: first.finalLoss,
      lossReductionPercent: first.lossReduction * 100,
      predictions: first.predictions,
      targets: first.targets,
      loggedLosses: first.loggedLosses,
      deterministicRepeat: true,
    },
    null,
    2,
  ),
);
