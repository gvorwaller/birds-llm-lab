import { resolve } from 'node:path';
import { loadCheckpoint } from '../server/checkpoint-store';
import { buildParameterOverview } from '../src/lib/parameters/explorer';
import { forwardStages } from '../src/lib/trace/forward-stages';

const loaded = await loadCheckpoint(resolve('checkpoints/ready-v1'));
const overview = buildParameterOverview(loaded.weightIndex, loaded.parameters, loaded.config.model);
const stages = new Set(forwardStages(loaded.config.model.nLayers).map((stage) => stage.name));
if (overview.entries.some((entry) => !stages.has(entry.traceStage))) {
  throw new Error('A registered parameter has no valid forward-trace link.');
}
if (
  overview.entries.some(
    (entry) => entry.histogram.reduce((sum, count) => sum + count, 0) !== entry.count,
  )
) {
  throw new Error('A parameter histogram does not conserve its value count.');
}
console.log(
  JSON.stringify(
    {
      status: 'passed',
      checkpoint: 'ready-v1',
      tensors: overview.tensorCount,
      values: overview.elementCount,
      expectedFromConfig: overview.expectedCount,
      groups: overview.groups.map((group) => ({
        name: group.name,
        tensors: group.entries.length,
        values: group.count,
      })),
    },
    null,
    2,
  ),
);
