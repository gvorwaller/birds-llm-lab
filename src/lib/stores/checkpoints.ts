import { writable } from 'svelte/store';
import type { CheckpointList } from '../data/api-types';

export const checkpointCatalog = writable<CheckpointList>({
  checkpoints: [],
  activeCheckpointId: null,
  recoveryWarning: null,
});
