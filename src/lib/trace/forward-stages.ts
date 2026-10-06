export interface ForwardStage {
  readonly name: string;
  readonly title: string;
  readonly description: string;
}

export function forwardStages(layerCount: number): ForwardStage[] {
  if (!Number.isInteger(layerCount) || layerCount < 1) {
    throw new Error('Forward stages require at least one layer.');
  }
  const stages: ForwardStage[] = [
    {
      name: 'embedding.token',
      title: 'Token embeddings',
      description: 'Look up one learned row for each token id.',
    },
    {
      name: 'embedding.position',
      title: 'Position embeddings',
      description: 'Look up a separate learned row for each absolute position.',
    },
    {
      name: 'embedding.sum',
      title: 'Starting residual stream',
      description: 'Add token and position vectors, component by component.',
    },
  ];
  for (let layer = 0; layer < layerCount; layer += 1) {
    const prefix = `blocks.${layer}`;
    stages.push(
      {
        name: `${prefix}.ln1`,
        title: `Layer ${layer} · attention normalization`,
        description: 'Center and scale each token vector before attention.',
      },
      {
        name: `${prefix}.attn.q`,
        title: `Layer ${layer} · query projection`,
        description: 'Multiply normalized vectors by the learned query matrix.',
      },
      {
        name: `${prefix}.attn.k`,
        title: `Layer ${layer} · key projection`,
        description: 'Multiply normalized vectors by the learned key matrix.',
      },
      {
        name: `${prefix}.attn.v`,
        title: `Layer ${layer} · value projection`,
        description: 'Multiply normalized vectors by the learned value matrix.',
      },
      {
        name: `${prefix}.attn.rawScores`,
        title: `Layer ${layer} · raw attention scores`,
        description: 'Dot each query with every key before scaling.',
      },
      {
        name: `${prefix}.attn.scores`,
        title: `Layer ${layer} · scaled attention scores`,
        description: 'Divide the query-key dot product by the square root of head width.',
      },
      {
        name: `${prefix}.attn.mask`,
        title: `Layer ${layer} · causal mask`,
        description: 'A query can use only itself and earlier unpadded tokens.',
      },
      {
        name: `${prefix}.attn.probabilities`,
        title: `Layer ${layer} · attention probabilities`,
        description: 'Mask future positions, then normalize each row with softmax.',
      },
      {
        name: `${prefix}.attn.weightedValue`,
        title: `Layer ${layer} · weighted values`,
        description: 'Average value vectors using attention probabilities.',
      },
      {
        name: `${prefix}.attn.merged`,
        title: `Layer ${layer} · merge heads`,
        description: 'Place each head’s output channels back into one model-width vector.',
      },
      {
        name: `${prefix}.attn.projection`,
        title: `Layer ${layer} · attention output projection`,
        description: 'Apply a learned projection to the merged head outputs.',
      },
      {
        name: `${prefix}.residual.attention`,
        title: `Layer ${layer} · attention residual`,
        description: 'Add projected attention to the incoming residual stream.',
      },
      {
        name: `${prefix}.ln2`,
        title: `Layer ${layer} · MLP normalization`,
        description: 'Center and scale the residual stream before the MLP.',
      },
      {
        name: `${prefix}.mlp.preactivation`,
        title: `Layer ${layer} · MLP expansion`,
        description: 'Project from model width to the wider hidden width.',
      },
      {
        name: `${prefix}.mlp.gelu`,
        title: `Layer ${layer} · GELU`,
        description: 'Apply the GELU nonlinearity to each expanded channel.',
      },
      {
        name: `${prefix}.mlp.output`,
        title: `Layer ${layer} · MLP contraction`,
        description: 'Project the activated wide vector back to model width.',
      },
      {
        name: `${prefix}.residual.mlp`,
        title: `Layer ${layer} · MLP residual`,
        description: 'Add the MLP output to the residual stream.',
      },
    );
  }
  stages.push(
    {
      name: 'final_ln',
      title: 'Final normalization',
      description: 'Normalize the last residual stream before token scoring.',
    },
    {
      name: 'logits',
      title: 'Top five next-token logits',
      description: 'Score every vocabulary token from the final prompt position.',
    },
  );
  return stages;
}
