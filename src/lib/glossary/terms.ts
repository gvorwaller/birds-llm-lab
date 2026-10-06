export interface GlossaryTerm {
  readonly id: string;
  readonly label: string;
  readonly plain: string;
  readonly formula: string | null;
  readonly detail: string;
  readonly related: readonly string[];
  readonly workbench: { readonly label: string; readonly href: string };
}

export const glossaryTerms: readonly GlossaryTerm[] = [
  {
    id: 'token',
    label: 'Token',
    plain:
      'One numbered piece of text the model can process or predict. A piece may be a byte, part of a word, or several characters.',
    formula: 'text → UTF-8 bytes → ranked BPE merges → token IDs',
    detail:
      'The tokenizer fixes the pieces before the model sees the text. A visible word can span several tokens.',
    related: ['vocabulary', 'context-window'],
    workbench: { label: 'Replay tokenization', href: '/tokenizer' },
  },
  {
    id: 'vocabulary',
    label: 'Vocabulary',
    plain: 'The complete set of token IDs the tokenizer can emit and the model can score.',
    formula: 'logits.shape = [vocabulary size]',
    detail:
      'The ready checkpoint has 1,024 token IDs, including reserved special tokens. The tokenizer page shows their stored byte sequences.',
    related: ['token', 'logit'],
    workbench: { label: 'Inspect vocabulary', href: '/tokenizer' },
  },
  {
    id: 'embedding',
    label: 'Embedding',
    plain:
      'A learned row of numbers looked up for a token or position. It becomes the model’s starting vector for that input.',
    formula: 'start[position] = token_embedding[token_id] + position_embedding[position]',
    detail:
      'Rows are stored weights. The Embeddings view compares selected trained rows with their reconstructed initial values.',
    related: ['token', 'parameter-weight', 'residual-stream'],
    workbench: { label: 'Inspect embedding rows', href: '/embeddings' },
  },
  {
    id: 'parameter-weight',
    label: 'Parameter / weight',
    plain:
      'A stored number the optimizer can change during training. Together these numbers determine the model’s calculations.',
    formula: 'parameter count = sum(product(shape) for each registered tensor)',
    detail:
      'A weight often multiplies an input; other parameters include biases and normalization scales. The explorer covers every named tensor in the checkpoint.',
    related: ['bias', 'gradient', 'matrix-multiply'],
    workbench: { label: 'Browse stored parameters', href: '/parameters' },
  },
  {
    id: 'bias',
    label: 'Bias',
    plain:
      'A learned number added after a weighted sum. It can shift an output even when the input is zero.',
    formula: 'output[j] = sum_i(input[i] × weight[i,j]) + bias[j]',
    detail:
      'This model includes biases in its projections when useBias is enabled. Bias tensors appear by name in the parameter explorer.',
    related: ['parameter-weight', 'matrix-multiply'],
    workbench: { label: 'Find bias tensors', href: '/parameters' },
  },
  {
    id: 'matrix-multiply',
    label: 'Matrix multiply',
    plain:
      'A collection of weighted sums that transforms one vector or table of numbers into another.',
    formula: 'Y[i,j] = sum_k(X[i,k] × W[k,j])',
    detail:
      'Query, key, value, attention output, and MLP projections use this operation. The forward trace exposes their terms for a selected cell.',
    related: ['dot-product', 'parameter-weight', 'mlp'],
    workbench: { label: 'Inspect a projection', href: '/forward-pass?stage=blocks.0.attn.q' },
  },
  {
    id: 'dot-product',
    label: 'Dot product',
    plain: 'Multiply matching entries in two vectors and add the products to get one number.',
    formula: 'a · b = sum_i(a[i] × b[i])',
    detail: 'Attention compares a query and key with a dot product before scaling and softmax.',
    related: ['matrix-multiply', 'attention'],
    workbench: {
      label: 'Inspect raw attention scores',
      href: '/forward-pass?stage=blocks.0.attn.rawScores',
    },
  },
  {
    id: 'attention',
    label: 'Attention',
    plain:
      'For each token position, the model computes weights over allowed earlier positions and mixes their value vectors.',
    formula: 'weights = softmax(mask(QKᵀ / √d_head)); output = weights × V',
    detail:
      '“Looks at” is a metaphor. Literally, each allowed key receives a probability and its value vector contributes that fraction to the weighted sum.',
    related: ['head', 'softmax', 'dot-product'],
    workbench: { label: 'Inspect attention probabilities', href: '/attention' },
  },
  {
    id: 'head',
    label: 'Head',
    plain: 'One parallel set of query, key, and value channels inside an attention layer.',
    formula: 'head output = softmax(mask(Q_head K_headᵀ / √d_head)) × V_head',
    detail:
      'Heads use different channel slices. Their outputs are joined and projected; the Attention view separates their measured probability maps.',
    related: ['attention', 'matrix-multiply'],
    workbench: { label: 'Compare heads', href: '/attention' },
  },
  {
    id: 'residual-stream',
    label: 'Residual stream',
    plain:
      'The running vector at each token position. A block adds its attention and MLP outputs to this vector.',
    formula: 'next = current + attention_output; later = next + mlp_output',
    detail:
      'The addition preserves a direct path for earlier values and gradients through the block.',
    related: ['attention', 'mlp', 'backpropagation'],
    workbench: {
      label: 'Trace residual additions',
      href: '/forward-pass?stage=blocks.0.residual.attention',
    },
  },
  {
    id: 'layernorm',
    label: 'LayerNorm',
    plain:
      'Normalize channels within one token vector, then apply learned scale and offset per channel.',
    formula: 'y[i] = scale[i] × (x[i] − mean(x)) / √(variance(x) + ε) + offset[i]',
    detail:
      'This model uses LayerNorm before each attention and MLP sublayer and once after the final block.',
    related: ['parameter-weight', 'residual-stream'],
    workbench: { label: 'Trace normalization', href: '/forward-pass?stage=blocks.0.ln1' },
  },
  {
    id: 'mlp',
    label: 'MLP',
    plain:
      'A small feed-forward network applied separately to each token position after attention.',
    formula: 'MLP(x) = GELU(x × W_in + b_in) × W_out + b_out',
    detail:
      'It expands the channel width, applies GELU, contracts it, and adds the result to the residual stream.',
    related: ['gelu', 'matrix-multiply', 'residual-stream'],
    workbench: {
      label: 'Trace MLP expansion',
      href: '/forward-pass?stage=blocks.0.mlp.preactivation',
    },
  },
  {
    id: 'gelu',
    label: 'GELU',
    plain: 'A smooth activation that scales each MLP value according to its magnitude and sign.',
    formula: 'GELU(x) ≈ 0.5x × [1 + tanh(√(2/π) × (x + 0.044715x³))]',
    detail: 'This app uses the displayed tanh approximation, not an exact Gaussian integral.',
    related: ['mlp'],
    workbench: { label: 'Trace GELU output', href: '/forward-pass?stage=blocks.0.mlp.gelu' },
  },
  {
    id: 'logit',
    label: 'Logit',
    plain: 'A raw score for one possible next token before probabilities are computed.',
    formula: 'logit[token] = final_vector · output_weight[token]',
    detail:
      'Higher logits generally produce higher softmax probabilities, but a logit itself is not a probability.',
    related: ['softmax', 'vocabulary'],
    workbench: { label: 'Inspect next-token logits', href: '/generation' },
  },
  {
    id: 'softmax',
    label: 'Softmax',
    plain: 'Turn a set of scores into nonnegative probabilities that add to one.',
    formula: 'softmax(z)[i] = exp(z[i] − max(z)) / sum_j exp(z[j] − max(z))',
    detail:
      'Subtracting the maximum improves numerical stability without changing the probabilities.',
    related: ['logit', 'temperature', 'attention'],
    workbench: { label: 'Inspect probabilities', href: '/generation' },
  },
  {
    id: 'temperature',
    label: 'Temperature',
    plain: 'A sampling control that changes how spread out next-token probabilities are.',
    formula: 'p[i] = softmax(logit[i] / temperature)',
    detail:
      'Lower positive temperature sharpens the distribution; higher temperature flattens it. Top-k and top-p can then remove candidates and renormalize.',
    related: ['softmax', 'logit'],
    workbench: { label: 'Change sampling controls', href: '/generation' },
  },
  {
    id: 'cross-entropy',
    label: 'Cross-entropy',
    plain: 'The penalty for assigning low probability to the actual next token in training data.',
    formula: 'cross_entropy = −log(p[correct_token])',
    detail: 'Training averages this value over predicted token positions in a batch.',
    related: ['loss', 'softmax'],
    workbench: { label: 'Inspect training loss', href: '/training' },
  },
  {
    id: 'loss',
    label: 'Loss',
    plain:
      'A number measuring prediction error for the chosen training or validation examples; lower is better on those examples.',
    formula: 'batch_loss = sum(cross_entropy at predicted positions) / position_count',
    detail:
      'The Training view plots separate training and held-out validation losses. A falling training loss alone does not establish generalization.',
    related: ['cross-entropy', 'validation-set', 'overfitting'],
    workbench: { label: 'Compare loss curves', href: '/training' },
  },
  {
    id: 'gradient',
    label: 'Gradient',
    plain:
      'The derivative of loss with respect to a parameter. It describes how a small change in that value would change loss locally.',
    formula: 'gradient[i] = ∂loss / ∂parameter[i]',
    detail:
      'Backpropagation computes these derivatives; the optimizer uses them to update stored weights.',
    related: ['backpropagation', 'adam', 'parameter-weight'],
    workbench: { label: 'Inspect one update', href: '/training' },
  },
  {
    id: 'backpropagation',
    label: 'Backpropagation',
    plain:
      'Apply the chain rule backward through the recorded operations to calculate gradients for every used parameter.',
    formula: '∂loss/∂x = (∂loss/∂y) × (∂y/∂x)',
    detail: 'It computes gradients; AdamW then decides how far to move each parameter.',
    related: ['gradient', 'adam', 'loss'],
    workbench: { label: 'Inspect training steps', href: '/training' },
  },
  {
    id: 'learning-rate',
    label: 'Learning rate',
    plain: 'A multiplier controlling the size of an optimizer update at a training step.',
    formula: 'new_weight = old_weight − learning_rate × update_direction',
    detail: 'The live training display shows its scheduled value at each reported step.',
    related: ['adam', 'gradient'],
    workbench: { label: 'Inspect learning-rate curve', href: '/training' },
  },
  {
    id: 'adam',
    label: 'Adam (AdamW here)',
    plain:
      'An optimizer that tracks moving averages of gradients and squared gradients, then scales each parameter update.',
    formula:
      'm = β₁m_prev + (1−β₁)g; v = β₂v_prev + (1−β₂)g²; AdamW also applies separate weight decay',
    detail:
      'The slowed step shows clipping, moments, bias corrections, adaptive term, decay term, and the exact stored Float32 result.',
    related: ['gradient', 'learning-rate', 'parameter-weight'],
    workbench: { label: 'Inspect the slowed AdamW step', href: '/training' },
  },
  {
    id: 'overfitting',
    label: 'Overfitting',
    plain:
      'When a model improves on training examples but fails to improve, or worsens, on held-out examples.',
    formula: null,
    detail:
      'A widening train-versus-validation loss gap can be evidence of overfitting; one gap alone is not a complete diagnosis.',
    related: ['loss', 'validation-set'],
    workbench: { label: 'Compare train and validation', href: '/training' },
  },
  {
    id: 'validation-set',
    label: 'Validation set',
    plain:
      'Held-out examples used to check a model during training without updating its weights from those examples.',
    formula: null,
    detail:
      'The exported corpus labels train, validation, and test splits. Only train examples supply optimizer updates.',
    related: ['loss', 'overfitting'],
    workbench: { label: 'Inspect corpus splits', href: '/data' },
  },
  {
    id: 'context-window',
    label: 'Context window',
    plain: 'The maximum number of token positions a model call can use at once.',
    formula: 'visible_ids = most_recent_ids up to context_length',
    detail:
      'Generation drops older IDs when the prompt and continuation exceed this model’s configured context length.',
    related: ['token', 'attention'],
    workbench: { label: 'Watch context sliding', href: '/generation' },
  },
  {
    id: 'hallucination',
    label: 'Hallucination',
    plain:
      'A fluent generated statement that is unsupported or wrong for the question or reference being checked.',
    formula: 'next_token = sample(probabilities from current context)',
    detail:
      '“Hallucination” is a metaphor. Literally, the model samples tokens from learned probabilities; it does not consult a source of truth unless the application supplies and checks one. The Evidence demo labels its causal account as interpretation.',
    related: ['temperature', 'validation-set'],
    workbench: { label: 'Inspect the verified example', href: '/evidence' },
  },
];

export const glossaryById = new Map(glossaryTerms.map((term) => [term.id, term]));
