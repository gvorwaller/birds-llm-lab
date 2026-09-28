export interface LabRoute {
  path: string;
  label: string;
  eyebrow: string;
  title: string;
  summary: string;
  milestone: string;
}

export const routes: readonly LabRoute[] = [
  {
    path: '/',
    label: 'Overview',
    eyebrow: 'Laboratory notebook',
    title: 'See every number inside a tiny language model.',
    summary:
      'A local, inspectable transformer trained on a read-only export of bird reference text. No hidden APIs and no production data.',
    milestone: 'M0',
  },
  {
    path: '/data',
    label: 'Data',
    eyebrow: 'Corpus bench',
    title: 'Build and inspect the local teaching corpus.',
    summary:
      'Export bird articles from the local test database, inspect the manifest, and later train or select checkpoints.',
    milestone: 'M0',
  },
  {
    path: '/tokenizer',
    label: 'Tokenizer',
    eyebrow: 'Token bench',
    title: 'Watch text become byte-pair tokens.',
    summary: 'Replay deterministic merges and inspect every token id and byte sequence.',
    milestone: 'M1',
  },
  {
    path: '/forward-pass',
    label: 'Forward pass',
    eyebrow: 'Model bench',
    title: 'Trace one prompt through every layer.',
    summary: 'Inspect embeddings, attention, residual updates, and logits using real values.',
    milestone: 'M3',
  },
  {
    path: '/generation',
    label: 'Generation',
    eyebrow: 'Sampling bench',
    title: 'See one next-token decision at a time.',
    summary: 'Adjust sampling controls and follow the seeded draw through the probability mass.',
    milestone: 'M4',
  },
  {
    path: '/training',
    label: 'Training',
    eyebrow: 'Learning bench',
    title: 'Slow down one gradient update.',
    summary: 'Watch loss, gradients, optimizer state, and selected weights change.',
    milestone: 'M5',
  },
  {
    path: '/evidence',
    label: 'Evidence',
    eyebrow: 'Evidence bench',
    title: 'Compare generated claims with corpus evidence.',
    summary: 'Separate plausible continuation mechanics from retrieved source evidence.',
    milestone: 'M6',
  },
];

export function normalizePath(pathname: string): string {
  const normalized = pathname.replace(/\/+$/, '');
  return normalized || '/';
}

export function routeForPath(pathname: string): LabRoute {
  const path = normalizePath(pathname);
  return routes.find((route) => route.path === path) ?? routes[0];
}
