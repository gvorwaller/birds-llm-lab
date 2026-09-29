import type { CorpusRow } from '../data/schemas';

export function trainingDocumentText(row: CorpusRow): string {
  const lines = [
    `Common name: ${row.name}`,
    `Scientific name: ${row.sci}`,
    `Order: ${row.order ?? ''}`,
    `Family: ${row.family ?? ''}`,
    '',
    row.extract ?? '',
  ];
  for (const section of row.sections) lines.push('', section.title, section.text);
  return lines.join('\n');
}
