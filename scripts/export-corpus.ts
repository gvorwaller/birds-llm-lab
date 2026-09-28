import { exportCorpus } from '../server/export/corpus-export.js';

async function main(): Promise<void> {
  const result = await exportCorpus({
    onProgress(progress) {
      if (progress.phase === 'normalizing' && progress.processedRows % 1000 !== 1) return;
      const total = progress.totalRows === null ? '' : `/${progress.totalRows}`;
      console.log(`${progress.phase}: ${progress.processedRows}${total}`);
    },
  });
  console.log(
    `Export complete: ${result.manifest.emittedRows} rows, ${result.manifest.corpusBytes} bytes, SHA-256 ${result.manifest.corpusSha256}`,
  );
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : 'Unknown export failure.';
  console.error(`Export failed: ${message}`);
  process.exitCode = 1;
});
