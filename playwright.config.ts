import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  outputDir: './test-results',
  fullyParallel: false,
  reporter: 'line',
  use: {
    baseURL: 'http://127.0.0.1:5302',
    browserName: 'chromium',
    channel: 'chrome',
    colorScheme: 'light',
  },
  projects: [
    { name: 'chrome-1024', use: { viewport: { width: 1024, height: 900 } } },
    { name: 'chrome-1280', use: { viewport: { width: 1280, height: 900 } } },
  ],
  webServer: {
    command: 'npm run preview -- --port 5302',
    url: 'http://127.0.0.1:5302/tokenizer',
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
