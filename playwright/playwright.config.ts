import { defineConfig } from '@playwright/test';
import path from 'path';

const projectRoot = path.resolve(__dirname);

export default defineConfig({
  testDir: path.join(projectRoot, 'tests'),
  timeout: 30_000,
  expect: {
    timeout: 5_000,
  },
  use: {
    headless: true,
    viewport: { width: 1280, height: 720 },
    trace: 'on-first-retry',
    video: 'retain-on-failure',
  },
  workers: process.env.CI ? 2 : undefined,
  reporter: [['list'], ['html', { open: 'never' }]],
  metadata: {
    project: 'f1-winner-playwright',
  },
});
