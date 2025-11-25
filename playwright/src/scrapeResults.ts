import { chromium } from 'playwright';
import path from 'path';
import { promises as fs } from 'fs';
import { RaceResultsPage } from './page-objects/RaceResultsPage';

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const raceResults = new RaceResultsPage(page);

  await raceResults.goto();
  const rows = await raceResults.extractResults();

  const artifactDir = path.resolve(__dirname, '../artifacts');
  await fs.mkdir(artifactDir, { recursive: true });
  const artifactPath = path.join(artifactDir, 'mock-results.json');
  await fs.writeFile(artifactPath, JSON.stringify(rows, null, 2), 'utf-8');

  console.log(`Saved ${rows.length} race results to ${artifactPath}`);
  await browser.close();
}

main().catch((error) => {
  console.error('Unable to scrape mock results', error);
  process.exit(1);
});
