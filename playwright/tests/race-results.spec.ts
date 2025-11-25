import { expect, test } from '@playwright/test';
import { RaceResultsPage } from '../src/page-objects/RaceResultsPage';

test.describe('f1-winner Playwright automation', () => {
  test('extracts mock race results for downstream model features', async ({ page }) => {
    const raceResults = new RaceResultsPage(page);
    await raceResults.goto();

    await expect(raceResults.title).toHaveText(/2024 Live Timing Snapshot/);
    const rows = await raceResults.extractResults();
    expect(rows).toHaveLength(3);
    expect(rows[0]).toMatchObject({
      round: 1,
      grandPrix: 'Bahrain GP',
      driver: 'Max Verstappen',
      abbreviation: 'VER',
    });
  });

  test('filters the table via selectors and exposes empty states', async ({ page }) => {
    const raceResults = new RaceResultsPage(page);
    await raceResults.goto();

    await raceResults.filterByDriver('Ferrari');
    const ferrariRows = await raceResults.extractVisibleResults();
    expect(ferrariRows).toHaveLength(1);
    expect(ferrariRows[0].team).toContain('Ferrari');

    await raceResults.filterByDriver('HAM');
    await expect(raceResults.visibleRows).toHaveCount(0);
    await expect(raceResults.emptyState).toBeVisible();

    await raceResults.clearFilter();
    await expect(raceResults.visibleRows).toHaveCount(3);
  });

  test('waits for asynchronous refresh state to capture new winners', async ({ page }) => {
    const raceResults = new RaceResultsPage(page);
    await raceResults.goto();

    await raceResults.triggerRefresh();
    await expect(raceResults.statusBanner).toHaveAttribute('data-state', 'loading');
    await expect(raceResults.statusBanner).toHaveText(/Loading latest classification/);

    const norrisRow = raceResults.rows.filter({ hasText: 'Lando Norris' });
    await expect(norrisRow).toHaveCount(1);
    await expect(raceResults.statusBanner).toHaveAttribute('data-state', 'idle');
    await expect(raceResults.visibleRows).toHaveCount(4);
  });
});
