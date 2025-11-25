import path from 'path';
import { pathToFileURL } from 'url';
import type { Locator, Page } from 'playwright';

export type RaceResult = {
  round: number;
  grandPrix: string;
  driver: string;
  abbreviation: string;
  team: string;
  grid: number;
  finish: number;
  points: number;
};

const defaultFixturePath = path.resolve(__dirname, '../../fixtures/mock_results.html');

export class RaceResultsPage {
  private readonly page: Page;
  private readonly fixturePath: string;

  readonly title: Locator;
  readonly filterInput: Locator;
  readonly clearFilterButton: Locator;
  readonly refreshButton: Locator;
  readonly statusBanner: Locator;
  readonly emptyState: Locator;
  readonly rows: Locator;
  readonly visibleRows: Locator;

  constructor(page: Page, fixtureOverride?: string) {
    this.page = page;
    this.fixturePath = fixtureOverride ?? defaultFixturePath;

    this.title = page.getByTestId('page-title');
    this.filterInput = page.getByTestId('driver-filter');
    this.clearFilterButton = page.getByTestId('clear-filter');
    this.refreshButton = page.getByTestId('refresh-button');
    this.statusBanner = page.getByTestId('status-banner');
    this.emptyState = page.getByTestId('empty-state');
    this.rows = page.getByTestId('result-row');
    this.visibleRows = page.locator('[data-testid="result-row"]:not(.is-hidden)');
  }

  async goto() {
    const fileUrl = pathToFileURL(this.fixturePath).href;
    await this.page.goto(fileUrl);
  }

  async filterByDriver(query: string) {
    await this.filterInput.fill(query);
  }

  async clearFilter() {
    await this.clearFilterButton.click();
  }

  async triggerRefresh() {
    await this.refreshButton.click();
  }

  async extractResults(): Promise<RaceResult[]> {
    return this.rows.evaluateAll((elements) =>
      elements.map((row) => {
        const cells = Array.from(row.querySelectorAll('td')).map((cell) =>
          cell.textContent?.trim() ?? ''
        );
        const driverCell = row.querySelector('td:nth-child(3)')?.textContent ?? '';
        const abbreviation = (row.querySelector('.chip')?.textContent ?? '').trim();
        return {
          round: Number(cells[0]),
          grandPrix: cells[1],
          driver: driverCell.replace(abbreviation, '').trim(),
          abbreviation,
          team: cells[3],
          grid: Number(cells[4]),
          finish: Number(cells[5]),
          points: Number(cells[6]),
        };
      })
    );
  }

  async extractVisibleResults(): Promise<RaceResult[]> {
    return this.visibleRows.evaluateAll((elements) =>
      elements.map((row) => {
        const cells = Array.from(row.querySelectorAll('td')).map((cell) =>
          cell.textContent?.trim() ?? ''
        );
        const abbreviation = (row.querySelector('.chip')?.textContent ?? '').trim();
        const driverCell = row.querySelector('td:nth-child(3)')?.textContent ?? '';
        return {
          round: Number(cells[0]),
          grandPrix: cells[1],
          driver: driverCell.replace(abbreviation, '').trim(),
          abbreviation,
          team: cells[3],
          grid: Number(cells[4]),
          finish: Number(cells[5]),
          points: Number(cells[6]),
        };
      })
    );
  }
}
