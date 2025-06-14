import { test, expect } from '@playwright/test';

test('Load PDF and establish WebSocket', async ({ page }) => {
  // Navigate to app
  await page.goto('/');
  // Input an arXiv ID and click Load
  await page.fill('#arxiv_id', 'test123');
  await page.click('#loadBtn');

  // Verify iframe src includes PDF.js viewer
  const viewer = page.locator('#viewer');
  await expect(viewer).toHaveAttribute('src', /viewer\.html\?file=\/pdfs\/test123\.pdf/);

  // Wait for WebSocket connection status
  const status = page.locator('#status');
  await expect(status).toHaveText(/WebSocket connected/);
});
