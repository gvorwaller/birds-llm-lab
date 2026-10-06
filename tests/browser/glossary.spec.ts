import { expect, test } from '@playwright/test';

test('keeps the glossary reachable in a shorter window', async ({ page }) => {
  await page.setViewportSize({ width: page.viewportSize()!.width, height: 560 });
  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Lab screens' })
    .getByRole('link', { name: 'Glossary' })
    .click();
  await expect(page).toHaveURL(/\/glossary$/);
  await expect(page.getByRole('navigation', { name: 'Glossary term index' })).toBeVisible();
});

test('keeps the full glossary reachable by keyboard and links both ways', async ({ page }) => {
  await page.goto('/generation');
  await page.getByRole('link', { name: 'temperature' }).click();
  await expect(page).toHaveURL(/\/glossary#temperature$/);
  const target = page.locator('#temperature');
  await expect(target).toBeVisible();
  await expect(target).toBeFocused();
  await expect(target).toContainText('softmax(logit[i] / temperature)');
  const index = page.getByRole('navigation', { name: 'Glossary term index' });
  await expect(index.getByRole('link')).toHaveCount(26);
  await index.getByRole('link', { name: 'Attention', exact: true }).click();
  await expect(page.locator('#attention')).toBeFocused();
  await expect(page.locator('#attention')).toContainText('“Looks at” is a metaphor. Literally');
  await page
    .locator('#attention')
    .getByRole('link', { name: 'Inspect attention probabilities' })
    .click();
  await expect(page).toHaveURL(/\/attention$/);
  await expect(page.getByRole('link', { name: 'head' })).toHaveAttribute('href', '/glossary#head');
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    ),
  ).toBe(0);
});

test('glossary text remains contrasted and reduced-motion preference is honored', async ({
  page,
}) => {
  await page.emulateMedia({ colorScheme: 'light', reducedMotion: 'reduce' });
  await page.goto('/glossary#hallucination');
  await expect(page.locator('#hallucination')).toBeFocused();
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' });
    const result = await page.evaluate(() => {
      const styles = getComputedStyle(document.documentElement);
      const parse = (value: string) =>
        value.trim().startsWith('#')
          ? value
              .trim()
              .slice(1)
              .match(/../g)!
              .map((part) => parseInt(part, 16) / 255)
          : value
              .match(/[\d.]+/g)!
              .slice(0, 3)
              .map((part) => Number(part) / 255);
      const luminance = (value: string) => {
        const [r, g, b] = parse(value).map((channel) =>
          channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4,
        );
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
      };
      const contrast = (a: string, b: string) => {
        const left = luminance(a),
          right = luminance(b);
        return (Math.max(left, right) + 0.05) / (Math.min(left, right) + 0.05);
      };
      const probe = document.createElement('div');
      probe.style.transitionDuration = '1s';
      document.body.append(probe);
      const transitionDuration = getComputedStyle(probe).transitionDuration;
      probe.remove();
      return {
        signalText: contrast(
          styles.getPropertyValue('--signal-text'),
          styles.getPropertyValue('--paper-raised'),
        ),
        mutedText: contrast(
          styles.getPropertyValue('--muted'),
          styles.getPropertyValue('--paper-raised'),
        ),
        indexLink: contrast(
          getComputedStyle(document.querySelector('.term-index a')!).color,
          getComputedStyle(document.querySelector('.glossary .introduction')!).backgroundColor,
        ),
        transitionDuration,
        targetOutline: getComputedStyle(document.getElementById('hallucination')!).outlineStyle,
      };
    });
    expect(result.signalText).toBeGreaterThanOrEqual(4.5);
    expect(result.mutedText).toBeGreaterThanOrEqual(4.5);
    expect(result.indexLink).toBeGreaterThanOrEqual(4.5);
    expect(parseFloat(result.transitionDuration)).toBeLessThan(0.01);
    expect(result.targetOutline).toBe('solid');
  }
});
