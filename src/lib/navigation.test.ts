import { describe, expect, it } from 'vitest';
import { normalizePath, routeForPath } from './navigation';

describe('lab navigation', () => {
  it('normalizes root and trailing slashes', () => {
    expect(normalizePath('/')).toBe('/');
    expect(normalizePath('/data///')).toBe('/data');
  });

  it('falls back to the overview for an unknown path', () => {
    expect(routeForPath('/not-a-lab').path).toBe('/');
  });

  it('exposes the local corpus workflow as milestone zero', () => {
    expect(routeForPath('/data')).toMatchObject({ label: 'Data', milestone: 'M0' });
  });
});
