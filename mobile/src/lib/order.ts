/** A fixed-size hash of some text (FNV-1a, 32 bits), for orderings that must come out the same every time they are asked. */
export function hash32(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** A number from 0 up to (not including) 1 that depends only on `seed`. The same seed always gives the same number. */
export function seededUnit(seed: string): number {
  // the hash alone leaves similar seeds with similar low bits, so mix it once more
  let h = hash32(seed);
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b) >>> 0;
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35) >>> 0;
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/** The id with the smallest seeded value: a shuffle that costs one pass, and is the same all day for the same `seed`. Null for none. */
export function seededFirst(ids: string[], seed: string): string | null {
  let best: string | null = null;
  let bestValue = Infinity;
  for (const id of ids) {
    const value = seededUnit(`${seed}:${id}`);
    if (value < bestValue) {
      best = id;
      bestValue = value;
    }
  }
  return best;
}
