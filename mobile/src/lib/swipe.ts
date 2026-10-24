export type SwipeDirection = 'left' | 'right';

export const SWIPE_DISTANCE = 110; // px dragged before a release counts as a swipe
const FLICK_MIN_DISTANCE = 40; // a quick flick needs less distance...
const FLICK_MIN_VELOCITY = 0.6; // ...but this much speed (px/ms)

/** Decide what a released drag means. Right approves, left rejects, anything else springs back. */
export function decideSwipe(dx: number, vx: number): SwipeDirection | null {
  if (dx >= SWIPE_DISTANCE || (dx >= FLICK_MIN_DISTANCE && vx >= FLICK_MIN_VELOCITY)) return 'right';
  if (dx <= -SWIPE_DISTANCE || (dx <= -FLICK_MIN_DISTANCE && vx <= -FLICK_MIN_VELOCITY)) return 'left';
  return null;
}

/** A release with almost no movement is a tap (flip the card), not a drag. */
export function isTap(dx: number, dy: number): boolean {
  return Math.abs(dx) < 6 && Math.abs(dy) < 6;
}
