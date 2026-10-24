/** Which colours the app uses: follow the phone's light/dark setting, or always one of the two. */
export type Appearance = 'system' | 'light' | 'dark';
export type Scheme = 'light' | 'dark';

export const APPEARANCES: { value: Appearance; label: string }[] = [
  { value: 'system', label: 'Same as phone' },
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
];

/** A saved value read back from storage; anything unknown (or missing) means "follow the phone". */
export function parseAppearance(value: unknown): Appearance {
  return value === 'light' || value === 'dark' ? value : 'system';
}

/** The colours to draw with, given the choice and what the phone is currently set to (null when it has no preference). */
export function resolveScheme(preference: Appearance, system: string | null | undefined): Scheme {
  if (preference === 'light' || preference === 'dark') return preference;
  return system === 'dark' ? 'dark' : 'light';
}
