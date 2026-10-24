import { useAppearance } from '@/lib/appearance-provider';

/** 'light' or 'dark': the user's choice from Settings, or the phone's own setting when they have not chosen. */
export function useColorScheme() {
  return useAppearance().scheme;
}
