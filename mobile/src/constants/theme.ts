/**
 * Below are the colors that are used in the app. The colors are defined in the light and dark mode.
 * There are many other ways to style your app. For example, [Nativewind](https://www.nativewind.dev/), [Tamagui](https://tamagui.dev/), [unistyles](https://unistyl.es/), etc.
 */

import '@/global.css';

import { Platform } from 'react-native';

export const Colors = {
  light: {
    text: '#0E0F12',
    background: '#ffffff',
    backgroundElement: '#F3F4F7',
    backgroundSelected: '#E3E5EB',
    textSecondary: '#5B6068',
    border: '#E1E3E9',
    tint: '#4F46E5',
    onTint: '#FFFFFF',
    success: '#15803D',
    successBg: '#E6F4EA',
    danger: '#B91C1C',
    dangerBg: '#FDECEC',
    warning: '#8A5A00',
    warningBg: '#FFF4DB',
  },
  dark: {
    text: '#F5F6F8',
    background: '#0B0C0F',
    backgroundElement: '#17181C',
    backgroundSelected: '#262830',
    textSecondary: '#A9AEB7',
    border: '#2A2C33',
    tint: '#8B93FF',
    onTint: '#0B0C1A',
    success: '#6EE7A0',
    successBg: '#12301F',
    danger: '#FF8A8A',
    dangerBg: '#3A1717',
    warning: '#F5C15A',
    warningBg: '#3A2D0E',
  },
} as const;

export type ThemeColor = keyof typeof Colors.light & keyof typeof Colors.dark;

export const Fonts = Platform.select({
  ios: {
    /** iOS `UIFontDescriptorSystemDesignDefault` */
    sans: 'system-ui',
    /** iOS `UIFontDescriptorSystemDesignSerif` */
    serif: 'ui-serif',
    /** iOS `UIFontDescriptorSystemDesignRounded` */
    rounded: 'ui-rounded',
    /** iOS `UIFontDescriptorSystemDesignMonospaced` */
    mono: 'ui-monospace',
  },
  default: {
    sans: 'normal',
    serif: 'serif',
    rounded: 'normal',
    mono: 'monospace',
  },
  web: {
    sans: 'var(--font-display)',
    serif: 'var(--font-serif)',
    rounded: 'var(--font-rounded)',
    mono: 'var(--font-mono)',
  },
});

export const Spacing = {
  half: 2,
  one: 4,
  two: 8,
  three: 16,
  four: 24,
  five: 32,
  six: 64,
} as const;

export const BottomTabInset = Platform.select({ ios: 50, android: 80 }) ?? 0;
export const MaxContentWidth = 800;
