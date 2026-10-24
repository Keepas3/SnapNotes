import type { ReactNode } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  View,
  type RefreshControlProps,
  type StyleProp,
  type ViewStyle,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { MaxContentWidth, Spacing, type ThemeColor } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import type { Tone } from '@/lib/steps';

import { ThemedText } from './themed-text';

// -- layout --------------------------------------------------------------------------------

interface ScreenProps {
  children: ReactNode;
  /** Content pinned below the scrolling area (primary action bar). */
  footer?: ReactNode;
  refreshControl?: React.ReactElement<RefreshControlProps>;
  scroll?: boolean;
  /** Set false while something is being dragged, so the page does not scroll under the finger. */
  scrollEnabled?: boolean;
}

export function Screen({ children, footer, refreshControl, scroll = true, scrollEnabled = true }: ScreenProps) {
  const theme = useTheme();
  const insets = useSafeAreaInsets();
  const body = <View style={styles.column}>{children}</View>;
  return (
    <View style={[styles.fill, { backgroundColor: theme.background }]}>
      {scroll ? (
        <ScrollView
          style={styles.fill}
          contentContainerStyle={styles.scrollContent}
          refreshControl={refreshControl}
          scrollEnabled={scrollEnabled}
          keyboardShouldPersistTaps="handled"
          contentInsetAdjustmentBehavior="automatic">
          {body}
        </ScrollView>
      ) : (
        <View style={[styles.fill, styles.scrollContent]}>{body}</View>
      )}
      {footer ? (
        <View
          style={[
            styles.footer,
            { borderTopColor: theme.border, backgroundColor: theme.background, paddingBottom: Math.max(insets.bottom, Spacing.three) },
          ]}>
          <View style={styles.footerInner}>{footer}</View>
        </View>
      ) : null}
    </View>
  );
}

export function Section({ title, right, children }: { title: string; right?: ReactNode; children: ReactNode }) {
  return (
    <View style={styles.section}>
      <View style={styles.sectionHeader}>
        <ThemedText type="heading">{title}</ThemedText>
        {right}
      </View>
      {children}
    </View>
  );
}

export function Card({
  children,
  onPress,
  style,
  accessibilityLabel,
}: {
  children: ReactNode;
  onPress?: () => void;
  style?: StyleProp<ViewStyle>;
  accessibilityLabel?: string;
}) {
  const theme = useTheme();
  const base = [styles.card, { backgroundColor: theme.backgroundElement, borderColor: theme.border }, style];
  if (!onPress) return <View style={base}>{children}</View>;
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel}
      onPress={onPress}
      style={({ pressed }) => [base, pressed && styles.pressed]}>
      {children}
    </Pressable>
  );
}

// -- controls ------------------------------------------------------------------------------

type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost';

export function Button({
  title,
  onPress,
  variant = 'primary',
  disabled,
  loading,
  style,
}: {
  title: string;
  onPress: () => void;
  variant?: ButtonVariant;
  disabled?: boolean;
  loading?: boolean;
  style?: StyleProp<ViewStyle>;
}) {
  const theme = useTheme();
  const palette: Record<ButtonVariant, { bg: string; fg: ThemeColor; border?: string }> = {
    primary: { bg: theme.tint, fg: 'onTint' },
    secondary: { bg: theme.backgroundElement, fg: 'text', border: theme.border },
    danger: { bg: theme.dangerBg, fg: 'danger' },
    ghost: { bg: 'transparent', fg: 'tint' },
  };
  const p = palette[variant];
  const inactive = disabled || loading;
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ disabled: !!inactive, busy: !!loading }}
      disabled={inactive}
      onPress={onPress}
      style={({ pressed }) => [
        styles.button,
        { backgroundColor: p.bg, borderColor: p.border ?? 'transparent' },
        inactive && styles.disabled,
        pressed && styles.pressed,
        style,
      ]}>
      {loading ? <ActivityIndicator color={theme[p.fg]} /> : <ThemedText type="smallBold" themeColor={p.fg} style={styles.buttonText}>{title}</ThemedText>}
    </Pressable>
  );
}

/** A small round button with one symbol (settings, close, delete). The label is what a screen reader says. */
export function IconButton({
  glyph,
  label,
  onPress,
  tone = 'neutral',
  loading,
  disabled,
}: {
  glyph: string;
  label: string;
  onPress: () => void;
  tone?: 'neutral' | 'danger';
  loading?: boolean;
  disabled?: boolean;
}) {
  const theme = useTheme();
  const fg: ThemeColor = tone === 'danger' ? 'danger' : 'textSecondary';
  const bg = tone === 'danger' ? theme.dangerBg : theme.backgroundSelected;
  const inactive = disabled || loading;
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ disabled: !!inactive, busy: !!loading }}
      disabled={inactive}
      hitSlop={10} // the circle is small, the touch target is still about 48 points
      onPress={onPress}
      style={({ pressed }) => [styles.iconButton, { backgroundColor: bg }, inactive && styles.disabled, pressed && styles.pressed]}>
      {loading ? (
        <ActivityIndicator size="small" color={theme[fg]} />
      ) : (
        <ThemedText type="smallBold" themeColor={fg} style={styles.iconGlyph}>
          {glyph}
        </ThemedText>
      )}
    </Pressable>
  );
}

export function Chip({ label, selected, onPress }: { label: string; selected?: boolean; onPress?: () => void }) {
  const theme = useTheme();
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ selected: !!selected }}
      onPress={onPress}
      style={[
        styles.chip,
        { backgroundColor: selected ? theme.tint : theme.backgroundElement, borderColor: selected ? theme.tint : theme.border },
      ]}>
      <ThemedText type="small" themeColor={selected ? 'onTint' : 'text'}>
        {label}
      </ThemedText>
    </Pressable>
  );
}

// -- feedback ------------------------------------------------------------------------------

export type { Tone };

function toneColors(theme: ReturnType<typeof useTheme>, tone: Tone): { bg: string; fg: ThemeColor } {
  switch (tone) {
    case 'success':
      return { bg: theme.successBg, fg: 'success' };
    case 'danger':
      return { bg: theme.dangerBg, fg: 'danger' };
    case 'warning':
      return { bg: theme.warningBg, fg: 'warning' };
    case 'info':
      return { bg: theme.backgroundSelected, fg: 'tint' };
    default:
      return { bg: theme.backgroundSelected, fg: 'textSecondary' };
  }
}

export function Badge({ label, tone = 'neutral' }: { label: string; tone?: Tone }) {
  const theme = useTheme();
  const c = toneColors(theme, tone);
  return (
    <View style={[styles.badge, { backgroundColor: c.bg }]}>
      <ThemedText type="caption" themeColor={c.fg}>
        {label}
      </ThemedText>
    </View>
  );
}

export function Banner({
  tone = 'danger',
  message,
  actionLabel,
  onAction,
}: {
  tone?: Tone;
  message: string;
  actionLabel?: string;
  onAction?: () => void;
}) {
  const theme = useTheme();
  const c = toneColors(theme, tone);
  return (
    <View accessibilityRole="alert" style={[styles.banner, { backgroundColor: c.bg }]}>
      <ThemedText type="small" themeColor={c.fg} style={styles.bannerText}>
        {message}
      </ThemedText>
      {actionLabel && onAction ? <Button title={actionLabel} variant="ghost" onPress={onAction} /> : null}
    </View>
  );
}

export function Centered({ children }: { children: ReactNode }) {
  return <View style={styles.centered}>{children}</View>;
}

export function Spinner() {
  const theme = useTheme();
  return <ActivityIndicator color={theme.tint} size="large" />;
}

export function EmptyState({ title, message, children }: { title: string; message?: string; children?: ReactNode }) {
  return (
    <View style={styles.empty}>
      <ThemedText type="heading" style={styles.center}>
        {title}
      </ThemedText>
      {message ? (
        <ThemedText themeColor="textSecondary" style={styles.center}>
          {message}
        </ThemedText>
      ) : null}
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  scrollContent: { padding: Spacing.three, flexGrow: 1 },
  column: { width: '100%', maxWidth: MaxContentWidth, alignSelf: 'center', gap: Spacing.three },
  footer: { borderTopWidth: StyleSheet.hairlineWidth, paddingTop: Spacing.three, paddingHorizontal: Spacing.three },
  footerInner: { width: '100%', maxWidth: MaxContentWidth, alignSelf: 'center', gap: Spacing.two },
  section: { gap: Spacing.two },
  sectionHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  card: { borderRadius: 16, borderWidth: StyleSheet.hairlineWidth, padding: Spacing.three, gap: Spacing.two },
  pressed: { opacity: 0.75 },
  disabled: { opacity: 0.45 },
  button: {
    minHeight: 48,
    borderRadius: 14,
    borderWidth: StyleSheet.hairlineWidth,
    paddingHorizontal: Spacing.three,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonText: { fontSize: 16, lineHeight: 22 },
  iconButton: { width: 28, height: 28, borderRadius: 14, alignItems: 'center', justifyContent: 'center' },
  iconGlyph: { fontSize: 13, lineHeight: 16, textAlign: 'center' },
  chip: { borderRadius: 999, borderWidth: StyleSheet.hairlineWidth, paddingHorizontal: 14, paddingVertical: 8, minHeight: 36, justifyContent: 'center' },
  badge: { alignSelf: 'flex-start', borderRadius: 999, paddingHorizontal: 10, paddingVertical: 2 },
  banner: { borderRadius: 12, padding: Spacing.three, gap: Spacing.one },
  bannerText: { fontWeight: 600 },
  centered: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: Spacing.four, gap: Spacing.three },
  empty: { alignItems: 'center', gap: Spacing.two, paddingVertical: Spacing.five },
  center: { textAlign: 'center' },
});
