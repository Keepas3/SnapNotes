import { StyleSheet, View } from 'react-native';

import { Spacing, type ThemeColor } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import type { CalendarCell } from '@/lib/stats';

import { ThemedText } from './themed-text';
import { Card } from './ui';

/** One headline number with a short label and an optional note under it. */
export function StatTile({ label, value, note, tone }: { label: string; value: string; note?: string; tone?: ThemeColor }) {
  return (
    <Card style={styles.tile} accessibilityLabel={`${label}: ${value}${note ? `. ${note}` : ''}`}>
      <ThemedText type="caption" themeColor="textSecondary">
        {label}
      </ThemedText>
      <ThemedText type="heading" themeColor={tone}>
        {value}
      </ThemedText>
      {note ? (
        <ThemedText type="caption" themeColor="textSecondary">
          {note}
        </ThemedText>
      ) : null}
    </Card>
  );
}

export interface Bar {
  /** what is shown under the bar */
  label: string;
  value: number;
  /** read out for screen readers, e.g. "Mar 10: 24 answers" */
  describe: string;
  /** shown above the bar when there is room (a percentage, say); defaults to the value */
  caption?: string;
  /** a bar with no data at all (a week with no reviews) is drawn as a faint stub instead of zero */
  empty?: boolean;
}

const CHART_HEIGHT = 120;

/** Vertical bars drawn with plain views, scaled so the tallest fills the chart (or to `max`, e.g. 100 for percentages). */
export function BarChart({ bars, max, tone = 'tint', showCaptions = false, summary }: { bars: Bar[]; max?: number; tone?: ThemeColor; showCaptions?: boolean; summary: string }) {
  const theme = useTheme();
  const top = max ?? Math.max(1, ...bars.map((b) => b.value));
  return (
    <View accessible accessibilityRole="image" accessibilityLabel={summary} style={styles.chart}>
      <View style={[styles.bars, { borderBottomColor: theme.border }]}>
        {bars.map((bar, i) => {
          const height = bar.empty ? 3 : bar.value <= 0 ? 0 : Math.max(3, (bar.value / top) * CHART_HEIGHT);
          return (
            <View key={i} style={styles.barSlot} accessibilityLabel={bar.describe}>
              {showCaptions ? (
                <ThemedText type="caption" themeColor="textSecondary" numberOfLines={1} style={styles.barCaption}>
                  {bar.empty ? '' : (bar.caption ?? (bar.value > 0 ? String(bar.value) : ''))}
                </ThemedText>
              ) : null}
              <View style={[styles.bar, { height, backgroundColor: bar.empty ? theme.border : theme[tone] }]} />
            </View>
          );
        })}
      </View>
      <View style={styles.labels}>
        {bars.map((bar, i) => (
          <View key={i} style={styles.barSlot}>
            <ThemedText type="caption" themeColor="textSecondary" numberOfLines={1} style={styles.barLabel}>
              {bar.label}
            </ThemedText>
          </View>
        ))}
      </View>
    </View>
  );
}

export interface Segment {
  label: string;
  value: number;
  tone: ThemeColor;
}

/** One horizontal bar split into parts, with a legend that gives each part's count (so colour is never the only clue). */
export function StackedBar({ segments, summary }: { segments: Segment[]; summary: string }) {
  const theme = useTheme();
  const total = segments.reduce((n, s) => n + s.value, 0);
  return (
    <View style={styles.stacked} accessible accessibilityRole="image" accessibilityLabel={summary}>
      <View style={[styles.stackTrack, { backgroundColor: theme.backgroundSelected }]}>
        {total > 0
          ? segments
              .filter((s) => s.value > 0)
              .map((s) => <View key={s.label} style={{ flex: s.value, backgroundColor: theme[s.tone] }} />)
          : null}
      </View>
      <View style={styles.legend}>
        {segments.map((s) => (
          <View key={s.label} style={styles.legendItem}>
            <View style={[styles.swatch, { backgroundColor: theme[s.tone] }]} />
            <ThemedText type="caption">{`${s.label} ${s.value}`}</ThemedText>
          </View>
        ))}
      </View>
    </View>
  );
}

const LEVEL_OPACITY = [0, 0.3, 0.5, 0.75, 1] as const;

/** A grid of small squares, one per day (a column per week), darker the more was answered that day. */
export function CalendarGrid({ columns, summary }: { columns: CalendarCell[][]; summary: string }) {
  const theme = useTheme();
  return (
    <View accessible accessibilityRole="image" accessibilityLabel={summary} style={styles.calendar}>
      {columns.map((column, c) => (
        <View key={c} style={styles.calendarColumn}>
          {column.map((cell) => (
            <View
              key={cell.day}
              style={[
                styles.cell,
                cell.future
                  ? undefined
                  : cell.level === 0
                    ? { backgroundColor: theme.backgroundSelected }
                    : { backgroundColor: theme.tint, opacity: LEVEL_OPACITY[cell.level] },
              ]}
            />
          ))}
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  tile: { flexGrow: 1, flexBasis: '45%', gap: 2 },
  chart: { gap: Spacing.one },
  bars: { flexDirection: 'row', alignItems: 'flex-end', height: CHART_HEIGHT + 20, borderBottomWidth: StyleSheet.hairlineWidth, gap: 2 },
  barSlot: { flex: 1, alignItems: 'center', justifyContent: 'flex-end', minWidth: 0 },
  bar: { width: '70%', maxWidth: 28, borderTopLeftRadius: 4, borderTopRightRadius: 4 },
  barCaption: { fontSize: 10, lineHeight: 14 },
  labels: { flexDirection: 'row', gap: 2 },
  barLabel: { fontSize: 10, lineHeight: 14 },
  stacked: { gap: Spacing.two },
  stackTrack: { flexDirection: 'row', height: 16, borderRadius: 8, overflow: 'hidden' },
  legend: { flexDirection: 'row', flexWrap: 'wrap', columnGap: Spacing.three, rowGap: Spacing.one },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: Spacing.one },
  swatch: { width: 10, height: 10, borderRadius: 3 },
  calendar: { flexDirection: 'row', gap: 4, justifyContent: 'center' },
  calendarColumn: { gap: 4 },
  cell: { width: 18, height: 18, borderRadius: 4 },
});
