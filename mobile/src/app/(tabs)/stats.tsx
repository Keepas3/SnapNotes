import { useQueryClient } from '@tanstack/react-query';
import { useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { StyleSheet, View } from 'react-native';

import { BarChart, CalendarGrid, StackedBar, StatTile, type Bar } from '@/components/charts';
import { ThemedText } from '@/components/themed-text';
import { Banner, Card, Centered, Chip, EmptyState, Screen, Section, Spinner } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { plural } from '@/lib/format';
import { calendarGrid, dayLabel, fillDays, formatDuration, retentionByWeek, shortDate, streaks, summarize } from '@/lib/stats';
import { useStudyStats } from '@/study/hooks';
import { LOCAL_KEY } from '@/sync/provider';

const RANGES = [7, 30, 90] as const;
const WEEKS_OF_RETENTION = 8;
const WEEKS_OF_CALENDAR = 12;

/** How you are doing, worked out from the answers saved on this phone. */
export default function StatsScreen() {
  const queryClient = useQueryClient();
  const stats = useStudyStats();
  const [range, setRange] = useState<(typeof RANGES)[number]>(30);

  // Coming back to this tab (after studying, or after the day has rolled over) shows fresh numbers.
  useFocusEffect(
    useCallback(() => {
      queryClient.invalidateQueries({ queryKey: [...LOCAL_KEY, 'stats'] });
    }, [queryClient]),
  );

  if (stats.isPending) {
    return (
      <Centered>
        <Spinner />
      </Centered>
    );
  }
  if (stats.isError) {
    return (
      <Screen>
        <Banner message={stats.error.message} actionLabel="Try again" onAction={() => stats.refetch()} />
      </Screen>
    );
  }

  const { days, today, cards, forecast } = stats.data;
  if (days.length === 0) {
    return (
      <Screen>
        <EmptyState title="No answers yet" message="Study some cards and your progress will show up here. Everything on this page is worked out on your phone from your own answers." />
      </Screen>
    );
  }

  const inRange = summarize(days, today, range);
  const run = streaks(days, today);
  const learned = cards.total - cards.new;
  const perDay = fillDays(days, today, range);
  const labelEvery = range <= 7 ? 1 : range <= 30 ? 5 : 15;
  const answerBars: Bar[] = perDay.map((d, i) => ({
    label: (perDay.length - 1 - i) % labelEvery === 0 ? dayLabel(d.day, range <= 7 ? 'weekday' : 'date') : '',
    value: d.answers,
    describe: `${shortDate(d.day)}: ${plural(d.answers, 'answer')}`,
  }));
  const weeks = retentionByWeek(days, today, WEEKS_OF_RETENTION);
  const weekBars: Bar[] = weeks.map((w) => ({
    label: shortDate(w.from),
    value: w.percent ?? 0,
    caption: w.percent === null ? '' : `${w.percent}%`,
    empty: w.percent === null,
    describe: w.percent === null ? `Week from ${shortDate(w.from)}: no reviews` : `Week from ${shortDate(w.from)}: ${w.percent}% remembered of ${plural(w.reviewAnswers, 'review')}`,
  }));
  const dueBars: Bar[] = forecast.map((f, i) => ({
    label: i === 0 ? 'Now' : dayLabel(f.day, 'weekday'),
    value: f.due,
    describe: `${i === 0 ? 'Today' : shortDate(f.day)}: ${plural(f.due, 'card')} due`,
  }));
  const dueSoon = forecast.reduce((n, f) => n + f.due, 0);

  return (
    <Screen>
      <View style={styles.chips} accessibilityRole="radiogroup" accessibilityLabel="Time range">
        {RANGES.map((r) => (
          <Chip key={r} label={`${r} days`} selected={range === r} onPress={() => setRange(r)} />
        ))}
      </View>

      <View style={styles.tiles}>
        <StatTile
          label="Retention"
          value={inRange.retention === null ? '—' : `${inRange.retention}%`}
          note={
            inRange.reviewAnswers === 0
              ? 'Shows after you review cards you already know.'
              : `${inRange.remembered} of ${plural(inRange.reviewAnswers, 'review')} remembered${inRange.retentionIsShaky ? '. Too few so far to rely on.' : ''}`
          }
        />
        <StatTile label="Cards learned" value={`${learned}`} note={`of ${cards.total}, ${cards.mature} mature`} />
        <StatTile label="Days studied" value={`${inRange.daysStudied}`} note={`of the last ${range}`} />
        <StatTile label="Streak" value={plural(run.current, 'day')} note={`Longest ${plural(run.longest, 'day')}. ${plural(run.studied, 'day')} studied in all.`} />
        <StatTile label="Answers" value={`${inRange.answers}`} note={`in the last ${range} days`} />
        <StatTile label="Time studied" value={formatDuration(inRange.seconds)} note="Each answer counts at most a minute." />
      </View>

      <Section title="Answers per day">
        <Card>
          <BarChart bars={answerBars} summary={`Answers per day over the last ${range} days: ${plural(inRange.answers, 'answer')} in total.`} />
        </Card>
      </Section>

      <Section title="Remembered, week by week">
        <Card>
          <BarChart
            bars={weekBars}
            max={100}
            tone="success"
            showCaptions
            summary={`Share of reviews remembered in each of the last ${WEEKS_OF_RETENTION} weeks.`}
          />
          <ThemedText type="caption" themeColor="textSecondary">
            Of the answers to cards you already knew, how many were not “Missed it”. Anki users aim for about 90%.
          </ThemedText>
        </Card>
      </Section>

      <Section title="Your cards">
        <Card>
          <StackedBar
            segments={[
              { label: 'New', value: cards.new, tone: 'textSecondary' },
              { label: 'Learning', value: cards.learning, tone: 'warning' },
              { label: 'Young', value: cards.young, tone: 'tint' },
              { label: 'Mature', value: cards.mature, tone: 'success' },
            ]}
            summary={`${cards.total} cards: ${cards.new} new, ${cards.learning} learning, ${cards.young} young, ${cards.mature} mature.`}
          />
          <ThemedText type="caption" themeColor="textSecondary">
            Young cards come back within three weeks; mature ones wait 21 days or longer.
          </ThemedText>
        </Card>
      </Section>

      <Section title="Due in the next 14 days">
        <Card>
          <BarChart bars={dueBars} showCaptions summary={`${plural(dueSoon, 'card')} due in the next 14 days, ${plural(forecast[0].due, 'card')} of them now or overdue.`} />
          <ThemedText type="caption" themeColor="textSecondary">
            Cards you have studied and not paused. Overdue ones are counted under Now. New cards are not included.
          </ThemedText>
        </Card>
      </Section>

      <Section title="Study calendar">
        <Card>
          <CalendarGrid
            columns={calendarGrid(days, today, WEEKS_OF_CALENDAR)}
            summary={`The last ${WEEKS_OF_CALENDAR} weeks: you studied on ${run.studied} days in all.`}
          />
          <ThemedText type="caption" themeColor="textSecondary" style={styles.center}>
            Each square is a day, darker when you answered more. A day starts at 4 a.m.
          </ThemedText>
        </Card>
      </Section>

      <ThemedText type="caption" themeColor="textSecondary" style={styles.center}>
        Worked out on this phone from your saved answers. Practice mode records nothing, so it is not counted.
      </ThemedText>
    </Screen>
  );
}

const styles = StyleSheet.create({
  chips: { flexDirection: 'row', gap: Spacing.two },
  tiles: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.two },
  center: { textAlign: 'center' },
});
