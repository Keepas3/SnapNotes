import { StyleSheet, View } from 'react-native';

import type { Card } from '@/api/types';
import { Spacing } from '@/constants/theme';
import { parseCloze, clozeQuestion } from '@/lib/cloze';
import { originLabel, sourceName } from '@/lib/origin';

import { ThemedText } from './themed-text';
import { Badge } from './ui';

export type Side = 'front' | 'back';

/** The text content of one side of a card; works for all three card types. */
export function CardFace({ card, side }: { card: Pick<Card, 'type' | 'front' | 'back'>; side: Side }) {
  if (card.type === 'cloze') {
    if (side === 'front') {
      return <ThemedText style={styles.prompt}>{clozeQuestion(card.front)}</ThemedText>;
    }
    return (
      <View style={styles.stack}>
        <ThemedText style={styles.prompt}>
          {parseCloze(card.front).map((seg, i) =>
            seg.blank ? (
              <ThemedText key={i} style={styles.prompt} themeColor="tint">
                {seg.text}
              </ThemedText>
            ) : (
              seg.text
            ),
          )}
        </ThemedText>
        {card.back ? <ThemedText themeColor="textSecondary">{card.back}</ThemedText> : null}
      </View>
    );
  }

  if (side === 'front') {
    return (
      <ThemedText style={card.type === 'definition' ? styles.term : styles.prompt}>{card.front}</ThemedText>
    );
  }
  return (
    <View style={styles.stack}>
      <ThemedText themeColor="textSecondary">{card.front}</ThemedText>
      <View style={styles.rule} />
      <ThemedText style={styles.prompt}>{card.back}</ThemedText>
    </View>
  );
}

const TYPE_LABEL: Record<Card['type'], string> = { basic: 'Question', cloze: 'Fill in the blank', definition: 'Definition' };

/** Where the card came from: the student's notes, a cited web page, or (clearly marked) the AI's own knowledge. */
export function CardMeta({ card }: { card: Pick<Card, 'type' | 'origin' | 'source_ref'> }) {
  const { label, tone } = originLabel(card.origin);
  return (
    <View style={styles.meta}>
      <Badge label={label} tone={tone} />
      <Badge label={TYPE_LABEL[card.type]} />
      {card.source_ref ? (
        <ThemedText type="caption" themeColor="textSecondary" numberOfLines={1} style={styles.ref}>
          {card.origin === 'web' ? sourceName(card.source_ref) : card.source_ref}
        </ThemedText>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  prompt: { fontSize: 20, lineHeight: 28, fontWeight: 600 },
  term: { fontSize: 26, lineHeight: 34, fontWeight: 700 },
  stack: { gap: Spacing.two },
  rule: { height: StyleSheet.hairlineWidth, backgroundColor: '#8884', marginVertical: Spacing.one },
  meta: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: Spacing.two },
  ref: { flexShrink: 1 },
});
