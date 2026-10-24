import { useState } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Card } from '@/components/ui';
import { Spacing } from '@/constants/theme';
import { splitSuspectClaim } from '@/lib/format';

/**
 * Statements in the student's notes that the fact check thinks are wrong. No card is made from them as written, so this list is the
 * only place they appear. The wording is deliberately cautious: it is the AI's own knowledge, not a source.
 *
 * `collapsible` keeps it to one tappable line (used above the card review, where space is tight and the cards are the point).
 */
export function SuspectClaims({
  claims,
  note,
  collapsible = false,
}: {
  claims: readonly string[] | undefined;
  /** Why the web search behind the corrections did not work, when it did not. */
  note?: string;
  collapsible?: boolean;
}) {
  const [open, setOpen] = useState(!collapsible);
  if (!claims || claims.length === 0) return null;
  const one = claims.length === 1;
  const title = one ? 'A statement in your notes looks wrong' : `${claims.length} statements in your notes look wrong`;

  return (
    <Card>
      {collapsible ? (
        <Pressable accessibilityRole="button" accessibilityState={{ expanded: open }} onPress={() => setOpen((o) => !o)} hitSlop={8}>
          <ThemedText type="smallBold">{title}</ThemedText>
          <ThemedText type="caption" themeColor="textSecondary">
            No card states {one ? 'it' : 'them'} as written. {open ? 'Tap to hide.' : 'Tap to see which.'}
          </ThemedText>
        </Pressable>
      ) : (
        <ThemedText type="smallBold">{title}</ThemedText>
      )}
      {open ? (
        <>
          {claims.map((text, i) => {
            const { claim, correction } = splitSuspectClaim(text);
            return (
              <View key={i} style={styles.item}>
                <ThemedText type="small">“{claim}”</ThemedText>
                {correction ? (
                  <ThemedText type="caption" themeColor="textSecondary">
                    {correction}
                  </ThemedText>
                ) : null}
              </View>
            );
          })}
          {note ? (
            <ThemedText type="caption" themeColor="warning">
              {note}
            </ThemedText>
          ) : null}
          {!collapsible ? (
            <ThemedText type="caption" themeColor="textSecondary">
              No card states {one ? 'it' : 'them'} as written. Where a source or the AI knew the right fact, a corrected card is made and labelled. The AI can be
              mistaken, so check before you rely on it.
            </ThemedText>
          ) : null}
        </>
      ) : null}
    </Card>
  );
}

const styles = StyleSheet.create({
  item: { gap: Spacing.half },
});
