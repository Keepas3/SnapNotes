import { StyleSheet, View, type ColorValue } from 'react-native';

/** Small icons for the bottom bar, drawn from plain shapes so they look the same on every phone and need no icon font. `color` is the bar's tint. */
const SIZE = 24;

/** Two stacked cards. */
export function DecksIcon({ color }: { color: ColorValue }) {
  return (
    <View style={styles.box}>
      <View style={[styles.deckBack, { borderColor: color }]} />
      <View style={[styles.deckFront, { borderColor: color, backgroundColor: color }]} />
    </View>
  );
}

/** Three bars of different heights. */
export function StatsIcon({ color }: { color: ColorValue }) {
  return (
    <View style={[styles.box, styles.statsRow]}>
      {[10, 18, 14].map((height, i) => (
        <View key={i} style={{ width: 5, height, borderRadius: 2, backgroundColor: color }} />
      ))}
    </View>
  );
}

/** A ring with a dot, like a dial. */
export function SettingsIcon({ color }: { color: ColorValue }) {
  return (
    <View style={[styles.box, styles.center]}>
      <View style={[styles.ring, { borderColor: color }]}>
        <View style={[styles.dot, { backgroundColor: color }]} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  box: { width: SIZE, height: SIZE },
  center: { alignItems: 'center', justifyContent: 'center' },
  deckBack: { position: 'absolute', left: 2, top: 2, width: 15, height: 18, borderRadius: 3, borderWidth: 2, opacity: 0.55 },
  deckFront: { position: 'absolute', left: 7, top: 5, width: 15, height: 18, borderRadius: 3, borderWidth: 2 },
  statsRow: { flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'center', gap: 3 },
  ring: { width: 20, height: 20, borderRadius: 10, borderWidth: 2, alignItems: 'center', justifyContent: 'center' },
  dot: { width: 6, height: 6, borderRadius: 3 },
});
