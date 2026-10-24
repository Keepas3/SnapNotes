import { useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState, type ReactNode, type Ref } from 'react';
import { Animated, PanResponder, StyleSheet, View, useWindowDimensions } from 'react-native';

import { Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { decideSwipe, isTap, type SwipeDirection } from '@/lib/swipe';

import { ThemedText } from './themed-text';

export interface SwipeCardHandle {
  /** Fling the card off-screen programmatically (for the approve/reject buttons). */
  swipe: (direction: SwipeDirection) => void;
}

interface Props {
  children: ReactNode;
  onSwiped: (direction: SwipeDirection) => void;
  onTap: () => void;
  ref?: Ref<SwipeCardHandle>;
  /** The words stamped on the card while it is dragged right and left. */
  rightLabel?: string;
  leftLabel?: string;
  /** When false the card does not follow a drag at all (a tap still works): used while the answer has not been seen yet. */
  swipeable?: boolean;
  accessibilityHint?: string;
}

/**
 * Draggable card: right is one answer, left the other, tap flips. Built on the core Animated/PanResponder APIs
 * so it needs no native modules (works in Expo Go and in the web preview).
 * Mount it with a `key` that changes for every card shown so each one starts centred.
 */
export function SwipeCard({
  children,
  onSwiped,
  onTap,
  ref,
  rightLabel = 'APPROVE',
  leftLabel = 'REJECT',
  swipeable = true,
  accessibilityHint = 'Tap to flip. Swipe right to approve, left to reject.',
}: Props) {
  const theme = useTheme();
  const { width } = useWindowDimensions();
  const [pan] = useState(() => new Animated.ValueXY());
  const busy = useRef(false);

  // The gesture handlers are created once, so they read the latest props through a ref updated after each render.
  const latest = useRef({ onSwiped, onTap, width, swipeable });
  useEffect(() => {
    latest.current = { onSwiped, onTap, width, swipeable };
  });

  const fling = useCallback(
    (direction: SwipeDirection) => {
      if (busy.current) return;
      busy.current = true;
      Animated.timing(pan, {
        toValue: { x: (direction === 'right' ? 1 : -1) * latest.current.width * 1.2, y: 0 },
        duration: 180,
        useNativeDriver: false,
      }).start(() => latest.current.onSwiped(direction));
    },
    [pan],
  );

  useImperativeHandle(ref, () => ({ swipe: fling }), [fling]);

  const responder = useMemo(() => {
    const track = Animated.event([null, { dx: pan.x, dy: pan.y }], { useNativeDriver: false });
    // These handlers run from gesture events, never during render; the lint rule cannot see that.
    /* eslint-disable react-hooks/refs */
    return PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: (_, g) => Math.abs(g.dx) > 6 || Math.abs(g.dy) > 6,
      onPanResponderMove: (e, g) => {
        if (latest.current.swipeable) track(e, g); // before the answer is shown the card stays put
      },
      onPanResponderRelease: (_, g) => {
        if (isTap(g.dx, g.dy)) {
          pan.setValue({ x: 0, y: 0 });
          latest.current.onTap();
          return;
        }
        const direction = latest.current.swipeable ? decideSwipe(g.dx, g.vx) : null;
        if (direction) fling(direction);
        else Animated.spring(pan, { toValue: { x: 0, y: 0 }, useNativeDriver: false, friction: 6 }).start();
      },
      onPanResponderTerminate: () => {
        Animated.spring(pan, { toValue: { x: 0, y: 0 }, useNativeDriver: false }).start();
      },
    });
    /* eslint-enable react-hooks/refs */
  }, [pan, fling]);

  const rotate = pan.x.interpolate({ inputRange: [-200, 0, 200], outputRange: ['-8deg', '0deg', '8deg'], extrapolate: 'clamp' });
  const approveOpacity = pan.x.interpolate({ inputRange: [20, 110], outputRange: [0, 1], extrapolate: 'clamp' });
  const rejectOpacity = pan.x.interpolate({ inputRange: [-110, -20], outputRange: [1, 0], extrapolate: 'clamp' });

  return (
    <Animated.View
      {...responder.panHandlers}
      accessibilityHint={accessibilityHint}
      style={[
        styles.card,
        { backgroundColor: theme.backgroundElement, borderColor: theme.border },
        { transform: [{ translateX: pan.x }, { translateY: pan.y }, { rotate }] },
      ]}>
      {children}
      <Animated.View pointerEvents="none" style={[styles.stamp, styles.stampRight, { opacity: approveOpacity, borderColor: theme.success }]}>
        <ThemedText type="smallBold" themeColor="success">
          {rightLabel}
        </ThemedText>
      </Animated.View>
      <Animated.View pointerEvents="none" style={[styles.stamp, styles.stampLeft, { opacity: rejectOpacity, borderColor: theme.danger }]}>
        <ThemedText type="smallBold" themeColor="danger">
          {leftLabel}
        </ThemedText>
      </Animated.View>
    </Animated.View>
  );
}

/** Static look-alike shown behind the active card so the pile reads as a stack. */
export function CardBackdrop() {
  const theme = useTheme();
  return <View style={[styles.backdrop, { backgroundColor: theme.backgroundSelected }]} />;
}

const styles = StyleSheet.create({
  card: {
    width: '100%',
    minHeight: 300,
    borderRadius: 24,
    borderWidth: StyleSheet.hairlineWidth,
    padding: Spacing.four,
    justifyContent: 'space-between',
    gap: Spacing.three,
    // The web preview would otherwise select text while dragging.
    userSelect: 'none',
  },
  stamp: { position: 'absolute', top: 18, borderWidth: 2, borderRadius: 8, paddingHorizontal: 10, paddingVertical: 2 },
  stampRight: { left: 18, transform: [{ rotate: '-8deg' }] },
  stampLeft: { right: 18, transform: [{ rotate: '8deg' }] },
  backdrop: { position: 'absolute', left: 12, right: 12, top: 12, bottom: -10, borderRadius: 24, opacity: 0.6 },
});
