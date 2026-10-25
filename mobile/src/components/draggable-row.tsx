import * as Haptics from 'expo-haptics';
import { useEffect, useMemo, useRef, type ReactNode } from 'react';
import { PanResponder, Platform, View, type StyleProp, type ViewProps, type ViewStyle } from 'react-native';

/** How long a finger must stay down, without moving, before the row can be dragged. */
export const HOLD_MS = 400;
/** Moving further than this before the hold is up means the finger is scrolling or tapping, not lifting the row. */
const SLOP = 8;
/** If the hold is up but the finger never starts moving and no "lifted" signal arrives (a mouse has no touch-end), let go after this. */
const GIVE_UP_MS = 2500;

interface Props {
  children: ReactNode;
  style?: StyleProp<ViewStyle>;
  onLayout?: ViewProps['onLayout'];
  /** The finger has stayed down long enough: lock the page's scrolling so it cannot take the gesture. */
  onHold: () => void;
  /** The held row starts to move. */
  onLift: () => void;
  /** `dy` is how far the finger is above (negative) or below (positive) where it first touched. */
  onMove: (dy: number) => void;
  onDrop: (dy: number) => void;
  /** The hold ended without a drop (let go without moving, or the gesture was interrupted): unlock scrolling again. */
  onCancel: () => void;
}

/**
 * Wraps a row so it can be dragged up and down after pressing and holding anywhere on it. A quick tap, a press on a button
 * inside the row, and scrolling the page all work as before: nothing is taken until the finger has stayed put for HOLD_MS.
 * Built on core PanResponder like the swipe card (no native modules).
 */
export function DraggableRow({ children, style, onLayout, onHold, onLift, onMove, onDrop, onCancel }: Props) {
  const latest = useRef({ onHold, onLift, onMove, onDrop, onCancel });
  useEffect(() => {
    latest.current = { onHold, onLift, onMove, onDrop, onCancel };
  });
  const state = useRef({
    armed: false,
    active: false,
    /** Where on the page the finger first touched. The drag distance is measured from here, not from where the drag was picked up. */
    startY: 0,
    hold: null as ReturnType<typeof setTimeout> | null,
    giveUp: null as ReturnType<typeof setTimeout> | null,
  });

  const responder = useMemo(() => {
    const s = state.current;
    const clearTimers = () => {
      if (s.hold) clearTimeout(s.hold);
      if (s.giveUp) clearTimeout(s.giveUp);
      s.hold = s.giveUp = null;
    };
    /** Back to rest. Tells the screen only if it had been told about the hold. */
    const release = () => {
      clearTimers();
      const wasHeld = s.armed || s.active;
      s.armed = s.active = false;
      if (wasHeld) latest.current.onCancel();
    };
    // These handlers run from gesture events, never during render; the lint rule cannot see that.
    /* eslint-disable react-hooks/refs */
    return {
      release,
      handlers: PanResponder.create({
        // Seen on the way down, before any child gets the touch; returning false leaves the touch to the buttons inside.
        onStartShouldSetPanResponderCapture: (e) => {
          release();
          s.startY = e.nativeEvent.pageY;
          // A mouse (the browser preview) has no touch-end event on the row, so listen for the button coming up anywhere.
          if (Platform.OS === 'web') globalThis.addEventListener?.('pointerup', () => !s.active && release(), { once: true });
          s.hold = setTimeout(() => {
            s.hold = null;
            s.armed = true;
            Haptics.selectionAsync().catch(() => {}); // a small buzz where there is a motor; nothing elsewhere
            latest.current.onHold();
            s.giveUp = setTimeout(release, GIVE_UP_MS);
          }, HOLD_MS);
          return false;
        },
        // Once the hold is up, the first movement is taken from whatever would have had it (a button, the scrolling page).
        onMoveShouldSetPanResponderCapture: (_, g) => {
          if (s.armed) return true;
          if (Math.abs(g.dx) > SLOP || Math.abs(g.dy) > SLOP) clearTimers(); // it is a scroll or a swipe, not a hold
          return false;
        },
        onPanResponderTerminationRequest: () => false,
        onPanResponderGrant: () => {
          clearTimers();
          s.active = true;
          latest.current.onLift();
        },
        // The gesture tracker restarts its own distance count when it takes over (after the hold), which would drop the first bit of
        // movement and leave the row trailing the finger, so the distance is taken from where the finger first touched.
        onPanResponderMove: (_, g) => latest.current.onMove(g.moveY - s.startY),
        onPanResponderRelease: (_, g) => {
          clearTimers();
          const wasActive = s.active;
          s.armed = s.active = false;
          if (wasActive) latest.current.onDrop(g.moveY - s.startY);
        },
        onPanResponderTerminate: release,
      }),
    };
    /* eslint-enable react-hooks/refs */
  }, []);

  // Clear any pending timer if the row goes away mid-hold.
  useEffect(() => () => responder.release(), [responder]);

  return (
    <View
      {...responder.handlers.panHandlers}
      onLayout={onLayout}
      // A touch that ends while the hold is up but before any movement never reaches the responder handlers above.
      onTouchEnd={() => {
        if (!state.current.active) responder.release();
      }}
      onTouchCancel={() => {
        if (!state.current.active) responder.release();
      }}
      style={style}>
      {children}
    </View>
  );
}
