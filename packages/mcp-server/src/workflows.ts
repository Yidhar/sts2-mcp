import { AppError } from "./errors";
import type { BridgeAction, BridgeState } from "./types";

export function actionsFromState(state: BridgeState): BridgeAction[] {
  return Array.isArray(state.available_actions)
    ? state.available_actions.filter(
        (action): action is BridgeAction => Boolean(action && typeof action === "object")
      )
    : [];
}

export function actionId(action: BridgeAction): string | null {
  const value = action.handle ?? action.action_handle ?? action.action_id ?? action.action;
  return typeof value === "string" && value.trim() ? value : null;
}

export function actionKind(action: BridgeAction): string {
  const value = action.kind ?? action.type;
  return typeof value === "string" ? value.toLowerCase() : "";
}

export function actionCoordinate(action: BridgeAction): { x: number; y: number } | null {
  const candidate = action.coord;
  if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) return null;
  const value = candidate as Record<string, unknown>;
  const x = Number.isSafeInteger(value.x) ? value.x : value.col;
  const y = Number.isSafeInteger(value.y) ? value.y : value.row;
  return Number.isSafeInteger(x) && Number.isSafeInteger(y)
    ? { x: x as number, y: y as number }
    : null;
}

export function actionOptionIndex(action: BridgeAction): number | null {
  const value = action.option_index ?? action.index;
  return Number.isSafeInteger(value) && (value as number) >= 0 ? (value as number) : null;
}

export function findAction(
  state: BridgeState,
  options: {
    actionId?: string;
    kinds?: readonly string[];
    optionIndex?: number;
    x?: number;
    y?: number;
  }
): BridgeAction {
  const actions = actionsFromState(state);
  if (options.actionId) {
    const exact = actions.find((action) => actionId(action) === options.actionId);
    if (exact) return exact;
    throw new AppError("action_not_available", `Action '${options.actionId}' is not currently legal.`, {
      details: { action_id: options.actionId, state_version: state.state_version ?? null }
    });
  }

  let candidates = actions;
  if (options.kinds && options.kinds.length > 0) {
    const accepted = new Set(options.kinds.map((kind) => kind.toLowerCase()));
    candidates = candidates.filter((action) => accepted.has(actionKind(action)));
  }
  if (Number.isSafeInteger(options.x) && Number.isSafeInteger(options.y)) {
    candidates = candidates.filter((action) => {
      const coord = actionCoordinate(action);
      return coord?.x === options.x && coord?.y === options.y;
    });
  }
  if (Number.isSafeInteger(options.optionIndex)) {
    const exact = candidates.find(
      (action) => actionOptionIndex(action) === (options.optionIndex as number)
    );
    if (exact) return exact;
    // Old v1 actions did not expose a stable option index. Positional fallback
    // is deliberately limited to that legacy wire shape.
    const selected = candidates.every((action) => action.handle === undefined)
      ? candidates[options.optionIndex as number]
      : undefined;
    if (selected) return selected;
  }
  if (candidates.length === 1) return candidates[0] as BridgeAction;
  if (candidates.length === 0) {
    throw new AppError("workflow_action_unavailable", "No legal action matches this workflow.", {
      details: { state_version: state.state_version ?? null, kinds: options.kinds ?? [] }
    });
  }
  throw new AppError("workflow_action_ambiguous", "Multiple legal actions match; pass action_id explicitly.", {
    details: {
      state_version: state.state_version ?? null,
      candidates: candidates.map((action) => actionId(action)).filter(Boolean)
    }
  });
}

export async function waitForStateChange(
  readState: () => Promise<BridgeState>,
  afterVersion: number,
  timeoutMs: number,
  pollMs: number
): Promise<{ changed: boolean; state: BridgeState }> {
  const deadline = Date.now() + timeoutMs;
  let state = await readState();
  while (Date.now() < deadline) {
    if (Number.isSafeInteger(state.state_version) && (state.state_version as number) > afterVersion) {
      return { changed: true, state };
    }
    await new Promise((resolve) => setTimeout(resolve, pollMs));
    state = await readState();
  }
  return { changed: false, state };
}
