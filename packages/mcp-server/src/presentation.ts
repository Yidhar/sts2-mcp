import type { BridgeState } from "./types";
import { redactSecrets, safeStringify } from "./redaction";

export function toolResult(payload: unknown, isError = false, secrets: readonly string[] = []) {
  const safePayload = redactSecrets(payload, secrets);
  return {
    isError,
    content: [{ type: "text" as const, text: safeStringify(safePayload) }],
    ...(safePayload && typeof safePayload === "object" && !Array.isArray(safePayload)
      ? { structuredContent: safePayload as Record<string, unknown> }
      : {})
  };
}

export function stateVersion(state: BridgeState): number {
  if (!Number.isSafeInteger(state.state_version)) {
    throw new Error("Bridge state does not contain an integer state_version.");
  }
  return state.state_version as number;
}

export function compactState(state: BridgeState): Record<string, unknown> {
  return {
    state_version: Number.isSafeInteger(state.state_version) ? state.state_version : null,
    screen: typeof state.screen === "string" ? state.screen : null,
    run: state.run ?? null,
    player: state.player ?? null,
    combat: state.combat ?? null,
    map: state.map ?? null,
    reward: state.reward ?? state.rewards ?? null,
    card_selection: state.card_selection ?? null,
    rest_site: state.rest_site ?? null,
    shop: state.shop ?? null,
    available_actions: Array.isArray(state.available_actions) ? state.available_actions : []
  };
}

export function listActions(state: BridgeState): Record<string, unknown> {
  return {
    state_version: Number.isSafeInteger(state.state_version) ? state.state_version : null,
    screen: typeof state.screen === "string" ? state.screen : null,
    actions: Array.isArray(state.available_actions) ? state.available_actions : []
  };
}
