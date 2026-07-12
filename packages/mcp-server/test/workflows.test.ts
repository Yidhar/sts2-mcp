import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { actionId, findAction } from "../src/workflows";
import type { BridgeState } from "../src/types";

const envelope = JSON.parse(
  fs.readFileSync(
    path.resolve(__dirname, "../../../..", "contracts", "fixtures", "state.player-control.json"),
    "utf8"
  )
) as { state_version: number; state: BridgeState; legal_actions: BridgeState["available_actions"] };
const realV2State: BridgeState = {
  ...envelope.state,
  state_version: envelope.state_version,
  available_actions: envelope.legal_actions
};
const fixtureActions = realV2State.available_actions ?? [];

test("canonical v2 handle is the authoritative action identity", () => {
  const expected = fixtureActions[1]?.handle as string;
  const selected = findAction(realV2State, { actionId: expected });
  assert.equal(actionId(selected), expected);
});

test("map workflow reads canonical top-level coordinate metadata", () => {
  const selected = findAction(realV2State, { kinds: [fixtureActions[0]?.kind as string], x: 1, y: 2 });
  assert.equal(actionId(selected), fixtureActions[0]?.handle);
});

test("option workflow uses explicit canonical option index instead of array position", () => {
  const selected = findAction(realV2State, { kinds: ["event_option"], optionIndex: 0 });
  assert.equal(actionId(selected), fixtureActions[1]?.handle);
});
