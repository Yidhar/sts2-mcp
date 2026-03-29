"use strict";

const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const serverPath = path.join(__dirname, "index.js");
const args = new Set(process.argv.slice(2));
const timeoutMs = 90 * 60 * 1000;
const changeTimeoutMs = 10000;
const actionWaitMs = 1500;
const maxConsecutiveErrors = 6;
const maxRecoveryAttemptsPerSnapshot = 3;
const autoslayLogPath = path.join(
  process.env.APPDATA || path.join(process.env.USERPROFILE || "", "AppData", "Roaming"),
  "SlayTheSpire2",
  "bridge",
  "autoslay.log"
);

let stdoutBuffer = "";
let nextId = 1;
const pending = new Map();

const child = spawn(process.execPath, [serverPath], {
  stdio: ["pipe", "pipe", "inherit"],
  env: {
    ...process.env
  }
});

child.stdout.setEncoding("utf8");
child.stdout.on("data", (chunk) => {
  stdoutBuffer += chunk;
  drainMessages();
});

child.on("exit", (code, signal) => {
  if (pending.size === 0) {
    return;
  }

  const error = new Error(
    `MCP child exited before all responses arrived (code=${code}, signal=${signal})`
  );
  for (const [, entry] of pending) {
    entry.reject(error);
  }
  pending.clear();
});

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function parseMessages() {
  const messages = [];

  while (true) {
    const newlineIndex = stdoutBuffer.indexOf("\n");
    if (newlineIndex === -1) {
      return messages;
    }

    let line = stdoutBuffer.slice(0, newlineIndex);
    stdoutBuffer = stdoutBuffer.slice(newlineIndex + 1);

    if (line.endsWith("\r")) {
      line = line.slice(0, -1);
    }

    if (!line.trim()) {
      continue;
    }

    messages.push(JSON.parse(line));
  }
}

function drainMessages() {
  for (const message of parseMessages()) {
    if (!Object.prototype.hasOwnProperty.call(message, "id")) {
      continue;
    }

    const entry = pending.get(message.id);
    if (!entry) {
      continue;
    }

    pending.delete(message.id);
    entry.resolve(message);
  }
}

function sendMessage(message) {
  child.stdin.write(`${JSON.stringify(message)}\n`);
}

function sendRequest(method, params) {
  const id = nextId++;
  sendMessage({
    jsonrpc: "2.0",
    id,
    method,
    params
  });

  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
  });
}

async function initialize() {
  await sendRequest("initialize", {
    protocolVersion: "2025-03-26",
    capabilities: {},
    clientInfo: {
      name: "sts2-autoslay-runner",
      version: "0.4.0"
    }
  });

  sendMessage({
    jsonrpc: "2.0",
    method: "notifications/initialized",
    params: {}
  });
}

async function callTool(name, argumentsObject) {
  const message = await sendRequest("tools/call", {
    name,
    arguments: argumentsObject
  });

  if (message.error) {
    throw new Error(`Tool call failed for ${name}: ${message.error.message}`);
  }

  const result = message.result;
  if (!result || !Array.isArray(result.content) || result.content.length === 0) {
    throw new Error(`Tool ${name} returned no content.`);
  }

  const textItem = result.content.find((item) => item.type === "text");
  if (!textItem || typeof textItem.text !== "string") {
    throw new Error(`Tool ${name} returned no text content.`);
  }

  const payload = JSON.parse(textItem.text);
  if (result.isError) {
    const errorMessage =
      payload && typeof payload.message === "string"
        ? payload.message
        : `Tool ${name} returned an error result.`;
    const error = new Error(errorMessage);
    error.payload = payload;
    throw error;
  }

  return payload;
}

function extractState(payload) {
  if (!isPlainObject(payload)) {
    return null;
  }

  if (isPlainObject(payload.final_state)) {
    return payload.final_state;
  }

  if (isPlainObject(payload.state_after)) {
    return payload.state_after;
  }

  if (isPlainObject(payload.state)) {
    return payload.state;
  }

  return null;
}

async function getState() {
  return await callTool("sts2_get_state", {});
}

async function performAction(actionId, stateVersion) {
  const payload = await callTool("sts2_perform_action", {
    action_id: actionId,
    ...(Number.isInteger(stateVersion) ? { expected_state_version: stateVersion } : {}),
    wait_after_ms: actionWaitMs,
    return_state_after: true
  });
  const state = extractState(payload);
  return state ? { ...payload, state } : payload;
}

async function waitForChange(state) {
  try {
    return await callTool("sts2_wait_for_change", {
      ...(Number.isInteger(state?.state_version)
        ? { baseline_state_version: state.state_version }
        : {}),
      timeout_ms: changeTimeoutMs
    });
  } catch (error) {
    if (error?.payload?.error === "timeout") {
      return error.payload;
    }

    throw error;
  }
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function formatCoord(coord) {
  return Number.isInteger(coord?.col) && Number.isInteger(coord?.row)
    ? `${coord.col},${coord.row}`
    : null;
}

function getStateActionSurface(state) {
  if (isPlainObject(state?.rewards)) {
    if (state.rewards?.card_reward_selection?.visible === true) {
      return "card_reward";
    }

    if (
      state.rewards.in_reward_flow === true ||
      state.rewards?.rewards?.visible === true ||
      state.rewards?.has_proceed === true
    ) {
      return "reward";
    }
  }

  if (
    isPlainObject(state?.card_selection) &&
    (state.card_selection.in_card_selection_flow === true ||
      state.card_selection?.card_selection?.visible === true)
  ) {
    return "card_selection";
  }

  if (isPlainObject(state?.rest_site)) {
    if (state.rest_site?.deck_upgrade_selection?.visible === true) {
      return "deck_upgrade";
    }

    if (
      state.rest_site.in_rest_site_flow === true ||
      state.rest_site?.rest_site?.visible === true
    ) {
      return "rest_site";
    }
  }

  if (isPlainObject(state?.shop) && (state.shop.is_open === true || state.shop.visible === true)) {
    return "shop";
  }

  if (isPlainObject(state?.crystal_sphere)) {
    return "event_crystal_sphere";
  }

  if (state?.event_options?.visible === true) {
    return "event";
  }

  if (
    isPlainObject(state?.map) &&
    state.map.is_open === true &&
    state.map.is_traveling !== true &&
    getTravelablePoints(state).length > 0
  ) {
    return "map";
  }

  if (
    state?.screen === "COMBAT" &&
    state?.combat?.in_progress === true &&
    state?.combat?.is_play_phase === true &&
    state?.combat?.player_actions_disabled !== true
  ) {
    return "combat";
  }

  return null;
}

function summarizeState(state) {
  const hp =
    Number.isFinite(state?.player?.current_hp) && Number.isFinite(state?.player?.max_hp)
      ? `${state.player.current_hp}/${state.player.max_hp}`
      : "-";
  const summary = {
    screen: typeof state?.screen === "string" ? state.screen : null,
    surface: getStateActionSurface(state),
    game_over: state?.run?.is_game_over === true,
    act: typeof state?.run?.act === "string" ? state.run.act : null,
    total_floor: Number.isFinite(state?.run?.total_floor) ? state.run.total_floor : null,
    act_floor: Number.isFinite(state?.run?.act_floor) ? state.run.act_floor : null,
    room_type: typeof state?.run?.room_type === "string" ? state.run.room_type : null,
    current_coord: formatCoord(state?.run?.current_map_coord),
    hp,
    action_count: Array.isArray(state?.available_actions) ? state.available_actions.length : 0
  };

  if (summary.surface === "card_selection") {
    summary.card_selection = {
      min_select: Number.isFinite(state?.card_selection?.card_selection?.min_select)
        ? state.card_selection.card_selection.min_select
        : null,
      max_select: Number.isFinite(state?.card_selection?.card_selection?.max_select)
        ? state.card_selection.card_selection.max_select
        : null,
      selected_count: Number.isFinite(state?.card_selection?.card_selection?.selected_count)
        ? state.card_selection.card_selection.selected_count
        : null,
      option_count: Array.isArray(state?.card_selection?.card_selection?.options)
        ? state.card_selection.card_selection.options.length
        : 0
    };
  }

  if (summary.surface === "reward" || summary.surface === "card_reward") {
    summary.rewards = {
      reward_count: Array.isArray(state?.rewards?.rewards?.entries)
        ? state.rewards.rewards.entries.length
        : 0,
      card_option_count: Array.isArray(state?.rewards?.card_reward_selection?.options)
        ? state.rewards.card_reward_selection.options.length
        : 0
    };
  }

  if (summary.surface === "rest_site" || summary.surface === "deck_upgrade") {
    summary.rest_site = {
      option_count: Array.isArray(state?.rest_site?.rest_site?.options)
        ? state.rest_site.rest_site.options.length
        : 0,
      upgrade_option_count: Array.isArray(state?.rest_site?.deck_upgrade_selection?.options)
        ? state.rest_site.deck_upgrade_selection.options.length
        : 0
    };
  }

  if (summary.surface === "shop") {
    summary.shop = {
      is_open: state?.shop?.is_open === true,
      item_count: Array.isArray(state?.shop?.items) ? state.shop.items.length : 0
    };
  }

  if (summary.surface === "event") {
    summary.event_options = {
      option_count: Array.isArray(state?.event_options?.options)
        ? state.event_options.options.length
        : 0
    };
  }

  if (summary.surface === "event_crystal_sphere") {
    summary.crystal_sphere = {
      divinations_left: Number.isFinite(state?.crystal_sphere?.divinations_left)
        ? state.crystal_sphere.divinations_left
        : null,
      current_tool:
        typeof state?.crystal_sphere?.current_tool === "string"
          ? state.crystal_sphere.current_tool
          : null,
      hidden_cell_count: Number.isFinite(state?.crystal_sphere?.hidden_cell_count)
        ? state.crystal_sphere.hidden_cell_count
        : null,
      cell_count: Array.isArray(state?.crystal_sphere?.actions?.cells)
        ? state.crystal_sphere.actions.cells.length
        : 0
    };
  }

  if (summary.surface === "map") {
    summary.map = {
      travelable_count: getTravelablePoints(state).length
    };
  }

  return summary;
}

function buildStateFingerprint(state) {
  return JSON.stringify(summarizeState(state));
}

function printRecord(prefix, state, extra = {}) {
  console.log(
    JSON.stringify(
      {
        prefix,
        at: new Date().toISOString(),
        ...(isPlainObject(state) ? summarizeState(state) : {}),
        ...extra
      },
      null,
      2
    )
  );
}

function hasAction(state, actionId) {
  return (
    Array.isArray(state?.available_actions) &&
    state.available_actions.some((action) => action && action.action_id === actionId)
  );
}

function canStartAutoSlayInState(state) {
  return typeof state?.screen === "string" && state.screen === "MAIN_MENU";
}

async function getCanonicalState(fallbackPayload = null) {
  try {
    return await getState();
  } catch (error) {
    const fallbackState = extractState(fallbackPayload);
    if (fallbackState) {
      return fallbackState;
    }

    throw error;
  }
}

function getLogSignature(logFilePath) {
  if (!logFilePath || !fs.existsSync(logFilePath)) {
    return null;
  }

  const stats = fs.statSync(logFilePath);
  return {
    size: stats.size,
    mtime_ms: stats.mtimeMs
  };
}

function didLogChange(initialSignature, logFilePath) {
  const currentSignature = getLogSignature(logFilePath);
  if (!currentSignature) {
    return false;
  }

  if (!initialSignature) {
    return true;
  }

  return (
    currentSignature.size !== initialSignature.size ||
    currentSignature.mtime_ms !== initialSignature.mtime_ms
  );
}

function classifyAutoSlayOutcome(logFilePath) {
  const logTail = readAutoSlayLogTail(logFilePath, 40);
  if (!logTail) {
    return null;
  }

  if (
    logTail.includes("Run completed successfully") ||
    logTail.includes("Victory! Run completed and returned to main menu")
  ) {
    return "victory";
  }

  if (
    logTail.includes("Run failed") ||
    logTail.includes("Defeat") ||
    logTail.includes("Game over")
  ) {
    return "failure";
  }

  return null;
}

function readAutoSlayLogTail(logFilePath, lineCount) {
  if (!logFilePath || !fs.existsSync(logFilePath)) {
    return null;
  }

  const text = fs.readFileSync(logFilePath, "utf8");
  return text
    .split(/\r?\n/)
    .filter(Boolean)
    .slice(-lineCount)
    .join("\n");
}

function getFreshAutoSlayOutcome(initialSignature) {
  if (!didLogChange(initialSignature, autoslayLogPath)) {
    return null;
  }

  return classifyAutoSlayOutcome(autoslayLogPath);
}

function isActionNotAvailableError(error) {
  return error?.payload?.error === "action_not_available";
}

function isRecoverableRecoveryError(error) {
  const code = error?.payload?.error;
  return code === "state_version_conflict" || code === "action_not_available";
}

async function tryStartAutoSlay(state, prefix = "autoslay_started") {
  if (!canStartAutoSlayInState(state)) {
    return null;
  }

  try {
    const result = await performAction("automation:start_autoslay", state?.state_version);
    const nextState = await getCanonicalState(result);
    printRecord(prefix, nextState, {
      action_id: "automation:start_autoslay"
    });
    return nextState;
  } catch (error) {
    if (isActionNotAvailableError(error)) {
      return null;
    }

    throw error;
  }
}

async function ensureAutoSlayStarted(initialState) {
  let state = initialState ?? (await getState());

  if (args.has("--fresh-run") && hasAction(state, "main_menu:continue")) {
    if (!hasAction(state, "main_menu:abandon_current_game")) {
      throw new Error("Current save exists, but abandon_current_game is not available.");
    }

    const abandonResult = await performAction(
      "main_menu:abandon_current_game",
      state.state_version
    );
    state = abandonResult.state || (await getState());
    printRecord("fresh_run_abandon_requested", state);

    if (!hasAction(state, "main_menu:confirm_abandon_run")) {
      throw new Error("Abandon confirmation did not appear.");
    }

    const confirmResult = await performAction(
      "main_menu:confirm_abandon_run",
      state.state_version
    );
    state = confirmResult.state || (await getState());
    printRecord("fresh_run_abandoned", state);
  }

  const startedState = await tryStartAutoSlay(state, "autoslay_started");
  if (startedState) {
    return startedState;
  }

  printRecord("autoslay_start_deferred", state, {
    reason: canStartAutoSlayInState(state)
      ? "automation:start_autoslay unavailable in compact tool surface"
      : "autoslay_start_only_supported_from_main_menu"
  });
  return state;
}

function getTravelablePoints(state) {
  const compactPoints = Array.isArray(state?.map?.travelable_points)
    ? state.map.travelable_points
    : [];
  if (compactPoints.length > 0) {
    return compactPoints
      .filter((point) => Number.isInteger(point?.coord?.col) && Number.isInteger(point?.coord?.row))
      .map((point) => ({
        coord: {
          col: point.coord.col,
          row: point.coord.row
        },
        point_type: typeof point?.point_type === "string" ? point.point_type : null
      }));
  }

  const rawPoints = Array.isArray(state?.map?.points) ? state.map.points : [];
  return rawPoints
    .filter(
      (point) =>
        (point?.is_travelable === true || point?.state === "Travelable") &&
        Number.isInteger(point?.coord?.col) &&
        Number.isInteger(point?.coord?.row)
    )
    .map((point) => ({
      coord: {
        col: point.coord.col,
        row: point.coord.row
      },
      point_type: typeof point?.point_type === "string" ? point.point_type : null
    }));
}

function chooseTravelTarget(state) {
  const travelablePoints = getTravelablePoints(state).sort((left, right) => {
    if (left.coord.row !== right.coord.row) {
      return left.coord.row - right.coord.row;
    }

    return left.coord.col - right.coord.col;
  });
  return travelablePoints[0] ?? null;
}

function chooseGenericEventOptionIndex(state) {
  const options = Array.isArray(state?.event_options?.options) ? state.event_options.options : [];
  const indexedOptions = options.filter((option) => Number.isInteger(option?.index));
  if (indexedOptions.length <= 0) {
    return null;
  }

  const enabledOptions = indexedOptions.filter(
    (option) => option?.is_enabled !== false && option?.action_available !== false
  );
  const candidateOptions = enabledOptions.length > 0 ? enabledOptions : indexedOptions;
  const nonProceedOption =
    candidateOptions.find((option) => option?.is_proceed !== true) ?? null;
  const proceedOption =
    candidateOptions.find((option) => option?.is_proceed === true) ?? null;
  return (nonProceedOption ?? proceedOption)?.index ?? null;
}

function parseIndexedActionDescriptor(value) {
  if (typeof value !== "string") {
    return null;
  }

  const match = /^(\d+):([^*]+)(\*)?$/.exec(value.trim());
  if (!match) {
    return null;
  }

  return {
    index: Number.parseInt(match[1], 10),
    label: match[2],
    selected: match[3] === "*"
  };
}

function chooseCrystalSphereOptionIndex(state) {
  const sphere = isPlainObject(state?.crystal_sphere) ? state.crystal_sphere : null;
  if (!sphere) {
    return null;
  }

  const actions = isPlainObject(sphere.actions) ? sphere.actions : {};
  const controls = Array.isArray(actions.controls)
    ? actions.controls.map(parseIndexedActionDescriptor).filter((entry) => entry !== null)
    : [];
  const proceed = parseIndexedActionDescriptor(actions.proceed);
  const cellActionStartIndex = Number.isInteger(actions.cell_action_start_index)
    ? actions.cell_action_start_index
    : null;
  const hasCells = Array.isArray(actions.cells) && actions.cells.length > 0;
  const currentTool =
    typeof sphere.current_tool === "string" ? sphere.current_tool.toLowerCase() : null;
  const selectedControl = controls.find((control) => control.selected) ?? null;
  const smallControl =
    controls.find((control) => control.label.toLowerCase() === "small") ?? null;
  const bigControl =
    controls.find((control) => control.label.toLowerCase() === "big") ?? null;

  if (
    hasCells &&
    cellActionStartIndex !== null &&
    (currentTool === "small" || currentTool === "big" || selectedControl !== null)
  ) {
    return cellActionStartIndex;
  }

  if (smallControl && currentTool !== "small") {
    return smallControl.index;
  }

  if (bigControl && currentTool !== "big" && !smallControl) {
    return bigControl.index;
  }

  if (hasCells && cellActionStartIndex !== null) {
    return cellActionStartIndex;
  }

  if (proceed) {
    return proceed.index;
  }

  if (smallControl) {
    return smallControl.index;
  }

  if (bigControl) {
    return bigControl.index;
  }

  return null;
}

function chooseRestSiteOptionIndex(state) {
  const options = Array.isArray(state?.rest_site?.rest_site?.options)
    ? state.rest_site.rest_site.options
    : [];
  if (options.length <= 0) {
    return 0;
  }

  const hpRatio =
    Number.isFinite(state?.player?.current_hp) && Number.isFinite(state?.player?.max_hp)
      ? state.player.current_hp / state.player.max_hp
      : null;
  const smithOption =
    options.find(
      (option) =>
        option?.option_type === "smith" ||
        /smith|锻造|升级/i.test(`${option?.title ?? ""} ${option?.description ?? ""}`)
    ) ?? null;
  const restOption =
    options.find(
      (option) =>
        option?.option_type === "rest" ||
        /rest|休息|恢复|治疗/i.test(`${option?.title ?? ""} ${option?.description ?? ""}`)
    ) ?? null;

  if (hpRatio !== null && hpRatio >= 0.65 && smithOption && Number.isInteger(smithOption.index)) {
    return smithOption.index;
  }

  if (restOption && Number.isInteger(restOption.index)) {
    return restOption.index;
  }

  const firstIndexedOption = options.find((option) => Number.isInteger(option?.index)) ?? null;
  return firstIndexedOption?.index ?? 0;
}

function buildCardSelectionRecoveryRequest(state) {
  const cardSelection = isPlainObject(state?.card_selection?.card_selection)
    ? state.card_selection.card_selection
    : null;
  if (!cardSelection || cardSelection.visible !== true) {
    return null;
  }

  const selectedCount = Number.isFinite(cardSelection.selected_count)
    ? cardSelection.selected_count
    : 0;
  const minSelect = Number.isFinite(cardSelection.min_select) ? cardSelection.min_select : 0;
  const selectableIndices = Array.isArray(cardSelection.options)
    ? cardSelection.options
      .filter((option) => Number.isInteger(option?.index) && option?.is_selected !== true)
      .map((option) => option.index)
    : [];
  const remainingRequiredSelections = Math.max(0, minSelect - selectedCount);

  if (remainingRequiredSelections > 0) {
    if (selectableIndices.length < remainingRequiredSelections) {
      return null;
    }

    return {
      select_indices: selectableIndices.slice(0, remainingRequiredSelections),
      terminal_action: cardSelection.confirm_visible === true ? "confirm" : "none",
      ...(Number.isFinite(cardSelection.min_select)
        ? { expected_min_select: cardSelection.min_select }
        : {}),
      ...(Number.isFinite(cardSelection.max_select)
        ? { expected_max_select: cardSelection.max_select }
        : {})
    };
  }

  if (cardSelection.confirm_visible === true) {
    return {
      terminal_action: "confirm"
    };
  }

  if (cardSelection.skip_visible === true) {
    return {
      terminal_action: "skip"
    };
  }

  if (cardSelection.close_visible === true) {
    return {
      terminal_action: "close"
    };
  }

  if (cardSelection.cancel_visible === true) {
    return {
      terminal_action: "cancel"
    };
  }

  if (selectableIndices.length > 0) {
    return {
      select_indices: [selectableIndices[0]],
      terminal_action: "none"
    };
  }

  return null;
}

async function recoverSurface(state) {
  const surface = getStateActionSurface(state);
  if (!surface || surface === "combat") {
    return null;
  }

  switch (surface) {
    case "reward":
    case "card_reward": {
      const payload = await callTool("sts2_resolve_room_rewards", {
        claim_all_safe_rewards: true,
        take_potions: true,
        pick_card_index: 0,
        auto_proceed: true
      });
      return {
        surface,
        strategy: "resolve_room_rewards:first_card",
        payload,
        state: await getCanonicalState(payload)
      };
    }

    case "rest_site":
    case "deck_upgrade": {
      const upgradeOptions = Array.isArray(state?.rest_site?.deck_upgrade_selection?.options)
        ? state.rest_site.deck_upgrade_selection.options
        : [];
      const payload = await callTool("sts2_resolve_rest_site", {
        option_index: chooseRestSiteOptionIndex(state),
        ...(upgradeOptions.length > 0 ? { upgrade_card_index: 0 } : {}),
        auto_proceed: true
      });
      return {
        surface,
        strategy: upgradeOptions.length > 0 ? "resolve_rest_site:first_upgrade" : "resolve_rest_site:auto",
        payload,
        state: await getCanonicalState(payload)
      };
    }

    case "card_selection": {
      const request = buildCardSelectionRecoveryRequest(state);
      if (!request) {
        return null;
      }

      const payload = await callTool("sts2_resolve_card_selection", request);
      return {
        surface,
        strategy: "resolve_card_selection:auto",
        payload,
        state: await getCanonicalState(payload)
      };
    }

    case "shop": {
      const payload = await callTool("sts2_resolve_shop_visit", {
        open_shop: false,
        close_inventory: true,
        leave_shop: true,
        purchases: []
      });
      return {
        surface,
        strategy: "resolve_shop_visit:leave",
        payload,
        state: await getCanonicalState(payload)
      };
    }

    case "event": {
      const index = chooseGenericEventOptionIndex(state);
      if (!Number.isInteger(index)) {
        return null;
      }

      const payload = await callTool("sts2_pick_option", {
        index,
        surface: "event"
      });
      return {
        surface,
        strategy: "pick_option:event_first_enabled",
        payload,
        state: await getCanonicalState(payload)
      };
    }

    case "event_crystal_sphere": {
      const index = chooseCrystalSphereOptionIndex(state);
      if (!Number.isInteger(index)) {
        return null;
      }

      const payload = await callTool("sts2_pick_option", {
        index,
        surface: "event"
      });
      return {
        surface,
        strategy: "pick_option:crystal_sphere",
        payload,
        state: await getCanonicalState(payload)
      };
    }

    case "map": {
      const target = chooseTravelTarget(state);
      if (!target) {
        return null;
      }

      const payload = await callTool("sts2_travel_to_coordinate", {
        col: target.coord.col,
        row: target.coord.row,
        wait_after_ms: actionWaitMs
      });
      return {
        surface,
        strategy: "travel_to_coordinate:leftmost",
        payload,
        state: await getCanonicalState(payload)
      };
    }

    default:
      return null;
  }
}

function getRecoverySnapshotKey(state, surface) {
  return `${surface}|${buildStateFingerprint(state)}`;
}

function getTerminalPrefix(state, logOutcome) {
  if (state?.run?.is_game_over === true) {
    return logOutcome === "victory" ? "terminal_victory" : "terminal_game_over";
  }

  if (typeof state?.screen === "string" && state.screen === "MAIN_MENU") {
    if (logOutcome === "victory") {
      return "terminal_victory_from_log";
    }

    if (logOutcome === "failure") {
      return "terminal_game_over_from_log";
    }
  }

  return null;
}

async function monitorUntilTerminal(initialState) {
  const startedAt = Date.now();
  const initialLogSignature = getLogSignature(autoslayLogPath);
  let consecutiveErrors = 0;
  let timeoutCount = 0;
  let state = initialState ?? (await getState());
  let lastPrintedFingerprint = buildStateFingerprint(state);
  let lastRecoveryKey = null;
  let lastRecoveryAttemptCount = 0;

  while (Date.now() - startedAt < timeoutMs) {
    const logOutcome = getFreshAutoSlayOutcome(initialLogSignature);
    const terminalPrefix = getTerminalPrefix(state, logOutcome);
    if (terminalPrefix) {
      printRecord(terminalPrefix, state);
      return;
    }

    if (!args.has("--monitor-only")) {
      const surface = getStateActionSurface(state);
      if (surface && surface !== "combat") {
        const recoveryKey = getRecoverySnapshotKey(state, surface);
        if (recoveryKey !== lastRecoveryKey) {
          lastRecoveryKey = recoveryKey;
          lastRecoveryAttemptCount = 0;
        }
        if (lastRecoveryAttemptCount < maxRecoveryAttemptsPerSnapshot) {
          lastRecoveryAttemptCount += 1;
          let recovery = null;
          try {
            recovery = await recoverSurface(state);
          } catch (error) {
            if (isRecoverableRecoveryError(error)) {
              state = await getCanonicalState(error?.payload);
              const nextFingerprint = buildStateFingerprint(state);
              if (nextFingerprint !== lastPrintedFingerprint) {
                printRecord("recovery_race", state, {
                  surface,
                  attempt: lastRecoveryAttemptCount,
                  reason: error?.payload?.error ?? "unknown"
                });
                lastPrintedFingerprint = nextFingerprint;
              }
              continue;
            }

            throw error;
          }
          if (recovery) {
            state = recovery.state;
            timeoutCount = 0;
            const nextFingerprint = buildStateFingerprint(state);
            if (nextFingerprint !== lastPrintedFingerprint) {
              printRecord("recovery", state, {
                surface: recovery.surface,
                strategy: recovery.strategy,
                attempt: lastRecoveryAttemptCount,
                resolved: recovery.payload?.resolved === true,
                ok: recovery.payload?.ok !== false
              });
              lastPrintedFingerprint = nextFingerprint;
            } else {
              printRecord("recovery_noop", state, {
                surface: recovery.surface,
                strategy: recovery.strategy,
                attempt: lastRecoveryAttemptCount,
                resolved: recovery.payload?.resolved === true,
                ok: recovery.payload?.ok !== false
              });
            }
            continue;
          }
        }
      }
    }

    let changePayload;
    try {
      changePayload = await waitForChange(state);
      consecutiveErrors = 0;
    } catch (error) {
      consecutiveErrors += 1;
      console.error(`[monitor] wait_for_change failed (${consecutiveErrors}): ${error.message}`);

      if (error.payload) {
        console.error(JSON.stringify(error.payload, null, 2));
      }

      const logOutcomeAfterError = getFreshAutoSlayOutcome(initialLogSignature);
      if (logOutcomeAfterError === "victory") {
        console.log(
          JSON.stringify(
            {
              prefix: "terminal_victory_from_log",
              at: new Date().toISOString(),
              log_file_path: autoslayLogPath
            },
            null,
            2
          )
        );
        return;
      }

      if (consecutiveErrors >= maxConsecutiveErrors) {
        throw new Error("Too many consecutive wait_for_change failures.");
      }

      continue;
    }

    const nextState = extractState(changePayload) || changePayload?.state;
    if (isPlainObject(nextState)) {
      state = nextState;
    }

    if (changePayload?.changed === true) {
      timeoutCount = 0;
      const nextFingerprint = buildStateFingerprint(state);
      if (nextFingerprint !== lastPrintedFingerprint) {
        printRecord("update", state);
        lastPrintedFingerprint = nextFingerprint;
      } else {
        await sleep(250);
      }
      continue;
    }

    timeoutCount += 1;
    if (!args.has("--monitor-only") && canStartAutoSlayInState(state)) {
      const restartedState = await tryStartAutoSlay(state, "autoslay_restarted");
      if (restartedState) {
        state = restartedState;
        lastPrintedFingerprint = buildStateFingerprint(state);
        timeoutCount = 0;
        continue;
      }
    }

    if (timeoutCount % 3 === 0) {
      printRecord("waiting", state, {
        consecutive_timeouts: timeoutCount
      });
    }
  }

  throw new Error("Timed out waiting for AutoSlay to reach a terminal state.");
}

(async () => {
  await initialize();

  let state = await getState();
  printRecord("initial", state);

  if (!args.has("--monitor-only")) {
    state = await ensureAutoSlayStarted(state);
  }

  await monitorUntilTerminal(state);
})()
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  })
  .finally(() => {
    child.kill();
  });
