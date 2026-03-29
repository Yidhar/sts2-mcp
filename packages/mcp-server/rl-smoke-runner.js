"use strict";

const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const DEFAULT_EPISODES = 3;
const DEFAULT_MAX_STEPS = 200;
const DEFAULT_HEALTH_TIMEOUT_MS = 10_000;
const DEFAULT_RESET_TIMEOUT_MS = 30_000;
const DEFAULT_STEP_TIMEOUT_MS = 15_000;
const DEFAULT_BRIDGE_STARTUP_TIMEOUT_MS = 120_000;
const DEFAULT_POLL_INTERVAL_MS = 1_000;
const DEFAULT_RESTART_DELAY_MS = 3_000;
const DEFAULT_HTTP_TIMEOUT_GRACE_MS = 5_000;
const DEFAULT_MAX_RESET_ATTEMPTS = 3;
const DEFAULT_MAX_CONSECUTIVE_STEP_ERRORS = 3;
const DEFAULT_MAX_STARTUP_ATTEMPTS = 3;
const DEFAULT_READY_STABLE_POLLS = 2;
const DEFAULT_LOG_DIR = path.join(__dirname, "rl-smoke-logs");
const DEFAULT_GAME_EXE_CANDIDATES = buildDefaultGameExeCandidates();
const TRANSIENT_BRIDGE_ERROR_CODES = new Set([
  "dispatcher_not_ready",
  "bridge_request_failed"
]);
const SESSION_FILE_PATH = path.join(
  process.env.APPDATA ||
    path.join(process.env.USERPROFILE || "", "AppData", "Roaming"),
  "SlayTheSpire2",
  "bridge",
  "session.json"
);

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const logFilePath = resolveLogFilePath(options.logFile);
  fs.mkdirSync(path.dirname(logFilePath), { recursive: true });
  const logStream = fs.createWriteStream(logFilePath, { flags: "a" });
  const rng = createMulberry32(options.seed);

  const log = (type, data = {}) => {
    const entry = {
      ts: new Date().toISOString(),
      type,
      ...data
    };
    logStream.write(`${JSON.stringify(entry)}\n`);
  };

  try {
    log("runner_started", {
      options: redactOptionsForLog(options),
      session_file_path: SESSION_FILE_PATH
    });

    let session = await ensureBridgeReady(options, log, "startup");
    const spec = await bridgeRequestJson(session, "env/spec", {
      method: "GET",
      timeoutMs: options.healthTimeoutMs
    });
    log("env_spec", { spec });

    console.log(`rl-smoke-runner started. log=${logFilePath}`);
    let stopRunner = false;

    for (let episodeNumber = 1; episodeNumber <= options.episodes; episodeNumber++) {
      if (stopRunner) {
        break;
      }

      if (options.restartBeforeEpisode) {
        session = await hardRestartGame(options, log, `episode_${episodeNumber}_pre_reset_restart`);
      }

      const resetResult = await resetEpisodeWithRecovery(session, options, log, episodeNumber);
      session = resetResult.session;

      let current = resetResult.response;
      let stepErrors = 0;
      let episodeEnded = false;
      let alreadyRestartedThisEpisode = false;

      log("episode_started", {
        episode_number: episodeNumber,
        episode_id: current.episode_id,
        obs: current.obs,
        legal_actions: current.legal_actions,
        reset_actions: current.info?.reset_actions || []
      });

      console.log(
        `episode ${episodeNumber}/${options.episodes} reset ok: phase=${current.obs?.phase} episode_id=${current.episode_id}`
      );

      for (let stepNumber = 0; stepNumber < options.maxStepsPerEpisode; stepNumber++) {
        if (current.done) {
          const outcome = inferEpisodeOutcome(current);
          log("episode_done", {
            episode_number: episodeNumber,
            episode_id: current.episode_id,
            step_index: current.step_index,
            outcome,
            response: current
          });
          console.log(
            `episode ${episodeNumber} done at step_index=${current.step_index} outcome=${outcome}`
          );
          episodeEnded = true;
          break;
        }

        if (!Array.isArray(current.legal_actions) || current.legal_actions.length === 0) {
          log("episode_stalled", {
            episode_number: episodeNumber,
            episode_id: current.episode_id,
            response: current
          });
          console.log(`episode ${episodeNumber} stalled: no legal actions`);
          break;
        }

        if (
          options.chaosRestartProb > 0 &&
          stepNumber > 0 &&
          rng() < options.chaosRestartProb
        ) {
          log("episode_chaos_restart", {
            episode_number: episodeNumber,
            episode_id: current.episode_id,
            step_number: stepNumber,
            phase: current.obs?.phase
          });
          console.log(`episode ${episodeNumber} chaos restart at step ${stepNumber}`);
          session = await hardRestartGame(options, log, `episode_${episodeNumber}_chaos_restart`);
          alreadyRestartedThisEpisode = true;
          break;
        }

        const chosenAction = chooseRandomAction(current.legal_actions, rng);
        log("step_selected", {
          episode_number: episodeNumber,
          episode_id: current.episode_id,
          step_number: stepNumber,
          action: chosenAction
        });

        try {
          const next = await bridgeRequestJson(session, "env/step", {
            method: "POST",
            timeoutMs: options.stepTimeoutMs + DEFAULT_HTTP_TIMEOUT_GRACE_MS,
            body: {
              episode_id: current.episode_id,
              action_id: chosenAction.action_id,
              timeout_ms: options.stepTimeoutMs
            }
          });

          current = next;
          stepErrors = 0;

          log("step_result", {
            episode_number: episodeNumber,
            episode_id: current.episode_id,
            step_number: stepNumber,
            action_id: chosenAction.action_id,
            response: current
          });

          console.log(
            `episode ${episodeNumber} step ${stepNumber}: ${chosenAction.action_id} -> phase=${current.obs?.phase} reward=${formatScalar(current.reward)} done=${current.done === true}`
          );
        } catch (error) {
          const recoveredCurrent = tryRecoverCurrentFromActionError(current, error);
          if (recoveredCurrent) {
            current = recoveredCurrent;
            stepErrors = 0;
            log("step_surface_recovered", {
              episode_number: episodeNumber,
              episode_id: current.episode_id,
              step_number: stepNumber,
              action_id: chosenAction.action_id,
              recovered_phase: current.obs?.phase,
              recovered_legal_action_count: Array.isArray(current.legal_actions)
                ? current.legal_actions.length
                : 0,
              error: serializeError(error)
            });
            stepNumber--;
            continue;
          }

          if (isTransientBridgeError(error)) {
            log("step_wait_retry", {
              episode_number: episodeNumber,
              episode_id: current.episode_id,
              step_number: stepNumber,
              action_id: chosenAction.action_id,
              error: serializeError(error)
            });
            await sleep(options.pollIntervalMs);
            stepNumber--;
            continue;
          }

          stepErrors++;
          log("step_error", {
            episode_number: episodeNumber,
            episode_id: current.episode_id,
            step_number: stepNumber,
            action_id: chosenAction.action_id,
            error: serializeError(error)
          });

          console.log(
            `episode ${episodeNumber} step ${stepNumber} error: ${error.message}`
          );

          if (stepErrors >= options.maxConsecutiveStepErrors) {
            log("episode_aborted", {
              episode_number: episodeNumber,
              episode_id: current.episode_id,
              reason: "too_many_step_errors",
              error_count: stepErrors
            });
            console.log(`episode ${episodeNumber} aborted after repeated step errors`);
            stopRunner = true;
            break;
          }
        }
      }

      if (options.restartAfterEpisode && !alreadyRestartedThisEpisode) {
        session = await hardRestartGame(options, log, `episode_${episodeNumber}_post_restart`);
      }
    }

    log("runner_finished", { ok: true });
    console.log("rl-smoke-runner finished.");
  } finally {
    logStream.end();
  }
}

function parseArgs(argv) {
  const options = {
    episodes: DEFAULT_EPISODES,
    maxStepsPerEpisode: DEFAULT_MAX_STEPS,
    healthTimeoutMs: DEFAULT_HEALTH_TIMEOUT_MS,
    resetTimeoutMs: DEFAULT_RESET_TIMEOUT_MS,
    stepTimeoutMs: DEFAULT_STEP_TIMEOUT_MS,
    bridgeStartupTimeoutMs: DEFAULT_BRIDGE_STARTUP_TIMEOUT_MS,
    pollIntervalMs: DEFAULT_POLL_INTERVAL_MS,
    restartDelayMs: DEFAULT_RESTART_DELAY_MS,
    maxResetAttempts: DEFAULT_MAX_RESET_ATTEMPTS,
    maxConsecutiveStepErrors: DEFAULT_MAX_CONSECUTIVE_STEP_ERRORS,
    maxStartupAttempts: DEFAULT_MAX_STARTUP_ATTEMPTS,
    readyStablePolls: DEFAULT_READY_STABLE_POLLS,
    launchGame: true,
    restartBeforeEpisode: false,
    restartAfterEpisode: false,
    defensiveBuffs: true,
    chaosRestartProb: 0,
    character: null,
    gameExe: null,
    logFile: null,
    seed: Date.now() & 0xffffffff
  };

  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      printHelpAndExit(0);
    }

    const [flag, inlineValue] = arg.startsWith("--") && arg.includes("=")
      ? [arg.slice(0, arg.indexOf("=")), arg.slice(arg.indexOf("=") + 1)]
      : [arg, null];

    switch (flag) {
      case "--episodes":
        options.episodes = parsePositiveInteger(readValue(argv, index, inlineValue, flag), flag);
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--max-steps":
        options.maxStepsPerEpisode = parsePositiveInteger(
          readValue(argv, index, inlineValue, flag),
          flag
        );
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--health-timeout-ms":
        options.healthTimeoutMs = parsePositiveInteger(readValue(argv, index, inlineValue, flag), flag);
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--reset-timeout-ms":
        options.resetTimeoutMs = parsePositiveInteger(readValue(argv, index, inlineValue, flag), flag);
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--step-timeout-ms":
        options.stepTimeoutMs = parsePositiveInteger(readValue(argv, index, inlineValue, flag), flag);
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--bridge-startup-timeout-ms":
        options.bridgeStartupTimeoutMs = parsePositiveInteger(
          readValue(argv, index, inlineValue, flag),
          flag
        );
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--poll-interval-ms":
        options.pollIntervalMs = parsePositiveInteger(readValue(argv, index, inlineValue, flag), flag);
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--restart-delay-ms":
        options.restartDelayMs = parsePositiveInteger(readValue(argv, index, inlineValue, flag), flag);
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--max-reset-attempts":
        options.maxResetAttempts = parsePositiveInteger(readValue(argv, index, inlineValue, flag), flag);
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--max-consecutive-step-errors":
        options.maxConsecutiveStepErrors = parsePositiveInteger(
          readValue(argv, index, inlineValue, flag),
          flag
        );
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--max-startup-attempts":
        options.maxStartupAttempts = parsePositiveInteger(readValue(argv, index, inlineValue, flag), flag);
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--ready-stable-polls":
        options.readyStablePolls = parsePositiveInteger(readValue(argv, index, inlineValue, flag), flag);
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--chaos-restart-prob":
        options.chaosRestartProb = parseProbability(readValue(argv, index, inlineValue, flag), flag);
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--character":
        options.character = requireNonEmptyString(readValue(argv, index, inlineValue, flag), flag);
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--game-exe":
        options.gameExe = requireNonEmptyString(readValue(argv, index, inlineValue, flag), flag);
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--log-file":
        options.logFile = requireNonEmptyString(readValue(argv, index, inlineValue, flag), flag);
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--seed":
        options.seed = parseNonNegativeInteger(readValue(argv, index, inlineValue, flag), flag);
        if (inlineValue === null) {
          index++;
        }
        break;
      case "--launch-game":
        options.launchGame = true;
        break;
      case "--no-launch-game":
        options.launchGame = false;
        break;
      case "--restart-before-episode":
        options.restartBeforeEpisode = true;
        break;
      case "--restart-after-episode":
        options.restartAfterEpisode = true;
        break;
      case "--defensive-buffs":
        options.defensiveBuffs = true;
        break;
      case "--no-defensive-buffs":
        options.defensiveBuffs = false;
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  options.gameExe = resolveGameExePath(options.gameExe);
  if (!options.gameExe && options.launchGame) {
    throw new Error(
      "Could not resolve SlayTheSpire2.exe. Pass --game-exe <path> or set STS2_GAME_EXE."
    );
  }

  return options;
}

function printHelpAndExit(code) {
  const lines = [
    "Usage: node rl-smoke-runner.js [options]",
    "",
    "Options:",
    "  --episodes <n>                    Number of episodes to run.",
    "  --max-steps <n>                   Maximum env/step calls per episode.",
    "  --character <id>                  Optional env/reset character id, e.g. CHARACTER.REGENT.",
    "  --game-exe <path>                 Explicit SlayTheSpire2.exe path.",
    "  --launch-game / --no-launch-game  Allow the runner to start the game when needed.",
    "  --max-startup-attempts <n>        Max launch/relaunch attempts while stabilizing startup.",
    "  --ready-stable-polls <n>          Consecutive healthy state polls required before startup is treated as ready.",
    "  --restart-before-episode          Hard-restart the game before each episode reset.",
    "  --restart-after-episode           Hard-restart the game after each episode.",
    "  --defensive-buffs / --no-defensive-buffs",
    "                                    Toggle RL-only defensive buffs during episodes. Defaults to on.",
    "  --chaos-restart-prob <0..1>       Randomly hard-restart mid-episode to test restart handling.",
    "  --log-file <path>                 JSONL output path.",
    "  --seed <n>                        RNG seed for reproducible random action selection."
  ];

  console.log(lines.join("\n"));
  process.exit(code);
}

function resolveLogFilePath(explicitPath) {
  if (explicitPath) {
    return path.resolve(explicitPath);
  }

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return path.join(DEFAULT_LOG_DIR, `rl-smoke-${stamp}.jsonl`);
}

function redactOptionsForLog(options) {
  return {
    episodes: options.episodes,
    max_steps_per_episode: options.maxStepsPerEpisode,
    health_timeout_ms: options.healthTimeoutMs,
    reset_timeout_ms: options.resetTimeoutMs,
    step_timeout_ms: options.stepTimeoutMs,
    bridge_startup_timeout_ms: options.bridgeStartupTimeoutMs,
    poll_interval_ms: options.pollIntervalMs,
    restart_delay_ms: options.restartDelayMs,
    max_reset_attempts: options.maxResetAttempts,
    max_consecutive_step_errors: options.maxConsecutiveStepErrors,
    max_startup_attempts: options.maxStartupAttempts,
    ready_stable_polls: options.readyStablePolls,
    launch_game: options.launchGame,
    restart_before_episode: options.restartBeforeEpisode,
    restart_after_episode: options.restartAfterEpisode,
    defensive_buffs: options.defensiveBuffs,
    chaos_restart_prob: options.chaosRestartProb,
    character: options.character,
    game_exe: options.gameExe,
    seed: options.seed
  };
}

function readValue(argv, currentIndex, inlineValue, flag) {
  if (inlineValue !== null) {
    return inlineValue;
  }

  const nextIndex = currentIndex + 1;
  if (nextIndex >= argv.length) {
    throw new Error(`Missing value for ${flag}`);
  }

  return argv[nextIndex];
}

function parsePositiveInteger(rawValue, flag) {
  const value = Number.parseInt(rawValue, 10);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${flag} must be a positive integer.`);
  }
  return value;
}

function parseNonNegativeInteger(rawValue, flag) {
  const value = Number.parseInt(rawValue, 10);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${flag} must be a non-negative integer.`);
  }
  return value;
}

function parseProbability(rawValue, flag) {
  const value = Number.parseFloat(rawValue);
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${flag} must be between 0 and 1.`);
  }
  return value;
}

function requireNonEmptyString(value, flag) {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${flag} must be a non-empty string.`);
  }
  return value.trim();
}

function buildDefaultGameExeCandidates() {
  const candidates = [];
  if (process.env.STS2_GAME_EXE) {
    candidates.push(process.env.STS2_GAME_EXE);
  }

  const driveRoots = ["C:\\", "D:\\", "E:\\", "F:\\", "G:\\"];
  const steamSuffixes = [
    path.join("Program Files (x86)", "Steam", "steamapps", "common", "Slay the Spire 2", "SlayTheSpire2.exe"),
    path.join("Program Files", "Steam", "steamapps", "common", "Slay the Spire 2", "SlayTheSpire2.exe"),
    path.join("SteamLibrary", "steamapps", "common", "Slay the Spire 2", "SlayTheSpire2.exe")
  ];

  for (const envKey of ["ProgramFiles(x86)", "ProgramFiles"]) {
    const root = process.env[envKey];
    if (root) {
      candidates.push(path.join(root, "Steam", "steamapps", "common", "Slay the Spire 2", "SlayTheSpire2.exe"));
    }
  }

  for (const drive of driveRoots) {
    for (const suffix of steamSuffixes) {
      candidates.push(path.join(drive, suffix));
    }
  }

  return dedupePaths(candidates);
}

function dedupePaths(values) {
  const seen = new Set();
  const output = [];
  for (const value of values) {
    const normalized = path.normalize(value);
    if (seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    output.push(normalized);
  }
  return output;
}

function resolveGameExePath(explicitPath) {
  const candidates = explicitPath ? [explicitPath, ...DEFAULT_GAME_EXE_CANDIDATES] : DEFAULT_GAME_EXE_CANDIDATES;
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) {
      return path.resolve(candidate);
    }
  }
  return null;
}

function createMulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function ensureBridgeReady(options, log, reason) {
  const ready = await waitForRuntimeReady(options, log);
  if (ready) {
    return ready;
  }

  if (!options.launchGame) {
    throw new Error("Bridge is not healthy and launch_game is disabled.");
  }

  let lastError = null;
  for (let attempt = 1; attempt <= options.maxStartupAttempts; attempt++) {
    try {
      return await hardRestartGame(options, log, `${reason}_startup_attempt_${attempt}`);
    } catch (error) {
      lastError = error;
      log("startup_attempt_failed", {
        reason,
        attempt,
        error: serializeError(error)
      });
    }
  }

  throw lastError || new Error("Unable to stabilize the game runtime.");
}

async function resetEpisodeWithRecovery(session, options, log, episodeNumber) {
  let lastError = null;
  for (let attempt = 1; attempt <= options.maxResetAttempts; attempt++) {
    try {
      const response = await bridgeRequestJson(session, "env/reset", {
        method: "POST",
        timeoutMs: options.resetTimeoutMs + DEFAULT_HTTP_TIMEOUT_GRACE_MS,
        body: {
          ...(options.character ? { character: options.character } : {}),
          defensive_buffs: options.defensiveBuffs,
          timeout_ms: options.resetTimeoutMs
        }
      });

      log("episode_reset_ok", {
        episode_number: episodeNumber,
        attempt,
        episode_id: response.episode_id,
        response
      });

      return { session, response };
    } catch (error) {
      if (isTransientBridgeError(error)) {
        lastError = error;
        log("episode_reset_wait_retry", {
          episode_number: episodeNumber,
          attempt,
          error: serializeError(error)
        });
        await sleep(options.pollIntervalMs);
        attempt--;
        continue;
      }

      lastError = error;
      log("episode_reset_error", {
        episode_number: episodeNumber,
        attempt,
        error: serializeError(error)
      });

      if (!options.gameExe || attempt >= options.maxResetAttempts) {
        break;
      }

      session = await hardRestartGame(
        options,
        log,
        `episode_${episodeNumber}_reset_retry_${attempt}`
      );
    }
  }

  throw lastError || new Error("env/reset failed for an unknown reason.");
}

async function waitForRuntimeReady(options, log) {
  const deadline = Date.now() + options.bridgeStartupTimeoutMs;
  let stableKey = null;
  let stableCount = 0;
  while (Date.now() < deadline) {
    const session = readSession();
    if (!session || !session.base_url || !session.token) {
      stableKey = null;
      stableCount = 0;
      await sleep(options.pollIntervalMs);
      continue;
    }

    if (!isProcessAlive(session.pid)) {
      stableKey = null;
      stableCount = 0;
      await sleep(options.pollIntervalMs);
      continue;
    }

    try {
      const health = await bridgeRequestJson(session, "health", {
        method: "GET",
        timeoutMs: options.healthTimeoutMs
      });
      const state = await bridgeRequestJson(session, "state", {
        method: "GET",
        timeoutMs: options.healthTimeoutMs
      });
      if (!state?.screen || state.screen === "UNKNOWN") {
        stableKey = null;
        stableCount = 0;
        await sleep(options.pollIntervalMs);
        continue;
      }
      const currentKey = `${session.pid}|${session.base_url}|${state?.screen || "UNKNOWN"}`;
      if (currentKey === stableKey) {
        stableCount++;
      } else {
        stableKey = currentKey;
        stableCount = 1;
      }

      if (stableCount < options.readyStablePolls) {
        await sleep(options.pollIntervalMs);
        continue;
      }

      log("bridge_ready", {
        pid: session.pid,
        base_url: session.base_url,
        screen: state?.screen,
        health
      });
      return session;
    } catch (error) {
      if (!isTransientBridgeError(error)) {
        log("bridge_wait_retry", {
          error: serializeError(error)
        });
      }
      stableKey = null;
      stableCount = 0;
      await sleep(options.pollIntervalMs);
    }
  }

  return null;
}

function readSession() {
  try {
    const raw = fs.readFileSync(SESSION_FILE_PATH, "utf8");
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function isProcessAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false;
  }

  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return !!(error && typeof error === "object" && error.code === "EPERM");
  }
}

async function launchGame(options, log, reason) {
  if (!options.gameExe) {
    throw new Error("No game executable path is available for launch.");
  }

  log("game_launch", {
    reason,
    game_exe: options.gameExe
  });

  const child = spawn(options.gameExe, [], {
    detached: true,
    stdio: "ignore"
  });
  child.unref();
}

async function hardRestartGame(options, log, reason) {
  log("game_restart_begin", { reason });
  await killGame(log, reason);
  await sleep(options.restartDelayMs);
  await launchGame(options, log, reason);
  const session = await waitForRuntimeReady(options, log);
  if (!session) {
    throw new Error("Timed out waiting for a stable runtime after hard restart.");
  }
  log("game_restart_ready", {
    reason,
    pid: session.pid,
    base_url: session.base_url
  });
  return session;
}

async function killGame(log, reason) {
  log("game_kill", { reason });
  if (process.platform === "win32") {
    await runCommand("taskkill", ["/IM", "SlayTheSpire2.exe", "/T", "/F"], true);
    return;
  }

  await runCommand("pkill", ["-f", "SlayTheSpire2"], true);
}

async function runCommand(command, args, ignoreFailure = false) {
  await new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: "ignore"
    });

    child.on("error", (error) => {
      if (ignoreFailure) {
        resolve();
        return;
      }
      reject(error);
    });

    child.on("exit", (code) => {
      if (code === 0 || ignoreFailure) {
        resolve();
        return;
      }
      reject(new Error(`${command} exited with code ${code}`));
    });
  });
}

async function bridgeRequestJson(session, endpointPath, options) {
  const method = options?.method || "GET";
  const timeoutMs = options?.timeoutMs || DEFAULT_HEALTH_TIMEOUT_MS;
  const url = new URL(ensureTrailingSlash(session.base_url) + endpointPath.replace(/^\/+/, ""));
  const headers = {
    Authorization: `Bearer ${session.token}`
  };
  const requestOptions = {
    method,
    headers,
    signal: AbortSignal.timeout(timeoutMs)
  };

  if (Object.prototype.hasOwnProperty.call(options || {}, "body")) {
    requestOptions.body = JSON.stringify(options.body);
    requestOptions.headers = {
      ...headers,
      "Content-Type": "application/json"
    };
  }

  let response;
  try {
    response = await fetch(url, requestOptions);
  } catch (error) {
    throw new BridgeHttpError("bridge_request_failed", String(error), 0, null);
  }

  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new BridgeHttpError("invalid_bridge_response", "Bridge response was not valid JSON.", response.status, {
        raw_text: text
      });
    }
  }

  if (!response.ok) {
    throw new BridgeHttpError(
      payload?.error || "bridge_http_error",
      payload?.message || `Bridge request failed with status ${response.status}.`,
      response.status,
      payload
    );
  }

  return payload;
}

class BridgeHttpError extends Error {
  constructor(code, message, status, payload) {
    super(message);
    this.name = "BridgeHttpError";
    this.code = code;
    this.status = status;
    this.payload = payload;
  }
}

function ensureTrailingSlash(value) {
  return value.endsWith("/") ? value : `${value}/`;
}

function chooseRandomAction(legalActions, rng) {
  const index = Math.floor(rng() * legalActions.length);
  return legalActions[index];
}

function inferEpisodeOutcome(response) {
  const breakdown = response?.info?.reward_breakdown;
  if (breakdown?.death) {
    return "death";
  }
  if (breakdown?.victory) {
    return "victory";
  }
  if (response?.truncated) {
    return "truncated";
  }
  return "done";
}

function formatScalar(value) {
  return Number.isFinite(value) ? Number(value).toFixed(3) : String(value);
}

function serializeError(error) {
  return {
    name: error?.name || "Error",
    message: error?.message || String(error),
    code: error?.code || null,
    status: error?.status || null,
    payload: error?.payload || null
  };
}

function tryRecoverCurrentFromActionError(current, error) {
  if (!error || typeof error !== "object" || error.code !== "action_not_available") {
    return null;
  }

  const details = error.payload?.details;
  if (!details || !Array.isArray(details.legal_actions) || !details.phase) {
    return null;
  }

  return {
    ...current,
    done: false,
    truncated: false,
    legal_actions: details.legal_actions,
    obs: {
      ...(current?.obs || {}),
      phase: details.phase
    },
    info: {
      ...(current?.info || {}),
      recovered_from_action_not_available: true
    }
  };
}

function isTransientBridgeError(error) {
  if (!error || typeof error !== "object") {
    return false;
  }

  if (TRANSIENT_BRIDGE_ERROR_CODES.has(error.code)) {
    return true;
  }

  if (error.code === "env_reset_no_reset_path") {
    const phase = error.payload?.details?.phase;
    const screen = error.payload?.details?.screen;
    const legalActions = error.payload?.details?.legal_actions;
    return phase === "settling" && screen === "UNKNOWN" && Array.isArray(legalActions) && legalActions.length === 0;
  }

  return false;
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
