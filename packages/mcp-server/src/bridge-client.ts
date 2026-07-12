import crypto from "node:crypto";
import { AppError } from "./errors";
import { API_VERSION, SCHEMA_VERSION } from "./generated/contract-versions";
import {
  supportsControlV2,
  tokenForCapability,
  type BridgeCapability
} from "./session";
import type { BridgeAction, BridgeSession, BridgeState } from "./types";

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const MAX_ACTION_HANDLE_LENGTH = 512;
const MAX_PLAYER_COMMAND_DEADLINE_MS = 120_000;

export interface BridgeClientOptions {
  timeoutMs?: number;
  maxResponseBytes?: number;
  fetchImpl?: typeof fetch;
  requestIdFactory?: () => string;
}

export interface BridgeRequestOptions {
  method?: "GET" | "POST";
  body?: unknown;
  timeoutMs?: number;
  capability?: BridgeCapability;
}

export interface ExecuteActionRequest {
  actionId: string;
  expectedStateVersion: number;
  waitAfterMs?: number;
  timeoutMs?: number;
  requestId?: string;
  /** Reuse this exact value when explicitly replaying an idempotent request_id. */
  deadlineUtc?: string;
}

export interface ExecuteActionResult {
  protocol: "control-v2" | "training-v2" | "legacy-v1";
  request_id: string;
  status: string | null;
  replayed_result: boolean;
  committed_state_version: number | null;
  payload: unknown;
}

function plainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function validateRelativePath(relativePath: string): string {
  const normalized = relativePath.replace(/^\/+/, "");
  if (!normalized || normalized.includes("..") || /^[a-z]+:/i.test(normalized)) {
    throw new AppError("invalid_bridge_path", "Bridge request path is invalid.");
  }
  return normalized;
}

function normalizeRequestId(value: string): string {
  const normalized = value.trim().toLowerCase();
  if (!UUID_PATTERN.test(normalized)) {
    throw new AppError("invalid_request_id", "request_id must be a UUID.");
  }
  return normalized;
}

async function readResponsePayload(response: Response, maxResponseBytes: number): Promise<unknown> {
  const declaredLength = Number.parseInt(response.headers.get("content-length") ?? "", 10);
  if (Number.isFinite(declaredLength) && declaredLength > maxResponseBytes) {
    throw new AppError(
      "bridge_response_too_large",
      `Bridge response exceeds the ${maxResponseBytes}-byte client limit.`
    );
  }
  const reader = response.body?.getReader();
  if (!reader) return null;
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    total += value.byteLength;
    if (total > maxResponseBytes) {
      await reader.cancel().catch(() => undefined);
      throw new AppError(
        "bridge_response_too_large",
        `Bridge response exceeds the ${maxResponseBytes}-byte client limit.`
      );
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  const text = new TextDecoder("utf-8", { fatal: false }).decode(bytes);
  if (text.trim().length === 0) return null;
  try {
    return JSON.parse(text);
  } catch {
    return { text: text.slice(0, 2_000) };
  }
}

function summarizeCommandResult(
  protocol: ExecuteActionResult["protocol"],
  requestId: string,
  payload: unknown
): ExecuteActionResult {
  const value = plainObject(payload) ? payload : {};
  const status = typeof value.status === "string" ? value.status : protocol === "legacy-v1" ? "committed" : null;
  const committed = value.committed_state_version;
  return {
    protocol,
    request_id: requestId,
    status,
    replayed_result: value.replayed_result === true,
    committed_state_version: Number.isSafeInteger(committed) ? (committed as number) : null,
    payload
  };
}

function commandEnvelopeError(
  code: string,
  message: string,
  requestId: string,
  status: string | null,
  payload: Record<string, unknown>
): AppError {
  const nested = plainObject(payload.error) ? payload.error : {};
  return new AppError(code, message, {
    retryable: false,
    details: {
      request_id: requestId,
      status,
      status_query_supported: true,
      bridge_error_code:
        typeof nested.code === "string"
          ? nested.code
          : typeof payload.error_code === "string"
            ? payload.error_code
            : null
    }
  });
}

function requireCommittedV2Command(
  protocol: "control-v2" | "training-v2",
  requestId: string,
  payload: unknown
): ExecuteActionResult {
  if (!plainObject(payload)) {
    throw new AppError("malformed_command_result", "Bridge command result must be an object.", {
      details: { request_id: requestId }
    });
  }
  const envelopeRequestId = payload.request_id;
  if (envelopeRequestId !== requestId) {
    throw new AppError("command_request_id_mismatch", "Bridge returned a different request_id.", {
      details: { request_id: requestId, returned_request_id: envelopeRequestId }
    });
  }
  if (payload.api_version !== API_VERSION || payload.schema_version !== SCHEMA_VERSION) {
    throw new AppError(
      "incompatible_bridge_contract",
      "Bridge command result does not match the required API/schema identity.",
      {
        details: {
          request_id: requestId,
          expected_api_version: API_VERSION,
          returned_api_version: payload.api_version ?? null,
          expected_schema_version: SCHEMA_VERSION,
          returned_schema_version: payload.schema_version ?? null
        }
      }
    );
  }
  if (typeof payload.replayed_result !== "boolean") {
    throw new AppError("malformed_command_result", "Bridge command result has no replayed_result flag.", {
      details: { request_id: requestId }
    });
  }
  const status = typeof payload.status === "string" ? payload.status : null;
  const nested = plainObject(payload.error) ? payload.error : {};
  const message =
    (typeof nested.message === "string" && nested.message) ||
    (typeof payload.error_message === "string" && payload.error_message) ||
    "The Bridge command did not commit.";

  if (status === "committed") {
    if (payload.ok !== true || !plainObject(payload.result)) {
      throw new AppError(
        "malformed_command_result",
        "A committed Bridge command must contain ok=true and an object result.",
        { details: { request_id: requestId, status } }
      );
    }
    return summarizeCommandResult(protocol, requestId, payload);
  }
  if (status === "outcome_unknown") {
    throw commandEnvelopeError("command_outcome_unknown", message, requestId, status, payload);
  }
  if (status === "accepted" || status === "executing") {
    throw commandEnvelopeError(
      "command_pending",
      "The Bridge command is still pending; query its status with the same request_id.",
      requestId,
      status,
      payload
    );
  }
  if (status === "rejected_before_execution" || status === "rejected") {
    const rejectionCode =
      (typeof nested.code === "string" && nested.code) ||
      (typeof payload.error_code === "string" && payload.error_code) ||
      "command_rejected";
    throw commandEnvelopeError(rejectionCode, message, requestId, status, payload);
  }
  throw new AppError("malformed_command_result", "Bridge returned an unknown command status.", {
    details: { request_id: requestId, status }
  });
}

function normalizeV2LegalActions(value: unknown): BridgeState["available_actions"] {
  if (!Array.isArray(value)) {
    throw new AppError("invalid_bridge_state", "Bridge v2 state has no legal_actions array.");
  }
  return value.map((item, index) => {
    if (
      !plainObject(item) ||
      typeof item.handle !== "string" ||
      item.handle.trim().length === 0 ||
      item.handle.length > 512 ||
      typeof item.kind !== "string" ||
      item.kind.trim().length === 0 ||
      item.kind.length > 96 ||
      !("label" in item) ||
      (item.label !== null && typeof item.label !== "string") ||
      (typeof item.label === "string" && item.label.length > 512)
    ) {
      throw new AppError(
        "invalid_bridge_state",
        "Every Bridge v2 legal action must contain canonical handle, kind, and label fields.",
        { details: { action_index: index } }
      );
    }
    const coord = plainObject(item.coord) ? item.coord : undefined;
    const optionCandidate = [
      item.option_index,
      item.index
    ].find((candidate) => Number.isSafeInteger(candidate));
    const optionIndex =
      typeof optionCandidate === "number" ? optionCandidate : undefined;
    const normalized: BridgeAction = { ...(item as BridgeAction), handle: item.handle };
    if (coord) normalized.coord = coord as BridgeAction["coord"];
    if (optionIndex !== undefined) normalized.option_index = optionIndex;
    return normalized;
  });
}

export class BridgeClient {
  private readonly defaultTimeoutMs: number;
  private readonly maxResponseBytes: number;
  private readonly fetchImpl: typeof fetch;
  private readonly requestIdFactory: () => string;

  constructor(options: BridgeClientOptions = {}) {
    this.defaultTimeoutMs = options.timeoutMs ?? 10_000;
    this.maxResponseBytes = options.maxResponseBytes ?? 16 * 1024 * 1024;
    if (!Number.isSafeInteger(this.maxResponseBytes) || this.maxResponseBytes <= 0) {
      throw new AppError("invalid_response_limit", "Bridge response limit must be a positive integer.");
    }
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch;
    this.requestIdFactory = options.requestIdFactory ?? (() => crypto.randomUUID());
  }

  async request<T = unknown>(
    session: BridgeSession,
    relativePath: string,
    options: BridgeRequestOptions = {}
  ): Promise<T> {
    const requestPath = validateRelativePath(relativePath);
    const timeoutMs = options.timeoutMs ?? this.defaultTimeoutMs;
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) {
      throw new AppError("invalid_timeout", "Bridge timeout must be a positive integer.");
    }
    const token = tokenForCapability(session, options.capability);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const url = `${session.base_url}/${requestPath}`;
    try {
      const response = await this.fetchImpl(url, {
        method: options.method ?? "GET",
        headers: {
          Accept: "application/json",
          ...(options.body !== undefined ? { "Content-Type": "application/json" } : {}),
          ...(token ? { Authorization: `Bearer ${token}` } : {})
        },
        ...(options.body !== undefined ? { body: JSON.stringify(options.body) } : {}),
        signal: controller.signal
      });
      const payload = await readResponsePayload(response, this.maxResponseBytes);
      if (!response.ok) {
        throw new AppError("bridge_http_error", `Bridge returned HTTP ${response.status}.`, {
          httpStatus: response.status,
          retryable: false,
          details: { status: response.status, path: requestPath, payload }
        });
      }
      return payload as T;
    } catch (cause) {
      if (cause instanceof AppError) throw cause;
      if (controller.signal.aborted || (cause as Error)?.name === "AbortError") {
        throw new AppError("bridge_timeout", `Bridge request timed out after ${timeoutMs} ms.`, {
          retryable: false,
          details: { path: requestPath, timeout_ms: timeoutMs },
          cause
        });
      }
      throw new AppError("bridge_unreachable", "Bridge request failed before a response was received.", {
        retryable: false,
        details: { path: requestPath },
        cause
      });
    } finally {
      clearTimeout(timer);
    }
  }

  async health(session: BridgeSession): Promise<unknown> {
    return this.request(session, supportsControlV2(session) ? "v2/health" : "health", {
      capability: supportsControlV2(session) ? "player-control" : undefined
    });
  }

  async state(session: BridgeSession): Promise<BridgeState> {
    const isV2 = supportsControlV2(session);
    const payload = await this.request<BridgeState | { state?: BridgeState }>(
      session,
      isV2 ? "v2/state" : "state",
      { capability: isV2 ? "player-control" : undefined }
    );
    if (isV2 && plainObject(payload) && plainObject(payload.state)) {
      const envelope = payload as Record<string, unknown>;
      const expectedSessionId =
        (typeof session.session_id === "string" && session.session_id) ||
        (typeof session.instance_id === "string" && session.instance_id) ||
        null;
      if (
        !Number.isSafeInteger(envelope.state_version) ||
        (envelope.state_version as number) < 0 ||
        envelope.visibility !== "player" ||
        typeof envelope.captured_at_utc !== "string" ||
        !Number.isFinite(Date.parse(envelope.captured_at_utc)) ||
        expectedSessionId === null ||
        envelope.session_id !== expectedSessionId
      ) {
        throw new AppError(
          "invalid_bridge_state",
          "Bridge v2 state identity, revision, timestamp, or visibility is invalid."
        );
      }
      const inner = envelope.state as BridgeState;
      return {
        ...inner,
        state_version: Number.isSafeInteger(envelope.state_version)
          ? (envelope.state_version as number)
          : inner.state_version,
        available_actions: normalizeV2LegalActions(envelope.legal_actions),
        visibility: envelope.visibility ?? "player",
        session_id: envelope.session_id,
        captured_at_utc: envelope.captured_at_utc
      } as BridgeState;
    }
    if (isV2) {
      throw new AppError("invalid_bridge_state", "Bridge v2 state envelope is malformed.");
    }
    return payload as BridgeState;
  }

  async executeAction(
    session: BridgeSession,
    request: ExecuteActionRequest
  ): Promise<ExecuteActionResult> {
    const actionHandle = request.actionId.trim();
    if (!actionHandle || actionHandle.length > MAX_ACTION_HANDLE_LENGTH) {
      throw new AppError(
        "invalid_action_id",
        `action_id must be non-empty and no longer than ${MAX_ACTION_HANDLE_LENGTH} characters.`
      );
    }
    if (!Number.isSafeInteger(request.expectedStateVersion) || request.expectedStateVersion < 0) {
      throw new AppError(
        "expected_state_version_required",
        "Every mutation requires a non-negative integer expected_state_version. Refresh state before retrying."
      );
    }
    const waitAfterMs = request.waitAfterMs ?? 0;
    if (!Number.isSafeInteger(waitAfterMs) || waitAfterMs < 0 || waitAfterMs > 5_000) {
      throw new AppError("invalid_wait_after_ms", "wait_after_ms must be between 0 and 5000.");
    }
    const requestId = normalizeRequestId(request.requestId || this.requestIdFactory());
    const timeoutMs = request.timeoutMs ?? this.defaultTimeoutMs + waitAfterMs;
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) {
      throw new AppError("invalid_timeout", "Bridge timeout must be a positive integer.");
    }

    if (supportsControlV2(session)) {
      if (timeoutMs > MAX_PLAYER_COMMAND_DEADLINE_MS) {
        throw new AppError(
          "invalid_command_deadline",
          `A v2 player command timeout cannot exceed ${MAX_PLAYER_COMMAND_DEADLINE_MS} ms.`
        );
      }
      const sessionId =
        (typeof session.session_id === "string" && session.session_id.trim()) ||
        (typeof session.instance_id === "string" && session.instance_id.trim());
      if (!sessionId) {
        throw new AppError("session_id_missing", "A v2 Bridge session must expose session_id.");
      }
      const now = Date.now();
      const deadlineUtc = request.deadlineUtc ?? new Date(now + timeoutMs).toISOString();
      const deadlineTimestamp = Date.parse(deadlineUtc);
      if (
        !Number.isFinite(deadlineTimestamp) ||
        deadlineTimestamp <= now ||
        deadlineTimestamp > now + MAX_PLAYER_COMMAND_DEADLINE_MS
      ) {
        throw new AppError(
          "invalid_command_deadline",
          `deadlineUtc must be a valid future timestamp within ${MAX_PLAYER_COMMAND_DEADLINE_MS} ms.`
        );
      }
      try {
        const payload = await this.request(session, "v2/commands", {
          method: "POST",
          timeoutMs,
          capability: "player-control",
          body: {
            request_id: requestId,
            session_id: sessionId,
            capability: "player-control",
            expected_state_version: request.expectedStateVersion,
            deadline_utc: deadlineUtc,
            command: {
              kind: "perform_action",
              action_handle: actionHandle,
              wait_after_ms: waitAfterMs
            }
          }
        });
        return requireCommittedV2Command("control-v2", requestId, payload);
      } catch (error) {
        if (
          error instanceof AppError &&
          ["bridge_timeout", "bridge_unreachable"].includes(error.code)
        ) {
          throw new AppError(
            "command_outcome_unknown",
            "The v2 command response was not observed. Query command status with the same request_id; do not submit a new command.",
            {
              retryable: false,
              details: { request_id: requestId, status_query_supported: true },
              cause: error
            }
          );
        }
        throw error;
      }
    }

    // Legacy v1 has no server-side idempotency key. It is intentionally attempted exactly once.
    try {
      const payload = await this.request(session, "action", {
        method: "POST",
        timeoutMs,
        body: {
          action_id: actionHandle,
          expected_state_version: request.expectedStateVersion,
          wait_after_ms: waitAfterMs
        }
      });
      return summarizeCommandResult("legacy-v1", requestId, payload);
    } catch (error) {
      if (
        error instanceof AppError &&
        ["bridge_timeout", "bridge_unreachable"].includes(error.code)
      ) {
        throw new AppError(
          "legacy_action_outcome_unknown",
          "The legacy action response was not observed. Refresh and reconcile state; never retry automatically.",
          {
            retryable: false,
            details: { local_request_id: requestId, status_query_supported: false },
            cause: error
          }
        );
      }
      throw error;
    }
  }

  async executeEnvironmentCommand(
    session: BridgeSession,
    relativePath: "v2/env/reset" | "v2/env/step",
    requestId: string,
    body: unknown,
    timeoutMs: number
  ): Promise<ExecuteActionResult> {
    const normalized = normalizeRequestId(requestId);
    try {
      const payload = await this.request(session, relativePath, {
        method: "POST",
        timeoutMs,
        capability: "training",
        body
      });
      return requireCommittedV2Command("training-v2", normalized, payload);
    } catch (error) {
      if (
        error instanceof AppError &&
        ["bridge_timeout", "bridge_unreachable"].includes(error.code)
      ) {
        throw new AppError(
          "command_outcome_unknown",
          "The training command response was not observed. Query status with the same request_id; do not resubmit it.",
          {
            retryable: false,
            details: { request_id: normalized, status_query_supported: true },
            cause: error
          }
        );
      }
      throw error;
    }
  }

  async executeLegacyEnvironmentCommand(
    session: BridgeSession,
    relativePath: "env/reset" | "env/step",
    requestId: string,
    body: unknown,
    timeoutMs: number
  ): Promise<ExecuteActionResult> {
    const normalized = normalizeRequestId(requestId);
    try {
      const payload = await this.request(session, relativePath, {
        method: "POST",
        timeoutMs,
        body
      });
      return summarizeCommandResult("legacy-v1", normalized, payload);
    } catch (error) {
      if (
        error instanceof AppError &&
        ["bridge_timeout", "bridge_unreachable"].includes(error.code)
      ) {
        throw new AppError(
          "legacy_environment_outcome_unknown",
          "The legacy training mutation response was not observed. Refresh and reconcile environment state; never retry automatically.",
          {
            retryable: false,
            details: {
              local_request_id: normalized,
              operation: relativePath,
              status_query_supported: false
            },
            cause: error
          }
        );
      }
      throw error;
    }
  }

  async commandStatus(
    session: BridgeSession,
    requestId: string,
    capability: BridgeCapability = "player-control"
  ): Promise<unknown> {
    if (!supportsControlV2(session)) {
      throw new AppError(
        "command_status_unavailable",
        "The active legacy bridge does not support command status queries."
      );
    }
    const normalized = normalizeRequestId(requestId);
    return this.request(session, `v2/commands/${encodeURIComponent(normalized)}`, {
      capability
    });
  }
}
