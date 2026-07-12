import crypto from "node:crypto";
import fs from "node:fs";
import { AppError } from "./errors";
import {
  ACTION_SCHEMA_VERSION,
  API_VERSION,
  LEGAL_ACTION_ORDERING_VERSION,
  SCHEMA_VERSION
} from "./generated/contract-versions";
import type { BridgeGameCompatibility, BridgeSession } from "./types";

export type BridgeCapability = "player-control" | "training";

export interface SessionStoreOptions {
  processAlive?: (pid: number) => boolean;
  readText?: (file: string) => string;
}

function defaultProcessAlive(pid: number): boolean {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

function isLoopbackHost(hostname: string): boolean {
  const normalized = hostname.replace(/^\[|\]$/g, "").toLowerCase();
  return (
    normalized === "localhost" ||
    normalized === "::1" ||
    normalized === "0:0:0:0:0:0:0:1" ||
    normalized.startsWith("127.")
  );
}

function plainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function hasExactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value);
  return actual.length === expected.length && expected.every((key) => Object.hasOwn(value, key));
}

function boundedString(value: unknown, maximum: number): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= maximum;
}

export function validateBridgeBaseUrl(value: unknown): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new AppError("invalid_session_file", "Bridge session has no base_url.");
  }
  let url: URL;
  try {
    url = new URL(value);
  } catch (cause) {
    throw new AppError("invalid_session_file", "Bridge base_url is not a valid URL.", { cause });
  }
  if (!["http:", "https:"].includes(url.protocol) || !isLoopbackHost(url.hostname)) {
    throw new AppError(
      "unsafe_bridge_url",
      "Bridge base_url must use HTTP(S) on a loopback host.",
      { details: { protocol: url.protocol, hostname: url.hostname } }
    );
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new AppError(
      "unsafe_bridge_url",
      "Bridge base_url must not contain credentials, query parameters, or fragments."
    );
  }
  return url.toString().replace(/\/$/, "");
}

const V2_CAPABILITIES = new Set(["player-control", "training", "legacy-privileged"]);
const MAX_SESSION_FILE_BYTES = 1024 * 1024;
const GAME_COMPATIBILITY_KEYS = [
  "health",
  "startup_allowed",
  "error_code",
  "error_message",
  "profile_id",
  "assembly",
  "probes"
] as const;
const GAME_ASSEMBLY_IDENTITY_KEYS = [
  "name",
  "assembly_version",
  "informational_version",
  "module_version_id"
] as const;
const GAME_COMPATIBILITY_PROBE_KEYS = ["capability", "passed", "code", "detail"] as const;
const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function readCapabilityTokens(value: unknown): Record<string, string> | undefined {
  if (!plainObject(value)) return undefined;
  const tokens: Record<string, string> = {};
  for (const [name, token] of Object.entries(value)) {
    if (!V2_CAPABILITIES.has(name) || typeof token !== "string" || token.length < 32) {
      throw new AppError(
        "invalid_session_file",
        "Bridge capability_tokens contains an unsupported scope or malformed credential."
      );
    }
    tokens[name] = token;
  }
  return Object.keys(tokens).length > 0 ? tokens : undefined;
}

function requireIsoTimestamp(value: unknown, field: string): void {
  if (typeof value !== "string" || value.length === 0 || !Number.isFinite(Date.parse(value))) {
    throw new AppError("invalid_session_file", `Bridge session has an invalid ${field}.`);
  }
}

function readReadyGameCompatibility(value: unknown): BridgeGameCompatibility {
  if (!plainObject(value) || !hasExactKeys(value, GAME_COMPATIBILITY_KEYS)) {
    throw new AppError(
      "invalid_session_file",
      "Bridge v2 session has a malformed game_compatibility record."
    );
  }
  if (
    value.health !== "ready" ||
    value.startup_allowed !== true ||
    value.error_code !== "" ||
    value.error_message !== "" ||
    !boundedString(value.profile_id, 128)
  ) {
    throw new AppError(
      "incompatible_game_assembly",
      "Bridge v2 session does not carry a successful fail-closed game compatibility decision."
    );
  }

  const assembly = value.assembly;
  if (
    !plainObject(assembly) ||
    !hasExactKeys(assembly, GAME_ASSEMBLY_IDENTITY_KEYS) ||
    !boundedString(assembly.name, 128) ||
    !boundedString(assembly.assembly_version, 128) ||
    !boundedString(assembly.informational_version, 256) ||
    typeof assembly.module_version_id !== "string" ||
    !UUID_PATTERN.test(assembly.module_version_id)
  ) {
    throw new AppError(
      "invalid_session_file",
      "Bridge v2 session has a malformed game assembly compatibility identity."
    );
  }

  if (!Array.isArray(value.probes) || value.probes.length === 0) {
    throw new AppError(
      "incompatible_game_assembly",
      "Bridge v2 session contains no successful game compatibility probes."
    );
  }
  for (const probe of value.probes) {
    if (
      !plainObject(probe) ||
      !hasExactKeys(probe, GAME_COMPATIBILITY_PROBE_KEYS) ||
      !boundedString(probe.capability, 256) ||
      probe.passed !== true ||
      probe.code !== "capability_present" ||
      !boundedString(probe.detail, 1024)
    ) {
      throw new AppError(
        "incompatible_game_assembly",
        "Bridge v2 session contains an unsuccessful or malformed game compatibility probe."
      );
    }
  }

  return value as unknown as BridgeGameCompatibility;
}

function validateV2SessionDescriptor(
  parsed: Record<string, unknown>,
  capabilityTokens: Record<string, string> | undefined
): BridgeGameCompatibility | undefined {
  const versions = Array.isArray(parsed.api_versions)
    ? parsed.api_versions.filter((value): value is string => typeof value === "string")
    : [];
  const capabilities = Array.isArray(parsed.capabilities)
    ? parsed.capabilities.filter((value): value is string => typeof value === "string")
    : [];
  const tokenScopes = Object.keys(capabilityTokens ?? {});
  const allowedVersions = new Set([API_VERSION, "legacy-v1"]);
  if (
    (parsed.api_versions !== undefined &&
      (!Array.isArray(parsed.api_versions) ||
        versions.length !== parsed.api_versions.length ||
        new Set(versions).size !== versions.length ||
        versions.some((value) => !allowedVersions.has(value)))) ||
    (typeof parsed.api_version === "string" &&
      parsed.api_version !== API_VERSION &&
      parsed.api_version !== "legacy-v1")
  ) {
    throw new AppError(
      "incompatible_session_contract",
      "Bridge session advertises an unsupported API identity."
    );
  }
  const claimsV2 =
    versions.includes(API_VERSION) ||
    parsed.api_version === API_VERSION ||
    capabilities.some((value) => V2_CAPABILITIES.has(value)) ||
    tokenScopes.some((value) => V2_CAPABILITIES.has(value));
  if (!claimsV2) return undefined;

  if (
    !Array.isArray(parsed.api_versions) ||
    !versions.includes(API_VERSION) ||
    versions.length !== parsed.api_versions.length ||
    new Set(versions).size !== versions.length ||
    versions.some((value) => !allowedVersions.has(value))
  ) {
    throw new AppError(
      "incompatible_session_contract",
      `Bridge session must advertise the supported API ${API_VERSION} without unknown versions.`
    );
  }
  if (
    parsed.schema_version !== SCHEMA_VERSION ||
    parsed.action_schema_version !== ACTION_SCHEMA_VERSION ||
    parsed.legal_action_ordering_version !== LEGAL_ACTION_ORDERING_VERSION
  ) {
    throw new AppError(
      "incompatible_session_contract",
      "Bridge session schema/action identities do not match this MCP build."
    );
  }
  if (typeof parsed.session_id !== "string" || parsed.session_id.trim().length === 0) {
    throw new AppError("invalid_session_file", "Bridge v2 session has no session_id.");
  }
  requireIsoTimestamp(parsed.process_started_at_utc, "process_started_at_utc");
  requireIsoTimestamp(parsed.created_at_utc, "created_at_utc");
  if (
    !Array.isArray(parsed.capabilities) ||
    capabilities.length !== parsed.capabilities.length ||
    new Set(capabilities).size !== capabilities.length ||
    !capabilities.includes("player-control") ||
    capabilities.some((value) => !V2_CAPABILITIES.has(value))
  ) {
    throw new AppError(
      "incompatible_session_contract",
      "Bridge v2 session capabilities are missing player-control or contain an unsupported scope."
    );
  }
  if (!capabilityTokens?.["player-control"]) {
    throw new AppError(
      "capability_token_missing",
      "Bridge v2 session has no player-control capability credential."
    );
  }

  const trainingEnabled = capabilities.includes("training");
  if (trainingEnabled !== Boolean(capabilityTokens.training)) {
    throw new AppError(
      "invalid_session_file",
      "Bridge training capability and credential declarations do not agree."
    );
  }
  const legacyEnabled = versions.includes("legacy-v1");
  if (
    legacyEnabled !== capabilities.includes("legacy-privileged") ||
    legacyEnabled !== Boolean(capabilityTokens["legacy-privileged"]) ||
    legacyEnabled !== (typeof parsed.token === "string" && parsed.token.length >= 32)
  ) {
    throw new AppError(
      "invalid_session_file",
      "Bridge legacy-v1 version, capability, and credential declarations do not agree."
    );
  }

  return readReadyGameCompatibility(parsed.game_compatibility);
}

export class SessionStore {
  private readonly processAlive: (pid: number) => boolean;
  private readonly readText: (file: string) => string;

  constructor(readonly sessionFile: string, options: SessionStoreOptions = {}) {
    this.processAlive = options.processAlive ?? defaultProcessAlive;
    this.readText = options.readText ?? ((file) => fs.readFileSync(file, "utf8"));
  }

  read(options: { requireAlive?: boolean } = {}): BridgeSession {
    let raw: string;
    try {
      raw = this.readText(this.sessionFile);
    } catch (cause) {
      const errno = cause as NodeJS.ErrnoException;
      if (errno.code === "ENOENT") {
        throw new AppError("session_file_missing", "Bridge session file was not found.", {
          details: { session_file: this.sessionFile },
          cause
        });
      }
      throw new AppError("session_file_unreadable", "Bridge session file could not be read.", {
        details: { session_file: this.sessionFile },
        cause
      });
    }

    if (Buffer.byteLength(raw, "utf8") > MAX_SESSION_FILE_BYTES) {
      throw new AppError(
        "invalid_session_file",
        `Bridge session file exceeds the ${MAX_SESSION_FILE_BYTES}-byte discovery limit.`
      );
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch (cause) {
      throw new AppError("invalid_session_file", "Bridge session file is not valid JSON.", {
        details: { session_file: this.sessionFile },
        cause
      });
    }
    if (!plainObject(parsed)) {
      throw new AppError("invalid_session_file", "Bridge session file must contain an object.");
    }

    const pid = parsed.pid;
    if (!Number.isSafeInteger(pid) || (pid as number) <= 0) {
      throw new AppError("invalid_session_file", "Bridge session has an invalid pid.");
    }
    const baseUrl = validateBridgeBaseUrl(parsed.base_url);
    const token = typeof parsed.token === "string" && parsed.token.length > 0 ? parsed.token : undefined;
    const capabilityTokens = readCapabilityTokens(parsed.capability_tokens);
    const gameCompatibility = validateV2SessionDescriptor(parsed, capabilityTokens);
    const session: BridgeSession = {
      ...parsed,
      base_url: baseUrl,
      pid: pid as number,
      ...(token ? { token } : {}),
      ...(capabilityTokens ? { capability_tokens: capabilityTokens } : {}),
      ...(gameCompatibility ? { game_compatibility: gameCompatibility } : {})
    } as BridgeSession;

    if (options.requireAlive !== false && !this.processAlive(session.pid)) {
      throw new AppError("stale_session", "The recorded bridge process is not alive.", {
        details: {
          session_file: this.sessionFile,
          pid: session.pid,
          session_id_hash: sessionIdentityHash(session)
        }
      });
    }
    return session;
  }

  secrets(): string[] {
    try {
      const session = this.read({ requireAlive: false });
      return [session.token, ...Object.values(session.capability_tokens ?? {})].filter(
        (value): value is string => typeof value === "string" && value.length > 0
      );
    } catch {
      return [];
    }
  }

  describe(session: BridgeSession): Record<string, unknown> {
    const schemaVersions = plainObject(session.schema_versions) ? session.schema_versions : {};
    return {
      session_id_hash: sessionIdentityHash(session),
      pid: session.pid,
      base_url: session.base_url,
      token_present: Boolean(session.token),
      capability_tokens_present: Object.keys(session.capability_tokens ?? {}),
      api_version: session.api_version ?? null,
      api_versions: Array.isArray(session.api_versions) ? session.api_versions : [],
      schema_version:
        session.schema_version ??
        (typeof schemaVersions.control === "string" ? schemaVersions.control : null),
      capabilities: capabilityNames(session),
      process_started_at_utc: session.process_started_at_utc ?? null
    };
  }
}

export function sessionIdentityHash(session: BridgeSession): string {
  const identity =
    (typeof session.session_id === "string" && session.session_id) ||
    (typeof session.instance_id === "string" && session.instance_id) ||
    `${session.pid}@${session.base_url}`;
  return crypto.createHash("sha256").update(identity).digest("hex").slice(0, 12);
}

export function capabilityNames(session: BridgeSession): string[] {
  if (Array.isArray(session.capabilities)) {
    return session.capabilities.filter((value): value is string => typeof value === "string");
  }
  if (!plainObject(session.capabilities)) return [];
  return Object.entries(session.capabilities)
    .filter(([, descriptor]) => {
      if (descriptor === true) return true;
      return plainObject(descriptor) && descriptor.enabled === true;
    })
    .map(([name, descriptor]) => {
      if (plainObject(descriptor) && typeof descriptor.scope === "string") return descriptor.scope;
      return name.replaceAll("_", "-");
    });
}

export function supportsControlV2(session: BridgeSession): boolean {
  const versions = [
    ...(Array.isArray(session.api_versions) ? session.api_versions : []),
    ...(typeof session.api_version === "string" ? [session.api_version] : [])
  ];
  return (
    versions.some(
      (value) =>
        /^2(?:\.\d+){0,2}$/.test(value) ||
        /(?:^|[-_/])v?2(?:$|[-_/])/i.test(value) ||
        value.toLowerCase() === "control-v2"
    ) ||
    capabilityNames(session).includes("player-control")
  );
}

export function tokenForCapability(
  session: BridgeSession,
  capability?: BridgeCapability
): string | undefined {
  if (capability) {
    const scoped = session.capability_tokens?.[capability];
    if (scoped) return scoped;
    if (supportsControlV2(session)) {
      throw new AppError(
        "capability_token_missing",
        `The active v2 session does not expose a scoped ${capability} token.`
      );
    }
  }
  // The top-level token belongs exclusively to a legacy-only session. Never
  // downgrade a v2 capability request to a potentially privileged v1 token.
  return session.token;
}
