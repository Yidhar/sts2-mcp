import os from "node:os";
import fs from "node:fs";
import path from "node:path";
import { API_VERSION, SCHEMA_VERSION } from "./generated/contract-versions";

export const PROFILE_NAMES = ["minimal", "strategic", "debug"] as const;
export type ToolProfile = (typeof PROFILE_NAMES)[number];

export interface ServerConfig {
  profile: ToolProfile;
  sessionFile: string;
  requestTimeoutMs: number;
  waitPollMs: number;
  serverName: string;
  serverVersion: string;
  apiVersion: string;
  schemaVersion: string;
}

function packageVersion(packageRoot: string): string {
  const value = JSON.parse(
    fs.readFileSync(path.join(packageRoot, "package.json"), "utf8")
  ) as { version?: unknown };
  if (typeof value.version !== "string" || value.version.length === 0) {
    throw new Error("packages/mcp-server/package.json has no valid version");
  }
  return value.version;
}

export function normalizeProfile(value: unknown): ToolProfile | null {
  if (typeof value !== "string" || value.trim().length === 0) {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  if ((PROFILE_NAMES as readonly string[]).includes(normalized)) {
    return normalized as ToolProfile;
  }
  return null;
}

export function resolveProfile(
  argv: readonly string[] = process.argv.slice(2),
  env: NodeJS.ProcessEnv = process.env
): ToolProfile {
  const profileArg = argv.find(
    (arg) => arg.startsWith("--profile=") || arg.startsWith("--tool-profile=")
  );
  if (profileArg !== undefined) {
    const cliValue = profileArg.slice((profileArg.indexOf("=") || 0) + 1);
    return normalizeProfile(cliValue) ?? "minimal";
  }
  for (const key of ["STS2_MCP_PROFILE", "STS2_TOOL_PROFILE", "STS2_MCP_TOOL_PROFILE"] as const) {
    if (env[key] !== undefined) return normalizeProfile(env[key]) ?? "minimal";
  }
  return "minimal";
}

export function defaultSessionFile(env: NodeJS.ProcessEnv = process.env): string {
  if (env.STS2_BRIDGE_SESSION_FILE) {
    return path.resolve(env.STS2_BRIDGE_SESSION_FILE);
  }
  const appData = env.APPDATA ?? path.join(os.homedir(), "AppData", "Roaming");
  return path.join(appData, "SlayTheSpire2", "bridge", "session.json");
}

function parsePositiveInteger(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback;
}

export function loadConfig(
  argv: readonly string[] = process.argv.slice(2),
  env: NodeJS.ProcessEnv = process.env
): ServerConfig {
  const packageRoot = path.resolve(__dirname, "..", "..");
  return {
    profile: resolveProfile(argv, env),
    sessionFile: defaultSessionFile(env),
    requestTimeoutMs: parsePositiveInteger(env.STS2_MCP_HTTP_TIMEOUT_MS, 10_000),
    waitPollMs: parsePositiveInteger(env.STS2_MCP_WAIT_POLL_MS, 200),
    serverName: "sts2",
    serverVersion: packageVersion(packageRoot),
    apiVersion: API_VERSION,
    schemaVersion: SCHEMA_VERSION
  };
}
