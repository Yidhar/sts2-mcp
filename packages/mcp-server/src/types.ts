export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export interface BridgeGameAssemblyIdentity {
  name: string;
  assembly_version: string;
  informational_version: string;
  module_version_id: string;
}

export interface BridgeGameCompatibilityProbe {
  capability: string;
  passed: true;
  code: "capability_present";
  detail: string;
}

/**
 * Fail-closed retail adapter evidence published only after the Bridge startup
 * gate has matched an audited game build and every required probe has passed.
 */
export interface BridgeGameCompatibility {
  health: "ready";
  startup_allowed: true;
  error_code: "";
  error_message: "";
  profile_id: string;
  assembly: BridgeGameAssemblyIdentity;
  probes: BridgeGameCompatibilityProbe[];
}

export interface BridgeSession {
  session_id?: string;
  instance_id?: string;
  base_url: string;
  token?: string;
  pid: number;
  api_version?: string;
  api_versions?: string[];
  schema_version?: string;
  capabilities?: string[] | Record<string, unknown>;
  capability_tokens?: Record<string, string>;
  schema_versions?: Record<string, string>;
  process_started_at_utc?: string;
  process_started_at?: string;
  game_compatibility?: BridgeGameCompatibility;
  [key: string]: unknown;
}

export interface BridgeState {
  state_version?: number;
  state_hash?: string;
  screen?: string;
  available_actions?: BridgeAction[];
  [key: string]: unknown;
}

export interface BridgeAction {
  /** Canonical control-v2 legal-action identity. */
  handle?: string;
  action_handle?: string;
  action_id?: string;
  action?: string;
  kind?: string;
  type?: string;
  label?: string;
  target_handle?: string | null;
  coord?: { x?: number; y?: number; col?: number; row?: number } | null;
  option_index?: number | null;
  index?: number;
  selection_id?: string | null;
  card_ref?: string | null;
  slot_index?: number | null;
  [key: string]: unknown;
}

export interface ToolErrorBody {
  ok: false;
  error: {
    code: string;
    message: string;
    retryable: boolean;
    details?: Record<string, unknown>;
  };
}
