import type { ServerConfig } from "./config";
import type { BridgeClient } from "./bridge-client";
import type { SessionStore } from "./session";

export interface ToolContext {
  config: ServerConfig;
  sessions: SessionStore;
  bridge: BridgeClient;
}
