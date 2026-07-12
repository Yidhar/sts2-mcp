import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { BridgeClient } from "../bridge-client";
import { loadConfig, type ServerConfig } from "../config";
import { SessionStore } from "../session";
import type { ToolContext } from "../tool-context";
import { registerTools } from "../tool-registry";

export interface ServerDependencies {
  sessions?: SessionStore;
  bridge?: BridgeClient;
}

export function createMcpServer(
  config: ServerConfig = loadConfig(),
  dependencies: ServerDependencies = {}
): { server: McpServer; context: ToolContext; toolNames: ReadonlySet<string> } {
  const context: ToolContext = {
    config,
    sessions: dependencies.sessions ?? new SessionStore(config.sessionFile),
    bridge: dependencies.bridge ?? new BridgeClient({ timeoutMs: config.requestTimeoutMs })
  };
  const server = new McpServer(
    { name: config.serverName, version: config.serverVersion },
    {
      instructions:
        `Contract API ${config.apiVersion}, schema ${config.schemaVersion}. Active profile: ${config.profile}. ` +
        "Observe state and legal actions first. " +
        "Every mutation is strict and one-shot; never retry a timed-out legacy mutation. " +
        "Journal, knowledge, observation persistence, and training tools are not part of the core profile."
    }
  );
  const toolNames = registerTools(server, context);
  return { server, context, toolNames };
}

export async function connectStdioServer(
  config: ServerConfig = loadConfig(),
  dependencies: ServerDependencies = {}
): Promise<McpServer> {
  const { server } = createMcpServer(config, dependencies);
  const transport = new StdioServerTransport();
  await server.connect(transport);
  return server;
}
