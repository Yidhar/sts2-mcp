import { loadConfig } from "./config";
import { errorBody } from "./errors";
import { connectStdioServer } from "./protocol/server";
import { safeStringify } from "./redaction";

export async function startServer(): Promise<void> {
  const config = loadConfig();
  const server = await connectStdioServer(config);
  const shutdown = async () => {
    await server.close();
    process.exitCode = 0;
  };
  process.once("SIGINT", shutdown);
  process.once("SIGTERM", shutdown);
}

if (require.main === module) {
  startServer().catch((error) => {
    process.stderr.write(`${safeStringify(errorBody(error))}\n`);
    process.exitCode = 1;
  });
}
