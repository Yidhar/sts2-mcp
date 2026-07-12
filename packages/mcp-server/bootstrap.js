"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const packageRoot = __dirname;
const compiledServer = path.join(packageRoot, "dist", "src", "server.js");
const contractSource = path.resolve(
  packageRoot,
  "..",
  "..",
  "contracts",
  "generated",
  "typescript",
  "versions.ts"
);
const contractCopy = path.join(packageRoot, "src", "generated", "contract-versions.ts");

function latestSourceMtime(directory) {
  let latest = 0;
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) latest = Math.max(latest, latestSourceMtime(fullPath));
    else if (entry.isFile() && entry.name.endsWith(".ts")) {
      latest = Math.max(latest, fs.statSync(fullPath).mtimeMs);
    }
  }
  return latest;
}

function assertContractCopyCurrent() {
  const normalize = (value) => value.replace(/\r\n/g, "\n");
  if (
    !fs.existsSync(contractSource) ||
    !fs.existsSync(contractCopy) ||
    normalize(fs.readFileSync(contractSource, "utf8")) !==
      normalize(fs.readFileSync(contractCopy, "utf8"))
  ) {
    throw new Error(
      "Generated MCP contract versions are stale. Run npm run sync:contracts and rebuild."
    );
  }
}

function ensureBuild() {
  assertContractCopyCurrent();
  const outputMtime = fs.existsSync(compiledServer) ? fs.statSync(compiledServer).mtimeMs : 0;
  const sourceMtime = latestSourceMtime(path.join(packageRoot, "src"));
  if (outputMtime >= sourceMtime) return;

  const compiler = path.join(packageRoot, "node_modules", "typescript", "bin", "tsc");
  if (!fs.existsSync(compiler)) {
    throw new Error("MCP dependencies are not installed. Run npm ci before starting the server.");
  }
  const result = spawnSync(process.execPath, [compiler, "-p", "tsconfig.json"], {
    cwd: packageRoot,
    stdio: ["ignore", "ignore", "pipe"],
    encoding: "utf8"
  });
  if (result.status !== 0) {
    const detail = (result.stderr || "TypeScript build failed.").trim();
    throw new Error(detail.slice(0, 4_000));
  }
}

module.exports = { ensureBuild };
