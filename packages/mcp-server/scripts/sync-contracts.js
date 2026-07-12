"use strict";

const fs = require("node:fs");
const path = require("node:path");

const source = path.resolve(
  __dirname,
  "..",
  "..",
  "..",
  "contracts",
  "generated",
  "typescript",
  "versions.ts"
);
const target = path.resolve(__dirname, "..", "src", "generated", "contract-versions.ts");
const expected = fs.readFileSync(source, "utf8").replace(/\r\n/g, "\n");
const checkOnly = process.argv.includes("--check");

if (checkOnly) {
  const actual = fs.existsSync(target)
    ? fs.readFileSync(target, "utf8").replace(/\r\n/g, "\n")
    : "";
  if (actual !== expected) {
    process.stderr.write("Generated MCP contract versions are stale. Run npm run sync:contracts.\n");
    process.exitCode = 1;
  }
} else {
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, expected, "utf8");
}
