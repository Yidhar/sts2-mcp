import test from "node:test";
import assert from "node:assert/strict";
import { redactSecrets, safeStringify } from "../src/redaction";

test("recursive redaction removes credential keys and bearer values", () => {
  const token = "top-secret-token";
  const value = {
    token,
    token_present: true,
    capability_tokens_present: ["player-control", "training"],
    capability_tokens: { "player-control": token, training: "other-secret" },
    clientSecret: "camel-secret",
    refresh_token: "refresh-secret",
    nested: {
      Authorization: `Bearer ${token}`,
      message: `request failed with Bearer ${token}`,
      safe: "visible"
    }
  };
  const json = safeStringify(redactSecrets(value, [token]));
  assert(!json.includes(token));
  assert(!json.includes("other-secret"));
  assert(!json.includes("camel-secret"));
  assert(!json.includes("refresh-secret"));
  assert.match(json, /\[REDACTED\]/);
  assert.match(json, /"token_present":true/);
  assert.match(json, /"capability_tokens_present":\["player-control","training"\]/);
  assert.match(json, /"safe":"visible"/);
});
