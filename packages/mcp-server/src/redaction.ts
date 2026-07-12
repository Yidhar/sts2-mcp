const SENSITIVE_KEY = /(?:^|[_-])(authorization|tokens?|secrets?|credentials?|passwords?|cookies?|api[-_]?keys?)(?:$|[_-])|(?:access|refresh|session|bearer)token$|clientsecret$|apikey$/i;
const SAFE_PRESENCE_KEY = /(?:^|[_-])(authorization|tokens?|secrets?|credentials?)(?:[_-]map)?[_-]present$/i;
const BEARER_VALUE = /\bBearer\s+[^\s,;"']+/gi;

export function redactSecrets(value: unknown, secrets: readonly string[] = []): unknown {
  const seen = new WeakSet<object>();
  const knownSecrets = secrets.filter((secret) => typeof secret === "string" && secret.length > 0);

  function visit(current: unknown, key?: string): unknown {
    if (key && SENSITIVE_KEY.test(key) && !SAFE_PRESENCE_KEY.test(key)) {
      return "[REDACTED]";
    }
    if (typeof current === "string") {
      let redacted = current.replace(BEARER_VALUE, "Bearer [REDACTED]");
      for (const secret of knownSecrets) {
        redacted = redacted.split(secret).join("[REDACTED]");
      }
      return redacted;
    }
    if (current === null || typeof current !== "object") {
      return current;
    }
    if (seen.has(current)) {
      return "[CIRCULAR]";
    }
    seen.add(current);
    if (Array.isArray(current)) {
      return current.map((entry) => visit(entry));
    }
    const result: Record<string, unknown> = {};
    for (const [entryKey, entryValue] of Object.entries(current)) {
      result[entryKey] = visit(entryValue, entryKey);
    }
    return result;
  }

  return visit(value);
}

export function safeStringify(value: unknown, secrets: readonly string[] = []): string {
  return JSON.stringify(redactSecrets(value, secrets));
}
