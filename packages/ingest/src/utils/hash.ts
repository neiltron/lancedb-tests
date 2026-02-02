import crypto from "node:crypto";

export function hashId(input: string): string {
  return crypto.createHash("sha1").update(input).digest("hex");
}
