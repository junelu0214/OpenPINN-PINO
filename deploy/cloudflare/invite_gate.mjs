// Cloudflare Pages advanced-mode worker.
// scripts/stage_demo_site.py copies this file to docs/_build/html/_worker.js
// before `wrangler pages deploy`. The worker owns every request: static Sphinx
// files are fetched only after the shared invite passphrase checks out.
// INVITE_PASSPHRASE is a Pages secret. It is not read from the repository.

export const COOKIE_NAME = "op_invite";
export const MAX_AGE_SECONDS = 12 * 60 * 60;

const MAX_PASSPHRASE_CHARS = 512;
const MAX_BODY_BYTES = 4096;
const EXPIRY_SKEW_SECONDS = 30;
const SAFE_NEXT = /^\/(?!\/)[A-Za-z0-9._~/-]*(\?[A-Za-z0-9._~%=&+-]*)?$/;

function timingSafeEqual(left, right) {
  if (!(left instanceof Uint8Array) || !(right instanceof Uint8Array)) return false;
  const length = Math.max(left.length, right.length);
  let diff = left.length === right.length ? 0 : 1;
  for (let index = 0; index < length; index += 1) {
    diff |= (left[index] || 0) ^ (right[index] || 0);
  }
  return diff === 0;
}

function bytesToBase64Url(bytes) {
  let binary = "";
  for (const value of bytes) binary += String.fromCharCode(value);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
}

function base64UrlToBytes(text) {
  if (typeof text !== "string" || !/^[A-Za-z0-9_-]+$/.test(text)) return null;
  const padded = text + "=".repeat((4 - (text.length % 4)) % 4);
  const binary = atob(padded.replaceAll("-", "+").replaceAll("_", "/"));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function configuredPassphrase(env) {
  const value = env && typeof env.INVITE_PASSPHRASE === "string" ? env.INVITE_PASSPHRASE : "";
  return value.trim() ? value : "";
}

async function sameSecret(left, right) {
  const digest = async (value) => new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)),
  );
  const [leftDigest, rightDigest] = await Promise.all([digest(String(left)), digest(String(right))]);
  return timingSafeEqual(leftDigest, rightDigest);
}

async function hmac(secret, message) {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    {name: "HMAC", hash: "SHA-256"},
    false,
    ["sign"],
  );
  const signature = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message));
  return new Uint8Array(signature);
}

export function safeNext(value) {
  if (typeof value !== "string" || value.length === 0 || value.length > 512) return "/";
  if (value.includes("\\") || value.includes("\r") || value.includes("\n") || value.includes("\0")) return "/";
  if (value.startsWith("/_invite") || value.includes("//") || !SAFE_NEXT.test(value)) return "/";
  if (/[?&]passphrase=/i.test(value)) return "/";
  try {
    const decoded = decodeURIComponent(value);
    if (decoded !== value && (decoded.includes("//") || decoded.includes("\\") || decoded.includes("://"))) return "/";
  } catch {
    return "/";
  }
  return value;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function cookieAttributes(request, maxAge) {
  const secure = new URL(request.url).protocol === "https:";
  const parts = ["HttpOnly", "SameSite=Lax", "Path=/", `Max-Age=${maxAge}`];
  if (secure) parts.push("Secure");
  return parts.join("; ");
}

function readCookie(request, name) {
  const header = request.headers.get("cookie");
  if (!header) return "";
  for (const part of header.split(";")) {
    const trimmed = part.trim();
    const separator = trimmed.indexOf("=");
    if (separator === -1) continue;
    if (trimmed.slice(0, separator) === name) return trimmed.slice(separator + 1);
  }
  return "";
}

export async function issueCookie(passphrase, options = {}) {
  const now = options.now ?? Math.floor(Date.now() / 1000);
  const ttl = options.ttl ?? MAX_AGE_SECONDS;
  const exp = now + ttl;
  const nonceBytes = new Uint8Array(18);
  crypto.getRandomValues(nonceBytes);
  const payload = `v1.${exp}.${bytesToBase64Url(nonceBytes)}`;
  const mac = bytesToBase64Url(await hmac(passphrase, payload));
  return `${payload}.${mac}`;
}

async function validSession(token, passphrase) {
  if (typeof token !== "string") return false;
  const parts = token.split(".");
  if (parts.length !== 4 || parts[0] !== "v1" || !/^[0-9]{10,12}$/.test(parts[1])) return false;
  if (!/^[A-Za-z0-9_-]{16,}$/.test(parts[2])) return false;
  const exp = Number(parts[1]);
  const now = Math.floor(Date.now() / 1000);
  if (exp + EXPIRY_SKEW_SECONDS < now) return false;
  if (exp > now + MAX_AGE_SECONDS + EXPIRY_SKEW_SECONDS) return false;
  const actual = base64UrlToBytes(parts[3]);
  if (!actual) return false;
  const expected = await hmac(passphrase, `v1.${parts[1]}.${parts[2]}`);
  return timingSafeEqual(expected, actual);
}

function htmlResponse(status, html, extraHeaders = []) {
  const headers = new Headers({
    "content-type": "text/html; charset=utf-8",
    "cache-control": "no-store",
    "cdn-cache-control": "no-store",
    "cloudflare-cdn-cache-control": "no-store",
    "x-robots-tag": "noindex, nofollow",
    "referrer-policy": "no-referrer",
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "content-security-policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
  });
  for (const [name, value] of extraHeaders) headers.append(name, value);
  return new Response(html, {status, headers});
}

function pageShell(title, body) {
  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>${escapeHtml(title)}</title>
<style>
  body { margin: 0; font: 16px/1.5 sans-serif; background: Canvas; color: CanvasText; }
  main { max-width: 36rem; margin: 4rem auto; padding: 0 1.25rem; }
  h1 { font-size: 1.5rem; font-weight: 600; }
  form { margin-top: 1.5rem; }
  label { display: block; font-weight: 600; }
  input { display: block; width: 100%; max-width: 24rem; margin-top: 0.35rem; padding: 0.5rem; box-sizing: border-box; }
  button { margin-top: 0.85rem; padding: 0.45rem 0.9rem; }
  p[role="alert"] { font-weight: 600; }
</style>
</head>
<body>
<main>
${body}
</main>
</body>
</html>`;
}

function unavailable(message) {
  return htmlResponse(503, pageShell(
    "OpenPINN demo unavailable",
    `<h1>OpenPINN 演示站暂不可用</h1>
<p>${escapeHtml(message)}</p>
<p>This demo is not available. The configurator is not served until the invite gate is configured.</p>`,
  ));
}

function invitePage(request, {rejected = false, next = "/"} = {}) {
  const destination = safeNext(next || new URL(request.url).pathname + new URL(request.url).search);
  const alert = rejected
    ? `<p role="alert">邀请口令不正确。That invite passphrase was not accepted.</p>`
    : "";
  return htmlResponse(401, pageShell(
    "OpenPINN demo invite",
    `<h1>OpenPINN 演示站</h1>
<p>向项目维护者另行索取统一邀请口令，不要把它写进仓库或公开 issue。站点只生成可在本地运行的 Python 脚本，不在浏览器中训练。</p>
<p>Ask the project operator for the shared invite passphrase out of band. Do not commit it or post it in a public issue. This site copies or downloads a <code>.py</code> file. It does not train in the browser.</p>
${alert}
<form method="post" action="/_invite">
  <input type="hidden" name="next" value="${escapeHtml(destination)}">
  <label for="passphrase">邀请口令 Invite passphrase</label>
  <input id="passphrase" name="passphrase" type="password" autocomplete="current-password" required maxlength="${MAX_PASSPHRASE_CHARS}">
  <button type="submit">进入 Continue</button>
</form>`,
  ));
}

function redirect(location, cookies = []) {
  const headers = new Headers({
    location,
    "cache-control": "no-store",
    "cdn-cache-control": "no-store",
    "cloudflare-cdn-cache-control": "no-store",
    "referrer-policy": "no-referrer",
    "x-robots-tag": "noindex, nofollow",
  });
  for (const cookie of cookies) headers.append("set-cookie", cookie);
  return new Response(null, {status: 303, headers});
}

function sessionCookie(request, token) {
  return `${COOKIE_NAME}=${token}; ${cookieAttributes(request, MAX_AGE_SECONDS)}`;
}

function clearedCookie(request) {
  return `${COOKIE_NAME}=; ${cookieAttributes(request, 0)}`;
}

function privatize(response) {
  const headers = new Headers();
  for (const [name, value] of response.headers) {
    if (name.toLowerCase() === "set-cookie") continue;
    headers.set(name, value);
  }
  if (typeof response.headers.getSetCookie === "function") {
    for (const cookie of response.headers.getSetCookie()) headers.append("set-cookie", cookie);
  }
  headers.set("cache-control", "private, no-store");
  headers.set("cdn-cache-control", "no-store");
  headers.set("cloudflare-cdn-cache-control", "no-store");
  headers.set("x-robots-tag", "noindex, nofollow");
  headers.set("referrer-policy", "no-referrer");
  headers.set("x-content-type-options", "nosniff");
  headers.set("x-frame-options", "DENY");
  headers.set("vary", "Cookie");
  const status = response.status;
  const body = status === 204 || status === 205 || status === 304 ? null : response.body;
  return new Response(body, {status, statusText: response.statusText, headers});
}

async function readLimitedBytes(request) {
  if (!request.body) return new Uint8Array();
  const reader = request.body.getReader();
  const chunks = [];
  let total = 0;
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > MAX_BODY_BYTES) {
      await reader.cancel();
      return null;
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function login(request, passphrase) {
  const type = (request.headers.get("content-type") || "").split(";")[0].trim().toLowerCase();
  if (type !== "application/x-www-form-urlencoded") return invitePage(request, {rejected: true});
  const declared = Number(request.headers.get("content-length") || "0");
  if (Number.isFinite(declared) && declared > MAX_BODY_BYTES) return invitePage(request, {rejected: true});
  const bytes = await readLimitedBytes(request);
  if (!bytes) return invitePage(request, {rejected: true});
  const params = new URLSearchParams(new TextDecoder("utf-8", {fatal: false}).decode(bytes));
  const submitted = params.get("passphrase") ?? "";
  const next = safeNext(params.get("next") || "/");
  if (submitted.length === 0 || submitted.length > MAX_PASSPHRASE_CHARS) {
    return invitePage(request, {rejected: true, next});
  }
  if (!(await sameSecret(submitted, passphrase))) return invitePage(request, {rejected: true, next});
  const token = await issueCookie(passphrase);
  return redirect(next, [sessionCookie(request, token)]);
}

function blockedAsset(pathname) {
  return pathname === "/_worker.js" || pathname.endsWith("/_worker.js") || pathname === "/.dev.vars";
}

export async function handleRequest(request, env, assetsFetch) {
  const passphrase = configuredPassphrase(env);
  const url = new URL(request.url);
  if (!passphrase) return unavailable("This demo is not available.");
  if (url.pathname === "/_invite/logout") return redirect("/", [clearedCookie(request)]);
  if (request.method === "POST" && url.pathname === "/_invite") return login(request, passphrase);
  if (await validSession(readCookie(request, COOKIE_NAME), passphrase)) {
    if (url.pathname === "/_invite" || url.pathname === "/_invite/") return redirect("/");
    if (blockedAsset(url.pathname)) {
      return new Response("Not found", {
        status: 404,
        headers: {"cache-control": "no-store", "content-type": "text/plain; charset=utf-8"},
      });
    }
    try {
      const upstream = await assetsFetch(request);
      if (!upstream) return unavailable("This page is unavailable.");
      return privatize(upstream);
    } catch {
      return unavailable("This page is unavailable.");
    }
  }
  return invitePage(request, {next: url.pathname + url.search});
}

export default {
  async fetch(request, env) {
    const bindings = env || {};
    if (!bindings.ASSETS || typeof bindings.ASSETS.fetch !== "function") {
      return handleRequest(request, bindings, async () => unavailable("Demo assets are unavailable."));
    }
    return handleRequest(request, bindings, (nextRequest) => env.ASSETS.fetch(nextRequest));
  },
};
