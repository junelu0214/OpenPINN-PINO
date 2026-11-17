// Loopback reproduction of the Pages invite gate. Production traffic is served
// by the copied _worker.js on Cloudflare Pages, not by this process.

import {createServer} from "node:http";
import {readFile, realpath, stat} from "node:fs/promises";
import path from "node:path";
import {pathToFileURL} from "node:url";

import {handleRequest} from "./invite_gate.mjs";

const CONTENT_TYPES = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".svg", "image/svg+xml"],
  [".json", "application/json"],
  [".png", "image/png"],
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"],
  [".gif", "image/gif"],
  [".webp", "image/webp"],
  [".ico", "image/x-icon"],
  [".txt", "text/plain; charset=utf-8"],
  [".map", "application/json"],
  [".woff", "font/woff"],
  [".woff2", "font/woff2"],
]);

function contentType(file) {
  return CONTENT_TYPES.get(path.extname(file).toLowerCase()) || "application/octet-stream";
}

function notFound() {
  return new Response("Not found", {
    status: 404,
    headers: {"content-type": "text/plain; charset=utf-8", "cache-control": "no-store"},
  });
}

export async function readAsset(root, request) {
  const url = new URL(request.url);
  let decoded;
  try {
    decoded = decodeURIComponent(url.pathname);
  } catch {
    return notFound();
  }
  if (decoded.includes("\0") || decoded.includes("\\")) return notFound();
  if (decoded.endsWith("/")) decoded += "index.html";
  const rootReal = await realpath(root);
  const candidate = path.resolve(rootReal, decoded.replace(/^\/+/, ""));
  if (candidate !== rootReal && !candidate.startsWith(rootReal + path.sep)) return notFound();
  let fileReal;
  try {
    fileReal = await realpath(candidate);
  } catch {
    return notFound();
  }
  if (fileReal !== rootReal && !fileReal.startsWith(rootReal + path.sep)) return notFound();
  const info = await stat(fileReal);
  const file = info.isDirectory() ? path.join(fileReal, "index.html") : fileReal;
  let finalReal;
  try {
    finalReal = await realpath(file);
  } catch {
    return notFound();
  }
  if (!finalReal.startsWith(rootReal + path.sep)) return notFound();
  const data = await readFile(finalReal);
  return new Response(data, {status: 200, headers: {"content-type": contentType(finalReal)}});
}

async function toFetchRequest(req) {
  if (!req.url || !req.url.startsWith("/")) return null;
  const chunks = [];
  let total = 0;
  for await (const chunk of req) {
    total += chunk.length;
    if (total > 1024 * 1024) return null;
    chunks.push(chunk);
  }
  const body = Buffer.concat(chunks);
  const headers = new Headers();
  if (req.headers.cookie) headers.set("cookie", req.headers.cookie);
  if (req.headers["content-type"]) headers.set("content-type", req.headers["content-type"]);
  const host = req.headers.host && /^127\.0\.0\.1(?::\d+)?$/.test(req.headers.host)
    ? req.headers.host
    : "127.0.0.1";
  const method = req.method || "GET";
  const init = {method, headers};
  if (method !== "GET" && method !== "HEAD") init.body = body;
  return new Request(`http://${host}${req.url}`, init);
}

async function writeNodeResponse(res, response) {
  res.statusCode = response.status;
  response.headers.forEach((value, key) => {
    if (key.toLowerCase() === "set-cookie") return;
    res.setHeader(key, value);
  });
  if (typeof response.headers.getSetCookie === "function") {
    const cookies = response.headers.getSetCookie();
    if (cookies.length) res.setHeader("set-cookie", cookies);
  }
  const bytes = Buffer.from(await response.arrayBuffer());
  res.end(bytes);
}

export function createPreviewServer({root, passphrase, host = "127.0.0.1"}) {
  if (typeof passphrase !== "string" || !passphrase.trim()) {
    throw new Error("INVITE_PASSPHRASE is required. Refusing to serve an ungated site.");
  }
  const resolvedRoot = path.resolve(root);
  return createServer((req, res) => {
    Promise.resolve()
      .then(async () => {
        const request = await toFetchRequest(req);
        if (!request) {
          res.statusCode = 400;
          res.end("Bad request");
          return;
        }
        const response = await handleRequest(
          request,
          {INVITE_PASSPHRASE: passphrase},
          (nextRequest) => readAsset(resolvedRoot, nextRequest),
        );
        await writeNodeResponse(res, response);
      })
      .catch(() => {
        if (!res.headersSent) {
          res.statusCode = 500;
          res.setHeader("content-type", "text/plain; charset=utf-8");
          res.end("Preview failed.");
        } else {
          res.destroy();
        }
      });
  });
}

export function listenPreview({root, passphrase, host = "127.0.0.1", port = 0}) {
  const server = createPreviewServer({root, passphrase, host});
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, () => resolve(server));
  });
}

async function main() {
  const passphrase = process.env.INVITE_PASSPHRASE || "";
  if (!passphrase.trim()) {
    console.error("INVITE_PASSPHRASE is not set. Refusing to serve an ungated site.");
    process.exit(1);
  }
  const root = path.resolve(process.argv[2] || "docs/_build/html");
  try {
    const info = await stat(root);
    if (!info.isDirectory()) throw new Error("not a directory");
  } catch {
    console.error(`HTML directory not found: ${root}`);
    console.error("Build the site first with: python scripts/build_docs.py");
    process.exit(1);
  }
  const port = Number(process.env.PORT || 8765);
  const server = await listenPreview({root, passphrase, port});
  const address = server.address();
  console.log(`Gated preview: http://127.0.0.1:${address.port}/`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
