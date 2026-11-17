import assert from "node:assert/strict";
import {spawnSync} from "node:child_process";
import {mkdtemp, mkdir, readFile, symlink, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import path from "node:path";
import {test} from "node:test";
import {fileURLToPath} from "node:url";

import worker, {
  COOKIE_NAME,
  MAX_AGE_SECONDS,
  handleRequest,
  issueCookie,
} from "../../deploy/cloudflare/invite_gate.mjs";
import {listenPreview} from "../../deploy/cloudflare/preview_server.mjs";

const SECRET = "fixture-invite-!";
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const MARKER = "CONFIGURATOR_ASSET_MARKER";

function env(passphrase = SECRET) {
  return {INVITE_PASSPHRASE: passphrase};
}

function assets() {
  const calls = [];
  return {
    calls,
    async fetch(request) {
      calls.push(new URL(request.url).pathname);
      return new Response(`${MARKER} data-role="download-code"`, {
        status: 200,
        headers: {
          "content-type": "text/html; charset=utf-8",
          "cache-control": "public, max-age=3600",
        },
      });
    },
  };
}

function cookieHeader(token) {
  return {cookie: `${COOKIE_NAME}=${token}`};
}

async function textOf(response) {
  return response.text();
}

test("a missing passphrase fails closed and does not fetch assets", async () => {
  const site = assets();
  const response = await handleRequest(
    new Request("https://demo.example/method_configurator.html"),
    {},
    site.fetch,
  );
  assert.equal(response.status, 503);
  assert.equal(response.headers.get("www-authenticate"), null);
  const body = await textOf(response);
  assert.equal(body.includes(MARKER), false);
  assert.equal(body.includes("download-code"), false);
  assert.equal(site.calls.length, 0);
});

test("an unauthenticated visitor receives the invite page and not the configurator", async () => {
  const site = assets();
  const response = await handleRequest(
    new Request("https://demo.example/method_configurator.html"),
    env(),
    site.fetch,
  );
  assert.equal(response.status, 401);
  assert.equal(response.headers.get("www-authenticate"), null);
  assert.equal(response.headers.get("cache-control"), "no-store");
  const body = await textOf(response);
  assert.match(body, /邀请口令/);
  assert.match(body, /does not train/);
  assert.equal(body.includes(MARKER), false);
  assert.equal(body.includes("download-code"), false);
  assert.equal(site.calls.length, 0);
});

test("static assets and a passphrase in the query string stay gated", async () => {
  const site = assets();
  const response = await handleRequest(
    new Request(`https://demo.example/_static/method_configurator.js?passphrase=${encodeURIComponent(SECRET)}`),
    env(),
    site.fetch,
  );
  assert.equal(response.status, 401);
  const body = await textOf(response);
  assert.equal(body.includes(SECRET), false);
  assert.equal(body.includes(MARKER), false);
  assert.equal(site.calls.length, 0);
});

test("a wrong passphrase is rejected and is not reflected", async () => {
  const site = assets();
  const body = new URLSearchParams({passphrase: "wrong-invite", next: "/method_configurator.html"});
  const response = await handleRequest(
    new Request("https://demo.example/_invite", {
      method: "POST",
      headers: {"content-type": "application/x-www-form-urlencoded; charset=UTF-8"},
      body,
    }),
    env(),
    site.fetch,
  );
  assert.equal(response.status, 401);
  assert.equal(response.headers.get("set-cookie"), null);
  const html = await textOf(response);
  assert.match(html, /was not accepted/);
  assert.equal(html.includes("wrong-invite"), false);
  assert.equal(html.includes(MARKER), false);
  assert.equal(site.calls.length, 0);
});

test("posting the passphrase to a configurator URL does not authenticate", async () => {
  const site = assets();
  const response = await handleRequest(
    new Request("https://demo.example/method_configurator.html", {
      method: "POST",
      headers: {"content-type": "application/x-www-form-urlencoded"},
      body: new URLSearchParams({passphrase: SECRET}),
    }),
    env(),
    site.fetch,
  );
  assert.equal(response.status, 401);
  assert.equal(site.calls.length, 0);
});

test("the correct passphrase sets a secure cookie and does not embed the secret", async () => {
  const site = assets();
  const response = await handleRequest(
    new Request("https://demo.example/_invite", {
      method: "POST",
      headers: {"content-type": "application/x-www-form-urlencoded"},
      body: new URLSearchParams({passphrase: SECRET, next: "/method_configurator.html"}),
    }),
    env(),
    site.fetch,
  );
  assert.equal(response.status, 303);
  assert.equal(response.headers.get("location"), "/method_configurator.html");
  const cookie = response.headers.get("set-cookie") || "";
  const flags = cookie.split(";").slice(1).map((part) => part.trim());
  assert.match(cookie, new RegExp(`^${COOKIE_NAME}=`));
  assert.ok(flags.includes("HttpOnly"));
  assert.ok(flags.includes("SameSite=Lax"));
  assert.ok(flags.includes("Secure"));
  assert.equal(cookie.includes(SECRET), false);
  assert.equal(site.calls.length, 0);
});

test("an open redirect is dropped", async () => {
  for (const next of ["https://evil.example/phish", "//evil.example", "/\\evil", "/%2f%2fevil.example"]) {
    const response = await handleRequest(
      new Request("https://demo.example/_invite", {
        method: "POST",
        headers: {"content-type": "application/x-www-form-urlencoded"},
        body: new URLSearchParams({passphrase: SECRET, next}),
      }),
      env(),
      async () => {
        throw new Error("assets should not be fetched");
      },
    );
    assert.equal(response.status, 303, next);
    assert.equal(response.headers.get("location"), "/", next);
  }
});

test("a valid cookie serves the asset with private caching", async () => {
  const token = await issueCookie(SECRET);
  const site = assets();
  const response = await handleRequest(
    new Request("https://demo.example/method_configurator.html", {headers: cookieHeader(token)}),
    env(),
    site.fetch,
  );
  assert.equal(response.status, 200);
  assert.equal(await textOf(response), `${MARKER} data-role="download-code"`);
  assert.equal(response.headers.get("cache-control"), "private, no-store");
  assert.equal(response.headers.get("cdn-cache-control"), "no-store");
  assert.deepEqual(site.calls, ["/method_configurator.html"]);
});

test("a tampered, expired, or overlong cookie does not fetch assets", async () => {
  const token = await issueCookie(SECRET);
  const parts = token.split(".");
  const nonce = parts[2].split("");
  nonce[0] = nonce[0] === "A" ? "B" : "A";
  parts[2] = nonce.join("");
  const tampered = parts.join(".");
  const expired = await issueCookie(SECRET, {ttl: -120});
  const overlong = await issueCookie(SECRET, {ttl: MAX_AGE_SECONDS + 120});
  for (const [label, candidate] of [
    ["tampered", tampered],
    ["expired", expired],
    ["overlong", overlong],
    ["malformed", "v1.not-a-cookie"],
  ]) {
    const site = assets();
    const response = await handleRequest(
      new Request("https://demo.example/method_configurator.html", {headers: cookieHeader(candidate)}),
      env(),
      site.fetch,
    );
    assert.equal(response.status, 401, label);
    assert.equal(site.calls.length, 0, label);
  }
});

test("logout clears the session and the worker script is not an asset", async () => {
  const token = await issueCookie(SECRET);
  const site = assets();
  const loggedOut = await handleRequest(
    new Request("https://demo.example/_invite/logout", {headers: cookieHeader(token)}),
    env(),
    site.fetch,
  );
  assert.equal(loggedOut.status, 303);
  assert.match(loggedOut.headers.get("set-cookie") || "", /Max-Age=0/);
  assert.equal(site.calls.length, 0);

  const workerFile = await handleRequest(
    new Request("https://demo.example/_worker.js", {headers: cookieHeader(token)}),
    env(),
    site.fetch,
  );
  assert.equal(workerFile.status, 404);
  assert.equal(site.calls.length, 0);
});

test("the pages worker entrypoint uses env.ASSETS.fetch only after authentication", async () => {
  const site = assets();
  const anonymous = await worker.fetch(new Request("https://demo.example/plate_with_hole.html"), {
    INVITE_PASSPHRASE: SECRET,
    ASSETS: {fetch: site.fetch},
  });
  assert.equal(anonymous.status, 401);
  assert.equal(site.calls.length, 0);

  const token = await issueCookie(SECRET);
  const admitted = await worker.fetch(
    new Request("https://demo.example/method_configurator.html", {headers: cookieHeader(token)}),
    {INVITE_PASSPHRASE: SECRET, ASSETS: {fetch: site.fetch}},
  );
  assert.equal(admitted.status, 200);
  assert.match(await textOf(admitted), new RegExp(MARKER));

  let fetched = false;
  const broken = await worker.fetch(
    new Request("https://demo.example/method_configurator.html", {headers: cookieHeader(token)}),
    {
      INVITE_PASSPHRASE: SECRET,
      ASSETS: {
        fetch: async () => {
          fetched = true;
          throw new Error("asset backend path");
        },
      },
    },
  );
  assert.equal(broken.status, 503);
  assert.equal(fetched, true);
  assert.equal((await textOf(broken)).includes("asset backend path"), false);
});

test("the loopback preview gates files and refuses paths outside the site root", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "openpinn-gate-"));
  const site = path.join(directory, "site");
  await mkdir(site);
  await writeFile(
    path.join(site, "method_configurator.html"),
    `<p>${MARKER}</p><button data-role="download-code">Download .py</button>`,
  );
  await writeFile(path.join(directory, "secret.txt"), "outside-the-site");
  await symlink(path.join(directory, "secret.txt"), path.join(site, "leak.txt"));

  const server = await listenPreview({root: site, passphrase: SECRET, port: 0});
  const port = server.address().port;
  try {
    const anonymous = await fetch(`http://127.0.0.1:${port}/method_configurator.html`);
    assert.equal(anonymous.status, 401);
    assert.equal((await anonymous.text()).includes(MARKER), false);

    const rejected = await fetch(`http://127.0.0.1:${port}/_invite`, {
      method: "POST",
      redirect: "manual",
      headers: {"content-type": "application/x-www-form-urlencoded"},
      body: new URLSearchParams({passphrase: "nope", next: "/method_configurator.html"}),
    });
    assert.equal(rejected.status, 401);

    const admitted = await fetch(`http://127.0.0.1:${port}/_invite`, {
      method: "POST",
      redirect: "manual",
      headers: {"content-type": "application/x-www-form-urlencoded"},
      body: new URLSearchParams({passphrase: SECRET, next: "/method_configurator.html"}),
    });
    assert.equal(admitted.status, 303);
    const setCookie = admitted.headers.get("set-cookie") || "";
    const flags = setCookie.split(";").slice(1).map((part) => part.trim());
    assert.ok(flags.includes("HttpOnly"));
    assert.equal(flags.includes("Secure"), false);
    assert.equal(setCookie.includes(SECRET), false);
    const pair = setCookie.split(";")[0];

    const page = await fetch(`http://127.0.0.1:${port}/method_configurator.html`, {
      headers: {cookie: pair},
    });
    assert.equal(page.status, 200);
    assert.match(await page.text(), new RegExp(MARKER));
    assert.equal(page.headers.get("cache-control"), "private, no-store");

    for (const route of ["/%2e%2e/secret.txt", "/leak.txt", "/_worker.js"]) {
      const blocked = await fetch(`http://127.0.0.1:${port}${route}`, {headers: {cookie: pair}});
      const body = await blocked.text();
      assert.equal(body.includes("outside-the-site"), false, route);
      assert.equal(body.includes("handleRequest"), false, route);
    }
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("the preview process refuses to start without a passphrase or an HTML directory", () => {
  const script = path.join(ROOT, "deploy/cloudflare/preview_server.mjs");
  const missingSecret = spawnSync(process.execPath, [script], {
    cwd: ROOT,
    encoding: "utf8",
    env: {...process.env, INVITE_PASSPHRASE: ""},
  });
  assert.notEqual(missingSecret.status, 0);
  assert.match(missingSecret.stderr, /INVITE_PASSPHRASE/);

  const missingDir = spawnSync(process.execPath, [script, path.join(tmpdir(), "openpinn-missing-html")], {
    cwd: ROOT,
    encoding: "utf8",
    env: {...process.env, INVITE_PASSPHRASE: SECRET},
  });
  assert.notEqual(missingDir.status, 0);
  assert.match(missingDir.stderr, /HTML directory not found/);
});

test("gate sources and docs do not contain the test passphrase or a training route", async () => {
  const files = [
    "deploy/cloudflare/invite_gate.mjs",
    "deploy/cloudflare/preview_server.mjs",
    "docs/demo_site.md",
    "README.md",
    ".github/workflows/pages.yml",
  ];
  for (const relative of files) {
    const text = await readFile(path.join(ROOT, relative), "utf8");
    assert.equal(text.includes(SECRET), false, relative);
  }
  const worker = await readFile(path.join(ROOT, "deploy/cloudflare/invite_gate.mjs"), "utf8");
  assert.equal(worker.includes("train_operator"), false);
  assert.equal(worker.includes("node:"), false);
});
