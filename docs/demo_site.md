# Gated demo site

The external demo is the existing Sphinx configurator on Cloudflare Pages. A site-wide Worker asks for one shared invite passphrase before it serves any HTML, script, or stylesheet. The configurator still only copies or downloads a `.py` file. Nothing trains in the browser.

This is not Cloudflare Access. Access policies are per person (one-time PIN, an identity provider, or a service token). M7 uses one shared invite, so the gate is a passphrase page in the Pages Worker. It is also not public GitHub Pages. The old `build-and-publish` step that pushed to `xiwang0706/openpinn-docs` is gone.

## Demo URL

[https://openpinn-upstream-demo.pages.dev](https://openpinn-upstream-demo.pages.dev)

The gated demo is deployed at that hostname. An unauthenticated request receives the invite page with HTTP 401 rather than Sphinx assets. If the project name has to change, update this URL, the `--project-name` in `.github/workflows/pages.yml`, and the commands in this file together.

## How to request and use the invite

Ask the OpenPINN-upstream operator for the passphrase out of band. Do not request it in a public GitHub issue, and do not commit it.

Open the demo URL. The response is HTTP 401 and an invite page until the passphrase is accepted. Enter the one shared passphrase. It is not a personal account. The session cookie lasts 12 hours. Sign out by opening `/_invite/logout`.

Then open `method_configurator.html`. Choose the operator path, select DeepONet or FNO and `data_only`, `physics_only`, or `hybrid` training, design the custom 1D or 2D PDE and boundaries, and copy or download the `.py` file. Data-only and hybrid require a local training-observation CSV. Physics-only hides that entry. An optional post-train reference CSV is separate and is not reused as the observation. PINO is not a separate model option. Run that file locally; the page does not train or upload data. Built-in Poisson, Burgers, tunnel, and plate examples remain tutorials, not operator-path solution options.

A missing or blank Pages secret fails closed with HTTP 503 and does not serve the configurator. A wrong passphrase stays on the invite page and does not serve it either.

## Secrets

Nothing in this list is committed.

GitHub repository secrets (Settings → Secrets and variables → Actions):

| Name | Where it is used |
|---|---|
| `CLOUDFLARE_API_TOKEN` | Wrangler authentication for the deploy job. Create a custom token with Account → Cloudflare Pages → Edit. |
| `CLOUDFLARE_ACCOUNT_ID` | The account that owns the Pages project. Copy it from the Workers & Pages overview. |

Cloudflare Pages secret (Workers & Pages → `openpinn-upstream-demo` → Settings → Variables and Secrets → Production), **not** a GitHub secret:

| Name | Where it is used |
|---|---|
| `INVITE_PASSPHRASE` | The shared invite. The Worker reads it at request time and uses it as the HMAC key for the session cookie. |

The deploy job skips publication when either GitHub secret is unset, and it still exits successfully after the Sphinx build. It never reads `INVITE_PASSPHRASE`, so the passphrase cannot land in an Actions log. A deploy without that Pages secret stays closed.

## Project setup reference

The current Pages project is already deployed. Operators need this setup
procedure only when recreating the project or restoring its deployment
credentials; routine invite rotation uses the later rotation section.

Export the API token and account id for the current shell. Do not put them in a file that is committed.

```bash
read -r -s CLOUDFLARE_API_TOKEN
export CLOUDFLARE_API_TOKEN
read -r CLOUDFLARE_ACCOUNT_ID
export CLOUDFLARE_ACCOUNT_ID
npx --yes wrangler@4.153.0 pages project create openpinn-upstream-demo --production-branch main --compatibility-date 2024-11-11
```

Generate a passphrase locally and share it with demo visitors out of band:

```bash
openssl rand -base64 24
npx --yes wrangler@4.153.0 pages secret put INVITE_PASSPHRASE --project-name openpinn-upstream-demo
```

`pages secret put` prompts for the value. Paste the passphrase once. Do not pass it as a command-line argument. The comparison is exact, so do not add quotes or a trailing space. Wrangler drops the single newline from the prompt.

Add the two GitHub secrets, then push to `main` or run the `CI` workflow with `workflow_dispatch` on `main`.

## What the workflow does

`.github/workflows/pages.yml` job `build-and-publish` (push to `main`, or `workflow_dispatch`):

1. Install the documentation extra and build both strict Sphinx language trees, the same way the pull-request `docs` job does:

```bash
python scripts/build_docs.py
```

The default Chinese site is written to `docs/_build/html/`; the English counterpart is nested at `docs/_build/html/en/` and is deployed with it.

2. Refuse the build if `docs/_build/html/_modules` exists.
3. Run `python scripts/stage_demo_site.py`, which copies `deploy/cloudflare/invite_gate.mjs` to `docs/_build/html/_worker.js`.
4. If `CLOUDFLARE_API_TOKEN` or `CLOUDFLARE_ACCOUNT_ID` is missing, print a notice and stop without deploying.
5. Otherwise upload that directory:

```bash
npx --yes wrangler@4.153.0 pages deploy docs/_build/html \
  --project-name openpinn-upstream-demo \
  --branch "${GITHUB_REF_NAME}" \
  --commit-dirty=true \
  --commit-hash "${GITHUB_SHA}"
```

`--commit-dirty=true` is required because the HTML output is gitignored. The Worker is advanced mode: it handles every path, and it calls `env.ASSETS.fetch` only after the cookie checks out. Unauthenticated requests never receive Sphinx files, including `/_static/*`.

Pull requests run the `docs` job only. They do not deploy.

## Check the gate

```bash
curl -s -o /tmp/openpinn-gate.html -w '%{http_code}\n' \
  https://openpinn-upstream-demo.pages.dev/method_configurator.html
```

Expect `401` when the passphrase is set, or `503` when the Pages secret is missing. The saved HTML is the invite page or the unavailable page. It is not the configurator. Then use a browser: reject a wrong passphrase, accept the real one, and download one operator script after checking its PDE, boundary, observation, and visualization controls.

## Human verify once

The operator who holds the passphrase should complete this authenticated check
once after a deployment. Never paste the passphrase into git, an issue, a PR,
documentation, or CI output.

1. Open the demo URL in a private window and confirm the invite page appears.
2. Enter the shared passphrase received out of band and confirm the Sphinx site opens.
3. Open `method_configurator.html`; choose DeepONet or FNO plus one training mode, design a custom 1D or 2D PDE and boundaries, configure a local CSV Path when the mode needs observations, and copy or download a `.py` file.
4. Confirm a built-in plate or solid example remains reachable as a tutorial rather than an operator solution-path option.
5. In a fresh private window, confirm a wrong passphrase remains gated.
6. Open `/_invite/logout` and confirm the session returns to the invite page.

If any step fails, report the blocker to the OpenPINN operator without including
the passphrase.

## Rotate the invite

1. Generate a new passphrase with `openssl rand -base64 24`.
2. Run `npx --yes wrangler@4.153.0 pages secret put INVITE_PASSPHRASE --project-name openpinn-upstream-demo` and paste the new value when prompted.
3. Tell current demo visitors the new passphrase out of band.
4. Existing cookies stop working because the HMAC key is the passphrase. A code deploy is not required.
5. In a private window, confirm the previous passphrase is rejected and the new one opens `method_configurator.html`.
6. If the previous passphrase still works after a few minutes, rerun the `CI` workflow on `main`.

There is no second passphrase and no per-user account. Rotating the GitHub API token does not change the invite. Delete an old API token in the Cloudflare dashboard after creating its replacement.

## Local reproduction

The loopback preview uses the same Worker code. It is not the external site.

```bash
python scripts/build_docs.py
read -r -s INVITE_PASSPHRASE
export INVITE_PASSPHRASE
node deploy/cloudflare/preview_server.mjs
```

Open `http://127.0.0.1:8765/`. The process listens on loopback only. If `INVITE_PASSPHRASE` is unset, it exits and does not serve the HTML. Session cookies on this HTTP preview omit the `Secure` flag; the Pages site is HTTPS and sets `Secure`.

## Out of scope

- Public GitHub Pages, and any retarget of the old `xiwang0706/openpinn-docs` repository
- Browser training, accounts, Basic-auth user lists, OAuth, and Cloudflare Access
- Changes to the OpenPINN Python API or to configurator behavior beyond the M6 copy/download paths
- Committing the passphrase, the API token, the account id, or Access client secrets
