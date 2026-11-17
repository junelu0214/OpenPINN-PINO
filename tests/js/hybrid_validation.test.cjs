"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const generic = require(path.resolve(__dirname, "../../docs/_static/generic_validation.js"));
const hybrid = require(path.resolve(__dirname, "../../docs/_static/hybrid_validation.js"));

const problem = {
  physics: "novel_reaction_transport", dim: 1,
  field: {kind: "scalar", components: ["u"]},
  geometry: {type: "interval", x_min: 0, x_max: 1},
  parameters: {alpha: {value: 0.1, unit: "m^2/s"}},
  pde: {expression: "alpha*dx(dx(u))"},
  boundaries: [{region: "left", type: "dirichlet", value: 0}],
};
const method = {path: "new_backend", method: "research_solver", training: {steps: 10}};

function payload(status = "no_concern_identified") {
  return {
    review_status: status,
    concerns: status === "concerns" ? [{
      code: "semantic.bc_coverage", path: "problem.boundaries",
      message: "Only one boundary is declared.",
      evidence: [{path: "problem.boundaries[0].region", quote_or_excerpt: "left", reasoning_nature: "input-boundary coverage cue"}],
    }] : [],
    unknowns: status === "insufficient_information" ? [{code: "semantic.domain", path: "problem.geometry", message: "More geometry context is needed."}] : [],
    information_needed: status === "insufficient_information" ? ["Right boundary condition"] : [],
    assumptions_noted: [{path: "problem.physics", message: "The free-form physics label is treated as user terminology."}],
    method_fit: {status: "unsupported", message: "The method is unregistered; physical validity is not inferred from that."},
  };
}
function response(content, status = 200) {
  return {ok: status >= 200 && status < 300, status, json: async () => ({choices: [{message: {content: typeof content === "string" ? content : JSON.stringify(content)}}]})};
}
function options(content) {
  return {baseUrl: "https://validation.example/v1", model: "semantic-model", apiKey: "secret-test-key",
    fetch: async (_url, request) => {
      assert.equal(request.headers.Authorization, "Bearer secret-test-key");
      assert.equal(request.body.includes("secret-test-key"), false);
      return response(content);
    }};
}

test("bounded semantic statuses bind evidence to input paths and never claim training", async () => {
  for (const status of hybrid.REVIEW_STATUSES) {
    const session = hybrid.createSession();
    const report = await session.run(problem, method, options(payload(status)));
    assert.equal(report.semantic.state, "complete");
    assert.equal(report.semantic.report.review_status, status);
    assert.equal(report.semantic.report.pair_fingerprint, report.deterministic.pair_fingerprint);
    assert.equal(report.training_export_unlocked, false);
    assert.equal(report.deterministic.layers.find(layer => layer.id === "L5").status, "unsupported");
    assert.equal(report.deterministic.summary.description_ready_for_review, true);
    assert.equal(JSON.stringify(report).includes("secret-test-key"), false);
  }
});

test("real-model field aliases normalize without weakening input evidence", () => {
  const deterministic = generic.validate(problem, method);
  const normalized = hybrid.validateSemanticPayload({
    review_status: "concerns",
    concerns: [{code: "steps", path: "method.training.steps", message: "Short run.",
      evidence: [{path: "method.training.steps", quote_or_excerpt: "10 steps", reasoning_nature: "method budget cue"}]}],
    unknowns: [{path: "deterministic_summary.executable_adapter", description: "Executable support is unknown."}],
    information_needed: [{description: "Additional calibration evidence"}],
    assumptions_noted: [{path: "problem.physics", assumption: "User-defined label."}],
    method_fit: {status: "concerns", rationale: "The backend is unregistered."},
  }, {problem: generic.normalizeProblem(problem), method, deterministic,
    pairFingerprint: deterministic.pair_fingerprint, requestId: "aliases", model: "m"});
  assert.equal(normalized.concerns[0].evidence[0].quote_or_excerpt, "10");
  assert.equal(normalized.unknowns[0].code, "semantic.unknown");
  assert.equal(normalized.information_needed[0], "Additional calibration evidence");
  assert.equal(normalized.method_fit.message, "The backend is unregistered.");
});

test("semantic paths accept bounded JSON Pointer syntax for objects and arrays", () => {
  const deterministic = generic.validate(problem, method);
  const normalizedProblem = generic.normalizeProblem(problem);
  const normalized = hybrid.validateSemanticPayload({
    review_status: "concerns",
    concerns: [{code: "equation", path: "/problem/equations/0", message: "Review the residual.", evidence: [{
      path: "/problem/equations/0/expression", quote_or_excerpt: "alpha*dx(dx(u))", reasoning_nature: "declared residual",
    }]}],
    unknowns: [{code: "adapter", path: "/deterministic_summary/executable_adapter", message: "Adapter support is separate."}],
    information_needed: [],
    assumptions_noted: [{path: "/problem/equations/0", message: "The declared residual is interpreted literally."}],
    method_fit: {status: "unsupported", message: "The research method is unregistered."},
  }, {problem: normalizedProblem, method, deterministic,
    pairFingerprint: deterministic.pair_fingerprint, requestId: "json-pointer", model: "m"});
  assert.equal(normalized.assumptions_noted[0].path, "/problem/equations/0");
  assert.equal(normalized.concerns[0].evidence[0].quote_or_excerpt, "alpha*dx(dx(u))");

  assert.throws(() => hybrid.validateSemanticPayload({
    ...payload(), assumptions_noted: [{path: "/problem/equations/99", message: "Missing."}],
  }, {problem: normalizedProblem, method, deterministic,
    pairFingerprint: deterministic.pair_fingerprint, requestId: "bad-pointer", model: "m"}), /path does not exist/);
});

test("missing key, HTTP auth, malformed JSON, and bad evidence remain incomplete with no fallback", async () => {
  const cases = [
    [{baseUrl: "https://validation.example/v1", model: "m", apiKey: "", fetch: async () => response(payload())}, "configuration.missing_key"],
    [{baseUrl: "https://validation.example/v1", model: "m", apiKey: "redact-me", fetch: async () => response({}, 401)}, "transport.auth"],
    [options("not json"), "schema.content"],
    [options({...payload("concerns"), concerns: [{code: "x", path: "problem.physics", message: "x", evidence: [{path: "problem.physics", quote_or_excerpt: "fabricated", reasoning_nature: "x"}]}]}), "schema.semantic"],
  ];
  for (const [config, code] of cases) {
    const report = await hybrid.createSession().run(problem, method, config);
    assert.equal(report.semantic.state, "incomplete");
    assert.equal(report.semantic.code, code);
    assert.equal(report.training_export_unlocked, false);
    assert.equal(JSON.stringify(report).includes(config.apiKey || "never"), false);
  }
});

test("timeout and explicit cancellation are incomplete", async () => {
  function abortingFetch(_url, request) {
    return new Promise((_resolve, reject) => request.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")), {once: true}));
  }
  const deterministic = generic.validate(problem, method);
  const timed = await hybrid.semanticRequest(generic.normalizeProblem(problem), method, deterministic,
    {baseUrl: "https://validation.example/v1", model: "m", apiKey: "k", timeoutMs: 5, fetch: abortingFetch}, "timeout-1");
  assert.equal(timed.code, "transport.timeout");
  const controller = new AbortController();
  const pending = hybrid.semanticRequest(generic.normalizeProblem(problem), method, deterministic,
    {baseUrl: "https://validation.example/v1", model: "m", apiKey: "k", fetch: abortingFetch}, "cancel-1", controller.signal);
  controller.abort();
  assert.equal((await pending).code, "transport.cancelled");
});

test("out-of-order responses are ignored and edits invalidate export", async () => {
  const resolvers = [];
  const config = {baseUrl: "https://validation.example/v1", model: "m", apiKey: "k",
    fetch: async () => new Promise(resolve => resolvers.push(resolve))};
  const session = hybrid.createSession();
  const first = session.run(problem, method, {...config, requestId: "first"});
  const second = session.run(problem, method, {...config, requestId: "second"});
  resolvers[1](response(payload()));
  const current = await second;
  resolvers[0](response(payload("concerns")));
  assert.deepEqual(await first, {state: "stale", ignored: true, request_id: "first"});
  assert.equal(current.request_id, "second");
  assert.equal(session.isCurrent(problem, method), true);
  assert.equal(session.canExportTraining(), false);
  const artifact = session.exportArtifacts(problem, method);
  assert.match(artifact.label, /NON-TRAINING/);
  assert.equal(JSON.stringify(artifact).includes("apiKey"), false);
  assert.equal(session.isCurrent(problem, {...method, training: {steps: 11}}), false);
  assert.throws(() => session.exportArtifacts(problem, {...method, training: {steps: 11}}), /stale/);
});
