"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const diagnostic = require(path.resolve(__dirname, "../../docs/_static/diagnostic_agent.js"));

function artifact(overrides = {}) {
  return {
    schema_version: 1, kind: "RunArtifact", pair_fingerprint: "fnv1a32:1234abcd",
    generator_id: "test-generator", backend_runnable: true, status: "completed",
    recorded_at: "2026-11-10T00:00:00Z",
    metrics: {loss_history: [1, 0.2, 0.05], final_loss: 0.05, error_metrics: {relative_l2: 0.03}},
    failure: {type: null, message: null, stack: null, truncated: false},
    ...overrides,
  };
}
function semanticPayload(assessment = "no_additional_concern") {
  return {assessment, findings: assessment === "concerns" ? [{code: "diag.loss", severity: "warning", path: "artifact.metrics", message: "Review the loss plateau."}] : [],
    repair_hints: [{message: "Inspect sampling coverage.", rationale: "This is a non-binding next check."}],
    human_summary: "The recorded metrics were reviewed; this is diagnostic guidance only."};
}
function response(content, status = 200) {
  return {ok: status >= 200 && status < 300, status,
    json: async () => ({choices: [{message: {content: typeof content === "string" ? content : JSON.stringify(content)}}]})};
}
function options(content = semanticPayload()) {
  return {baseUrl: "https://diagnostic.example/v1", model: "diagnostic-model", apiKey: "secret-diagnostic-key",
    fetch: async (_url, request) => {
      assert.equal(request.headers.Authorization, "Bearer secret-diagnostic-key");
      assert.equal(request.body.includes("secret-diagnostic-key"), false);
      return response(content);
    }};
}

test("completed run produces deterministic metrics and a dual-track DiagnosticReport", async () => {
  const input = artifact(), before = JSON.stringify(input);
  const report = await diagnostic.diagnose(input, input.pair_fingerprint, options());
  assert.equal(report.kind, "DiagnosticReport");
  assert.equal(report.deterministic.metrics_summary.loss_samples, 3);
  assert.equal(report.deterministic.metrics_summary.final_loss, 0.05);
  assert.equal(report.llm_track.state, "complete");
  assert.equal(report.healthy, false);
  assert.equal(report.deterministic.outcome, "completed_unverified");
  assert(report.deterministic.findings.some(entry => entry.code === "run.accuracy_unverified"));
  assert.equal(report.repair_performed, false);
  assert.equal(report.llm_track.report.repair_hints[0].binding, false);
  assert.equal(JSON.stringify(input), before, "diagnosis must not mutate the uploaded artifact");
  assert.equal(JSON.stringify(report).includes("secret-diagnostic-key"), false);
});

test("human-readable report contains charts, findings, method suggestions and standalone download HTML", async()=>{
  const input=artifact({status:"not_converged",metrics:{loss_history:[1,0.8,0.7,0.69],final_loss:0.69,error_metrics:{relative_l2:0.12}}});
  const report=await diagnostic.diagnose(input,input.pair_fingerprint,options(semanticPayload("concerns")));
  const body=diagnostic.reportBodyHtml(report,input),html=diagnostic.reportHtml(report,input);
  assert.match(body,/diagnostic-loss-chart/);
  assert.match(body,/RAR \/ RAD/);
  assert.match(body,/Adam → L-BFGS/);
  assert.match(body,/下一轮建议/);
  assert.match(html,/<!doctype html>/);
  assert.match(html,/原始 DiagnosticReport JSON/);
  assert.doesNotMatch(html,/<script/i);
});

test("uploaded secret-like fields and Bearer strings are redacted without mutating input", async () => {
  const input = artifact({metadata: {api_key: "artifact-secret"}, failure: {
    type: null, message: "Bearer artifact-token", stack: null, truncated: false,
  }}), before = JSON.stringify(input);
  const config = options();
  config.fetch = async (_url, request) => {
    assert.equal(request.body.includes("artifact-secret"), false);
    assert.equal(request.body.includes("artifact-token"), false);
    return response(semanticPayload());
  };
  const report = await diagnostic.diagnose(input, input.pair_fingerprint, config);
  assert.equal(JSON.stringify(report).includes("artifact-secret"), false);
  assert.equal(JSON.stringify(report).includes("artifact-token"), false);
  assert.equal(JSON.stringify(input), before);
});

test("stale fingerprint is a warning and does not block semantic diagnosis", async () => {
  let called = false;
  const result = await diagnostic.diagnose(artifact(), "fnv1a32:changed", {...options(), fetch: async () => { called = true; return response(semanticPayload()); }});
  assert.equal(result.healthy, false);
  assert.equal(result.deterministic.outcome, "completed_unverified");
  assert(result.deterministic.findings.some(entry => entry.code === "artifact.stale_fingerprint"));
  assert.equal(result.llm_track.state, "complete");
  assert.equal(called, true);
});

test("completed run without loss is invalid and cannot be healthy", () => {
  const result = diagnostic.evaluateArtifact(artifact({metrics: {loss_history: [], final_loss: null, error_metrics: {}}}), "fnv1a32:1234abcd");
  assert.equal(result.valid, false);
  assert.equal(result.healthy, false);
  assert(result.findings.some(entry => entry.code === "run.completed_loss_missing"));
});

test("failed run retains failure and truncated-stack findings", async () => {
  const failed = artifact({status: "failed", metrics: {loss_history: [1, 2], final_loss: 2, error_metrics: {}},
    failure: {type: "RuntimeError", message: "optimizer diverged", stack: "trace…", truncated: true}});
  const report = await diagnostic.diagnose(failed, failed.pair_fingerprint, options(semanticPayload("concerns")));
  assert.equal(report.deterministic.outcome, "failed");
  assert(report.deterministic.findings.some(entry => entry.code === "run.failed"));
  assert(report.deterministic.findings.some(entry => entry.code === "run.stack_truncated"));
  assert.equal(report.llm_track.report.assessment, "concerns");
  assert.equal(report.healthy, false);
});

test("not-runnable artifact is valid but never receives a healthy claim", async () => {
  const stopped = artifact({backend_runnable: false, status: "not_runnable",
    metrics: {loss_history: [], final_loss: null, error_metrics: {}}});
  const report = await diagnostic.diagnose(stopped, stopped.pair_fingerprint, options());
  assert.equal(report.deterministic.valid, true);
  assert.equal(report.deterministic.outcome, "not_runnable");
  assert.equal(report.healthy, false);
  assert.match(report.human_summary, /guidance only/);
});

test("bounded non-success terminal statuses remain diagnosable and never healthy", () => {
  for (const status of ["budget_exhausted", "not_converged", "interrupted"]) {
    const result = diagnostic.evaluateArtifact(artifact({status}), "fnv1a32:1234abcd");
    assert.equal(result.valid, true, status);
    assert.equal(result.outcome, status);
    assert.equal(result.healthy, false);
  }
});

test("smoke, fixture, and example artifacts receive a limited warning", () => {
  for (const override of [{smoke_only: true}, {fixture: true}, {example: true}, {artifact_origin: "fixture"}]) {
    const result = diagnostic.evaluateArtifact(artifact(override), "fnv1a32:1234abcd");
    assert.equal(result.valid, true);
    assert(result.findings.some(entry => entry.code === "artifact.non_product_run" || entry.code === "artifact.smoke_only"));
  }
});

test("runnable generator requires same-run provenance and accepts the full contract", () => {
  const base = artifact({
    generator_id: "openpinn.shared-pinn-engine.v1", artifact_origin: "local_generated_run",
    example: false, fixture: false, smoke_only: false, device: "cpu",
    budgets: {max_steps: 10, max_wall_seconds: 30}, steps_completed: 10,
    engine: {status: "budget_exhausted"}, status: "budget_exhausted",
  });
  assert.equal(diagnostic.evaluateArtifact(base, base.pair_fingerprint).valid, true);
  const missing = {...base}; delete missing.engine;
  const rejected = diagnostic.evaluateArtifact(missing, missing.pair_fingerprint);
  assert.equal(rejected.valid, false);
  assert(rejected.findings.some(entry => entry.code === "artifact.engine"));
  for (const [field, code] of [["artifact_origin", "artifact.origin"], ["device", "artifact.device"],
    ["budgets", "artifact.budgets"], ["steps_completed", "artifact.steps"]]) {
    const altered = {...base}; delete altered[field];
    assert(diagnostic.evaluateArtifact(altered, altered.pair_fingerprint).findings.some(entry => entry.code === code));
  }
});

test("auth, malformed content, semantic shape, and timeout remain incomplete", async () => {
  const input = artifact();
  const cases = [
    [{...options(), fetch: async () => response({}, 401)}, "transport.auth"],
    [options("not-json"), "schema.content"],
    [options({assessment: "concerns", findings: []}), "schema.semantic"],
    [options({...semanticPayload(), human_summary: "This proves correctness and will succeed."}), "schema.semantic"],
  ];
  for (const [config, code] of cases) {
    const report = await diagnostic.diagnose(input, input.pair_fingerprint, config);
    assert.equal(report.llm_track.state, "incomplete");
    assert.equal(report.llm_track.code, code);
    assert.equal(report.healthy, false);
  }
  const timed = await diagnostic.diagnose(input, input.pair_fingerprint, {...options(), timeoutMs: 5,
    fetch: async (_url, request) => new Promise((_resolve, reject) => request.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")), {once: true}))});
  assert.equal(timed.llm_track.code, "transport.timeout");
});

function storedZip(name, text) {
  const filename = Buffer.from(name), data = Buffer.from(text);
  const local = Buffer.alloc(30 + filename.length);
  local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(0, 8);
  local.writeUInt32LE(data.length, 18); local.writeUInt32LE(data.length, 22); local.writeUInt16LE(filename.length, 26); filename.copy(local, 30);
  const central = Buffer.alloc(46 + filename.length);
  central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6); central.writeUInt16LE(0, 10);
  central.writeUInt32LE(data.length, 20); central.writeUInt32LE(data.length, 24); central.writeUInt16LE(filename.length, 28); filename.copy(central, 46);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0); eocd.writeUInt16LE(1, 8); eocd.writeUInt16LE(1, 10);
  eocd.writeUInt32LE(central.length, 12); eocd.writeUInt32LE(local.length + data.length, 16);
  return Buffer.concat([local, data, central, eocd]);
}

test("JSON and ZIP upload readers return the contained RunArtifact", async () => {
  const input = artifact(), json = JSON.stringify(input);
  const jsonRead = await diagnostic.readArtifactFile({name: "run_artifact.json", arrayBuffer: async () => Buffer.from(json)});
  assert.deepEqual(jsonRead, input);
  const zip = storedZip("openpinn_run/run_artifact.json", json);
  const zipRead = await diagnostic.readArtifactFile({name: "openpinn_run.zip", arrayBuffer: async () => zip});
  assert.deepEqual(zipRead, input);
  await assert.rejects(() => diagnostic.readArtifactFile({name: "notes.txt", arrayBuffer: async () => Buffer.from("x")}), /Only .json and .zip/);
  await assert.rejects(() => diagnostic.readArtifactFile({name: "huge.json", size: 11 * 1024 * 1024, arrayBuffer: async () => Buffer.alloc(0)}), /10 MB safety limit/);
});
