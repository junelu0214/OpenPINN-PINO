"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const childProcess = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const root = path.resolve(__dirname, "../..");
const generic = require(path.join(root, "docs/_static/generic_validation.js"));
const hybridValidation = require(path.join(root, "docs/_static/hybrid_validation.js"));
const builder = require(path.join(root, "docs/_static/codebuilder_agent.js"));
const diagnostic = require(path.join(root, "docs/_static/diagnostic_agent.js"));

const method = {
  method: "baseline", variant: "plain_pinn", path: "pinn",
  sampling: {name: "uniform"}, loss: {formulation: "strong", weighting: "fixed"},
  optimizer: {name: "adam"}, network: {name: "mlp"},
  differentiator: {mode: "autodiff"}, training: {steps: 20},
};

const cards = {
  poisson_1d: {
    physics: "poisson", dim: 1, units: {space: "m"},
    field: {name: "u", kind: "scalar", components: ["u"], unit: "1"},
    geometry: {type: "interval", x_min: 0, x_max: 1},
    pde: {expression: "-dx(dx(u))=pi*pi*sin(pi*x)"},
    boundaries: [
      {location: "x=0", type: "dirichlet", value: 0},
      {location: "x=1", type: "dirichlet", value: 0},
    ],
  },
  heat_1d_unsteady: {
    physics: "heat_equation", dim: 1, kind: "dynamic", units: {space: "m", time: "s"},
    time_domain: {t_min: 0, t_max: 1, unit: "s"},
    field: {name: "u", kind: "scalar", components: ["u"], unit: "K"},
    parameters: [{name: "alpha", value: 0.01, unit: "m^2/s"}],
    geometry: {type: "interval", x_min: 0, x_max: 1},
    pde: {expression: "dt(u)-alpha*dx(dx(u))=0"},
    boundaries: [
      {location: "x=0", type: "dirichlet", value: 0},
      {location: "x=1", type: "neumann", value: 0},
    ],
    initial_condition: {field: "u", expression: "sin(pi*x/2)"},
  },
};

function hybrid(problem) {
  const normalized = generic.normalizeProblem(problem);
  const deterministic = generic.validate(normalized, method);
  return {schema_version: 1, kind: "HybridValidationReport",
    pair_fingerprint: deterministic.pair_fingerprint, request_id: "stage-g-fixture",
    request_version: "semantic-v1", snapshots: {problem: normalized, method}, deterministic,
    semantic: {state: "complete", report: {review_status: "no_concern_identified", concerns: [],
      unknowns: [], information_needed: [], method_fit: {status: "suitable", message: "Fixture response for deterministic e2e only."}}},
    training_export_unlocked: false};
}

function run(source, directory) {
  const script = path.join(directory, "openpinn_train.py");
  fs.writeFileSync(script, source);
  try {
    childProcess.execFileSync(path.join(root, ".venv/bin/python"), [script, "--max-steps", "2",
      "--max-wall-seconds", "30", "--device", "cpu", "--output-dir", "openpinn_run"], {
      cwd: directory, env: {...process.env, PYTHONPATH: path.join(root, "src")}, stdio: "pipe",
    });
  } catch (failure) {
    if (failure.status !== 0) throw failure;
  }
}

function storedZip(name, text) {
  const filename = Buffer.from(name), data = Buffer.from(text);
  const local = Buffer.alloc(30 + filename.length);
  local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(0, 8);
  local.writeUInt32LE(data.length, 18); local.writeUInt32LE(data.length, 22); local.writeUInt16LE(filename.length, 26); filename.copy(local, 30);
  const central = Buffer.alloc(46 + filename.length);
  central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6); central.writeUInt16LE(0, 10);
  central.writeUInt32LE(data.length, 20); central.writeUInt32LE(data.length, 24); central.writeUInt16LE(filename.length, 28); filename.copy(central, 46);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(1, 8); end.writeUInt16LE(1, 10);
  end.writeUInt32LE(central.length, 12); end.writeUInt32LE(local.length + data.length, 16);
  return Buffer.concat([local, data, central, end]);
}

for (const [backendId, problem] of Object.entries(cards)) {
  test(`CodeBuilder to local non-smoke run to JSON and ZIP diagnosis: ${backendId}`, async () => {
    const envelope = hybrid(problem);
    const built = builder.build(envelope, problem, method);
    assert.equal(built.filename, "openpinn_train.py");
    assert.equal(built.report.backend_runnable, true);
    assert.equal(built.report.generator_id, builder.RUNNABLE_GENERATOR_ID);
    assert.match(built.source, /op\.run_registered_pinn/);
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), `openpinn-stage-g-${backendId}-`));
    run(built.source, directory);
    const artifactPath = path.join(directory, "openpinn_run/run_artifact.json");
    assert.equal(fs.existsSync(artifactPath), true);
    const text = fs.readFileSync(artifactPath, "utf8"), artifact = JSON.parse(text);
    assert.equal(artifact.backend_id, backendId);
    assert.equal(artifact.pair_fingerprint, built.report.pair_fingerprint);
    assert.equal(artifact.status, "budget_exhausted");
    assert.equal(artifact.smoke_only, false);
    assert.equal(artifact.accuracy_verified, false);
    assert.equal(artifact.steps_completed, 2);
    assert.deepEqual(Object.keys(artifact.metrics.error_metrics).sort(), ["max_abs", "rel_l2"]);
    Object.values(artifact.metrics.error_metrics).forEach(value => assert.equal(Number.isFinite(value), true));
    const report = diagnostic.evaluateArtifact(artifact, built.report.pair_fingerprint);
    assert.equal(report.valid, true); assert.equal(report.outcome, "budget_exhausted"); assert.equal(report.healthy, false);
    assert(report.findings.some(entry => entry.code === "run.budget_exhausted" && /not an accuracy or convergence claim/.test(entry.message)));
    const jsonRead = await diagnostic.readArtifactFile({name: "run_artifact.json", arrayBuffer: async () => Buffer.from(text)});
    assert.deepEqual(jsonRead, artifact);
    const zip = storedZip("openpinn_run/run_artifact.json", text);
    const zipRead = await diagnostic.readArtifactFile({name: "openpinn_run.zip", arrayBuffer: async () => zip});
    assert.deepEqual(zipRead, artifact);
    assert.equal(diagnostic.evaluateArtifact(zipRead, built.report.pair_fingerprint).valid, true);
  });
}

test("generic scalar transport uses the same generated local-training path", () => {
  const problem = {
    physics: "viscous_scalar_transport", dim: 1, kind: "dynamic", units: {space: "m", time: "s"},
    time_domain: {t_min: 0, t_max: 1, unit: "s"},
    field: {name: "v", kind: "scalar", components: ["v"], unit: "1"},
    parameters: [{name: "nu", value: 0.01, unit: "m^2/s"}],
    geometry: {type: "interval", x_min: 0, x_max: 1},
    pde: {expression: "dt(v)+v*dx(v)-nu*dx(dx(v))=0"},
    boundaries: [{location: "x=0", type: "dirichlet", value: 0}, {location: "x=1", type: "dirichlet", value: 0}],
    initial_condition: {field: "v", expression: "sin(pi*x)"},
  };
  const envelope = hybrid(problem), built = builder.build(envelope, problem, method);
  assert.equal(built.report.execution_capability.backend_id, "scalar_strong_form_v1");
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "openpinn-generic-scalar-"));
  run(built.source, directory);
  const artifact = JSON.parse(fs.readFileSync(path.join(directory, "openpinn_run/run_artifact.json"), "utf8"));
  assert.equal(artifact.backend_id, "scalar_strong_form_v1");
  assert.equal(artifact.status, "budget_exhausted");
  assert.deepEqual(artifact.metrics.error_metrics, {});
  assert.equal(diagnostic.evaluateArtifact(artifact, built.report.pair_fingerprint).valid, true);
});

test("symbolic Robin target does not leak a nonnumeric value placeholder into runtime parameters", () => {
  const problem = {
    physics: "scalar_elliptic", dim: 1, units: {space: "1"},
    field: {name: "w", components: ["w"], unit: "1"},
    parameters: [
      {name: "alpha", value: 1, unit: "1"}, {name: "beta", value: 1, unit: "1"},
      {name: "value", value: "-pi", unit: "1"},
    ],
    geometry: {type: "interval", x_min: 0, x_max: 1},
    pde: {expression: "-dx(dx(w))-pi*pi*sin(pi*x)=0"},
    boundaries: [
      {location: "x=0", type: "dirichlet", value: 0},
      {location: "x=1", type: "robin", alpha: 1, beta: 1,
        expression: "alpha*w+beta*dx(w)=value", value_expression: "-pi"},
    ],
  };
  const envelope = hybrid(problem), built = builder.build(envelope, problem, method);
  assert.equal(built.report.execution_capability.backend_id, "scalar_strong_form_v1");
  assert.equal(built.report.backend_runnable, true);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "openpinn-generic-robin-expression-"));
  run(built.source, directory);
  const artifact = JSON.parse(fs.readFileSync(path.join(directory, "openpinn_run/run_artifact.json"), "utf8"));
  assert.equal(artifact.status, "budget_exhausted");
  assert.equal(artifact.backend_id, "scalar_strong_form_v1");
  assert.equal(diagnostic.evaluateArtifact(artifact, built.report.pair_fingerprint).valid, true);
});

test("2D coordinate boundary expressions and multiple residuals complete generated local training", () => {
  const problem = {
    physics: "scalar_first_order", dim: 2, units: {space: "1"},
    field: {name: "q", components: ["q"], unit: "1"},
    geometry: {type: "rectangle", x_min: 0, x_max: 1, y_min: 0, y_max: 1},
    pde: {residuals: [{expression: "dx(q)-1=0"}, {expression: "dy(q)-1=0"}]},
    boundaries: [
      {location: "x=0", type: "dirichlet", value_expression: "y"},
      {location: "x=1", type: "robin", alpha: 1, beta: 1, value_expression: "y+2"},
      {location: "y=0", type: "dirichlet", value_expression: "x"},
      {location: "y=1", type: "neumann", value: 1},
    ],
  };
  const envelope = hybrid(problem), built = builder.build(envelope, problem, method);
  assert.equal(built.report.execution_capability.backend_id, "scalar_strong_form_v1");
  assert.equal(built.report.backend_runnable, true);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "openpinn-generic-2d-expression-"));
  run(built.source, directory);
  const artifact = JSON.parse(fs.readFileSync(path.join(directory, "openpinn_run/run_artifact.json"), "utf8"));
  assert.equal(artifact.status, "budget_exhausted");
  assert.equal(artifact.backend_id, "scalar_strong_form_v1");
  assert.equal(diagnostic.evaluateArtifact(artifact, built.report.pair_fingerprint).valid, true);
});

test("Case 1 JSON Pointer semantic note completes Hybrid, codegen, local run, and diagnosis", async () => {
  const problem = {
    physics: "scalar_elliptic", dim: 1, units: {space: "1"},
    field: {name: "w", components: ["w"], unit: "1"},
    parameters: [{name: "alpha", value: 1, unit: "1"}, {name: "beta", value: 1, unit: "1"}],
    geometry: {type: "interval", x_min: 0, x_max: 1},
    pde: {expression: "-dx(dx(w))-pi*pi*sin(pi*x)=0", source: "pi*pi*sin(pi*x)"},
    boundaries: [
      {location: "x=0", type: "dirichlet", value: 0},
      {location: "x=1", type: "robin", alpha: 1, beta: 1, value_expression: "-pi"},
    ],
  };
  const semantic = {
    review_status: "no_concern_identified", concerns: [], unknowns: [], information_needed: [],
    assumptions_noted: [{path: "/problem/equations/0", message: "The declared residual is interpreted literally."}],
    method_fit: {status: "suitable", message: "The baseline strong-form PINN matches the declared problem."},
  };
  const envelope = await hybridValidation.createSession().run(problem, method, {
    baseUrl: "https://validation.example/v1", model: "semantic-model", apiKey: "fixture-key",
    fetch: async () => ({ok: true, status: 200, json: async () => ({choices: [{message: {content: JSON.stringify(semantic)}}]})}),
  });
  assert.equal(envelope.semantic.state, "complete");
  assert.equal(envelope.deterministic.execution_capability.backend_id, "scalar_strong_form_v1");
  const built = builder.build(envelope, problem, method);
  assert.equal(built.report.backend_runnable, true);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "openpinn-case1-pointer-semantic-"));
  run(built.source, directory);
  const artifact = JSON.parse(fs.readFileSync(path.join(directory, "openpinn_run/run_artifact.json"), "utf8"));
  assert.equal(artifact.status, "budget_exhausted");
  assert.equal(artifact.backend_id, "scalar_strong_form_v1");
  assert.equal(artifact.pair_fingerprint, built.report.pair_fingerprint);
  assert.equal(diagnostic.evaluateArtifact(artifact, built.report.pair_fingerprint).valid, true);
});

