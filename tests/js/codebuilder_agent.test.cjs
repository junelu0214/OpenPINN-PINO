const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");
const os = require("node:os");
const childProcess = require("node:child_process");

const root = path.resolve(__dirname, "../..");
const generic = require(path.join(root, "docs/_static/generic_validation.js"));
const builder = require(path.join(root, "docs/_static/codebuilder_agent.js"));
const diagnostic = require(path.join(root, "docs/_static/diagnostic_agent.js"));

function pair() {
  return {
    problem: {
      dim: 1, physics: "magneto_thermo_chemo_nonlinear_v9", kind: "dynamic",
      field: {kind: "temperature", components: ["T"]},
      parameters: {alpha: {value: 0.1, unit: "m^2/s"}},
      pde: {expression: "dt(T)-alpha*dx(dx(T))+T*T*T"},
      geometry: {type: "interval", x_min: 0, x_max: 1},
      boundaries: [{location: "x=0", type: "dirichlet", value: 0}],
      initial_condition: {expression: "sin(pi*x)"},
    },
    method: {
      method: "baseline", variant: "plain_pinn", path: "pinn",
      sampling: {name: "uniform", interior: 32, boundary: 8},
      loss: {formulation: "strong", weighting: "fixed"},
      optimizer: {name: "adam", lr: 0.001},
      network: {name: "mlp", hidden_layers: [8, 8], activation: "tanh"},
      differentiator: {mode: "autodiff"}, training: {steps: 10},
    },
  };
}

function hybrid(problem, method, semantic = {}) {
  const normalized = generic.normalizeProblem(problem);
  const deterministic = generic.validate(normalized, method);
  return {
    schema_version: 1, kind: "HybridValidationReport",
    pair_fingerprint: deterministic.pair_fingerprint,
    request_id: "semantic-test", request_version: "semantic-v1",
    snapshots: {problem: normalized, method}, deterministic,
    semantic: {state: "complete", report: {
      review_status: semantic.review_status || "no_concern_identified",
      concerns: semantic.concerns || [], unknowns: [], information_needed: [],
      method_fit: {status: "suitable", message: "No concern identified."},
    }},
    training_export_unlocked: false,
  };
}

test("valid arbitrary physics produces a deterministic non-runnable skeleton and BuildReport", () => {
  const {problem, method} = pair();
  const report = hybrid(problem, method);
  assert.equal(report.deterministic.adapter.status, "unsupported");
  const first = builder.build(report, problem, method);
  const second = builder.build(report, problem, method);
  assert.deepEqual(second, first);
  assert.equal(first.report.kind, "BuildReport");
  assert.equal(first.report.ok, true);
  assert.equal(first.report.pair_fingerprint, report.pair_fingerprint);
  assert.equal(first.report.backend_runnable, false);
  assert.equal(first.report.mapping_complete_for_later_training, false);
  assert(first.report.warnings.some(entry => entry.code === "build.mapping_incomplete"));
  assert.equal(first.report.artifact_kind, "generic_training_skeleton");
  assert.match(first.source, /BACKEND_RUNNABLE = False/);
  assert.match(first.source, /raise NotImplementedError/);
  assert.match(first.source, /def write_run_artifact/);
  assert.match(first.source, /openpinn_run.*run_artifact\.json/);
  assert.match(first.source, /"status": status/);
  assert.match(first.source, /artifact_path = write_run_artifact\(\)/);
  assert.match(first.source, /No training was started/);
  assert.doesNotMatch(first.source, /from_specs|validate_specs|linear_elasticity/);
  assert.doesNotMatch(JSON.stringify(first), /trainable.?true|backend_runnable.?true/i);
});

test("registered heat structure produces a runnable shared-engine script", () => {
  const problem = {
    physics: "heat_equation", dim: 1, kind: "dynamic", units: {space: "m", time: "s"},
    time_domain: {t_min: 0, t_max: 1, unit: "s"},
    field: {name: "u", kind: "scalar", components: ["u"], unit: "K"},
    parameters: [{name: "alpha", value: 0.01, unit: "m^2/s"}],
    geometry: {type: "interval", x_min: 0, x_max: 1},
    pde: {diffusivity: 0.01, expression: "dt(u)-alpha*dx(dx(u))=0"},
    boundaries: [{location: "x=0", type: "dirichlet", value: 0}, {location: "x=1", type: "neumann", value: 0}],
    initial_condition: {expression: "sin(pi*x/2)"},
  };
  const method = pair().method;
  const result = builder.build(hybrid(problem, method), problem, method);
  assert.equal(result.report.ok, true);
  assert.equal(result.report.mapping_complete_for_later_training, true);
  assert.equal(result.report.backend_runnable, true);
  assert.equal(result.report.generator_id, builder.RUNNABLE_GENERATOR_ID);
  assert.equal(result.report.execution_capability.backend_id, "heat_1d_unsteady");
  assert.equal(result.filename, "openpinn_train.py");
  assert.match(result.source, /BACKEND_RUNNABLE = True/);
  assert.match(result.source, /op\.run_registered_pinn/);
  assert.match(result.source, /--smoke/);
  assert.match(result.source, /--max-wall-seconds/);
  assert.doesNotMatch(result.source, /NotImplementedError/);
});

test("generic scalar strong-form capability emits the shared-engine script", () => {
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
  const selected = builder.build(hybrid(problem, pair().method), problem, pair().method);
  assert.equal(selected.report.backend_runnable, true);
  assert.equal(selected.report.execution_capability.backend_id, "scalar_strong_form_v1");
  assert.match(selected.source, /op\.run_registered_pinn/);
});

test("generated runnable script performs one real local smoke step and writes same-run artifact", () => {
  const problem = {
    physics: "poisson", dim: 1, units: {space: "m"},
    field: {name: "u", kind: "scalar", components: ["u"], unit: "1"},
    geometry: {type: "interval", x_min: 0, x_max: 1},
    pde: {expression: "-dx(dx(u))=pi*pi*sin(pi*x)"},
    boundaries: [{location: "x=0", type: "dirichlet", value: 0}, {location: "x=1", type: "dirichlet", value: 0}],
  };
  const method = pair().method;
  const result = builder.build(hybrid(problem, method), problem, method);
  assert.equal(result.report.backend_runnable, true);
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "openpinn-codebuilder-"));
  const script = path.join(temporary, result.filename);
  fs.writeFileSync(script, result.source);
  try {
    childProcess.execFileSync(path.join(root, ".venv/bin/python"), [script, "--smoke", "--max-wall-seconds", "30"], {
      cwd: temporary, env: {...process.env, PYTHONPATH: path.join(root, "src")}, stdio: "pipe",
    });
  } catch (failure) {
    // Managed sandboxes can report EPERM after a child exits successfully.
    if (failure.status !== 0) throw failure;
  }
  const artifact = JSON.parse(fs.readFileSync(path.join(temporary, "openpinn_run/run_artifact.json"), "utf8"));
  assert.equal(artifact.generator_id, builder.RUNNABLE_GENERATOR_ID);
  assert.equal(artifact.backend_id, "poisson_1d");
  assert.equal(artifact.artifact_origin, "local_generated_run");
  assert.equal(artifact.smoke_only, true);
  assert.equal(artifact.steps_completed, 1);
  assert.notEqual(artifact.status, "completed");
  assert.deepEqual(Object.keys(artifact.metrics.error_metrics).sort(), ["max_abs", "rel_l2"]);
  const rejected = diagnostic.evaluateArtifact(artifact, result.report.pair_fingerprint);
  assert.equal(rejected.valid, true);
  assert(rejected.findings.some(entry => entry.code === "artifact.smoke_only"));
});

test("incomplete mapping and drifted heat structure cannot emit a runnable script", () => {
  const incomplete = pair().problem;
  const incompleteResult = builder.build(hybrid(incomplete, pair().method), incomplete, pair().method);
  assert.equal(incompleteResult.report.execution_capability.status, "incomplete");
  assert.equal(incompleteResult.report.backend_runnable, false);
  assert.match(incompleteResult.source, /NotImplementedError/);
  const heat = {
    physics: "heat_equation", dim: 1, kind: "dynamic", units: {space: "m", time: "s"},
    time_domain: {t_min: 0, t_max: 1, unit: "s"}, field: {name: "u", components: ["u"], unit: "K"},
    parameters: [{name: "alpha", value: 0.01, unit: "m^2/s"}], geometry: {type: "interval", x_min: 0, x_max: 1},
    pde: {expression: "dt(u)-alpha*dx(dx(u))=0"},
    boundaries: [{location: "x=0", type: "dirichlet", value: 0}, {location: "x=1", type: "neumann", value: 0}],
    initial_condition: {expression: "sin(pi*x/2)"},
  };
  heat.boundaries[1].type = "periodic";
  const drifted = builder.build(hybrid(heat, pair().method), heat, pair().method);
  assert.equal(drifted.report.execution_capability.status, "unsupported");
  assert.equal(drifted.report.backend_runnable, false);
});

test("deterministic fail hard-blocks generation", () => {
  const {problem, method} = pair();
  problem.pde.expression = "dt(T)-missing_symbol";
  const result = builder.build(hybrid(problem, method), problem, method);
  assert.equal(result.report.ok, false);
  assert.equal(result.source, null);
  assert(result.report.errors.some(entry => entry.code === "build.deterministic_fail"));
});

test("an edited final pair makes the Hybrid fingerprint stale", () => {
  const {problem, method} = pair();
  const report = hybrid(problem, method);
  const edited = JSON.parse(JSON.stringify(problem));
  edited.parameters.alpha.value = 0.2;
  const result = builder.build(report, edited, method);
  assert.equal(result.report.ok, false);
  assert(result.report.errors.some(entry => entry.code === "build.stale_fingerprint"));
});

test("template-required fields block even when their deterministic layer is unknown", () => {
  const {problem, method} = pair();
  delete problem.field;
  const result = builder.build(hybrid(problem, method), problem, method);
  assert.equal(result.report.ok, false);
  assert(result.report.errors.some(entry => entry.code === "template.missing_fields"));
});

test("semantic incomplete hard-blocks without a rules fallback", () => {
  const {problem, method} = pair();
  const report = hybrid(problem, method);
  report.semantic = {state: "incomplete", code: "transport.timeout", message: "timed out"};
  const result = builder.build(report, problem, method);
  assert.equal(result.report.ok, false);
  assert.equal(result.source, null);
  assert(result.report.errors.some(entry => entry.code === "build.semantic_incomplete"));
});

test("semantic concerns allow generation but stay prominent in report and source", () => {
  const {problem, method} = pair();
  const concern = {path: "problem.equations[0]", message: "Confirm the cubic coupling sign.", severity: "medium"};
  const result = builder.build(hybrid(problem, method, {review_status: "concerns", concerns: [concern]}), problem, method);
  assert.equal(result.report.ok, true);
  assert.deepEqual(result.report.semantic_concerns_visible, [concern]);
  assert(result.report.warnings.some(entry => entry.code === "build.semantic_concerns"));
  assert.match(result.source, /Confirm the cubic coupling sign/);
});

test("missing method template fields and a missing Hybrid report are blocked", () => {
  const {problem, method} = pair();
  const report = hybrid(problem, method);
  delete report.snapshots.method.path;
  const noPath = builder.build(report);
  assert.equal(noPath.report.ok, false);
  assert(noPath.report.errors.some(entry => entry.code === "template.missing_path"));
  const missing = builder.build({}, problem, method);
  assert.equal(missing.report.ok, false);
  assert(missing.report.errors.some(entry => entry.code === "build.missing_hybrid_report"));
});

test("credentials and unrelated Hybrid metadata are never copied into outputs", () => {
  const {problem, method} = pair();
  const report = hybrid(problem, method);
  report.api_key = "super-secret-browser-key";
  report.semantic.transport = {authorization: "Bearer super-secret-browser-key"};
  const result = builder.build(report, problem, method);
  assert.equal(result.report.ok, true);
  assert.doesNotMatch(JSON.stringify(result), /super-secret-browser-key|authorization|api_key/i);
});
