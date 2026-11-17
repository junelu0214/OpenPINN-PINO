"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {JSDOM} = require("jsdom");

const root = path.resolve(__dirname, "../..");

async function main() {
  const html = fs.readFileSync(path.join(root, "docs/_build/html/method_configurator.html"), "utf8");
  const dom = new JSDOM(html, {runScripts: "outside-only", url: "https://openpinn.local/method_configurator.html"});
  const {window} = dom;
  const {document} = window;
  await new Promise(resolve => window.addEventListener("DOMContentLoaded", resolve, {once: true}));
  const errors = [];
  window.addEventListener("error", event => errors.push(event.message));
  window.OpenPINNUnderstandingTrial = {
    understandWithCloud: async () => ({draft: {dim: 2, physics: "linear_elasticity"},
      missing: ["geometry.outer"], questions: ["请补充矩形范围。"], warnings: [], ready_for_validation: false}),
    planWithCloud: () => Promise.reject(new Error("not used in this deterministic test")),
    assessFamily: () => ({missing: [], ready: true}),
    createConfiguratorHandoff: value => value,
  };
  for (const file of [
    "joint_handoff.js", "polygon_image_trace.js", "dxf_geometry_import.js", "physics_editor.js", "multiphysics_editor.js",
    "configurator_paths.js", "diagnostic_agent.js", "agent_pipeline.js", "method_configurator.js",
  ]) window.eval(fs.readFileSync(path.join(root, "docs/_static", file), "utf8"));
  document.dispatchEvent(new window.Event("DOMContentLoaded"));

  const agent = name => document.querySelector('[data-agent-role="' + name + '"]');
  const role = name => document.querySelector('[data-role="' + name + '"]');
  agent("understand").click();
  await Promise.resolve();
  await Promise.resolve();
  assert.match(agent("understanding-summary").textContent, /geometry\.outer/);
  assert.match(agent("understanding-summary").textContent, /请补充矩形范围/);
  const problem = {
    dim: 2,
    physics: "linear_elasticity",
    geometry: {outer: {type: "rectangle", x_min: 0, x_max: 3, y_min: -1, y_max: 1},
      holes: [{type: "circle", center: [1.25, 0], radius: 0.2}]},
    field: {kind: "displacement", components: ["ux", "uy"]},
    constitutive: {model: "isotropic_linear", assumption: "plane_stress", E: 70000000000, nu: 0.29, stress_voigt: true},
    boundaries: [{id: "traction_right", region: "right", type: "neumann", dofs: ["ux"], value: 250000}],
    units: {length: "m", force: "N", stress: "Pa"},
  };
  const method = {
    method: "baseline", variant: "plain_pinn", path: "pinn",
    sampling: {name: "uniform", interior: 48, boundary: 20},
    loss: {formulation: "strong", weights: {pde: 1, boundary: 10}},
    optimizer: {name: "adam", lr: 0.0005},
    network: {name: "mlp", hidden_layers: [24, 24], activation: "tanh"},
    differentiator: {mode: "autodiff"}, training: {steps: 3},
  };
  agent("understanding-json").value = JSON.stringify(problem);
  agent("planning-json").value = JSON.stringify(method);
  agent("build-joint").click();

  assert.equal(role("solution-path"), null);
  assert.equal(role("problem").value, "custom_equation_template");
  assert.equal(role("plate-panel").hidden, true);
  assert.equal(role("multiphysics-class").value, "solid_mechanics");
  assert.equal(role("multiphysics-profile").value, "linear_elasticity");
  assert.equal(role("plate-x-max").value, "3");
  assert.equal(role("plate-hole-x").value, "1.25");
  assert.equal(role("plate-hole-radius").value, "0.2");
  assert.equal(role("plate-young").value, "70000000000");
  assert.equal(role("plate-nu").value, "0.29");
  assert.equal(role("plate-traction").value, "250000");
  assert.equal(role("plate-steps").value, "3");
  assert.equal(role("plate-hidden").value, "24, 24");
  assert.equal(role("copy").disabled, false);
  assert.equal(role("download-code").disabled, false);
  assert.match(role("code").textContent, /from_specs\(/);
  assert.match(role("code").textContent, /['"]assumption['"]:\s*['"]plane_stress['"]/);
  assert.doesNotMatch(role("code").textContent, /Understanding 问题草稿仍缺少必要字段/);
  assert.match(role("code").textContent, /op\.solve\(/);
  assert.match(role("code").textContent, /write_diagnostic_run_artifact\(/);
  assert.match(agent("joint-json").value, /JointAgentHandoff/);

  const artifact = {
    schema_version: 1, kind: "RunArtifact", status: "not_converged",
    backend_runnable: true, generator_id: "test.generator", artifact_origin: "local_generated_run",
    export_snapshot_fingerprint: "test-fingerprint",
    metrics: {loss_history: [1, 0.7, 0.45, 0.31, 0.29], final_loss: 0.29,
      error_metrics: {relative_l2: 0.12}},
    failure: {type: null, message: null, stack: null, truncated: false},
    exported_pair_snapshot: {problem, method},
  };
  const deterministic = window.OpenPINNDiagnosticAgent.evaluateArtifact(artifact, null);
  const report = {
    schema_version: 1, kind: "DiagnosticReport", pair_fingerprint: "test-fingerprint",
    artifact_ref: {kind: "RunArtifact", status: "not_converged"}, healthy: false,
    deterministic,
    llm_track: {state: "complete", report: {assessment: "concerns", findings: [],
      repair_hints: [], human_summary: "损失下降后进入平台区，建议调整采样和优化策略。"}},
    human_summary: "损失下降后进入平台区，建议调整采样和优化策略。",
    repair_performed: false,
  };
  window.OpenPINNDiagnosticAgent.readArtifactFile = async () => artifact;
  window.OpenPINNDiagnosticAgent.diagnose = async () => report;
  const diagnosticFile = agent("diagnostic-file");
  Object.defineProperty(diagnosticFile, "files", {
    configurable: true,
    value: [new window.File([JSON.stringify(artifact)], "plate-with-hole-run.json", {type: "application/json"})],
  });
  diagnosticFile.dispatchEvent(new window.Event("change", {bubbles: true}));
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(agent("diagnose").disabled, false);
  agent("diagnose").click();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(agent("diagnostic-report-view").hidden, false);
  assert.ok(agent("diagnostic-report-view").querySelector(".diagnostic-loss-chart"));
  assert.match(agent("diagnostic-report-view").textContent, /Adam.*L-BFGS/);
  assert.match(agent("diagnostic-report-view").textContent, /RAR \/ RAD/);
  assert.equal(agent("download-diagnostic-html").disabled, false);
  assert.equal(agent("download-diagnostic-json").disabled, false);

  role("custom-x-max").value = "0.5";
  role("custom-x-max").dispatchEvent(new window.Event("input", {bubbles: true}));
  assert.equal(role("copy").disabled, false);
  assert.equal(role("download-code").disabled, false);
  assert.match(role("code").textContent, /NotImplementedError/);
  assert.deepEqual(errors, []);
  dom.window.close();
  console.log("AGENT_PIPELINE_OK: same-page handoff, plate codegen, readable diagnostic report, honest incomplete export");
}

main().catch(error => { console.error(error); process.exitCode = 1; });
