"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const api = require(path.resolve(__dirname, "../../docs/_static/generic_validation.js"));

test("the public status vocabulary is complete", () => {
  assert.deepEqual(api.STATUSES, ["pass", "fail", "unknown", "unsupported", "skipped"]);
});

test("heat diffusivity and physics-editor symbols survive normalization", () => {
  const heat = api.normalizeProblem({
    physics: "heat_equation", dim: 1,
    field: {kind: "temperature", components: ["u"]},
    geometry: {type: "interval", x_min: 0, x_max: 1},
    pde: {type: "diffusion", diffusivity: 0.01, expression: "dt(u)-alpha*dx(dx(u))"},
    boundaries: [], initial_condition: {value: 0},
  });
  assert(heat.parameters.some(parameter => parameter.name === "alpha" && parameter.value === 0.01));
  assert.equal(api.validate(heat, {path: "pinn", method: "baseline"}).layers.find(layer => layer.id === "L2").status, "pass");
  const editor = api.normalizeProblem({...heat.source_draft, parameters: [{id: "custom.alpha", symbol: "alpha", value: 0.02, unit: "m^2/s"}]});
  assert(editor.parameters.some(parameter => parameter.name === "alpha"));
  assert.equal(api.validate(editor, {path: "pinn", method: "baseline"}).layers.find(layer => layer.id === "L2").status, "pass");
});

const method = {
  path: "pinn", method: "baseline", variant: "plain_pinn",
  sampling: {name: "uniform"}, loss: {formulation: "strong", weighting: "fixed"},
  optimizer: {name: "adam"}, network: {name: "mlp"},
  differentiator: {mode: "autodiff"}, training: {steps: 10},
};

function trainableHeatDraft() {
  return {
    physics: "heat_equation", dim: 1, kind: "dynamic",
    units: {space: "m", time: "s"},
    time_domain: {t_min: 0, t_max: 1, unit: "s"},
    field: {name: "u", kind: "scalar", components: ["u"], unit: "K"},
    parameters: [{name: "alpha", value: 0.01, unit: "m^2/s"}],
    geometry: {type: "interval", x_min: 0, x_max: 1},
    pde: {type: "diffusion", diffusivity: 0.01, source: {expression: "0", value: 0}, expression: "dt(u)-alpha*dx(dx(u))=0"},
    boundaries: [
      {location: "x=0", type: "dirichlet", value: 0, unit: "K"},
      {location: "x=1", type: "neumann", value: 0, unit: "K/m"},
    ],
    initial_condition: {field: "u", expression: "sin(pi*x/2)", unit: "K"},
  };
}

test("Stage A heat contract lifts time bounds and explicitly canonicalizes diffusivity to alpha", () => {
  const problem = api.normalizeProblem(trainableHeatDraft());
  assert.deepEqual(problem.coordinates, [
    {name: "x", role: "space", unit: "m", bounds: {min: 0, max: 1}},
    {name: "t", role: "time", unit: "s", bounds: {min: 0, max: 1}},
  ]);
  assert.deepEqual(problem.fields, [{name: "u", kind: "scalar", components: ["u"], shape: [1], unit: "K"}]);
  assert.deepEqual(problem.parameters, [{name: "alpha", value: 0.01, unit: "m^2/s"}]);
  assert.equal(problem.equations[0].expression, "dt(u)-alpha*dx(dx(u))=0");
  assert.deepEqual(problem.equations[0].source, {expression: "0", value: 0});
  assert.deepEqual(problem.boundaries.map(boundary => [boundary.type, boundary.value]), [["dirichlet", 0], ["neumann", 0]]);
  assert.equal(problem.initial_conditions[0].expression, "sin(pi*x/2)");
  assert.equal(problem.opaque_blocks.some(block => JSON.stringify(block).includes("time_domain")), false);
  assert.equal(problem.trainable_contract.mapping_complete, true);
  assert(problem.trainable_contract.normalizations.some(entry => entry.code === "parameter.alias_normalized" && entry.from === "diffusivity" && entry.to === "alpha"));
  const report = api.validate(problem, method);
  assert.equal(report.summary.mapping_complete_for_later_training, true);
  assert.equal(report.summary.training_export_unlocked, false);
  assert.equal(report.execution_capability.status, "supported");
  assert.equal(report.execution_capability.backend_id, "heat_1d_unsteady");
});

test("execution registry matches frozen Poisson structure and fails closed on structural drift", () => {
  const draft = {
    physics: "poisson", dim: 1, units: {space: "m"},
    field: {name: "u", kind: "scalar", components: ["u"], unit: "1"},
    geometry: {type: "interval", x_min: 0, x_max: 1},
    pde: {expression: "-dx(dx(u))=pi*pi*sin(pi*x)"},
    boundaries: [
      {location: "x=0", type: "dirichlet", value: 0},
      {location: "x=1", type: "dirichlet", value: 0},
    ],
  };
  const accepted = api.validate(draft, method);
  assert.equal(accepted.execution_capability.status, "supported");
  assert.equal(accepted.execution_capability.backend_id, "poisson_1d");
  const drifted = structuredClone(draft);
  drifted.boundaries[1].type = "periodic";
  assert.equal(api.validate(drifted, method).execution_capability.status, "unsupported");
});

test("generic scalar registry supports 2D multiple residuals and Robin conditions", () => {
  const problem = api.normalizeProblem({
    physics: "scalar_elliptic", dim: 2, units: {space: "m"},
    field: {name: "v", components: ["v"], unit: "1"},
    geometry: {type: "rectangle", x_min: 0, x_max: 1, y_min: 0, y_max: 1},
    equations: [
      {id: "pde", form: "expression", expression: "dx(dx(v))+dy(dy(v))=0"},
      {id: "aux", form: "expression", expression: "dx(v)-dx(v)=0"},
    ],
    boundaries: [
      {location: "x=0", type: "dirichlet", value: 0},
      {location: "x=1", type: "robin", alpha: 1, beta: 1, value: 0},
      {location: "y=0", type: "dirichlet", value: 0},
      {location: "y=1", type: "neumann", value: 0},
    ],
  });
  const report = api.validate(problem, method);
  assert.equal(report.summary.has_fail, false);
  assert.equal(report.execution_capability.status, "supported");
  assert.equal(report.execution_capability.backend_id, "scalar_strong_form_v1");

  problem.boundaries[0].location = "t=0";
  assert.equal(api.validate(problem, method).execution_capability.status, "unsupported");
});

test("boundary target expressions and pde.residuals normalize into a trainable generic contract", () => {
  const problem = api.normalizeProblem({
    physics: "scalar_first_order", dim: 2, units: {space: "1"},
    field: {name: "q", components: ["q"], unit: "1"},
    parameters: [{name: "value", value: "-pi", unit: "1"}],
    geometry: {type: "rectangle", x_min: 0, x_max: 1, y_min: 0, y_max: 1},
    pde: {residuals: [{expression: "dx(q)-1=0"}, {expression: "dy(q)-1=0"}]},
    boundaries: [
      {location: "x=0", type: "dirichlet", value_expression: "y"},
      {location: "x=1", type: "robin", alpha: 1, beta: 1,
        expression: "alpha*q+beta*dx(q)=value", value_expression: "y+2"},
      {location: "y=0", type: "dirichlet", value_expression: "x"},
      {location: "y=1", type: "neumann", value: 1},
    ],
  });
  assert.deepEqual(problem.equations.map(entry => entry.expression), ["dx(q)-1=0", "dy(q)-1=0"]);
  assert.deepEqual(problem.boundaries.slice(0, 3).map(entry => entry.expression), ["y", "y+2", "x"]);
  assert.equal(problem.parameters.some(entry => entry.name === "value"), false);
  assert(problem.trainable_contract.normalizations.some(entry => entry.code === "parameter.boundary_placeholder_removed"));
  assert.equal(problem.trainable_contract.mapping_complete, true);
  assert.equal(api.validate(problem, method).execution_capability.backend_id, "scalar_strong_form_v1");

  const genericSnapshot = structuredClone(problem);
  genericSnapshot.boundaries[1].expression = "alpha*q+beta*dx(q)=value";
  assert.equal(api.normalizeProblem(genericSnapshot).boundaries[1].expression, "y+2");

  const unbound = structuredClone(problem.source_draft);
  unbound.boundaries[0].value_expression = "undeclared+1";
  const rejected = api.normalizeProblem(unbound);
  assert.equal(rejected.trainable_contract.mapping_complete, false);
  assert(rejected.trainable_contract.issues.some(entry => entry.code === "mapping.boundary_expression_unbound"));
});

test("time bounds retained only in opaque remain mapping-incomplete", () => {
  const draft = trainableHeatDraft();
  delete draft.time_domain;
  draft.opaque_blocks = [{kind: "unmapped_source_fields", value: {time_domain: {t_min: 0, t_max: 1, unit: "s"}}}];
  const problem = api.normalizeProblem(draft);
  const time = problem.coordinates.find(coordinate => coordinate.role === "time");
  assert.equal(time.bounds, null);
  assert.equal(problem.trainable_contract.mapping_complete, false);
  assert(problem.trainable_contract.issues.some(entry => entry.code === "mapping.coordinate_bounds"));
  assert(problem.trainable_contract.issues.some(entry => entry.code === "mapping.required_data_opaque"));
});

test("unresolved alpha and diffusivity conflicts are visible and ineligible", () => {
  const draft = trainableHeatDraft();
  draft.pde.diffusivity = 0.02;
  const problem = api.normalizeProblem(draft);
  assert(problem.parameters.some(parameter => parameter.name === "alpha" && parameter.value === 0.01));
  assert(problem.parameters.some(parameter => parameter.name === "diffusivity" && parameter.value === 0.02));
  assert.equal(problem.trainable_contract.mapping_complete, false);
  assert(problem.trainable_contract.issues.some(entry => entry.code === "parameter.synonym_conflict"));
  assert.equal(api.validate(problem, method).summary.mapping_complete_for_later_training, false);
});

test("missing boundary or initial conditions cannot satisfy the later-training mapping contract", () => {
  const noBoundary = trainableHeatDraft();
  noBoundary.boundaries = [];
  assert(api.normalizeProblem(noBoundary).trainable_contract.issues.some(entry => entry.code === "mapping.boundaries_missing"));
  const noInitial = trainableHeatDraft();
  delete noInitial.initial_condition;
  assert(api.normalizeProblem(noInitial).trainable_contract.issues.some(entry => entry.code === "mapping.initial_condition_missing"));
});

test("unbound expression parameters and missing units fail closed for mapping eligibility", () => {
  const unbound = trainableHeatDraft();
  unbound.pde.expression = "dt(u)-beta*dx(dx(u))=0";
  assert(api.normalizeProblem(unbound).trainable_contract.issues.some(entry => entry.code === "mapping.expression_unbound" && /beta/.test(entry.message)));
  const noParameterUnit = trainableHeatDraft();
  noParameterUnit.parameters[0].unit = null;
  assert(api.normalizeProblem(noParameterUnit).trainable_contract.issues.some(entry => entry.code === "mapping.parameter_unit"));
  const noFieldUnit = trainableHeatDraft();
  noFieldUnit.field.unit = null;
  assert(api.normalizeProblem(noFieldUnit).trainable_contract.issues.some(entry => entry.code === "mapping.field_unit"));
});

function openPhysics() {
  return {
    physics: "magneto_thermo_chemo_nonlinear_v9",
    dim: 1,
    coordinates: [{name: "x", role: "space", unit: "m"}, {name: "t", role: "time", unit: "s"}],
    fields: [
      {name: "temperature", kind: "scalar", components: ["T"], shape: [1], unit: "K"},
      {name: "flux", kind: "vector", components: ["q"], shape: [1], unit: null},
    ],
    parameters: [{name: "alpha", value: 0.1, unit: "m^2/s"}],
    equations: [{form: "expression", expression: "dt(T)-alpha*dx(dx(T))"}],
    geometry: {type: "opaque", description: "periodic porous line"},
    boundaries: [{kind: "opaque", text: "nonlocal periodic constraint"}],
    initial_conditions: [{field: "T", expression: "sin(pi*x)"}],
    couplings: [{from: "T", to: "q", law: "opaque constitutive map"}],
    opaque_blocks: [{kind: "weak_form_note", text: "research notation retained verbatim"}],
  };
}

test("free physics labels, multiple fields, couplings, and opaque blocks enter the layered report", () => {
  const problem = api.normalizeProblem(openPhysics());
  const report = api.validate(problem, method);
  assert.equal(problem.schema_version, 1);
  assert.equal(problem.physics, "magneto_thermo_chemo_nonlinear_v9");
  assert.equal(problem.fields.length, 2);
  assert.equal(problem.couplings.length, 1);
  assert.equal(problem.opaque_blocks[0].text, "research notation retained verbatim");
  assert.equal(report.kind, "LayeredValidationReport");
  assert.deepEqual(report.layers.map(layer => layer.id), ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]);
  assert.equal(report.layers.find(layer => layer.id === "L4").status, "unknown");
  assert.equal(report.summary.training_export_unlocked, false);
  assert.equal(report.summary.target_gate, "not_run_in_browser");
  assert.equal("ok" in report, false);
});

test("restricted expression checker locates illegal symbols, derivative coordinates, and shape errors without eval", () => {
  const problem = openPhysics();
  problem.fields[1].shape = [2];
  problem.equations[0].expression = "dt(T)+dz(T)+intruder";
  const report = api.validate(problem, method);
  const issues = report.layers.flatMap(layer => layer.items);
  assert(issues.some(issue => issue.code === "structure.field_shape" && issue.path === "problem.fields[1].shape"));
  assert(issues.some(issue => issue.code === "expression.derivative_coordinate" && issue.path === "problem.equations[0].expression"));
  assert(issues.some(issue => issue.code === "expression.unknown_symbol" && /intruder/.test(issue.message)));
  assert.equal(report.summary.has_fail, true);
  assert.doesNotMatch(require("node:fs").readFileSync(path.resolve(__dirname, "../../docs/_static/generic_validation.js"), "utf8"), /\beval\s*\(|new\s+Function|Function\s*\(/);
});

test("Stage B browser checker uses the same first-period function whitelist", () => {
  const supported = openPhysics();
  supported.equations[0].expression = "dt(T)-alpha*dx(dx(T))+sin(pi*x)+cos(x)+exp(-x)";
  const supportedIssues = api.validate(supported, method).layers.flatMap(layer => layer.items);
  assert(supportedIssues.some(issue => issue.code === "expression.restricted_syntax" && issue.status === "pass"));

  ["tan(T)", "log(T)", "T^2", "dy(T)"].forEach(expression => {
    const rejected = openPhysics();
    rejected.equations[0].expression = expression;
    const issues = api.validate(rejected, method).layers.flatMap(layer => layer.items);
    assert(issues.some(issue => issue.status === "fail" && issue.path === "problem.equations[0].expression"), expression);
  });
});

test("steady descriptions need no initial condition and opaque mathematics stays scoped", () => {
  const steady = {
    physics: "fractional_nonlocal_material", dim: 2,
    fields: [{name: "u", kind: "scalar", components: ["u"], shape: [1]}],
    equations: [{form: "weak", opaque: "fractional bilinear form"}],
    opaque_blocks: [{kind: "notation", value: "H^s"}],
  };
  const report = api.validate(steady, method);
  assert.equal(report.layers.some(layer => layer.items.some(issue => /initial/i.test(issue.path) && issue.status === "fail")), false);
  assert.equal(report.layers.find(layer => layer.id === "L2").status, "unsupported");
  assert.equal(api.normalizeProblem(steady).opaque_blocks[0].value, "H^s");
});

test("unsupported method is an L5 capability result, not an invalid problem", () => {
  const unsupported = {...method, path: "quantum_backend", method: "invented_solver"};
  const report = api.validate(openPhysics(), unsupported);
  const l5 = report.layers.find(layer => layer.id === "L5");
  assert.equal(l5.status, "unsupported");
  assert.equal(l5.items[0].path, "method.method");
  assert.equal(report.summary.description_ready_for_review, true);
});

test("fingerprint changes invalidate reports and training export always stays locked", () => {
  const problem = openPhysics();
  const session = api.createSession();
  session.validate(problem, method);
  assert.equal(session.isCurrent(problem, method), true);
  assert.equal(session.canExportTraining(), false);
  assert.match(session.exportArtifacts(problem, method).label, /NON-TRAINING/);
  const changed = structuredClone(problem);
  changed.parameters[0].value = 0.2;
  assert.equal(session.isCurrent(changed, method), false);
  assert.throws(() => session.exportArtifacts(changed, method), /stale/);
});

test("Executable adapter is exact-or-reject and never bypasses Python M34", () => {
  const elasticity = {
    dim: 2, physics: "linear_elasticity",
    geometry: {outer: {type: "rectangle", x_min: 0, x_max: 1, y_min: 0, y_max: 1}, holes: []},
    field: {kind: "displacement", components: ["ux", "uy"]},
    constitutive: {model: "isotropic_linear", E: 1, nu: 0.3},
    boundaries: [{id: "fix_left", region: "left", type: "dirichlet", dofs: ["ux", "uy"], value: [0, 0]}],
    units: {length: "m"},
  };
  const accepted = api.validate(elasticity, method);
  assert.equal(accepted.adapter.status, "pass");
  assert.equal(accepted.adapter.target_gate, "python_m34_required");
  assert.deepEqual(accepted.adapter.problem_spec, elasticity);
  assert.equal(accepted.summary.training_export_unlocked, false);
  const rejected = api.validate({...elasticity, physics: "novel_elasticity", opaque_blocks: [{raw: "keep"}]}, method);
  assert.equal(rejected.adapter.status, "unsupported");
  assert.equal(rejected.summary.training_export_unlocked, false);
});
