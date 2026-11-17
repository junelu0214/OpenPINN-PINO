"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const joint = require("../../docs/_static/joint_handoff.js");

const plate = {
  dim: 2,
  physics: "linear_elasticity",
  geometry: {outer: {type: "rectangle", x_min: 0, x_max: 2, y_min: 0, y_max: 2},
    holes: [{type: "circle", center: [1, 1], radius: 0.25}]},
  field: {kind: "displacement", components: ["ux", "uy"]},
  constitutive: {model: "isotropic_linear", E: 210e9, nu: 0.3},
  boundaries: [{id: "traction_right", region: "right", type: "neumann", dofs: ["ux"], value: 1e6}],
  units: {length: "m", stress: "Pa"},
};

test("joint handoff preserves Understanding problem and Planning method", () => {
  const method = {method: "baseline", variant: "plain_pinn", path: "pinn", training: {steps: 3}};
  const value = joint.build(
    {draft: plate, missing: [], warnings: [], ready_for_validation: true},
    {method_draft: method, rationale: ["bounded baseline"], warnings: [], ready_for_validation: true},
    {inputText: "plate with hole", understandingModel: "u-model", planningModel: "p-model"},
  );
  assert.equal(value.kind, "JointAgentHandoff");
  assert.deepEqual(value.mapping.applied.problem.field.components, ["ux", "uy"]);
  assert.deepEqual(value.mapping.applied.problem.geometry.holes[0].center, [1, 1]);
  assert.deepEqual(value.mapping.applied.problem.boundaries[0].dofs, ["ux"]);
  assert.equal(value.mapping.applied.problem.constitutive.E, 210e9);
  assert.equal(value.mapping.applied.method.training.steps, 3);
  assert.equal(value.sources.understanding.model, "u-model");
  assert.equal(value.sources.planning.model, "p-model");
  assert.equal(JSON.stringify(value).includes("api_key"), false);
  assert.equal(value.completeness.joint_complete, true);
});

test("missing source and conflicts are visible but non-blocking", () => {
  const partial = joint.build(null, {method_draft: {method: "baseline"}, warnings: [], ready_for_validation: false});
  assert.equal(partial.sources.understanding.present, false);
  assert(partial.issues.some(item => item.code === "handoff.missing_understanding"));
  assert(partial.issues.every(item => item.blocking_import === false && item.blocking_export === false));

  const conflict = joint.build(
    {draft: {method: "fourier", physics: "poisson"}, missing: [], warnings: [], ready_for_validation: true},
    {method_draft: {method: "baseline"}, warnings: [], ready_for_validation: true},
  );
  assert.equal(conflict.mapping.applied.method.method, "baseline");
  assert.equal(conflict.mapping.conflicts[0].understanding_value, "fourier");
  assert.equal(conflict.mapping.conflicts[0].planning_value, "baseline");
  assert.equal(conflict.mapping.conflicts[0].default_used, "planning");
  assert.deepEqual(joint.parse(JSON.stringify(conflict)).mapping.conflicts, conflict.mapping.conflicts);
});
