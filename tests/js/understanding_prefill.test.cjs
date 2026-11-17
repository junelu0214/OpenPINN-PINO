"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {JSDOM} = require("jsdom");

const root = path.resolve(__dirname, "../..");
const trial = require(path.join(root, "docs/_static/understanding_trial.js"));
const html = fs.readFileSync(
  path.join(root, "docs/_build/html/method_configurator.html"), "utf8"
);

function report(draft) {
  const assessment = trial.assessFamily(draft);
  return {
    draft,
    missing: assessment.missing,
    questions: [],
    warnings: [],
    ready_for_validation: assessment.ready,
  };
}

function role(document, name) {
  return document.querySelector('[data-role="' + name + '"]');
}

function loadConfigurator(sourceReport, methodHandoff) {
  const dom = new JSDOM(html, {
    runScripts: "outside-only",
    url: "https://openpinn.local/method_configurator.html",
  });
  const window = dom.window;
  if (sourceReport) {
    const handoff = trial.createConfiguratorHandoff(sourceReport, {language: "zh"});
    window.sessionStorage.setItem(trial.configuratorHandoffKey, JSON.stringify(handoff));
  }
  if (methodHandoff) {
    window.sessionStorage.setItem(trial.methodHandoffKey, JSON.stringify(methodHandoff));
  }
  for (const file of [
    "polygon_image_trace", "dxf_geometry_import", "physics_editor", "multiphysics_editor",
    "configurator_paths", "generic_validation", "understanding_trial", "method_configurator",
  ]) {
    window.eval(fs.readFileSync(path.join(root, "docs/_static", file + ".js"), "utf8"));
  }
  window.document.dispatchEvent(new window.Event("DOMContentLoaded"));
  return dom;
}

test("Planning handoff preserves Python validation warnings without blocking export", () => {
  const handoff = {
    schema_version: 1,
    source: "solver_planning",
    language: "zh",
    path: "pinn",
    ready_for_validation: true,
    problem_draft: heat,
    method_draft: {
      path: "pinn", method: "baseline", variant: "plain_pinn",
      sampling: {name: "uniform"}, loss: {formulation: "strong", weighting: "fixed"},
      optimizer: {name: "adam"}, network: {name: "mlp"},
      differentiator: {mode: "autodiff"}, training: {steps: 4321},
    },
    warnings: [],
  };
  const dom = loadConfigurator(null, handoff);
  try {
    const document = dom.window.document;
    assert.equal(role(document, "solution-path"), null);
    assert.equal(role(document, "steps").value, "4321");
    assert.equal(role(document, "problem").value, "custom_equation_template");
    assert.match(role(document, "physics-formula").value, /dt|dx/);
    const baseline = document.querySelector('[data-method-card="baseline"]');
    assert.equal(baseline.querySelector('[data-role="method-toggle"]').checked, true);
    assert.equal(baseline.querySelector('[data-role="variant"]').value, "plain_pinn");
    assert.match(role(document, "planning-method-status").textContent, /问题与方法草稿已联合导入/);
    assert.match(role(document, "planning-method-status").textContent, /不阻止原始代码导出/);
    assert.match(document.querySelector("[data-openpinn-configurator]").dataset.planningPairFingerprint, /^fnv1a32:/);
    assert.equal(role(document, "copy").disabled, false);
    assert.equal(role(document, "download-code").disabled, false);
    role(document, "steps").value = "5000";
    role(document, "steps").dispatchEvent(new dom.window.Event("input", {bubbles: true}));
    assert.equal(role(document, "copy").disabled, false);
    assert.equal(role(document, "download-code").disabled, false);
    assert.equal(document.querySelector("[data-openpinn-configurator]").dataset.planningPairFingerprint, undefined);
    assert.match(role(document, "planning-method-status").textContent, /旧 Hybrid Validation 报告已过期/);
    assert.match(role(document, "planning-method-status").textContent, /不阻止复制和下载/);
    const revalidate = role(document, "hybrid-revalidate-link");
    assert.equal(revalidate.hidden, false);
    revalidate.addEventListener("click", event => event.preventDefault());
    revalidate.click();
    const currentPair = JSON.parse(dom.window.sessionStorage.getItem("openpinn-hybrid-revalidate-v1"));
    assert.equal(currentPair.method_draft.training.steps, 5000);
    assert(currentPair.problem_draft.parameters.some(parameter => parameter.name === "alpha"));
    assert.equal(dom.window.sessionStorage.getItem(trial.methodHandoffKey), null);
  } finally {
    dom.window.close();
  }
});

test("Planning handoff without problem_draft exports an honest incomplete script", () => {
  const dom = loadConfigurator(null, {
    schema_version: 1, source: "solver_planning", language: "zh", path: "pinn",
    method_draft: {path: "pinn", method: "baseline", training: {steps: 20}},
  });
  try {
    const document = dom.window.document;
    assert.match(role(document, "planning-method-status").textContent, /预填失败/);
    assert.equal(role(document, "copy").disabled, false);
    assert.equal(role(document, "download-code").disabled, false);
    assert.equal(document.querySelector("[data-openpinn-configurator]").dataset.planningPairFingerprint, undefined);
  } finally {
    dom.window.close();
  }
});

const elasticity = {
  dim: 2,
  physics: "linear_elasticity",
  geometry: {
    outer: {type: "rectangle", x_min: 0, x_max: 2, y_min: 0, y_max: 2},
    holes: [{type: "circle", center: [1, 1], radius: 0.25}],
  },
  field: {kind: "displacement", components: ["ux", "uy"]},
  constitutive: {model: "isotropic_linear", E: 210e9, nu: 0.3},
  boundaries: [
    {region: "left", type: "dirichlet", dofs: ["ux", "uy"], value: [0, 0]},
    {region: "right", type: "neumann", dofs: ["ux"], value: 1e6},
    {region: "top", type: "neumann", dofs: ["ux", "uy"], value: [0, 0]},
    {region: "bottom", type: "neumann", dofs: ["ux", "uy"], value: [0, 0]},
    {region: "hole_0", type: "neumann", dofs: ["ux", "uy"], value: [0, 0]},
  ],
  units: {length: "m", stress: "Pa"},
};

const heat = {
  dim: 1,
  physics: "heat_equation",
  geometry: {type: "interval", x_min: 0, x_max: 1},
  field: {kind: "temperature", components: ["u"]},
  pde: {type: "diffusion", diffusivity: 0.01},
  boundaries: [
    {location: "x=0", type: "dirichlet", value: 0},
    {location: "x=1", type: "neumann", value: 0},
  ],
  initial_condition: {value: 1},
};

const poisson = {
  dim: 2,
  physics: "poisson",
  geometry: {type: "quadrilateral", bounds: {x: [0, 1], y: [0, 1]}},
  field: {kind: "scalar", components: ["u"]},
  pde: {equation: "-Δu = 1", source: {value: 1}},
  boundaries: ["left", "right", "top", "bottom"].map(region => ({
    region, type: "dirichlet", value: 0,
  })),
};

const wave = {
  dim: 1,
  physics: "wave_equation",
  geometry: {type: "interval", x_min: -1, x_max: 1},
  field: {kind: "scalar", components: ["u"]},
  pde: {equation: "dt(u)-c*dx(dx(u))=0", time_dependent: true},
  parameters: [{id: "wave.c", symbol: "c", value: 1, unit: "m^2/s"}],
  boundaries: [
    {region: "left", type: "dirichlet", value: 0},
    {region: "right", type: "dirichlet", value: 0},
  ],
  initial_condition: {value: 0},
};

for (const [name, draft] of [["elasticity", elasticity], ["heat", heat], ["poisson", poisson], ["other family", wave]]) {
  test(name + " Understanding prefill always lands in the custom PDE/BC editor", () => {
    const sourceReport = report(draft);
    assert.equal(sourceReport.ready_for_validation, true);
    const dom = loadConfigurator(sourceReport);
    try {
      const document = dom.window.document;
      assert.equal(role(document, "problem").value, "custom_equation_template");
      assert.equal(role(document, "custom-equation-mode").value, "expression");
      assert.equal(role(document, "custom-equation-mode").type, "hidden");
      assert.equal(role(document, "custom-template-field").hidden, true);
      assert.equal(role(document, "custom-problem-panel").hidden, false);
      assert.equal(role(document, "physics-editor").hidden, true);
      assert.match(role(document, "multiphysics-compiled-preview").dataset.latex, /R_\{/);
      assert.equal(role(document, "physics-template-controls").hidden, true);
      assert.equal(role(document, "physics-field-preset-controls").hidden, true);
      assert.equal(dom.window.sessionStorage.getItem(trial.configuratorHandoffKey), null);
      if (name === "elasticity") {
        assert.equal(document.querySelector('[data-role="plate-panel"]:not([hidden])'), null);
        assert.equal(role(document, "multiphysics-class").value, "solid_mechanics");
        assert.equal(role(document, "multiphysics-profile").value, "linear_elasticity");
        assert(Array.from(role(document, "multiphysics-fields-list").querySelectorAll("input")).some(node=>node.value === "位移"));
        assert.equal(role(document, "multiphysics-profile").value, "linear_elasticity");
        assert.deepEqual(Array.from(document.querySelectorAll('[data-role="multiphysics-fields-list"] input')).map(node=>node.value).filter(value=>value.includes("ux")), ["ux, uy"]);
        assert(role(document,"custom-preview").querySelector('[data-multiphysics-hole]'));
        assert.equal(role(document, "plate-x-max").value, "2");
        assert.match(role(document, "understanding-prefill-status").textContent, /线弹性草稿已填入自定义控制方程/);
        assert.doesNotMatch(role(document, "code").textContent, /NotImplementedError/);
        assert.match(role(document, "code").textContent, /from_specs\(/);
      } else {
        assert.equal(document.querySelector('[data-role="plate-panel"]:not([hidden])'), null);
        assert.match(role(document, "understanding-prefill-status").textContent, /自定义 PDE\/BC/);
        assert.equal(role(document, "physics-field").value, "u");
        assert(role(document, "physics-formula").value.length > 0);
        assert.match(role(document, "physics-status").textContent, /校验通过/);
        if (name === "heat") {
          assert.match(role(document, "physics-formula").value, /dt\(u\).*alpha/);
          assert.equal(role(document, "physics-parameter-0-symbol").value, "alpha");
          assert.equal(role(document, "physics-parameter-0-value").value, "0.01");
        }
        if (name === "poisson") {
          assert.doesNotMatch(role(document, "physics-formula").value, /Δ|∇/);
          assert.match(role(document, "physics-formula").value, /dx\(dx\(u\)\).*dy\(dy\(u\)\).*f/);
          assert.equal(role(document, "physics-parameter-0-symbol").value, "f");
          assert.equal(role(document, "physics-parameter-0-value").value, "1");
        }
      }
    } finally {
      dom.window.close();
    }
  });
}

test("physics interface selection distinguishes editable models from runnable adapters", () => {
  const dom = loadConfigurator(null);
  try {
    const document = dom.window.document;
    const physicsClass = role(document, "multiphysics-class");
    physicsClass.value = "fluid_mechanics";
    physicsClass.dispatchEvent(new dom.window.Event("change", {bubbles:true}));
    assert.equal(role(document, "multiphysics-profile").value, "incompressible_navier_stokes");
    const fluidValues = Array.from(role(document, "multiphysics-fields-list").querySelectorAll("input"), node=>node.value);
    assert(fluidValues.includes("速度"));
    assert(fluidValues.includes("压力"));
    assert.match(role(document, "code").textContent, /MULTIPHYSICS_MODEL/);
    assert.match(role(document, "code").textContent, /NotImplementedError/);
    assert.match(role(document, "multiphysics-editor").textContent, /执行适配待接入/);

    physicsClass.value = "solid_mechanics";
    physicsClass.dispatchEvent(new dom.window.Event("change", {bubbles:true}));
    assert.equal(role(document, "multiphysics-profile").value, "linear_elasticity");
    assert.doesNotMatch(role(document, "code").textContent, /NotImplementedError/);
    assert.match(role(document, "code").textContent, /from_specs\(/);
    assert.match(role(document, "code").textContent, /['"]assumption['"]:\s*['"]plane_stress['"]/);
    const mechanicsPreview = role(document, "custom-preview");
    assert.equal(mechanicsPreview.querySelectorAll('[data-mechanical-load-arrow="true"]').length, 3);
    assert(mechanicsPreview.querySelector('[data-mechanical-free-hole="true"]'));
    assert.match(mechanicsPreview.textContent, /固定 uₓ=uᵧ=0/);
    assert.match(mechanicsPreview.textContent, /右边：t=\(1\.00e\+6, 0\)/);
    assert.match(mechanicsPreview.textContent, /孔边：自由 t=0/);
  } finally {
    dom.window.close();
  }
});

test("seepage PINN and consolidation-family operator generate distinct inverse scripts", () => {
  const dom = loadConfigurator(null);
  try {
    const document = dom.window.document;
    const physicsClass = role(document, "multiphysics-class");
    physicsClass.value = "fluid_mechanics";
    physicsClass.dispatchEvent(new dom.window.Event("change", {bubbles:true}));
    const profile = role(document, "multiphysics-profile");
    profile.value = "darcy_inverse_seepage";
    profile.dispatchEvent(new dom.window.Event("change", {bubbles:true}));

    assert.equal(document.querySelector(".configurator-observation-panel").hidden, false);
    assert.equal(role(document, "observation-enable").checked, true);
    assert.equal(role(document, "observation-path").value, "examples/references/darcy_inverse_observations.csv");
    assert.equal(role(document, "observation-value").value, "p");
    const inverseRole = Array.from(role(document, "multiphysics-properties-list").querySelectorAll("select"))
      .find(node => node.value === "inverse");
    assert(inverseRole);
    assert.match(role(document, "code").textContent, /inverse_darcy_pinn/);
    assert.match(role(document, "code").textContent, /PERMEABILITY_BOUNDS = \[\s*0\.2,\s*5,\s*\]/);

    const model = document.querySelector('[data-ecosystem-control="model"]');
    model.value = "deeponet";
    model.dispatchEvent(new dom.window.Event("change", {bubbles:true}));
    assert.match(role(document, "code").textContent, /算子反分析请改选/);

    physicsClass.value = "multiphysics";
    physicsClass.dispatchEvent(new dom.window.Event("change", {bubbles:true}));
    profile.value = "consolidation_inverse_family";
    profile.dispatchEvent(new dom.window.Event("change", {bubbles:true}));
    assert.equal(role(document, "observation-path").value, "examples/references/consolidation_inverse_observations.csv");
    assert.equal(role(document, "observation-coordinates").value, "z,t");
    assert.equal(role(document, "observation-value").value, "u");
    assert.match(role(document, "code").textContent, /train_consolidation_family_operator/);
    assert.match(role(document, "code").textContent, /inverse_consolidation_operator/);
    assert.match(role(document, "code").textContent, /transient \(cv, U0\) consolidation family/);
    assert.doesNotMatch(role(document, "code").textContent, /inverse_darcy_pinn/);
  } finally { dom.window.close(); }
});

test("unified physics edits update compiled preview and generated code without exposing the legacy editor", () => {
  const dom = loadConfigurator(null);
  try {
    const document=dom.window.document;
    assert.equal(role(document,"physics-editor").hidden,true);
    assert.equal(role(document,"multiphysics-constitutive-model").tagName,"SELECT");
    assert.equal(document.querySelector(".multiphysics-constitutive-section").hidden,true);
    const equationInputs=Array.from(role(document,"multiphysics-equations-list").querySelectorAll("input"));
    const expression=equationInputs.find(node=>node.value.includes("dx("));
    expression.value="u=0";
    expression.dispatchEvent(new dom.window.Event("input",{bubbles:true}));
    assert.match(role(document,"multiphysics-compiled-preview").dataset.latex,/R_\{\\mathrm\{u\}\}.*\\mathrm\{u\}/);
    assert.match(role(document,"code").textContent,/expression_source[^\n]+u=0/);

    const physicsClass=role(document,"multiphysics-class");
    physicsClass.value="heat_transfer";physicsClass.dispatchEvent(new dom.window.Event("change",{bubbles:true}));
    const conductivity=Array.from(role(document,"multiphysics-properties-list").querySelectorAll('input[type="number"]')).find(node=>node.value==="45");
    assert(conductivity);conductivity.value="99";conductivity.dispatchEvent(new dom.window.Event("input",{bubbles:true}));
    assert.match(role(document,"code").textContent,/"symbol": "k"[\s\S]+"value": 99/);

    physicsClass.value="solid_mechanics";physicsClass.dispatchEvent(new dom.window.Event("change",{bubbles:true}));
    assert.equal(document.querySelector(".multiphysics-constitutive-section").hidden,false);
    assert.equal(role(document,"multiphysics-assumption-field").hidden,false);
    role(document,"multiphysics-dimension").value="3";
    role(document,"multiphysics-dimension").dispatchEvent(new dom.window.Event("change",{bubbles:true}));
    assert.equal(role(document,"multiphysics-assumption-field").hidden,true);

    physicsClass.value="fluid_mechanics";physicsClass.dispatchEvent(new dom.window.Event("change",{bubbles:true}));
    assert.equal(document.querySelector(".multiphysics-constitutive-section").hidden,true);
    assert.equal(role(document,"multiphysics-equations-list").querySelector("select").tagName,"SELECT");
    assert.equal(role(document,"multiphysics-boundaries-list").querySelectorAll("select").length>0,true);
  } finally {dom.window.close();}
});

test("unknown-family missing fields are visible but a locally valid configuration still exports code", () => {
  const sourceReport = report({
    dim: 2,
    geometry: {bounds: {x: [0, 1], y: [0, 1]}},
    field: {components: ["u"]},
    pde: {expression: "dx(dx(u))+dy(dy(u))=0"},
    boundaries: [{region: "left", type: "dirichlet", value: 0}],
  });
  assert.equal(sourceReport.ready_for_validation, false);
  assert.deepEqual(sourceReport.missing, ["physics"]);
  const dom = loadConfigurator(sourceReport);
  try {
    const document = dom.window.document;
    assert.equal(role(document, "problem").value, "custom_equation_template");
    assert.equal(role(document, "custom-equation-mode").value, "expression");
    assert.equal(role(document, "copy").disabled, false);
    assert.equal(role(document, "download-code").disabled, false);
    assert.doesNotMatch(role(document, "code").textContent, /NotImplementedError/);
    assert.match(role(document, "code").textContent, /op\.solve\(/);
    assert.match(role(document, "understanding-prefill-status").textContent, /尚缺字段：physics/);
    assert.match(role(document, "understanding-prefill-status").textContent, /部分预填|部分草稿|不代表代码可运行/);
    assert(role(document, "understanding-prefill-status").classList.contains("is-partial"));
    assert.equal(document.querySelector('[data-role="plate-panel"]:not([hidden])'), null);
  } finally {
    dom.window.close();
  }
});
