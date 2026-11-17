"use strict";

const assert = require("node:assert/strict");
const {execFileSync} = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const {JSDOM} = require("jsdom");

const root = path.resolve(__dirname, "../..");
const paths = require(path.join(root, "docs/_static/configurator_paths.js"));

function fixturePayload() {
  return {
    operator_hooks: [
      {
        name: "deeponet",
        title: "DeepONet",
        summary: "Branch/trunk coordinate model.",
        implemented: true,
        selectable: true,
        aliases: ["deep_operator_network"],
        parameters: {
          training_mode: {label: "Training mode", kind: "choice", default: "data_only", minimum: null, description: "mode", choices: ["data_only", "physics_only", "hybrid"]},
          n_dims: {label: "Query dimension", kind: "integer", default: 1, minimum: 1, description: "dims", choices: []},
          n_fields: {label: "Output fields", kind: "integer", default: 1, minimum: 1, description: "fields", choices: []},
          steps: {label: "Training steps", kind: "integer", default: 2, minimum: 1, description: "steps", choices: []},
          data_weight: {label: "Data weight", kind: "number", default: 1, minimum: 0, description: "data", choices: []},
          physics_weight: {label: "Physics weight", kind: "number", default: 1, minimum: 0, description: "physics", choices: []},
        },
        trainer_limits: {
          equals: {n_dims: 1, n_fields: 1},
          grid_size_at_least_twice_modes: false,
          weight_rules: true,
          messages: {
            equals: "train_operator supports the 1D scalar antiderivative (n_dims=1, n_fields=1). This page will not switch to build_deeponet or another operator.",
            hybrid_weights: "hybrid mode needs a positive data_weight or physics_weight. The training mode was not changed.",
            physics_weights: "physics_only mode needs a positive physics_weight. The training mode was not changed.",
          },
        },
      },
      {
        name: "stub_operator",
        title: "Stub",
        summary: "Not selectable.",
        implemented: false,
        selectable: false,
        aliases: [],
        parameters: {},
        trainer_limits: null,
      },
    ],
    operator_choices: [
      {value: "deeponet", label: "DeepONet", canonical: "deeponet", selectable: true},
    ],
    training_modes: ["data_only", "physics_only", "hybrid"],
    operator_extensions: {
      choices: {
        derivative: ["autodiff", "finitedifference", "grid"],
        sampler: ["uniform", "residualadaptive"],
        optimizer: ["adam", "adamw"],
      },
      defaults: {derivative: "grid", sampler: "uniform", optimizer: "adam"},
    },
    plate_contract: {
      problem: {
        dim: 2,
        physics: "linear_elasticity",
        geometry: {
          outer: {type: "rectangle", x_min: 0, x_max: 2, y_min: 0, y_max: 2},
          holes: [{type: "circle", center: [1, 1], radius: 0.25}],
        },
        field: {kind: "displacement", components: ["ux", "uy"]},
        constitutive: {model: "isotropic_linear", E: 210000000000, nu: 0.3, stress_voigt: true},
        boundaries: [
          {id: "fix_left", region: "left", type: "dirichlet", dofs: ["ux", "uy"], value: [0, 0]},
          {id: "traction_right", region: "right", type: "neumann", dofs: ["ux"], value: 1000000},
          {id: "shear_right", region: "right", type: "neumann", dofs: ["uy"], value: 0},
          {id: "free_top", region: "top", type: "neumann", dofs: ["ux", "uy"], value: [0, 0]},
          {id: "free_bottom", region: "bottom", type: "neumann", dofs: ["ux", "uy"], value: [0, 0]},
          {id: "free_hole", region: "hole_0", type: "neumann", dofs: ["ux", "uy"], value: [0, 0]},
        ],
        units: {length: "m", force: "N", stress: "Pa"},
      },
      method: {
        method: "baseline",
        variant: "plain_pinn",
        path: "pinn",
        sampling: {name: "uniform", interior: 32, boundary: 16},
        loss: {formulation: "strong", weights: {pde: 1, boundary: 10}},
        optimizer: {name: "adam", lr: 0.001},
        network: {name: "mlp", hidden_layers: [16, 16], activation: "tanh"},
        differentiator: {mode: "autodiff"},
        training: {steps: 2},
      },
      limits: {
        dims: [2],
        physics: ["linear_elasticity"],
        outer_types: ["rectangle"],
        constitutive: ["isotropic_linear"],
        field_kinds: ["displacement"],
        solver_paths: ["pinn"],
      },
      rejected_physics: ["poisson"],
      rejected_solver_paths: ["operator"],
      rejected_methods: ["not_a_method"],
      pinn_methods: [{method: "baseline", variant: "plain_pinn", title: "Baseline / plain"}],
      download_name: "plate_with_hole_config",
      operator_path_error: "[codegen.operator_path] method.path: operator path is rejected. No PINN method was substituted.",
    },
  };
}

function designerHtml() {
  return `<div data-openpinn-problem-designer>
    <select data-role="local-script"></select>
    <div data-role="designer-operator" hidden>
      <p data-role="operator-summary"></p>
      <div data-role="operator-parameters"></div>
      <select data-role="operator-training-mode"></select>
      <div data-role="operator-extension-fields"></div>
    </div>
    <p data-role="local-message"></p>
    <button type="button" data-role="local-copy" disabled>Copy .py</button>
    <button type="button" data-role="local-download" disabled>Download .py</button>
    <pre><code data-role="local-code"></code></pre>
  </div>`;
}

function role(document, name) {
  return document.querySelector('[data-role="' + name + '"]');
}

async function main() {
  const illegal = paths.operatorState(fixturePayload(), "stub_operator", {});
  assert.equal(illegal.ok, false);
  assert.match(illegal.errors.join(" "), /不可选/);
  assert.equal(illegal.source, undefined);

  const unknown = paths.operatorState(fixturePayload(), "missing_operator", {});
  assert.equal(unknown.ok, false);
  assert.match(unknown.errors.join(" "), /未选择备用 Operator/);

  const dom = new JSDOM(designerHtml(), {runScripts: "outside-only", url: "https://openpinn.local/problem_designer.html"});
  const window = dom.window;
  const document = window.document;
  Object.defineProperty(window, "isSecureContext", {configurable: true, value: true});
  let blobText = null;
  let downloadName = null;
  window.URL.createObjectURL = function (blob) { blobText = blob; return "blob:script"; };
  window.URL.revokeObjectURL = function () {};
  window.HTMLAnchorElement.prototype.click = function () { downloadName = this.download; };
  let copied = null;
  Object.defineProperty(window.navigator, "clipboard", {
    configurable: true,
    value: {writeText: async function (text) { copied = text; }},
  });
  paths.prepare(document.querySelector("[data-openpinn-problem-designer]"), fixturePayload());
  const code = role(document, "local-code");
  const copyButton = role(document, "local-copy");
  const downloadButton = role(document, "local-download");
  assert.equal(copyButton.disabled, false);
  assert.equal(downloadButton.disabled, false);
  assert.match(code.textContent, /OPERATOR = 'deeponet'/);
  assert.match(code.textContent, /'training_mode': 'data_only'/);
  assert.match(code.textContent, /EXTENSIONS =/);
  assert.match(code.textContent, /'derivative': 'grid'/);
  assert.match(code.textContent, /train_operator/);
  assert.doesNotMatch(code.textContent, /op\.solve/);

  copyButton.click();
  await Promise.resolve();
  assert.equal(copied, code.textContent);

  downloadButton.click();
  assert.equal(downloadName, "deeponet_config.py");
  const reader = new window.FileReader();
  const downloaded = await new Promise(function (resolve, reject) {
    reader.onload = function () { resolve(reader.result); };
    reader.onerror = reject;
    reader.readAsText(blobText);
  });
  assert.equal(downloaded, code.textContent);

  const dims = document.querySelector('[data-operator-parameter="n_dims"]');
  dims.value = "2";
  dims.dispatchEvent(new window.Event("input", {bubbles: true}));
  assert.equal(downloadButton.disabled, false);
  assert.match(code.textContent, /不符合内置训练器要求/);
  assert.match(code.textContent, /NotImplementedError/);
  assert.doesNotMatch(code.textContent, /build_operator_config/);

  const stale = blobText;
  downloadButton.click();
  copyButton.click();
  assert.notEqual(blobText, stale);
  assert.equal(copied, code.textContent);

  role(document, "local-script").value = "stub_operator";
  role(document, "local-script").dispatchEvent(new window.Event("change", {bubbles: true}));
  assert.equal(downloadButton.disabled, false);
  assert.match(code.textContent, /未知 Operator hook/);
  assert.doesNotMatch(code.textContent, /OPERATOR = 'deeponet'/);

  role(document, "local-script").value = "missing_operator";
  role(document, "local-script").dispatchEvent(new window.Event("change", {bubbles: true}));
  assert.equal(downloadButton.disabled, false);
  assert.match(code.textContent, /未知 Operator hook/);

  const designed = {dimension: 1, coordinateDimension: 1, source: "PROBLEM = None\n"};
  const observation = {
    path: "examples/references/observations.csv",
    columns: ["x"],
    scales: [1],
    valueColumn: "u",
    valueScale: 1,
    predictionScale: 1,
    fieldIndex: 0,
    fixedTime: null,
  };
  const reference = {
    path: "examples/references/holdout.csv",
    columns: ["x"],
    scales: [1],
    valueColumn: "v",
    valueScale: 2,
    predictionScale: 3,
    fieldIndex: 0,
    fixedTime: null,
  };
  const supplied = {training_mode: "data_only", steps: 1};
  const referenceOnly = paths.operatorState(fixturePayload(), "deeponet", supplied, designed, {}, null, reference);
  assert.equal(referenceOnly.ok, false);
  assert.match(referenceOnly.errors.join(" "), /观测 CSV 路径/);
  assert.equal(referenceOnly.source, undefined);
  const observationOnly = paths.operatorState(fixturePayload(), "deeponet", supplied, designed, {}, observation, null);
  assert.equal(observationOnly.ok, true, (observationOnly.errors || []).join(" "));
  assert.match(observationOnly.source, /OBSERVATION_FILE = Path\('examples\/references\/observations.csv'\)/);
  assert.match(observationOnly.source, /observation=OBSERVATION_FILE/);
  assert.doesNotMatch(observationOnly.source, /REFERENCE_FILE/);
  assert.doesNotMatch(observationOnly.source, /evaluate_operator_reference_csv/);
  const both = paths.operatorState(fixturePayload(), "deeponet", supplied, designed, {}, observation, reference);
  assert.equal(both.ok, true, (both.errors || []).join(" "));
  assert.match(both.source, /OBSERVATION_VALUE_COLUMN = 'u'/);
  assert.match(both.source, /REFERENCE_VALUE_COLUMN = 'v'/);
  assert.match(both.source, /REFERENCE_VALUE_SCALE = 2\.0/);
  assert.match(both.source, /evaluate_operator_reference_csv\(\n    RESULT, REFERENCE_FILE/);
  assert.doesNotMatch(both.source, /RESULT, OBSERVATION_FILE/);
  assert.doesNotMatch(both.source, /observation=REFERENCE_FILE/);
  const physicsReference = paths.operatorState(
    fixturePayload(),
    "deeponet",
    {training_mode: "physics_only", steps: 1},
    designed,
    {},
    observation,
    reference
  );
  assert.equal(physicsReference.ok, true, (physicsReference.errors || []).join(" "));
  assert.match(physicsReference.source, /REFERENCE_FILE = Path\('examples\/references\/holdout.csv'\)/);
  assert.match(physicsReference.source, /observation=None/);
  assert.doesNotMatch(physicsReference.source, /OBSERVATION_/);
  const hybridMissing = paths.operatorState(fixturePayload(), "deeponet", {training_mode: "hybrid", steps: 1}, designed, {}, null, null);
  assert.equal(hybridMissing.ok, false);
  assert.match(hybridMissing.errors.join(" "), /观测 CSV 路径/);

  compareWithPython();
  assertHiddenDeepONetFourierModes();
  dom.window.close();
  console.log("CONFIGURATOR_PATHS_OK");
}

function hiddenModesPayload() {
  function parameter(label, kind, value, extra) {
    return Object.assign({
      label: label,
      kind: kind,
      default: value,
      minimum: null,
      description: label,
      choices: [],
    }, extra || {});
  }
  return {
    operator_hooks: [
      {
        name: "deeponet",
        title: "DeepONet",
        summary: "Branch/trunk coordinate model.",
        implemented: true,
        selectable: true,
        aliases: [],
        parameters: {
          training_mode: parameter("Training mode", "choice", "data_only", {choices: ["data_only", "physics_only", "hybrid"]}),
          n_sensors: parameter("Sensor count", "integer", 16, {minimum: 2}),
          n_modes: parameter("Fourier modes", "integer", 3, {minimum: 1, ui_hidden: true}),
          steps: parameter("Training steps", "integer", 1, {minimum: 1}),
        },
        trainer_limits: {equals: {}, messages: {equals: "mismatch"}},
      },
      {
        name: "fno",
        title: "FNO",
        summary: "Fourier neural operator.",
        implemented: true,
        selectable: true,
        aliases: [],
        parameters: {
          training_mode: parameter("Training mode", "choice", "data_only", {choices: ["data_only", "physics_only", "hybrid"]}),
          n_modes: parameter("Fourier modes", "integer", 4, {minimum: 1}),
          n_harmonics: parameter("Harmonics", "integer", 2, {minimum: 1}),
          steps: parameter("Training steps", "integer", 1, {minimum: 1}),
        },
        trainer_limits: {equals: {}, messages: {equals: "mismatch"}},
      },
    ],
    operator_choices: [
      {value: "deeponet", label: "DeepONet", canonical: "deeponet", selectable: true},
      {value: "fno", label: "FNO", canonical: "fno", selectable: true},
    ],
    training_modes: ["data_only", "physics_only", "hybrid"],
    operator_extensions: {choices: {}, defaults: {}},
  };
}

function assertHiddenDeepONetFourierModes() {
  const payload = hiddenModesPayload();
  const dom = new JSDOM(designerHtml(), {runScripts: "outside-only", url: "https://openpinn.local/problem_designer.html"});
  const document = dom.window.document;
  const root = document.querySelector("[data-openpinn-problem-designer]");
  paths.prepare(root, payload);
  const parameters = role(document, "operator-parameters");
  assert.equal(document.querySelector('[data-operator-parameter="n_modes"]'), null);
  assert.doesNotMatch(parameters.textContent, /Fourier modes/);
  assert.match(role(document, "local-code").textContent, /'n_modes': 3/);
  const hidden = paths.operatorState(payload, "deeponet", {});
  assert.equal(hidden.ok, true, (hidden.errors || []).join(" "));
  assert.match(hidden.source, /'n_modes': 3/);

  role(document, "local-script").value = "fno";
  role(document, "local-script").dispatchEvent(new dom.window.Event("change", {bubbles: true}));
  const modes = document.querySelector('[data-operator-parameter="n_modes"]');
  const harmonics = document.querySelector('[data-operator-parameter="n_harmonics"]');
  assert(modes, "FNO Fourier modes control");
  assert.match(modes.parentElement.textContent, /Fourier modes/);
  assert.equal(modes.value, "4");
  assert(harmonics, "FNO harmonics control");
  assert.match(harmonics.parentElement.textContent, /Harmonics/);
  dom.window.close();
}

function compareWithPython() {
  const python = process.env.OPENPINN_PYTHON || "python3";
  let raw;
  try {
    raw = execFileSync(python, ["-c", [
      "import json",
      "from openpinn.scriptgen import browser_payload, render_operator_script, render_plate_with_hole_script",
      "payload = browser_payload()",
      "print(json.dumps({",
      '  "payload": payload,',
      '  "deeponet": render_operator_script("deeponet"),',
      '  "fno": render_operator_script("fno"),',
      '  "deep_physics": render_operator_script("deeponet", {"training_mode": "physics_only"}),',
      '  "fno_physics": render_operator_script("fno", {"training_mode": "physics_only"}),',
      '  "extensions": render_operator_script("fno", extensions={"optimizer": "adamw"}),',
      '  "plate": render_plate_with_hole_script(),',
      "}))",
    ].join("\n")], {encoding: "utf8", cwd: root});
  } catch (error) {
    console.log("PYTHON_COMPARE_SKIPPED");
    return;
  }
  const data = JSON.parse(raw);
  const cases = [
    ["deeponet", data.deeponet, {}],
    ["fno", data.fno, {}],
  ];
  cases.forEach(function (item) {
    const state = paths.operatorState(data.payload, item[0], item[2]);
    assert.equal(state.ok, true, item[0] + " " + (state.errors || []).join(" "));
    assert.equal(state.source, item[1], "JS operator script drifted for " + item[0]);
  });
  for (const item of [
    ["deeponet", data.deep_physics],
    ["fno", data.fno_physics],
  ]) {
    const physics = paths.operatorState(data.payload, item[0], {training_mode: "physics_only"});
    assert.equal(physics.ok, true, (physics.errors || []).join(" "));
    assert.equal(physics.source, item[1]);
  }
  const extensions = paths.operatorState(data.payload, "fno", {}, undefined, {optimizer: "adamw"});
  assert.equal(extensions.ok, true, (extensions.errors || []).join(" "));
  assert.equal(extensions.source, data.extensions);
  const pino = paths.operatorState(data.payload, "pino", {});
  assert.equal(pino.ok, false);
  assert.match(pino.errors.join(" "), /不可选/);
  const live = new JSDOM(designerHtml(), {runScripts: "outside-only", url: "https://openpinn.local/problem_designer.html"});
  const liveDocument = live.window.document;
  paths.prepare(liveDocument.querySelector("[data-openpinn-problem-designer]"), data.payload);
  assert.equal(liveDocument.querySelector('[data-operator-parameter="n_modes"]'), null);
  assert.doesNotMatch(role(liveDocument, "operator-parameters").textContent, /Fourier modes/);
  assert.match(role(liveDocument, "local-code").textContent, /'n_modes': 3/);
  role(liveDocument, "local-script").value = "fno";
  role(liveDocument, "local-script").dispatchEvent(new live.window.Event("change", {bubbles: true}));
  const liveModes = liveDocument.querySelector('[data-operator-parameter="n_modes"]');
  assert(liveModes, "FNO Fourier modes stay visible");
  assert.match(liveModes.parentElement.textContent, /Fourier modes/);
  assert.equal(liveModes.value, "4");
  const liveHarmonics = liveDocument.querySelector('[data-operator-parameter="n_harmonics"]');
  assert(liveHarmonics, "FNO harmonics stay visible");
  assert.match(liveHarmonics.parentElement.textContent, /Harmonics/);
  live.window.close();
  const contract = data.payload.plate_contract;
  const plate = paths.plateState(contract.problem, contract.method, {
    steps: contract.method.training.steps,
    seed: 0,
    device: "cpu",
    outputDir: "runs",
  }, contract);
  assert.equal(plate.ok, true, (plate.errors || []).join(" "));
  assert.equal(plate.source, data.plate);
  const validation = paths.createPlateValidationSession();
  const validated = validation.validate(contract.problem, contract.method, {
    steps: contract.method.training.steps, seed: 0, device: "cpu", outputDir: "runs",
  }, contract);
  assert.equal(validated.ok, true);
  assert.equal(validated.validation, "browser_contract_precheck");
  assert.equal(validated.fullPythonValidation, false);
  assert.equal(validation.isCurrent(contract.problem, contract.method), true);
  assert.equal(validation.authorizeExport(contract.problem, contract.method, {
    steps: contract.method.training.steps, seed: 0, device: "cpu", outputDir: "runs",
  }, contract).ok, true);
  const changedMethod = JSON.parse(JSON.stringify(contract.method));
  changedMethod.training.steps += 1;
  assert.equal(validation.isCurrent(contract.problem, changedMethod), false);
  assert.match(validation.authorizeExport(contract.problem, changedMethod, {
    steps: changedMethod.training.steps, seed: 0, device: "cpu", outputDir: "runs",
  }, contract).errors.join(" "), /changed after/);
  assert.match(validation.authorizeExport(contract.problem, contract.method, {
    steps: contract.method.training.steps, seed: 1, device: "cpu", outputDir: "runs",
  }, contract).errors.join(" "), /differs from the validated payload/);
  console.log("PYTHON_COMPARE_OK");
}

main().catch(function (error) {
  console.error(error);
  process.exitCode = 1;
});
