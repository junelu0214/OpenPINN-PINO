/* Exercise the operator path in the real built configurator. */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {JSDOM} = require("jsdom");

const root = path.resolve(__dirname, "../..");

async function main() {
  const html = fs.readFileSync(path.join(root, "docs/_build/html/method_configurator.html"), "utf8");
  const dom = new JSDOM(html, {runScripts: "outside-only", url: "https://openpinn.local/method_configurator.html"});
  const window = dom.window;
  const document = window.document;
  await new Promise(resolve => window.addEventListener("DOMContentLoaded", resolve, {once: true}));
  const errors = [];
  window.addEventListener("error", event => errors.push(event.message));

  let copied = null;
  let downloadName = null;
  let downloadedBlob = null;
  Object.defineProperty(window, "isSecureContext", {configurable: true, value: true});
  Object.defineProperty(window.navigator, "clipboard", {
    configurable: true,
    value: {writeText: async text => { copied = text; }},
  });
  window.URL.createObjectURL = blob => { downloadedBlob = blob; return "blob:operator-script"; };
  window.URL.revokeObjectURL = () => {};
  window.HTMLAnchorElement.prototype.click = function () { downloadName = this.download; };

  for (const file of [
    "polygon_image_trace.js",
    "dxf_geometry_import.js",
    "physics_editor.js",
    "configurator_paths.js",
    "method_configurator.js",
  ]) {
    window.eval(fs.readFileSync(path.join(root, "docs/_static", file), "utf8"));
  }
  document.dispatchEvent(new window.Event("DOMContentLoaded"));

  const node = role => document.querySelector('[data-role="' + role + '"]');
  const set = (role, value, event = "change") => {
    const element = node(role);
    assert(element, role);
    element.value = String(value);
    element.dispatchEvent(new window.Event(event, {bubbles: true}));
  };
  const setOperatorModel = value => {
    const select = document.querySelector('[data-ecosystem-control="model"]');
    assert(select, "unified model control");
    select.value = value;
    select.dispatchEvent(new window.Event("change", {bubbles: true}));
  };
  const setEcosystem = (key, value) => {
    const select = document.querySelector('[data-ecosystem-control="' + key + '"]');
    assert(select, "unified " + key + " control");
    select.value = value;
    select.dispatchEvent(new window.Event("change", {bubbles: true}));
  };
  assert.equal(node("solution-path"), null);
  assert.equal(document.querySelector("#configurator-solution-path"), null);
  assert.match(node("operator-hooks").querySelector("h3").textContent, /算子训练参数/);
  assert.match(node("operator-hooks").textContent, /训练模式/);
  assert.match(node("operator-hooks").textContent, /高级扩展/);
  assert.match(node("operator-hooks").textContent, /浏览器不会执行训练/);

  assert.deepEqual([...node("problem").options].map(option => option.value), ["custom_equation_template"]);
  setOperatorModel("deeponet");
  assert.equal(node("custom-problem-panel").hidden, false);
  assert.equal(document.querySelector(".configurator-observation-panel").hidden, false);
  assert.equal(document.querySelector(".configurator-reference-panel").hidden, false);
  assert.equal(node("copy").disabled, false);
  assert.match(node("message").textContent, /观测 CSV 路径/);
  assert.match(node("code").textContent, /NotImplementedError/);
  set("custom-dimension", 1);
  set("custom-name", "operator_1d_problem", "input");
  node("reference-enable").checked = true;
  node("reference-enable").dispatchEvent(new window.Event("change", {bubbles: true}));
  set("reference-path", "examples/references/holdout.csv", "input");
  set("reference-coordinates", "x", "input");
  set("reference-coordinate-scales", "1", "input");
  set("reference-value", "v", "input");
  set("reference-value-scale", "2", "input");
  assert.equal(node("observation-enable").checked, false);
  assert.equal(node("observation-path").value, "");
  assert.equal(node("copy").disabled, false, node("message").textContent);
  assert.match(node("message").textContent, /观测 CSV 路径/);
  assert.doesNotMatch(node("code").textContent, /OBSERVATION_FILE/);
  assert.doesNotMatch(node("code").textContent, /holdout\.csv/);
  assert.doesNotMatch(node("code").textContent, /evaluate_operator_reference_csv/);
  node("observation-enable").checked = true;
  node("observation-enable").dispatchEvent(new window.Event("change", {bubbles: true}));
  set("observation-path", path.join(root, "examples/references/poisson_reference.csv"), "input");
  set("observation-coordinates", "x", "input");
  set("observation-coordinate-scales", "1", "input");
  set("observation-value", "u", "input");
  set("observation-value-scale", "1", "input");
  node("reference-enable").checked = false;
  node("reference-enable").dispatchEvent(new window.Event("change", {bubbles: true}));
  const leftBoundary = document.querySelector('[data-role="custom-boundary-side"][data-side="left"] [data-role="boundary-value"]');
  leftBoundary.value = "7";
  leftBoundary.dispatchEvent(new window.Event("change", {bubbles: true}));
  assert(node("custom-preview").children.length > 0);
  assert.match(node("custom-preview-note").textContent, /边界/);

  const modelValues = [...node("operator-model").options].map(option => option.value);
  assert.deepEqual(modelValues, ["deeponet", "fno"]);
  assert(!modelValues.includes("pino"));
  assert.deepEqual(
    [...node("operator-training-mode").options].map(option => option.value),
    ["data_only", "physics_only", "hybrid"]
  );
  assert(node("operator-extensions").hasAttribute("open") === false);

  const programs = [];
  setOperatorModel("deeponet");
  set("operator-training-mode", "data_only");
  let steps = document.querySelector('[data-operator-parameter="steps"]');
  steps.value = "1";
  steps.dispatchEvent(new window.Event("input", {bubbles: true}));
  assert.equal(node("copy").disabled, false, node("message").textContent);
  let source = node("code").textContent;
  assert.match(source, /DESIGNED_PROBLEM = PROBLEM/);
  assert.match(source, /OBSERVATION_FILE = Path\(/);
  assert.match(source, /train_operator_problem\(/);
  assert.match(source, /observation=OBSERVATION_FILE/);
  assert.doesNotMatch(source, /REFERENCE_FILE/);
  assert.doesNotMatch(source, /evaluate_operator_reference_csv\(/);
  assert.match(source, /trainer_scope": "designed_problem/);
  assert.doesNotMatch(source, /registered_operator_smoke/);
  programs.push({name: "deeponet", dimension: 1, code: source});

  setOperatorModel("fno");
  set("operator-training-mode", "physics_only");
  set("custom-dimension", 2);
  set("custom-name", "operator_2d_problem", "input");
  steps = document.querySelector('[data-operator-parameter="steps"]');
  steps.value = "1";
  steps.dispatchEvent(new window.Event("input", {bubbles: true}));
  const gridSize = document.querySelector('[data-operator-parameter="grid_size"]');
  gridSize.value = "4";
  gridSize.dispatchEvent(new window.Event("input", {bubbles: true}));
  const modes = document.querySelector('[data-operator-parameter="n_modes"]');
  modes.value = "2";
  modes.dispatchEvent(new window.Event("input", {bubbles: true}));
  assert.equal(node("copy").disabled, false, node("message").textContent);
  source = node("code").textContent;
  assert.match(source, /OPERATOR = 'fno'/);
  assert.match(source, /'training_mode': 'physics_only'/);
  assert.match(source, /train_operator_problem\(/);
  assert.equal(document.querySelector(".configurator-observation-panel").hidden, true);
  assert.equal(node("observation-enable").disabled, true);
  assert.doesNotMatch(source, /OBSERVATION_FILE/);
  assert.doesNotMatch(source, /REFERENCE_FILE/);
  programs.push({name: "fno", dimension: 2, code: source});

  node("reference-enable").checked = true;
  node("reference-enable").dispatchEvent(new window.Event("change", {bubbles: true}));
  set("reference-path", "examples/references/holdout.csv", "input");
  set("reference-coordinates", "x,y", "input");
  set("reference-coordinate-scales", "1,1", "input");
  set("reference-value", "v", "input");
  assert.equal(node("copy").disabled, false, node("message").textContent);
  assert.match(node("code").textContent, /REFERENCE_FILE = Path\('examples\/references\/holdout.csv'\)/);
  assert.match(node("code").textContent, /evaluate_operator_reference_csv\(\n    RESULT, REFERENCE_FILE/);
  assert.match(node("code").textContent, /observation=None/);
  assert.doesNotMatch(node("code").textContent, /OBSERVATION_/);
  node("reference-enable").checked = false;
  node("reference-enable").dispatchEvent(new window.Event("change", {bubbles: true}));

  setOperatorModel("fno");
  set("operator-training-mode", "physics_only");
  assert.equal(node("copy").disabled, false, node("message").textContent);
  assert.match(node("code").textContent, /'training_mode': 'physics_only'/);
  assert.match(node("code").textContent, /GridPDEProblem/);

  setOperatorModel("deeponet");
  set("operator-training-mode", "physics_only");
  assert.equal(node("copy").disabled, false, node("message").textContent);
  assert.match(node("code").textContent, /query coordinates with autodiff/);
  assert.doesNotMatch(node("code").textContent, /GridPDEProblem/);

  setOperatorModel("fno");

  setEcosystem("optimizer", "adamw");
  assert.match(node("code").textContent, /'optimizer': 'adamw'/);
  assert.match(node("code").textContent, /EXTENSIONS =/);

  for (const dimension of [1, 2]) {
    set("custom-dimension", dimension);
    set("observation-coordinates", dimension === 1 ? "x" : "x,y", "input");
    set("observation-coordinate-scales", dimension === 1 ? "1" : "1,1", "input");
    for (const name of ["deeponet", "fno"]) {
      setOperatorModel(name);
      for (const mode of ["data_only", "physics_only", "hybrid"]) {
        set("operator-training-mode", mode);
        const observationPanel = document.querySelector(".configurator-observation-panel");
        assert.equal(observationPanel.hidden, mode === "physics_only");
        node("observation-enable").checked = mode !== "physics_only";
        node("observation-enable").dispatchEvent(new window.Event("change", {bubbles: true}));
        assert.equal(node("reference-enable").checked, false);
        assert.equal(node("copy").disabled, false, `${dimension}/${name}/${mode}: ${node("message").textContent}`);
        if (mode === "physics_only") {
          assert.doesNotMatch(node("code").textContent, /OBSERVATION_FILE/);
        } else {
          assert.match(node("code").textContent, /observation=OBSERVATION_FILE/);
          assert.doesNotMatch(node("code").textContent, /evaluate_operator_reference_csv\(/);
        }
      }
    }
  }

  set("custom-dimension", 1);
  setOperatorModel("deeponet");
  set("operator-training-mode", "data_only");
  node("observation-enable").checked = true;
  node("observation-enable").dispatchEvent(new window.Event("change", {bubbles: true}));
  set("observation-path", "examples/references/observations.csv", "input");
  set("observation-coordinates", "x", "input");
  set("observation-coordinate-scales", "1", "input");
  set("observation-value", "u", "input");
  set("observation-value-scale", "1", "input");
  node("reference-enable").checked = true;
  node("reference-enable").dispatchEvent(new window.Event("change", {bubbles: true}));
  set("reference-path", "examples/references/holdout.csv", "input");
  set("reference-coordinates", "x", "input");
  set("reference-coordinate-scales", "1", "input");
  set("reference-value", "v", "input");
  set("reference-value-scale", "2", "input");
  assert.equal(node("copy").disabled, false, node("message").textContent);
  source = node("code").textContent;
  assert.match(source, /OBSERVATION_FILE = Path\('examples\/references\/observations.csv'\)/);
  assert.match(source, /REFERENCE_FILE = Path\('examples\/references\/holdout.csv'\)/);
  assert.match(source, /OBSERVATION_VALUE_COLUMN = 'u'/);
  assert.match(source, /REFERENCE_VALUE_COLUMN = 'v'/);
  assert.match(source, /OBSERVATION_VALUE_SCALE = 1\.0/);
  assert.match(source, /REFERENCE_VALUE_SCALE = 2\.0/);
  assert.match(source, /observation=OBSERVATION_FILE/);
  assert.match(source, /evaluate_operator_reference_csv\(\n    RESULT, REFERENCE_FILE/);
  assert.doesNotMatch(source, /observation=REFERENCE_FILE/);
  assert.doesNotMatch(source, /RESULT, OBSERVATION_FILE/);
  assert.equal(node("observation-path").value, "examples/references/observations.csv");
  assert.equal(node("reference-path").value, "examples/references/holdout.csv");

  node("copy").click();
  await Promise.resolve();
  assert.equal(copied, node("code").textContent);
  node("download-code").click();
  assert.equal(downloadName, "deeponet_config.py");
  const reader = new window.FileReader();
  const downloaded = await new Promise((resolve, reject) => {
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsText(downloadedBlob);
  });
  assert.equal(downloaded, node("code").textContent);
  assert.deepEqual(errors, []);
  dom.window.close();
  return programs;
}

if (process.argv.includes("--json")) {
  main().then(result => process.stdout.write(JSON.stringify(result))).catch(error => {
    console.error(error);
    process.exitCode = 1;
  });
} else {
  require("node:test")("operator path trains 1D/2D designed PDEs with observation gating", async () => {
    await main();
  });
}
