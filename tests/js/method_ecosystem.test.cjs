/* Browser-DOM coverage for the paper-aligned five-axis method ecosystem. */
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
  Object.defineProperty(window.navigator, "clipboard", {
    configurable: true,
    value: {writeText: async text => { copied = text; }},
  });
  for (const file of [
    "polygon_image_trace.js",
    "dxf_geometry_import.js",
    "physics_editor.js",
    "multiphysics_editor.js",
    "configurator_paths.js",
    "method_configurator.js",
  ]) {
    window.eval(fs.readFileSync(path.join(root, "docs/_static", file), "utf8"));
  }
  document.dispatchEvent(new window.Event("DOMContentLoaded"));

  const axes = [...document.querySelectorAll(".method-ecosystem-axis")];
  assert.deepEqual(axes.map(node => node.dataset.axis), ["model", "sampler", "loss", "optimizer", "derivative"]);
  assert.equal(axes[0].open, true);
  assert(axes.every(axis => axis.querySelectorAll(".method-ecosystem-family").length > 0));
  assert(document.querySelectorAll(".method-ecosystem-family.is-built_in").length > 0);
  assert.equal(document.querySelectorAll(".method-ecosystem-family.is-extension").length, 0);
  assert.match(axes[0].textContent, /DeepONet/);
  assert.match(axes[0].textContent, /operator:deeponet/);
  assert.match(axes[0].textContent, /Physics-informed KAN/);
  assert.match(axes[4].textContent, /弱式 \/ 变分残差/);
  assert.match(axes[4].textContent, /分部积分求积/);
  assert.match(document.body.textContent, /五模块方法生态与参数/);
  assert.match(document.body.textContent, /算法边界：/);
  assert.match(document.body.textContent, /parameter -= lr \* grad \/ \(\|grad\| \+ damping\)/);
  assert.match(document.body.textContent, /缺少坐标 grad\/hessian 后拒绝训练/);
  assert.equal(document.querySelectorAll(".method-status-badge").length, document.querySelectorAll(".method-ecosystem-family.is-built_in").length);
  assert.equal(document.querySelectorAll(".method-family-documentation").length, 32);

  const node = role => document.querySelector('[data-role="' + role + '"]');
  const choose = (key, value) => {
    const select = document.querySelector('[data-ecosystem-control="' + key + '"]');
    assert(select, "missing ecosystem control " + key);
    select.value = value;
    select.dispatchEvent(new window.Event("change", {bubbles: true}));
  };
  assert.equal(document.querySelectorAll("[data-ecosystem-control]").length, 7);
  assert.equal(document.querySelectorAll(".ecosystem-axis-card").length, 5);
  assert.equal(document.querySelectorAll("[data-ecosystem-toggle]").length, 2);
  assert(document.querySelector('[data-role="operator-hooks"]').closest('.ecosystem-selection-panel'));
  assert(node("pinn-method-presets").closest('.ecosystem-selection-panel'));
  assert(node("parameter-pane").closest('.ecosystem-selection-panel'));
  assert.equal(document.querySelector('[data-ecosystem-toggle="feature"]').checked, false);
  assert.equal(document.querySelector('[data-ecosystem-toggle="constraint"]').checked, false);
  assert.equal(node("solution-path"), null);
  assert(document.querySelector('.configurator-custom-fields').closest('[data-role="multiphysics-geometry-slot"]'));
  assert(node("custom-dxf-file").closest('[data-role="multiphysics-geometry-slot"]'));
  assert(document.querySelector('.configurator-reference-panel').closest('[data-role="multiphysics-data-slot"]'));
  const depth=document.querySelector('[data-ecosystem-parameter="model.hidden_layers.depth"]');
  const widths=document.querySelector('[data-ecosystem-parameter="model.hidden_layers"]');
  assert(depth&&widths);
  depth.value='4';depth.dispatchEvent(new window.Event('change',{bubbles:true}));
  assert.equal(widths.value,'64, 64, 64, 64');
  assert.match(node("code").textContent,/"hidden_layers": \[[\s\S]*?64,[\s\S]*?64,[\s\S]*?64,[\s\S]*?64,[\s\S]*?\]/);
  choose("model", "gno");
  choose("sampler", "rar");
  choose("loss", "ntkbalanced");
  choose("optimizer", "naturalgradient");
  choose("derivative", "weakvariational");
  const generated = node("code").textContent;
  assert.match(generated, /ECOSYSTEM_OVERRIDES = \{/);
  assert.match(generated, /"name": "gno"/);
  assert.match(generated, /"name": "rar"/);
  assert.match(generated, /"candidate_pool": 512/);
  assert.match(generated, /"name": "ntkbalanced"/);
  assert.match(generated, /"name": "naturalgradient"/);
  assert.match(generated, /"mode": "weakvariational"/);
  assert.match(generated, /overrides=ECOSYSTEM_OVERRIDES/);
  assert.match(node("ecosystem-overrides").textContent, /"name": "gno"/);
  choose("sampler", "adaptivemovement");
  assert.equal(document.querySelector('[data-ecosystem-parameter="sampler.indicator"]').tagName, "SELECT");
  choose("optimizer", "adamlbfgs");
  assert(document.querySelector(".ecosystem-stage-editor"));
  const featureToggle=document.querySelector('[data-ecosystem-toggle="feature"]');
  featureToggle.checked=true;featureToggle.dispatchEvent(new window.Event("change",{bubbles:true}));
  choose("feature","fourier");
  const constraintToggle=document.querySelector('[data-ecosystem-toggle="constraint"]');
  constraintToggle.checked=true;constraintToggle.dispatchEvent(new window.Event("change",{bubbles:true}));
  choose("constraint","hard");
  assert.match(node("code").textContent,/"feature": \{/);
  assert.match(node("code").textContent,/"output_transform": \{/);
  featureToggle.checked=false;featureToggle.dispatchEvent(new window.Event("change",{bubbles:true}));
  assert.doesNotMatch(node("code").textContent,/"feature": \{/);
  assert.match(node("code").textContent,/"output_transform": \{/);

  // Every registered high-level method variant is selectable for at least one
  // physical problem exposed by the configurator.
  const allVariants = new Set();
  document.querySelectorAll("[data-method-card]").forEach(card => {
    const method = card.dataset.methodCard;
    card.querySelectorAll('[data-role="variant"] option').forEach(option => allVariants.add(method + "/" + option.value));
  });
  const reachable = new Set(), targetState = new Map();
  const rightBoundary=Array.from(node("multiphysics-boundaries-list").children).find(row=>row.querySelector("select")&&row.querySelector("select").value==="right");
  const rightValue=rightBoundary&&Array.from(rightBoundary.querySelectorAll("input")).find(input=>input.value==="1");
  if(rightValue){rightValue.value="0";rightValue.dispatchEvent(new window.Event("input",{bubbles:true}));}
  const problemStates = [
    {problem: "custom_equation_template", analysis: "stationary", dimension: "1"},
    {problem: "custom_equation_template", analysis: "transient", dimension: "1"},
  ];
  for (const state of problemStates) {
    node("problem").value = state.problem;
    node("problem").dispatchEvent(new window.Event("change", {bubbles: true}));
    if (state.analysis) {
      node("multiphysics-dimension").value=state.dimension;
      node("multiphysics-dimension").dispatchEvent(new window.Event("change",{bubbles:true}));
      const equation=Array.from(node("multiphysics-equations-list").querySelectorAll("input")).find(input=>input.value.includes("dx("));
      if(equation&&equation.value.includes("dy(")){equation.value="dx(dx(u))=0";equation.dispatchEvent(new window.Event("input",{bubbles:true}));}
      node("multiphysics-analysis").value = state.analysis;
      node("multiphysics-analysis").dispatchEvent(new window.Event("change", {bubbles: true}));
    }
    document.querySelectorAll("[data-method-card]").forEach(card => {
      const method = card.dataset.methodCard;
      card.querySelectorAll('[data-role="variant"] option:not(:disabled)').forEach(option => {
        const key = method + "/" + option.value; reachable.add(key); if (!targetState.has(key)) targetState.set(key, state);
      });
    });
  }
  assert.deepEqual([...reachable].sort(), [...allVariants].sort());
  for (const key of [...allVariants].sort()) {
    const [method, variant] = key.split("/");
    const state = targetState.get(key);
    node("problem").value = state.problem;
    node("problem").dispatchEvent(new window.Event("change", {bubbles: true}));
    if (state.analysis) { node("multiphysics-dimension").value=state.dimension;node("multiphysics-dimension").dispatchEvent(new window.Event("change",{bubbles:true}));node("multiphysics-analysis").value = state.analysis; node("multiphysics-analysis").dispatchEvent(new window.Event("change", {bubbles: true})); }
    const baseline = document.querySelector('[data-method-card="baseline"] [data-role="method-toggle"]');
    baseline.checked = true; baseline.dispatchEvent(new window.Event("change", {bubbles: true}));
    const card = document.querySelector('[data-method-card="' + method + '"]');
    const select = card.querySelector('[data-role="variant"]'); select.value = variant;
    const toggle = card.querySelector('[data-role="method-toggle"]');
    if (method === "baseline") select.dispatchEvent(new window.Event("change", {bubbles: true}));
    else { toggle.checked = true; toggle.dispatchEvent(new window.Event("change", {bubbles: true})); }
    assert.doesNotMatch(node("code").textContent, /NotImplementedError/, key);
    assert.match(node("code").textContent, /op\.method\(/, key);
    assert.match(node("code").textContent, new RegExp('[\\\"\']' + method + '[\\\"\']'), key);
    assert.match(node("code").textContent, new RegExp('[\\\"\']' + variant + '[\\\"\']'), key);
  }

  // Operator uses the same five cards; its model and grid-derivative choices
  // are path-specific and appear in the generated script.
  node("problem").value = "custom_equation_template";
  node("problem").dispatchEvent(new window.Event("change", {bubbles: true}));
  node("operator-training-mode").value = "physics_only";
  node("operator-training-mode").dispatchEvent(new window.Event("change", {bubbles: true}));
  assert.equal(document.querySelectorAll(".ecosystem-axis-card").length, 5);
  choose("model", "fno");
  choose("derivative", "grid");
  assert.doesNotMatch(node("code").textContent, /NotImplementedError/);
  assert.match(node("code").textContent, /OPERATOR = 'fno'/);
  assert.match(node("code").textContent, /'derivative': 'grid'/);
  node("extension-kind").value = "loss";
  node("extension-name").value = "my_loss";
  node("extension-module").value = "my_project.methods";
  node("extension-factory").value = "build_loss";
  node("extension-factory").dispatchEvent(new window.Event("input", {bubbles: true}));
  assert.equal(node("copy-extension").disabled, false);
  assert.match(node("extension-code").textContent, /registry\.register\("loss", "my_loss", build_loss\)/);
  assert.match(node("extension-code").textContent, /registry=registry/);
  assert.match(node("extension-code").textContent, /"loss": \{"weighting": "my_loss"\}/);
  node("extension-kind").value = "derivative";
  node("extension-name").value = "my_derivative";
  node("extension-factory").value = "build_derivative";
  node("extension-factory").dispatchEvent(new window.Event("input", {bubbles: true}));
  assert.match(node("extension-code").textContent, /registry\.create\("derivative", "my_derivative"/);
  assert.equal(node("extension-code").textContent.includes("differentiator"), false);
  node("extension-kind").value = "loss";
  node("extension-name").value = "my_loss";
  node("extension-factory").value = "build_loss";
  node("extension-factory").dispatchEvent(new window.Event("input", {bubbles: true}));
  node("copy-extension").click();
  await Promise.resolve();
  assert.equal(copied, node("extension-code").textContent);

  node("extension-module").value = "bad/module";
  node("extension-module").dispatchEvent(new window.Event("input", {bubbles: true}));
  assert.equal(node("copy-extension").disabled, true);
  assert.match(node("extension-status").textContent, /点分模块路径/);
  assert.deepEqual(errors, []);
  dom.window.close();
  console.log("METHOD_ECOSYSTEM_OK: five fully built-in axes, custom registration and validation");
}

main().catch(error => { console.error(error); process.exitCode = 1; });
