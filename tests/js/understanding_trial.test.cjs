const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {JSDOM} = require("jsdom");

const root = path.resolve(__dirname, "../..");
const fixtures = JSON.parse(fs.readFileSync(
  path.join(root, "tests/fixtures/understanding/cases.json"), "utf8"
));
const source = fs.readFileSync(
  path.join(root, "docs/_static/understanding_trial.js"), "utf8"
);
const genericSource = fs.readFileSync(
  path.join(root, "docs/_static/generic_validation.js"), "utf8"
);
const hybridSource = fs.readFileSync(
  path.join(root, "docs/_static/hybrid_validation.js"), "utf8"
);
const codebuilderSource = fs.readFileSync(
  path.join(root, "docs/_static/codebuilder_agent.js"), "utf8"
);
const diagnosticSource = fs.readFileSync(
  path.join(root, "docs/_static/diagnostic_agent.js"), "utf8"
);
const trial = require(path.join(root, "docs/_static/understanding_trial.js"));

test("shared complete and missing fixtures have the expected report shape", () => {
  for (const name of ["complete_zh", "complete_en"]) {
    const report = trial.understandProblem(fixtures[name]);
    assert.equal(report.ready_for_validation, true);
    assert.deepEqual(report.missing, []);
    assert(Object.keys(report.draft).length > 0);
  }
  const missing = trial.understandProblem(fixtures.missing_zh);
  assert.equal(missing.ready_for_validation, false);
  assert(missing.questions.length > 0);
  const answered = trial.understandProblem(fixtures.missing_zh, {
    answers: {"constitutive.E": 70e9, "constitutive.nu": 0.33}
  });
  assert.equal(answered.draft.constitutive.E, 70e9);
  assert(answered.questions.length > 0);
  const invalid = trial.understandProblem(fixtures.complete_zh, {
    answers: {"geometry.holes": [{type: "circle", center: [0, 0], radius: 2}]}
  });
  assert.equal(invalid.ready_for_validation, false);
  assert(invalid.questions.length > 0);
});

test("cloud API mode sends Bearer auth and fails closed without fallback", async () => {
  let request = null;
  const llmDraft = {
    draft: {
      dim: 1,
      physics: "heat_equation",
      geometry: {type: "interval", x_min: 0, x_max: 1},
      field: {kind: "temperature", components: ["u"]},
      pde: {type: "diffusion", diffusivity: 0.01},
      boundaries: [
        {location: "x=0", type: "dirichlet", value: null},
        {location: "x=1", type: "neumann", value: null}
      ],
      initial_condition: {expression: "sin(pi*x/L)"}
    },
    missing: ["boundaries[0].value", "boundaries[1].value", "L"],
    questions: ["What are the two boundary values and L?"],
    warnings: []
  };
  const report = await trial.understandWithCloud("一维热传导", {
    baseUrl: "https://compatible.example/v1",
    model: "example-model",
    apiKey: "test-browser-key",
    fetch: async (url, options) => {
      request = {url, options};
      return {
        ok: true,
        json: async () => ({choices: [{message: {content: JSON.stringify(llmDraft)}}]})
      };
    }
  });
  assert.equal(request.url, "https://compatible.example/v1/chat/completions");
  assert.equal(request.options.headers.Authorization, "Bearer test-browser-key");
  assert.equal(JSON.parse(request.options.body).model, "example-model");
  assert(!request.options.body.includes("test-browser-key"));
  assert.equal(report.draft.pde.diffusivity, 0.01);
  assert.equal(report.ready_for_validation, false);
  assert(report.questions.length > 0);

  await assert.rejects(() => trial.understandWithCloud(fixtures.complete_zh, {
    baseUrl: "https://compatible.example/v1", model: "example-model",
    apiKey: "test-browser-key",
    fetch: async () => { throw new Error("CORS blocked"); }
  }), /CORS blocked/);
  await assert.rejects(() => trial.understandWithCloud(fixtures.complete_zh, {
    baseUrl: "https://compatible.example/v1", model: "example-model",
    apiKey: "test-browser-key",
    fetch: async () => ({
      ok: true,
      json: async () => ({choices: [{message: {content: '{"draft": {}}'}}]})
    })
  }), /must contain draft, missing, questions, and warnings/);
  await assert.rejects(() => trial.understandWithCloud(fixtures.complete_zh, {
    baseUrl: "https://compatible.example/v1", model: "example-model",
    apiKey: ""
  }), /API Key is required/);

  const planPayload = {
    method_draft: {
      method: "baseline", variant: "plain_pinn", path: "pinn",
      sampling: {name: "uniform", interior: 512, boundary: 128},
      loss: {formulation: "strong", weighting: "fixed"},
      optimizer: {name: "adam", lr: 0.001},
      network: {name: "mlp", hidden_layers: [64, 64], activation: "tanh"},
      differentiator: {mode: "autodiff"}, training: {steps: 5000}
    },
    rationale: ["二维线弹性使用基线 PINN。"], warnings: []
  };
  const plan = await trial.planWithCloud(fixtures.complete_zh, {
    baseUrl: "https://compatible.example/v1", model: "example-model",
    apiKey: "test-browser-key",
    fetch: async (_url, options) => ({
      ok: true,
      json: async () => ({choices: [{message: {content: JSON.stringify(planPayload)}}]})
    })
  });
  assert.equal(plan.ready_for_validation, true);
  assert.equal(plan.method_draft.method, "baseline");
  assert.deepEqual(plan.rationale, ["二维线弹性使用基线 PINN。"]);
  await assert.rejects(() => trial.planWithCloud(fixtures.complete_zh, {
    baseUrl: "https://compatible.example/v1", model: "example-model",
    apiKey: "test-browser-key",
    fetch: async () => ({ok: true, json: async () => ({choices: [{message: {content: '{"method_draft": {"method": "invented"}}'}}]})})
  }), /rationale/);
});

test("planning JSON aliases and omitted optional fields are normalized", () => {
  const plan = trial.validatePlan({
    method_draft: {
      method: "baseline/plain_pinn", model: "mlp", sampler: "uniform",
      derivative: "autodiff", optimizer: "adam", loss: "strong",
      training: 3210, ignored_hint: "safe to drop"
    },
    rationale: "Use the baseline first."
  }, {path: "pinn"});
  assert.equal(plan.ready_for_validation, true);
  assert.equal(plan.method_draft.path, "pinn");
  assert.equal(plan.method_draft.variant, "plain_pinn");
  assert.equal(plan.method_draft.network.name, "mlp");
  assert.equal(plan.method_draft.training.steps, 3210);
  assert.equal("ignored_hint" in plan.method_draft, false);
  assert(plan.warnings.some(warning => warning.code === "planning.browser_normalized"));
});

test("planning replaces a nonnumeric training label with registered defaults", () => {
  const plan = trial.validatePlan({
    method_draft: {method: "baseline/plain_pinn", path: "pinn", training: "fixed"},
    rationale: ["Baseline first."], warnings: []
  });
  assert.equal(plan.ready_for_validation, true);
  assert.equal(plan.method_draft.training.steps, 5000);
});

test("default elasticity API aliases normalize to a ready problem draft", async () => {
  const extraction = {
    draft: {
      dim: 2, physics: "linear_elasticity", formulation: "plane_stress",
      geometry: {type: "rectangle_with_circular_hole", bounds: {x: [0, 2], y: [0, 2]}, hole: {center: [1, 1], radius: 0.25}},
      field: {kind: "displacement", components: ["ux", "uy"]},
      material: {youngs_modulus: 210e9, poisson_ratio: 0.3, units: "Pa"},
      pde: {body_force: [0, 0]},
      boundaries: [
        {region: "left", type: "dirichlet", value: {ux: 0, uy: 0}},
        {region: "right", type: "traction", value: {x: 1e6, y: 0}},
        {region: "top", type: "traction", value: [0, 0]},
        {region: "bottom", type: "traction", value: [0, 0]},
        {region: "hole_boundary", type: "traction", value: [0, 0]},
      ],
    }, missing: ["constitutive", "geometry.outer"], questions: [], warnings: []
  };
  const report = await trial.understandWithCloud("default elasticity", {
    baseUrl: "https://compatible.example/v1", model: "example-model", apiKey: "key",
    fetch: async () => ({ok: true, json: async () => ({choices: [{message: {content: JSON.stringify(extraction)}}]})})
  });
  assert.equal(report.ready_for_validation, true);
  assert.deepEqual(report.missing, []);
  assert.equal(report.draft.geometry.holes[0].type, "circle");
  assert.equal(report.draft.boundaries[4].region, "hole_0");
});

function loadBuilt(relative, fetchImpl) {
  const file = path.join(root, "docs/_build/html", relative);
  const dom = new JSDOM(fs.readFileSync(file, "utf8"), {
    runScripts: "outside-only",
    url: "https://openpinn.local/" + relative
  });
  if (fetchImpl) dom.window.fetch = fetchImpl;
  dom.window.eval(genericSource);
  dom.window.eval(hybridSource);
  dom.window.eval(codebuilderSource);
  dom.window.eval(diagnosticSource);
  dom.window.eval(source);
  dom.window.document.dispatchEvent(new dom.window.Event("DOMContentLoaded"));
  return dom;
}

test("Chinese trial stores and clears the Key with no rules/auto UI", async () => {
  let request = null;
  const complete = trial.understandProblem(fixtures.complete_zh);
  const response = {
    draft: complete.draft, missing: [], questions: [], warnings: []
  };
  const planResponse = {
    method_draft: {
      method: "baseline", variant: "plain_pinn", path: "pinn",
      sampling: {name: "uniform", interior: 512, boundary: 128},
      loss: {formulation: "strong", weighting: "fixed"},
      optimizer: {name: "adam", lr: 0.001},
      network: {name: "mlp", hidden_layers: [64, 64], activation: "tanh"},
      differentiator: {mode: "autodiff"}, training: {steps: 5000}
    },
    rationale: ["二维线弹性使用 MLP PINN。"], warnings: []
  };
  const dom = loadBuilt("understanding_trial.html", async (url, options) => {
    request = {url, options};
    const prompt = JSON.parse(options.body).messages[1].content;
    return {
      ok: true,
      json: async () => ({choices: [{message: {content: JSON.stringify(
        prompt.includes("Plan a solver") ? planResponse : response
      )}}]})
    };
  });
  try {
    const document = dom.window.document;
    const rootNode = document.querySelector("[data-openpinn-understanding-trial]");
    assert(rootNode);
    assert.equal(rootNode.querySelector('[data-role="backend"]'), null);
    assert.doesNotMatch(rootNode.textContent, /确定性规则|自动（|Deterministic rules|Auto \(/);
    const key = rootNode.querySelector('[data-role="api-key"]');
    assert.equal(key.type, "password");
    key.value = "test-browser-key";
    rootNode.querySelector('[data-role="plan-api-key"]').value = "test-planning-key";
    rootNode.querySelector('[data-role="run"]').click();
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    assert.equal(rootNode.querySelector('[data-role="ready"]').dataset.value, "true");
    assert.equal(dom.window.localStorage.getItem("openpinn-understanding-api-key"), "test-browser-key");
    assert.equal(request.options.headers.Authorization, "Bearer test-browser-key");
    const apply = rootNode.querySelector('[data-role="apply"]');
    assert.equal(apply.textContent, "填入配置器");
    apply.addEventListener("click", event => event.preventDefault());
    apply.click();
    const handoff = dom.window.sessionStorage.getItem(trial.configuratorHandoffKey);
    assert(handoff);
    assert.equal(handoff.includes("test-browser-key"), false);
    const planButton = rootNode.querySelector('[data-role="plan"]');
    assert.equal(planButton.disabled, false);
    planButton.click();
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    assert.equal(rootNode.querySelector('[data-role="plan-ready"]').dataset.value, "true");
    assert.equal(JSON.parse(rootNode.querySelector('[data-role="method-draft"]').textContent).method, "baseline");
    assert.match(rootNode.querySelector('[data-role="plan-rationale"]').textContent, /MLP PINN/);
    assert.match(rootNode.querySelector('[data-role="plan-status"]').textContent, /Specification Validation Agent/);
    assert.match(rootNode.querySelector('[data-role="plan-status"]').textContent, /浏览器结构预检/);
    const applyMethod = rootNode.querySelector('[data-role="apply-method"]');
    applyMethod.addEventListener("click", event => event.preventDefault());
    applyMethod.click();
    const methodHandoff = dom.window.sessionStorage.getItem(trial.methodHandoffKey);
    assert(methodHandoff);
    assert.equal(methodHandoff.includes("test-planning-key"), false);
    assert.equal(rootNode.querySelectorAll(
      '[data-role*="download"], [data-role*="fill"], [data-role*="codegen"], [data-role*="train"]'
    ).length, 3);
    assert.match(rootNode.querySelector('[data-role="download-generic"]').textContent, /非训练/);
    assert.equal(rootNode.querySelector('[data-role="validate-layered"]').disabled, false);
    rootNode.querySelector('[data-role="validate-layered"]').click();
    const layered = JSON.parse(rootNode.querySelector('[data-role="validation-report"]').textContent);
    assert.equal(layered.summary.training_export_unlocked, false);
    assert.equal(layered.summary.target_gate, "not_run_in_browser");
    rootNode.querySelector('[data-role="clear-key"]').click();
    assert.equal(key.value, "");
    assert.equal(dom.window.localStorage.getItem("openpinn-understanding-api-key"), null);
    assert.equal(rootNode.querySelector('[data-role="results"]').hidden, false);
    assert.equal(rootNode.querySelector('[data-role="ready"]').dataset.value, "true");
    rootNode.querySelector('[data-role="run"]').click();
    assert.match(rootNode.querySelector('[data-role="input-error"]').textContent, /API Key/);
    key.value = "test-browser-key";
    dom.window.fetch = async () => { throw new Error("CORS blocked"); };
    rootNode.querySelector('[data-role="run"]').click();
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    assert.match(rootNode.querySelector('[data-role="input-error"]').textContent, /Key.*接口地址.*CORS/);
    assert.doesNotMatch(rootNode.querySelector('[data-role="input-error"]').textContent, /方法规划失败/);
    assert.equal(rootNode.querySelector('[data-role="results"]').hidden, false);
    assert.equal(rootNode.querySelector('[data-role="ready"]').dataset.value, "true");
    rootNode.querySelector('[data-role="plan-clear-key"]').click();
    planButton.click();
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    assert.match(rootNode.querySelector('[data-role="plan-status"]').textContent, /API Key/);
    assert.equal(rootNode.querySelector('[data-role="ready"]').dataset.value, "true");
    rootNode.querySelector('[data-role="plan-api-key"]').value = "super-secret-planning-key";
    dom.window.fetch = async () => ({ok: false, status: 401, json: async () => ({})});
    planButton.click();
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    assert.match(rootNode.querySelector('[data-role="plan-status"]').textContent, /401/);
    assert.doesNotMatch(rootNode.querySelector('[data-role="plan-status"]').textContent, /super-secret/);
  } finally {
    dom.window.close();
  }
});

test("unlisted physics completes Understanding, Planning, and scoped layered web Validation", async () => {
  const problem = {
    dim: 1, physics: "magneto_thermo_chemo_nonlinear_v9", kind: "dynamic",
    geometry: {type: "interval", x_min: 0, x_max: 1},
    field: {kind: "temperature", components: ["T"]},
    parameters: {alpha: {value: 0.1, unit: "m^2/s"}},
    pde: {expression: "dt(T)-alpha*dx(dx(T))+T*T*T"},
    boundaries: [
      {location: "x=0", type: "dirichlet", value: 0},
      {location: "x=1", type: "neumann", value: 0},
    ],
    initial_condition: {expression: "sin(pi*x)"},
    opaque_blocks: [{kind: "research_note", text: "retained"}],
  };
  const method = {
    method: "baseline", variant: "plain_pinn", path: "pinn",
    sampling: {name: "uniform", interior: 32, boundary: 8},
    loss: {formulation: "strong", weighting: "fixed"}, optimizer: {name: "adam", lr: 0.001},
    network: {name: "mlp", hidden_layers: [8, 8], activation: "tanh"},
    differentiator: {mode: "autodiff"}, training: {steps: 10},
  };
  const apiCalls = [];
  const dom = loadBuilt("understanding_trial.html", async (_url, options) => {
    const request = JSON.parse(options.body);
    const prompt = request.messages[1].content;
    const semantic = request.messages[0].content.includes("bounded semantic reviewer");
    const diagnostic = request.messages[0].content.includes("post-run diagnostic reviewer");
    apiCalls.push({diagnostic, authorization: options.headers.Authorization});
    const content = diagnostic ? {
      assessment: "no_additional_concern", findings: [],
      repair_hints: [{message: "Review sampling if a runnable backend is added.", rationale: "Non-binding follow-up only."}],
      human_summary: "The skeleton recorded a non-runnable result; no training result is claimed.",
    } : semantic ? {
      review_status: "no_concern_identified", concerns: [], unknowns: [], information_needed: [],
      assumptions_noted: [{path: "problem.physics", message: "Free-form user terminology."}],
      method_fit: {status: "suitable", message: "No method-fit concern identified from the supplied snapshot."},
    } : prompt.includes("Plan a solver") ? {method_draft: method, rationale: ["scoped planning"], warnings: []}
      : {draft: problem, missing: [], questions: [], warnings: []};
    return {ok: true, json: async () => ({choices: [{message: {content: JSON.stringify(content)}}]})};
  });
  try {
    const document = dom.window.document;
    const role = name => document.querySelector('[data-role="' + name + '"]');
    role("text").value = "A one-dimensional dynamic magneto_thermo_chemo_nonlinear_v9 problem for T.";
    role("api-key").value = "mock-key";
    role("run").click();
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    assert.equal(role("ready").dataset.value, "true");
    assert.equal(JSON.parse(role("draft").textContent).physics, problem.physics);
    role("plan-api-key").value = "mock-plan-key";
    role("validation-reuse-planning").checked = true;
    role("plan").click();
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    role("validate-layered").click();
    const report = JSON.parse(role("validation-report").textContent);
    assert.equal(report.layers.find(layer => layer.id === "L2").status, "pass");
    assert.equal(report.layers.find(layer => layer.id === "L4").status, "unknown");
    assert.equal(report.adapter.status, "unsupported");
    assert.equal(report.summary.training_export_unlocked, false);
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    assert.equal(role("download-generic").disabled, false);
    const semanticReport = JSON.parse(role("semantic-report").textContent);
    assert.equal(semanticReport.state, "complete");
    assert.equal(semanticReport.report.review_status, "no_concern_identified");
    assert.match(role("execution-capability").textContent, /incomplete/);
    assert.match(role("validation-status").textContent, /没有已注册的可运行后端/);
    assert.equal(role("build-code").disabled, false);
    role("build-code").click();
    const buildReport = JSON.parse(role("build-report").textContent);
    assert.equal(buildReport.ok, true);
    assert.equal(buildReport.backend_runnable, false);
    assert.equal(buildReport.pair_fingerprint, report.pair_fingerprint);
    assert.equal(role("copy-code").disabled, false);
    assert.equal(role("download-codebuilder").disabled, false);
    assert.match(role("build-status").textContent, /backend_runnable=false/);
    assert.match(role("build-source").textContent, /raise NotImplementedError/);
    assert.doesNotMatch(role("build-source").textContent, /from_specs|linear_elasticity/);
    assert.equal(role("diagnostic-file").disabled, false);
    const runArtifact = {
      schema_version: 1, kind: "RunArtifact", pair_fingerprint: buildReport.pair_fingerprint,
      generator_id: buildReport.generator_id, backend_runnable: false, status: "not_runnable",
      metrics: {loss_history: [], final_loss: null, error_metrics: {}},
      failure: {type: null, message: null, stack: null, truncated: false},
    };
    dom.window.OpenPINNDiagnosticAgent.readArtifactFile = async () => runArtifact;
    Object.defineProperty(role("diagnostic-file"), "files", {configurable: true, value: [{name: "run_artifact.json"}]});
    role("diagnostic-file").dispatchEvent(new dom.window.Event("change"));
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    assert.equal(role("run-diagnostic").disabled, false);
    role("validation-base-url").value = "https://mock.openpinn.local/v1";
    role("validation-model").value = "mock-diagnostic";
    role("validation-api-key").value = "mock-diagnostic-key";
    role("diagnostic-reuse-validation").checked = true;
    role("diagnostic-api-key").value = "independent-key-to-clear";
    dom.window.localStorage.setItem("openpinn-diagnostic-api-key", "independent-key-to-clear");
    role("run-diagnostic").click();
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    await new Promise(resolve => dom.window.setTimeout(resolve, 0));
    const diagnosticReport = JSON.parse(role("diagnostic-report").textContent);
    assert.equal(diagnosticReport.kind, "DiagnosticReport");
    assert.equal(diagnosticReport.deterministic.outcome, "not_runnable");
    assert.equal(diagnosticReport.llm_track.state, "complete");
    assert.equal(diagnosticReport.healthy, false);
    assert.equal(diagnosticReport.repair_performed, false);
    assert.equal(apiCalls.find(call => call.diagnostic).authorization, "Bearer mock-plan-key");
    assert.equal(JSON.stringify(diagnosticReport).includes("mock-diagnostic-key"), false);
    assert.equal(role("download-diagnostic").disabled, false);
    assert.match(role("diagnostic-status").textContent, /repair_performed=false/);
    role("diagnostic-clear-key").click();
    assert.equal(role("diagnostic-api-key").value, "");
    assert.equal(dom.window.localStorage.getItem("openpinn-diagnostic-api-key"), null);
  } finally {
    dom.window.close();
  }
});

test("coordinate boundary expressions are ready and dotted clarification preserves the boundary array", async () => {
  const draft = {
    dim: 2, physics: "scalar_first_order", geometry: {type: "rectangle", x_min: 0, x_max: 1, y_min: 0, y_max: 1},
    field: {name: "q", components: ["q"], unit: "1"},
    pde: {residuals: [{expression: "dx(q)-1=0"}, {expression: "dy(q)-1=0"}]},
    boundaries: [
      {location: "x=0", type: "dirichlet"},
      {location: "x=1", type: "robin", alpha: 1, beta: 1},
      {location: "y=0", type: "dirichlet"},
      {location: "y=1", type: "neumann", value: 1},
    ],
  };
  const report = await trial.understandWithCloud("two dimensional scalar problem", {
    baseUrl: "https://compatible.example/v1", model: "example-model", apiKey: "key",
    answers: {"boundaries.0.value": "y", "boundaries.1.value": "y+2", "boundaries.2.value": "x"},
    fetch: async () => ({ok: true, json: async () => ({choices: [{message: {content: JSON.stringify({
      draft, missing: ["boundaries[0].value", "boundaries[1].value", "boundaries[2].value"],
      questions: ["Provide boundary values."], warnings: [],
    })}}]})}),
  });
  assert.equal(Array.isArray(report.draft.boundaries), true);
  assert.deepEqual(report.draft.boundaries.map(entry => entry.value), ["y", "y+2", "x", 1]);
  assert.equal(report.ready_for_validation, true);
  assert.deepEqual(report.missing, []);
});

test("Stage G page copy exposes registered capability and matchable frozen-card prompts", () => {
  const rst = fs.readFileSync(path.join(root, "docs/understanding_trial.rst"), "utf8");
  assert.match(rst, /-dx\(dx\(u\)\)=pi\*pi\*sin\(pi\*x\)/);
  assert.match(rst, /dt\(u\)-alpha\*dx\(dx\(u\)\)=0/);
  assert.match(rst, /python openpinn_train\.py --max-steps 20 --max-wall-seconds 30 --device cpu/);
  assert.match(source, /可进入 CodeBuilder 生成本地训练脚本（网页不训练）/);
  assert.doesNotMatch(source, /Hybrid Validation 完成；LLM 语义状态不是数学或物理证明。训练导出仍锁定。/);
});

test("trial is discoverable in both language trees and English chrome is localized", () => {
  for (const relative of ["index.html", "method_configurator.html", "en/index.html", "en/method_configurator.html"]) {
    const dom = new JSDOM(fs.readFileSync(path.join(root, "docs/_build/html", relative), "utf8"));
    try {
      assert(dom.window.document.querySelector('a[href*="understanding_trial.html"]'), relative);
    } finally {
      dom.window.close();
    }
  }

  const dom = loadBuilt("en/understanding_trial.html");
  try {
    const document = dom.window.document;
    assert.match(document.querySelector("h1").textContent, /Understanding Web Trial/i);
    assert.equal(document.querySelector('[data-role="run"]').textContent, "Understand");
    assert.equal(document.querySelector('[data-role="clear-key"]').textContent, "Clear Key");
    assert.match(document.querySelector('[data-role="text"]').value, /two-dimensional plane-stress linear elasticity/i);
  } finally {
    dom.window.close();
  }
});
