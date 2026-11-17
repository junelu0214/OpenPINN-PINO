"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "../..");
const trial = require(path.join(root, "docs/_static/understanding_trial.js"));
const generic = require(path.join(root, "docs/_static/generic_validation.js"));
const hybrid = require(path.join(root, "docs/_static/hybrid_validation.js"));
const diagnostic = require(path.join(root, "docs/_static/diagnostic_agent.js"));

const FIVE_MINUTES_MS = 300000;
const RETIRED_MS = [15000, 30000, 60000, 90000];

function fakeClock() {
  const scheduled = [];
  let next = 1;
  return {
    scheduled,
    scheduleTimeout(callback, delay) {
      const id = next++;
      scheduled.push({id, callback, delay, cleared: false, fired: false});
      return id;
    },
    cancelTimeout(id) {
      const entry = scheduled.find((item) => item.id === id);
      if (entry) entry.cleared = true;
    },
    fire(delay) {
      let fired = 0;
      for (const entry of scheduled) {
        if (!entry.cleared && entry.delay === delay) {
          entry.fired = true;
          entry.callback();
          fired += 1;
        }
      }
      return fired;
    },
    fireAtMost(maxDelay) {
      let fired = 0;
      for (const entry of scheduled) {
        if (!entry.cleared && entry.delay <= maxDelay) {
          entry.fired = true;
          entry.callback();
          fired += 1;
        }
      }
      return fired;
    },
  };
}

function clockOptions(clock) {
  return {scheduleTimeout: clock.scheduleTimeout, cancelTimeout: clock.cancelTimeout};
}

function hang(capture) {
  return (_url, request) => new Promise((_resolve, reject) => {
    capture.signal = request.signal;
    const fail = () => reject(new DOMException("signal is aborted without reason", "AbortError"));
    if (request.signal.aborted) fail();
    else request.signal.addEventListener("abort", fail, {once: true});
  });
}

function jsonResponse(content, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => ({choices: [{message: {content: JSON.stringify(content)}}]}),
  };
}

function assertFiveMinuteBudget(clock) {
  assert.deepEqual(clock.scheduled.map((entry) => entry.delay), [FIVE_MINUTES_MS]);
  assert.equal(RETIRED_MS.some((delay) => clock.scheduled.some((entry) => entry.delay === delay)), false);
}

const understandingDraft = {
  draft: {
    dim: 1,
    physics: "heat_equation",
    geometry: {type: "interval", x_min: 0, x_max: 1},
    field: {kind: "temperature", components: ["u"]},
    pde: {type: "diffusion", diffusivity: 0.01},
    boundaries: [
      {location: "x=0", type: "dirichlet", value: null},
      {location: "x=1", type: "neumann", value: null},
    ],
    initial_condition: {expression: "sin(pi*x/L)"},
  },
  missing: ["boundaries[0].value", "boundaries[1].value", "L"],
  questions: ["What are the two boundary values and L?"],
  warnings: [],
};

const planPayload = {
  method_draft: {
    method: "baseline", variant: "plain_pinn", path: "pinn",
    sampling: {name: "uniform", interior: 512, boundary: 128},
    loss: {formulation: "strong", weighting: "fixed"},
    optimizer: {name: "adam", lr: 0.001},
    network: {name: "mlp", hidden_layers: [64, 64], activation: "tanh"},
    differentiator: {mode: "autodiff"}, training: {steps: 5000},
  },
  rationale: ["二维线弹性使用基线 PINN。"],
  warnings: [],
};

const validationProblem = {
  physics: "novel_reaction_transport", dim: 1,
  field: {kind: "scalar", components: ["u"]},
  geometry: {type: "interval", x_min: 0, x_max: 1},
  parameters: {alpha: {value: 0.1, unit: "m^2/s"}},
  pde: {expression: "alpha*dx(dx(u))"},
  boundaries: [{region: "left", type: "dirichlet", value: 0}],
};
const validationMethod = {path: "new_backend", method: "research_solver", training: {steps: 10}};
const validationPayload = {
  review_status: "no_concern_identified",
  concerns: [],
  unknowns: [],
  information_needed: [],
  assumptions_noted: [{path: "problem.physics", message: "The free-form physics label is treated as user terminology."}],
  method_fit: {status: "unsupported", message: "The method is unregistered; physical validity is not inferred from that."},
};

function diagnosticArtifact() {
  return {
    schema_version: 1, kind: "RunArtifact", pair_fingerprint: "fnv1a32:1234abcd",
    generator_id: "test-generator", backend_runnable: true, status: "completed",
    recorded_at: "2026-11-10T00:00:00Z",
    metrics: {loss_history: [1, 0.2, 0.05], final_loss: 0.05, error_metrics: {relative_l2: 0.03}},
    failure: {type: null, message: null, stack: null, truncated: false},
  };
}
const diagnosticPayload = {
  assessment: "no_additional_concern",
  findings: [],
  repair_hints: [{message: "Inspect sampling coverage.", rationale: "This is a non-binding next check."}],
  human_summary: "The recorded metrics were reviewed; this is diagnostic guidance only.",
};

test("web LLM sources use a five-minute budget and drop the old waits", () => {
  const files = {
    "docs/_static/understanding_trial.js": fs.readFileSync(path.join(root, "docs/_static/understanding_trial.js"), "utf8"),
    "docs/_static/hybrid_validation.js": fs.readFileSync(path.join(root, "docs/_static/hybrid_validation.js"), "utf8"),
    "docs/_static/diagnostic_agent.js": fs.readFileSync(path.join(root, "docs/_static/diagnostic_agent.js"), "utf8"),
  };
  for (const [name, source] of Object.entries(files)) {
    assert.match(source, /var LLM_REQUEST_TIMEOUT_MS = 300000;/, name);
    assert.doesNotMatch(source, /timeout(?:Ms)?\s*(?:\|\||:)\s*(?:15000|30000|60000|90000)\b/, name);
  }
  const trialSource = files["docs/_static/understanding_trial.js"];
  assert.equal(trialSource.match(/options\.timeout \|\| LLM_REQUEST_TIMEOUT_MS/g).length, 2);
  assert.equal(trialSource.match(/timeoutMs: LLM_REQUEST_TIMEOUT_MS/g).length, 2);
});

test("Understanding initial and follow-up waits are five minutes and clear on completion or HTTP failure", async () => {
  for (const answers of [undefined, {"boundaries[0].value": 0}]) {
    const clock = fakeClock();
    const capture = {};
    const report = await trial.understandWithCloud("一维热传导", {
      baseUrl: "https://compatible.example/v1", model: "example-model", apiKey: "browser-key",
      answers, ...clockOptions(clock),
      fetch: async (_url, request) => {
        capture.signal = request.signal;
        return jsonResponse(understandingDraft);
      },
    });
    assert.equal(report.draft.pde.diffusivity, 0.01);
    assertFiveMinuteBudget(clock);
    assert.equal(clock.scheduled[0].cleared, true);
    assert.equal(clock.scheduled[0].fired, false);
    assert.equal(capture.signal.aborted, false);
  }

  const clock = fakeClock();
  const capture = {};
  await assert.rejects(() => trial.understandWithCloud("一维热传导", {
    baseUrl: "https://compatible.example/v1", model: "example-model", apiKey: "browser-key",
    ...clockOptions(clock),
    fetch: async (_url, request) => {
      capture.signal = request.signal;
      return jsonResponse({}, 503);
    },
  }), /HTTP 503/);
  assertFiveMinuteBudget(clock);
  assert.equal(clock.scheduled[0].cleared, true);
  assert.equal(capture.signal.aborted, false);
});

test("Understanding aborts at five minutes and a 15 second mark does not", async () => {
  const clock = fakeClock();
  const capture = {};
  const pending = trial.understandWithCloud("一维热传导", {
    baseUrl: "https://compatible.example/v1", model: "example-model", apiKey: "browser-key",
    ...clockOptions(clock), fetch: hang(capture),
  });
  assertFiveMinuteBudget(clock);
  assert.equal(clock.fireAtMost(15000), 0);
  assert.equal(capture.signal.aborted, false);
  assert.equal(clock.fire(FIVE_MINUTES_MS), 1);
  await assert.rejects(pending, (error) => error.name === "AbortError");
  assert.equal(capture.signal.aborted, true);
  assert.equal(clock.scheduled[0].cleared, true);
});

test("Solver Planning waits five minutes, clears on success or HTTP failure, and aborts at the budget", async () => {
  const successClock = fakeClock();
  const successCapture = {};
  const plan = await trial.planWithCloud({physics: "heat_equation"}, {
    baseUrl: "https://compatible.example/v1", model: "example-model", apiKey: "plan-key", path: "pinn",
    ...clockOptions(successClock),
    fetch: async (_url, request) => {
      successCapture.signal = request.signal;
      return jsonResponse(planPayload);
    },
  });
  assert.equal(plan.method_draft.method, "baseline");
  assertFiveMinuteBudget(successClock);
  assert.equal(successClock.scheduled[0].cleared, true);
  assert.equal(successCapture.signal.aborted, false);

  const failureClock = fakeClock();
  await assert.rejects(() => trial.planWithCloud({physics: "heat_equation"}, {
    baseUrl: "https://compatible.example/v1", model: "example-model", apiKey: "plan-key",
    ...clockOptions(failureClock),
    fetch: async () => jsonResponse({}, 502),
  }), /HTTP 502/);
  assert.equal(failureClock.scheduled[0].cleared, true);
  assert.equal(failureClock.scheduled[0].fired, false);

  const clock = fakeClock();
  const capture = {};
  const pending = trial.planWithCloud({physics: "heat_equation"}, {
    baseUrl: "https://compatible.example/v1", model: "example-model", apiKey: "plan-key",
    ...clockOptions(clock), fetch: hang(capture),
  });
  assert.equal(clock.fireAtMost(90000), 0);
  assert.equal(capture.signal.aborted, false);
  assert.equal(clock.fire(FIVE_MINUTES_MS), 1);
  await assert.rejects(pending, (error) => error.name === "AbortError");
  assert.equal(clock.scheduled[0].cleared, true);
});

test("Hybrid semantic review uses five minutes, keeps external cancel distinct, and clears the timer", async () => {
  const deterministic = generic.validate(validationProblem, validationMethod);
  const problem = generic.normalizeProblem(validationProblem);
  const base = {
    baseUrl: "https://validation.example/v1", model: "m", apiKey: "k",
  };

  const successClock = fakeClock();
  const success = await hybrid.semanticRequest(problem, validationMethod, deterministic, {
    ...base, ...clockOptions(successClock),
    fetch: async (_url, request) => {
      assert.equal(request.signal.aborted, false);
      return jsonResponse(validationPayload);
    },
  }, "success-1");
  assert.equal(success.state, "complete");
  assertFiveMinuteBudget(successClock);
  assert.equal(successClock.scheduled[0].cleared, true);

  const failureClock = fakeClock();
  const failed = await hybrid.semanticRequest(problem, validationMethod, deterministic, {
    ...base, ...clockOptions(failureClock),
    fetch: async () => jsonResponse({}, 500),
  }, "http-1");
  assert.equal(failed.code, "transport.http");
  assert.equal(failureClock.scheduled[0].cleared, true);
  assert.equal(failureClock.scheduled[0].fired, false);

  const cancelClock = fakeClock();
  const controller = new AbortController();
  const cancelCapture = {};
  const cancelling = hybrid.semanticRequest(problem, validationMethod, deterministic, {
    ...base, ...clockOptions(cancelClock), fetch: hang(cancelCapture),
  }, "cancel-1", controller.signal);
  assertFiveMinuteBudget(cancelClock);
  assert.equal(cancelClock.fireAtMost(90000), 0);
  assert.equal(cancelCapture.signal.aborted, false);
  controller.abort();
  assert.equal((await cancelling).code, "transport.cancelled");
  assert.equal(cancelClock.scheduled[0].fired, false);
  assert.equal(cancelClock.scheduled[0].cleared, true);

  const timeoutClock = fakeClock();
  const timingOut = hybrid.semanticRequest(problem, validationMethod, deterministic, {
    ...base, ...clockOptions(timeoutClock), fetch: hang({}),
  }, "timeout-1");
  assert.equal(timeoutClock.fire(FIVE_MINUTES_MS), 1);
  assert.equal((await timingOut).code, "transport.timeout");
  assert.equal(timeoutClock.scheduled[0].cleared, true);
});

test("Diagnostic semantic review uses five minutes and clears the timer on success, HTTP failure, and abort", async () => {
  const artifact = diagnosticArtifact();
  const base = {baseUrl: "https://diagnostic.example/v1", model: "diagnostic-model", apiKey: "k"};

  const successClock = fakeClock();
  const success = await diagnostic.diagnose(artifact, artifact.pair_fingerprint, {
    ...base, ...clockOptions(successClock),
    fetch: async (_url, request) => {
      assert.equal(request.signal.aborted, false);
      return jsonResponse(diagnosticPayload);
    },
  });
  assert.equal(success.llm_track.state, "complete");
  assertFiveMinuteBudget(successClock);
  assert.equal(successClock.scheduled[0].cleared, true);

  const failureClock = fakeClock();
  const failed = await diagnostic.diagnose(artifact, artifact.pair_fingerprint, {
    ...base, ...clockOptions(failureClock),
    fetch: async () => jsonResponse({}, 500),
  });
  assert.equal(failed.llm_track.code, "transport.http");
  assert.equal(failureClock.scheduled[0].cleared, true);
  assert.equal(failureClock.scheduled[0].fired, false);

  const clock = fakeClock();
  const pending = diagnostic.diagnose(artifact, artifact.pair_fingerprint, {
    ...base, ...clockOptions(clock), fetch: hang({}),
  });
  assert.equal(clock.fireAtMost(60000), 0);
  assert.equal(clock.fire(FIVE_MINUTES_MS), 1);
  assert.equal((await pending).llm_track.code, "transport.timeout");
  assert.equal(clock.scheduled[0].cleared, true);
});
