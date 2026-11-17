(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) module.exports = factory(require("./generic_validation.js"));
  else root.OpenPINNCodeBuilderAgent = factory(root.OpenPINNGenericValidation);
}(typeof globalThis !== "undefined" ? globalThis : this, function (generic) {
  "use strict";
  if (!generic) throw new Error("OpenPINNGenericValidation must load before CodeBuilder Agent");

  var GENERATOR_ID = "openpinn.generic-training-skeleton.v1";
  var RUNNABLE_GENERATOR_ID = "openpinn.shared-pinn-engine.v1";
  function clone(value) { return JSON.parse(JSON.stringify(value)); }
  function error(code, path, message) { return {code: code, path: path, message: message}; }
  function pythonJson(value) { return JSON.stringify(JSON.stringify(value)); }
  function requiredFields(problem, method) {
    var errors = [];
    if (!problem.physics) errors.push(error("template.missing_physics", "problem.physics", "A physics label is required by the generic skeleton template."));
    if (!Array.isArray(problem.fields) || !problem.fields.length) errors.push(error("template.missing_fields", "problem.fields", "At least one field declaration is required by the generic skeleton template."));
    if (!Array.isArray(problem.equations) || !problem.equations.length) errors.push(error("template.missing_equations", "problem.equations", "At least one equation declaration is required by the generic skeleton template."));
    if (!method || typeof method.method !== "string" || !method.method) errors.push(error("template.missing_method", "method.method", "A method label is required by the generic skeleton template."));
    if (!method || typeof method.path !== "string" || !method.path) errors.push(error("template.missing_path", "method.path", "A method path is required by the generic skeleton template."));
    return errors;
  }
  function renderSkeleton(problem, method, report) {
    var concernSummary = report.semantic_concerns_visible;
    return [
      '"""OpenPINN generic training skeleton.',
      "",
      "Generated deterministically from the validated generic Problem+Method snapshot.",
      "This artifact is an assembly skeleton, not evidence of a runnable backend,",
      'physical correctness, convergence, or successful training.',
      '"""',
      "",
      "import json",
      "from datetime import datetime, timezone",
      "from pathlib import Path",
      "",
      "GENERATOR_ID = " + JSON.stringify(GENERATOR_ID),
      "PAIR_FINGERPRINT = " + JSON.stringify(report.pair_fingerprint),
      "BACKEND_RUNNABLE = False",
      "PROBLEM_DESCRIPTION = json.loads(" + pythonJson(problem) + ")",
      "METHOD_DESCRIPTION = json.loads(" + pythonJson(method) + ")",
      "SEMANTIC_CONCERNS = json.loads(" + pythonJson(concernSummary) + ")",
      'RUN_ARTIFACT_PATH = Path("openpinn_run") / "run_artifact.json"',
      "",
      "def write_run_artifact(status=\"not_runnable\", loss_history=None, error_metrics=None, failure=None):",
      '    """Write the immutable post-run handoff consumed by Diagnostic Agent."""',
      "    losses = list(loss_history or [])",
      "    artifact = {",
      '        "schema_version": 1,',
      '        "kind": "RunArtifact",',
      '        "pair_fingerprint": PAIR_FINGERPRINT,',
      '        "generator_id": GENERATOR_ID,',
      '        "backend_runnable": BACKEND_RUNNABLE,',
      '        "status": status,',
      '        "recorded_at": datetime.now(timezone.utc).isoformat(),',
      '        "metrics": {',
      '            "loss_history": losses,',
      '            "final_loss": losses[-1] if losses else None,',
      '            "error_metrics": dict(error_metrics or {}),',
      "        },",
      '        "failure": failure if failure is not None else {"type": None, "message": None, "stack": None, "truncated": False},',
      '        "semantic_concerns": SEMANTIC_CONCERNS,',
      "    }",
      "    RUN_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)",
      '    RUN_ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")',
      "    return RUN_ARTIFACT_PATH",
      "",
      "def assemble_training_backend():",
      '    """Bind this validated description to a separately implemented backend."""',
      "    raise NotImplementedError(",
      '        "No runnable arbitrary-physics backend is provided by this skeleton. "',
      '        "Implement and validate an explicit backend before training."',
      "    )",
      "",
      "def main():",
      "    artifact_path = write_run_artifact()",
      '    print("Generic CodeBuilder skeleton:", GENERATOR_ID)',
      '    print("Pair fingerprint:", PAIR_FINGERPRINT)',
      '    print("Backend runnable:", BACKEND_RUNNABLE)',
      '    print("RunArtifact:", artifact_path)',
      "    if SEMANTIC_CONCERNS:",
      '        print("Semantic concerns requiring review:")',
      "        for concern in SEMANTIC_CONCERNS:",
      '            print("-", concern.get("path"), concern.get("message"))',
      '    print("No training was started. Implement assemble_training_backend() explicitly.")',
      "",
      'if __name__ == "__main__":',
      "    main()",
      "",
    ].join("\n");
  }
  function renderRunnable(problem, method, report) {
    var training = method && method.training || {};
    var defaultSteps = Number.isInteger(training.steps) && training.steps > 0 ? training.steps : 1000;
    var defaultWall = typeof training.max_wall_seconds === "number" && training.max_wall_seconds > 0 ? training.max_wall_seconds : 300;
    return [
      '\"\"\"Run a registered OpenPINN problem through the shared PINN engine.\"\"\"',
      "import argparse", "import json", "from pathlib import Path", "", "import openpinn as op", "",
      "GENERATOR_ID = " + JSON.stringify(RUNNABLE_GENERATOR_ID),
      "PAIR_FINGERPRINT = " + JSON.stringify(report.pair_fingerprint),
      "BACKEND_RUNNABLE = True",
      "PROBLEM_DESCRIPTION = json.loads(" + pythonJson(problem) + ")",
      "METHOD_DESCRIPTION = json.loads(" + pythonJson(method) + ")",
      "SEMANTIC_CONCERNS = json.loads(" + pythonJson(report.semantic_concerns_visible) + ")", "",
      "def parse_args():", "    parser = argparse.ArgumentParser()",
      "    parser.add_argument('--max-steps', type=int, default=" + defaultSteps + ")",
      "    parser.add_argument('--max-wall-seconds', type=float, default=" + defaultWall + ")",
      "    parser.add_argument('--device', default='cpu')",
      "    parser.add_argument('--output-dir', type=Path, default=Path('openpinn_run'))",
      "    parser.add_argument('--smoke', action='store_true', help='Run exactly one real optimization step.')",
      "    return parser.parse_args()", "",
      "def main():", "    args = parse_args()",
      "    if args.max_steps <= 0 or args.max_wall_seconds <= 0:",
      "        raise SystemExit('--max-steps and --max-wall-seconds must be positive')",
      "    result, artifact = op.run_registered_pinn(",
      "        PROBLEM_DESCRIPTION, METHOD_DESCRIPTION, output_dir=args.output_dir,",
      "        pair_fingerprint=PAIR_FINGERPRINT, max_steps=args.max_steps,",
      "        max_wall_seconds=args.max_wall_seconds, device=args.device, smoke=args.smoke,",
      "        semantic_concerns=SEMANTIC_CONCERNS, generator_id=GENERATOR_ID,",
      "    )",
      "    print('Run status:', result.status)",
      "    print('Steps completed:', result.steps_completed)",
      "    print('RunArtifact:', args.output_dir / 'run_artifact.json')",
      "    print('Measured errors:', json.dumps(artifact['metrics']['error_metrics'], sort_keys=True))",
      "    print('No frozen accuracy threshold is defined; measured errors are not a pass claim.')", "",
      "if __name__ == '__main__':", "    main()", "",
    ].join("\n");
  }
  function build(hybridInput, currentProblem, currentMethod) {
    var hybrid = clone(hybridInput || {}), errors = [], warnings = [];
    var snapshots = hybrid.snapshots || {};
    var problemSource = currentProblem || snapshots.problem;
    var method = clone(currentMethod || snapshots.method || {});
    var problem = null, currentValidation = null;
    try {
      problem = generic.normalizeProblem(problemSource);
      currentValidation = generic.validate(problem, method);
    } catch (failure) {
      errors.push(error("build.invalid_snapshot", "pair", failure.message));
    }
    if (hybrid.kind !== "HybridValidationReport") errors.push(error("build.missing_hybrid_report", "validation", "A HybridValidationReport is required."));
    if (currentValidation && hybrid.pair_fingerprint !== currentValidation.pair_fingerprint) {
      errors.push(error("build.stale_fingerprint", "validation.pair_fingerprint", "The validated fingerprint does not match the current Problem+Method snapshot."));
    }
    if (!hybrid.deterministic || hybrid.deterministic.summary && hybrid.deterministic.summary.has_fail) {
      errors.push(error("build.deterministic_fail", "validation.deterministic", "Deterministic Validation failures block skeleton generation."));
    }
    if (currentValidation && currentValidation.summary.has_fail) {
      errors.push(error("build.current_deterministic_fail", "pair", "The current final pair fails deterministic Validation."));
    }
    if (!hybrid.semantic || hybrid.semantic.state !== "complete" || !hybrid.semantic.report) {
      errors.push(error("build.semantic_incomplete", "validation.semantic", "A complete semantic review is required before training-skeleton export."));
    }
    if (problem) errors = errors.concat(requiredFields(problem, method));
    var semanticReport = hybrid.semantic && hybrid.semantic.report;
    var concerns = semanticReport && semanticReport.review_status === "concerns" ? clone(semanticReport.concerns || []) : [];
    var mappingComplete = !!(problem && problem.trainable_contract && problem.trainable_contract.mapping_complete);
    var capability = problem ? generic.executionCapability(problem, method) : {status: "unsupported", backend_id: null, reason: "No valid snapshot is available."};
    if (concerns.length) warnings.push({code: "build.semantic_concerns", path: "validation.semantic.concerns",
      message: "Codegen is allowed, but semantic concerns must remain visible and require human review."});
    if (capability.status !== "supported") warnings.push({code: "build.backend_not_runnable", path: "backend_runnable",
      message: "No registered executable backend matched; the generic template emits an honest non-runnable skeleton."});
    if (!mappingComplete) warnings.push({code: "build.mapping_incomplete", path: "problem.trainable_contract",
      message: "The Stage A first-class mapping contract is incomplete; this skeleton is not eligible for a later training path."});
    var runnable = errors.length === 0 && mappingComplete && capability.status === "supported";
    var report = {
      schema_version: 1, kind: "BuildReport", ok: errors.length === 0,
      pair_fingerprint: currentValidation ? currentValidation.pair_fingerprint : hybrid.pair_fingerprint || null,
      validation_ref: {kind: hybrid.kind || null, request_id: hybrid.request_id || null,
        request_version: hybrid.request_version || null, semantic_status: semanticReport && semanticReport.review_status || null},
      generator_id: runnable ? RUNNABLE_GENERATOR_ID : GENERATOR_ID,
      artifact_kind: runnable ? "local_training_script" : "generic_training_skeleton",
      mapping_complete_for_later_training: mappingComplete,
      execution_capability: capability, backend_runnable: runnable, errors: errors, warnings: warnings,
      semantic_concerns_visible: concerns,
    };
    return {schema_version: 1, kind: "CodeBuildResult", report: report,
      source: report.ok ? (runnable ? renderRunnable(problem, method, report) : renderSkeleton(problem, method, report)) : null,
      filename: report.ok ? (runnable ? "openpinn_train.py" : "openpinn_generic_training_skeleton.py") : null};
  }
  return {GENERATOR_ID: GENERATOR_ID, RUNNABLE_GENERATOR_ID: RUNNABLE_GENERATOR_ID, build: build};
}));
