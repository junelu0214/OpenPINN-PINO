(function (root, factory) {
  "use strict";
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.OpenPINNJointHandoff = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var STORAGE_KEY = "openpinn-joint-handoff-v1";
  function object(value) { return value && typeof value === "object" && !Array.isArray(value); }
  function clone(value) { return value == null ? value : JSON.parse(JSON.stringify(value)); }
  function source(state, kind, options) {
    options = options || {};
    state = object(state) ? state : {};
    var present = Boolean(object(state) && object(kind === "understanding" ? state.draft : state.method_draft));
    return {
      present: present,
      source_kind: kind === "understanding" ? "understanding_trial" : "solver_planning",
      input_text: kind === "understanding" ? String(options.inputText || "") || null : undefined,
      draft: present ? clone(kind === "understanding" ? state.draft : state.method_draft) : null,
      rationale: kind === "planning" && Array.isArray(state.rationale) ? clone(state.rationale) : [],
      missing: kind === "understanding" && Array.isArray(state.missing) ? clone(state.missing) : [],
      warnings: present && Array.isArray(state.warnings) ? clone(state.warnings) : [],
      ready_for_validation: present && state.ready_for_validation === true,
      model: String(options.model || "") || null,
    };
  }
  function issue(code, path, severity, message) {
    return {code: code, path: path, severity: severity, message: message,
      blocking_import: false, blocking_export: false};
  }
  function build(understandingState, planningState, options) {
    options = options || {};
    var understanding = source(understandingState, "understanding", {
      inputText: options.inputText, model: options.understandingModel,
    });
    var planning = source(planningState, "planning", {model: options.planningModel});
    var issues = [], unmapped = [], conflicts = [];
    if (!understanding.present) issues.push(issue("handoff.missing_understanding", "sources.understanding", "warning", "缺少 Understanding 问题草稿；已保留 Planning 的部分导入。"));
    if (!planning.present) issues.push(issue("handoff.missing_planning", "sources.planning", "warning", "缺少 Planning 方法草稿；已保留 Understanding 的部分导入。"));
    (understanding.missing || []).forEach(function (path) {
      unmapped.push({source: "understanding", path: String(path), reason: "Understanding 标记为缺失。"});
    });
    (understanding.warnings || []).forEach(function (warning, index) {
      unmapped.push({source: "understanding", path: "warnings." + index, reason: typeof warning === "string" ? warning : JSON.stringify(warning)});
    });
    var problem = understanding.present ? clone(understanding.draft) : {};
    var method = planning.present ? clone(planning.draft) : {};
    var methodFields = ["method", "variant", "path", "training", "sampling", "loss", "optimizer", "network", "differentiator"];
    if (understanding.present && planning.present) methodFields.forEach(function (name) {
      if (Object.prototype.hasOwnProperty.call(understanding.draft, name) &&
          JSON.stringify(understanding.draft[name]) !== JSON.stringify(planning.draft[name])) {
        conflicts.push({path: "method." + name, understanding_value: clone(understanding.draft[name]),
          planning_value: clone(planning.draft[name]), default_used: "planning",
          message: "问题草稿与方法草稿都提供了该方法字段；默认采用 Planning，两个值均保留在 envelope 中。"});
      }
    });
    conflicts.forEach(function (entry) { issues.push(issue("handoff.conflict", entry.path, "warning", entry.message)); });
    var problemComplete = understanding.present && understanding.missing.length === 0;
    var methodComplete = planning.present && planning.ready_for_validation;
    return {
      schema_version: 1, kind: "JointAgentHandoff", created_at: new Date().toISOString(),
      sources: {understanding: understanding, planning: planning},
      mapping: {applied: {problem: problem, method: method}, unmapped: unmapped, conflicts: conflicts},
      issues: issues,
      completeness: {problem_fields_complete: problemComplete, method_fields_complete: methodComplete,
        joint_complete: problemComplete && methodComplete},
    };
  }
  function parse(raw) {
    var value = typeof raw === "string" ? JSON.parse(raw) : clone(raw);
    if (!object(value) || value.schema_version !== 1 || value.kind !== "JointAgentHandoff" ||
        !object(value.sources) || !object(value.mapping) || !object(value.mapping.applied)) {
      throw new Error("JointAgentHandoff schema_version=1 is required.");
    }
    return value;
  }
  return {STORAGE_KEY: STORAGE_KEY, build: build, parse: parse};
}));
