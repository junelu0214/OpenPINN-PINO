(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.OpenPINNDiagnosticAgent = factory();
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var LLM_REQUEST_TIMEOUT_MS = 300000;
  var STATUSES = ["not_runnable", "completed", "budget_exhausted", "not_converged", "failed", "interrupted", "unknown"];
  var ASSESSMENTS = ["no_additional_concern", "concerns", "insufficient_information"];
  var DISCLAIMER = "Diagnostic output is bounded guidance, not proof of correctness, well-posedness, trainability, convergence, or successful Repair.";
  function object(value) { return value && typeof value === "object" && !Array.isArray(value); }
  function sanitize(value, configuredKey, fieldName) {
    if (fieldName && /api[_-]?key|authorization|access[_-]?token|password|secret/i.test(fieldName)) return "[redacted]";
    if (Array.isArray(value)) return value.map(function (entry) { return sanitize(entry, configuredKey, ""); });
    if (object(value)) {
      var result = {};
      Object.keys(value).forEach(function (key) { result[key] = sanitize(value[key], configuredKey, key); });
      return result;
    }
    if (typeof value === "string") {
      var text = value.replace(/Bearer\s+[^\s,;]+/gi, "Bearer [redacted]");
      return configuredKey ? text.split(String(configuredKey)).join("[redacted]") : text;
    }
    return value;
  }
  function finding(severity, code, path, message) { return {severity: severity, code: code, path: path, message: message}; }
  function finite(value) { return typeof value === "number" && Number.isFinite(value); }
  function redact(message, key) {
    var text = String(message || "Unknown diagnostic error");
    if (key) text = text.split(String(key)).join("[redacted]");
    return text.replace(/Bearer\s+[^\s,;]+/gi, "Bearer [redacted]");
  }
  function endpoint(baseUrl) {
    var url = new URL(String(baseUrl || "").trim());
    if (url.protocol !== "https:" && !(url.protocol === "http:" && ["localhost", "127.0.0.1"].indexOf(url.hostname) >= 0)) {
      throw new Error("Diagnostic API URL must use HTTPS (or localhost HTTP for tests).");
    }
    var path = url.pathname.replace(/\/+$/, "");
    if (!/\/chat\/completions$/.test(path)) path += "/chat/completions";
    url.pathname = path; url.search = ""; url.hash = "";
    return url.toString();
  }
  function metricsSummary(metrics) {
    metrics = object(metrics) ? metrics : {};
    var losses = Array.isArray(metrics.loss_history) ? metrics.loss_history : [];
    var numeric = losses.filter(finite);
    var finalLoss = finite(metrics.final_loss) ? metrics.final_loss : (numeric.length ? numeric[numeric.length - 1] : null);
    var initialLoss = numeric.length ? numeric[0] : null;
    var errors = {};
    if (object(metrics.error_metrics)) Object.keys(metrics.error_metrics).forEach(function (key) {
      if (finite(metrics.error_metrics[key])) errors[key] = metrics.error_metrics[key];
    });
    return {
      loss_samples: numeric.length, initial_loss: initialLoss, final_loss: finalLoss,
      minimum_loss: numeric.length ? Math.min.apply(Math, numeric) : null,
      relative_loss_change: finite(initialLoss) && initialLoss !== 0 && finite(finalLoss) ? (finalLoss - initialLoss) / Math.abs(initialLoss) : null,
      error_metrics: errors,
    };
  }
  function evaluateArtifact(input, expectedFingerprint) {
    var findings = [], artifact = object(input) ? sanitize(input, "", "") : null;
    if (!artifact) findings.push(finding("error", "artifact.not_object", "artifact", "RunArtifact must be a JSON object."));
    if (artifact && (artifact.schema_version !== 1 || artifact.kind !== "RunArtifact")) {
      findings.push(finding("error", "artifact.schema", "artifact", "RunArtifact schema_version=1 and kind=RunArtifact are required."));
    }
    var artifactFingerprint = artifact && (artifact.export_snapshot_fingerprint || artifact.pair_fingerprint);
    if (artifact && (typeof artifactFingerprint !== "string" || !artifactFingerprint)) {
      findings.push(finding("warning", "artifact.fingerprint_missing", "artifact.export_snapshot_fingerprint", "No export snapshot fingerprint was recorded; run evidence remains diagnosable but cannot be matched to a web export."));
    }
    if (artifact && expectedFingerprint && artifactFingerprint && artifactFingerprint !== expectedFingerprint) {
      findings.push(finding("warning", "artifact.stale_fingerprint", "artifact.export_snapshot_fingerprint", "RunArtifact does not match the current web snapshot. Its own status/loss/failure remain usable, but equivalence to the current page is unverified."));
    }
    if (artifact && STATUSES.indexOf(artifact.status) < 0) {
      findings.push(finding("error", "artifact.status", "artifact.status", "RunArtifact status is outside the supported run-state contract."));
    }
    if (artifact && typeof artifact.backend_runnable !== "boolean") {
      findings.push(finding("warning", "artifact.backend_flag", "artifact.backend_runnable", "backend_runnable was not recorded; Diagnostic will rely on status and available run evidence."));
    }
    if (artifact && (typeof artifact.generator_id !== "string" || !artifact.generator_id)) {
      findings.push(finding("warning", "artifact.generator_missing", "artifact.generator_id", "generator_id is missing; this does not invalidate a self-contained user-script run."));
    }
    if (artifact && (!object(artifact.metrics) || !Array.isArray(artifact.metrics.loss_history) ||
        !object(artifact.metrics.error_metrics) || !Object.prototype.hasOwnProperty.call(artifact.metrics, "final_loss"))) {
      findings.push(finding("error", "artifact.metrics_schema", "artifact.metrics", "RunArtifact requires loss_history, final_loss, and error_metrics slots."));
    }
    if (artifact && (!object(artifact.failure) || ["type", "message", "stack", "truncated"].some(function (key) {
      return !Object.prototype.hasOwnProperty.call(artifact.failure, key);
    }))) {
      findings.push(finding("error", "artifact.failure_schema", "artifact.failure", "RunArtifact requires type, message, stack, and truncated failure slots."));
    }
    if (artifact && artifact.backend_runnable === false && artifact.status !== "not_runnable") {
      findings.push(finding("error", "artifact.nonrunnable_status", "artifact.status", "A non-runnable backend cannot claim a completed or failed training run."));
    }
    if (artifact && artifact.backend_runnable === true && artifact.status === "not_runnable") {
      findings.push(finding("error", "artifact.runnable_status", "artifact.status", "A runnable same-run artifact cannot use not_runnable."));
    }
    if (artifact && (artifact.example === true || artifact.fixture === true || ["example", "fixture"].indexOf(artifact.artifact_origin) >= 0)) {
      findings.push(finding("warning", "artifact.non_product_run", "artifact.artifact_origin", "Example and fixture artifacts receive a limited diagnosis and are not evidence of a complete product run."));
    }
    if (artifact && artifact.smoke_only === true) {
      findings.push(finding("warning", "artifact.smoke_only", "artifact.smoke_only", "A smoke run is diagnosable only as a short path check, not a complete evaluation."));
    }
    if (artifact && artifact.code_origin === "user-modified") {
      findings.push(finding("warning", "artifact.user_modified", "artifact.code_origin", "The script was user-modified. Trust recorded run evidence, but do not assume equivalence to the exported web configuration."));
    }
    if (artifact && !object(artifact.exported_pair_snapshot)) {
      findings.push(finding("warning", "artifact.snapshot_missing", "artifact.exported_pair_snapshot", "The exported Problem/Method snapshot is missing; physics/configuration diagnosis is limited."));
    }
    if (artifact && artifact.generator_id === "openpinn.shared-pinn-engine.v1") {
      if (artifact.artifact_origin !== "local_generated_run") findings.push(finding("error", "artifact.origin", "artifact.artifact_origin", "The runnable generator requires a same-run local_generated_run origin."));
      if (typeof artifact.device !== "string" || !artifact.device) findings.push(finding("error", "artifact.device", "artifact.device", "The runnable generator requires its actual device."));
      if (!object(artifact.budgets) || !finite(artifact.budgets.max_steps) || artifact.budgets.max_steps <= 0 || !finite(artifact.budgets.max_wall_seconds) || artifact.budgets.max_wall_seconds <= 0) findings.push(finding("error", "artifact.budgets", "artifact.budgets", "The runnable generator requires positive step and wall-time budgets."));
      if (!Number.isInteger(artifact.steps_completed) || artifact.steps_completed < 0) findings.push(finding("error", "artifact.steps", "artifact.steps_completed", "The runnable generator requires a non-negative integer step count."));
      if (!object(artifact.engine) || artifact.engine.status !== artifact.status) findings.push(finding("error", "artifact.engine", "artifact.engine", "The embedded engine result must come from the same run and match its status."));
    }
    var summary = metricsSummary(artifact && artifact.metrics);
    if (artifact && object(artifact.metrics) && Array.isArray(artifact.metrics.loss_history) &&
        artifact.metrics.loss_history.some(function (value) { return !finite(value); })) {
      findings.push(finding("error", "run.loss_nonfinite", "artifact.metrics.loss_history", "Loss history contains non-finite entries."));
    }
    if (artifact && artifact.status === "not_runnable") {
      findings.push(finding("info", "run.not_runnable", "artifact.status", "No runnable backend was bound; no training result is claimed."));
    }
    if (artifact && artifact.status === "completed") {
      if (!summary.loss_samples || !finite(summary.final_loss)) {
        findings.push(finding("error", "run.completed_loss_missing", "artifact.metrics", "A completed run requires finite loss_history and final_loss data."));
      }
      if (finite(summary.relative_loss_change) && summary.relative_loss_change >= 0) {
        findings.push(finding("warning", "run.loss_not_decreased", "artifact.metrics.loss_history", "Recorded loss did not decrease from its initial value."));
      }
      findings.push(finding("info", "run.accuracy_unverified", "artifact.accuracy_threshold", "The run completed, but no frozen accuracy threshold exists; measured errors do not establish accuracy."));
    }
    if (artifact && ["budget_exhausted", "not_converged", "interrupted", "unknown"].indexOf(artifact.status) >= 0) {
      findings.push(finding(artifact.status === "interrupted" ? "warning" : "info", "run." + artifact.status,
        "artifact.status", "The run ended with status " + artifact.status + "; this is not an accuracy or convergence claim."));
    }
    if (artifact && artifact.status === "failed") {
      if (!object(artifact.failure) || typeof artifact.failure.message !== "string" || !artifact.failure.message) {
        findings.push(finding("error", "run.failure_missing", "artifact.failure", "A failed run requires a failure message."));
      } else {
        findings.push(finding("warning", "run.failed", "artifact.failure.message", "The run recorded a failure: " + artifact.failure.message));
        if (!artifact.failure.stack) findings.push(finding("warning", "run.stack_missing", "artifact.failure.stack", "No failure stack was recorded."));
        else if (artifact.failure.truncated === true) findings.push(finding("warning", "run.stack_truncated", "artifact.failure.stack", "The recorded failure stack is truncated."));
      }
    }
    var errors = findings.filter(function (entry) { return entry.severity === "error"; });
    var warnings = findings.filter(function (entry) { return entry.severity === "warning"; });
    var outcome = errors.length ? "invalid_artifact" : artifact && artifact.status === "not_runnable" ? "not_runnable" :
      artifact && artifact.status === "failed" ? "failed" : artifact && artifact.status === "completed" ? "completed_unverified" :
      artifact && ["budget_exhausted", "not_converged", "interrupted"].indexOf(artifact.status) >= 0 ? artifact.status : warnings.length ? "concerns" : "unverified";
    return {schema_version: 1, kind: "DeterministicDiagnosticReport", valid: !errors.length,
      healthy: false, outcome: outcome,
      pair_fingerprint: artifactFingerprint || null, metrics_summary: summary,
      findings: findings, disclaimer: DISCLAIMER};
  }
  function incomplete(code, message, model) {
    return {state: "incomplete", code: code, message: message, model: model || "", disclaimer: DISCLAIMER};
  }
  function exactKeys(value, allowed, label) {
    Object.keys(value).forEach(function (key) { if (allowed.indexOf(key) < 0) throw new Error(label + " contains unsupported field " + key + "."); });
  }
  function semanticPayload(payload, model) {
    if (!object(payload)) throw new Error("Diagnostic response must be a JSON object.");
    if (/\b(proves? correctness|proven correct|guarantees? success|is guaranteed|definitely correct|will succeed)\b|已证明|保证成功|一定成功/i.test(JSON.stringify(payload))) {
      throw new Error("Diagnostic response contains a prohibited proof or guaranteed-success claim.");
    }
    exactKeys(payload, ["assessment", "findings", "repair_hints", "human_summary"], "Diagnostic response");
    if (ASSESSMENTS.indexOf(payload.assessment) < 0) throw new Error("Diagnostic assessment is invalid.");
    if (!Array.isArray(payload.findings) || !Array.isArray(payload.repair_hints)) throw new Error("Diagnostic findings and repair_hints must be arrays.");
    var findings = payload.findings.map(function (entry, index) {
      if (!object(entry)) throw new Error("findings[" + index + "] must be an object.");
      exactKeys(entry, ["code", "severity", "path", "message"], "findings[" + index + "]");
      if (["info", "warning", "error"].indexOf(entry.severity) < 0) throw new Error("Diagnostic finding severity is invalid.");
      ["code", "path", "message"].forEach(function (key) { if (typeof entry[key] !== "string" || !entry[key].trim()) throw new Error("Diagnostic finding " + key + " is required."); });
      return {code: entry.code.trim(), severity: entry.severity, path: entry.path.trim(), message: entry.message.trim()};
    });
    var hints = payload.repair_hints.map(function (entry, index) {
      if (!object(entry)) throw new Error("repair_hints[" + index + "] must be an object.");
      exactKeys(entry, ["message", "rationale"], "repair_hints[" + index + "]");
      if (typeof entry.message !== "string" || !entry.message.trim()) throw new Error("repair hint message is required.");
      return {message: entry.message.trim(), rationale: typeof entry.rationale === "string" ? entry.rationale.trim() : "", binding: false};
    });
    if (typeof payload.human_summary !== "string" || !payload.human_summary.trim()) throw new Error("human_summary is required.");
    return {state: "complete", report: {schema_version: 1, kind: "SemanticDiagnosticReport", assessment: payload.assessment,
      findings: findings, repair_hints: hints, human_summary: payload.human_summary.trim(), model: model, disclaimer: DISCLAIMER}};
  }
  function semanticRequest(artifact, deterministic, options) {
    options = options || {};
    var apiKey = String(options.apiKey || "").trim(), model = String(options.model || "").trim();
    if (!apiKey) return Promise.resolve(incomplete("configuration.missing_key", "Diagnostic API Key is required; no rules fallback was used.", model));
    if (!model) return Promise.resolve(incomplete("configuration.missing_model", "Diagnostic model is required; no rules fallback was used.", model));
    var fetchImpl = options.fetch || (typeof fetch === "function" ? fetch : null), url;
    if (!fetchImpl) return Promise.resolve(incomplete("transport.unavailable", "fetch() is unavailable; semantic diagnosis was not run.", model));
    try { url = endpoint(options.baseUrl); }
    catch (failure) { return Promise.resolve(incomplete("configuration.endpoint", redact(failure.message, apiKey), model)); }
    var prompt = [
      "Diagnose this immutable RunArtifact and deterministic report. Do not rewrite specs, code, or hyperparameters.",
      "Return exactly {assessment, findings, repair_hints, human_summary} as JSON.",
      "assessment is no_additional_concern, concerns, or insufficient_information.",
      "Each finding is exactly {code, severity, path, message}; severity is info, warning, or error.",
      "Each repair hint is exactly {message, rationale}, is non-binding, and must not claim success.",
      "Do not claim proof, correctness, trainability, convergence, or guaranteed Repair.",
      JSON.stringify({artifact: artifact, deterministic: deterministic}),
    ].join("\n");
    var scheduleTimeout = typeof options.scheduleTimeout === "function" ? options.scheduleTimeout : setTimeout;
    var cancelTimeout = typeof options.cancelTimeout === "function" ? options.cancelTimeout : clearTimeout;
    var controller = typeof AbortController === "function" ? new AbortController() : null;
    var timedOut = false, timer = null, timeoutMs = Number(options.timeoutMs || LLM_REQUEST_TIMEOUT_MS);
    if (controller && Number.isFinite(timeoutMs) && timeoutMs > 0) timer = scheduleTimeout(function () { timedOut = true; controller.abort(); }, timeoutMs);
    return fetchImpl(url, {method: "POST", headers: {"Content-Type": "application/json", "Accept": "application/json", "Authorization": "Bearer " + apiKey},
      body: JSON.stringify({model: model, temperature: 0, response_format: {type: "json_object"}, messages: [
        {role: "system", content: "You are a bounded post-run diagnostic reviewer. JSON only. Diagnose without mutating inputs or promising Repair."},
        {role: "user", content: prompt},
      ]}), signal: controller && controller.signal}).then(function (response) {
        if (!response.ok) throw {code: response.status === 401 || response.status === 403 ? "transport.auth" : "transport.http", message: "Diagnostic API returned HTTP " + response.status + "."};
        return response.json().catch(function () { throw {code: "schema.envelope", message: "Diagnostic API response is not JSON."}; });
      }).then(function (envelope) {
        var content;
        try { content = envelope.choices[0].message.content; } catch (_error) { content = null; }
        if (typeof content !== "string") throw {code: "schema.envelope", message: "Diagnostic API response has an invalid chat envelope."};
        var parsed;
        try { parsed = JSON.parse(content); } catch (_error) { throw {code: "schema.content", message: "Diagnostic response content is not JSON."}; }
        try { return semanticPayload(parsed, model); }
        catch (failure) { throw {code: "schema.semantic", message: failure.message}; }
      }).catch(function (failure) {
        var code = failure && failure.name === "AbortError" ? (timedOut ? "transport.timeout" : "transport.cancelled") : failure.code || "transport.network";
        return incomplete(code, redact(failure && failure.message, apiKey), model);
      }).finally(function () { if (timer) cancelTimeout(timer); });
  }
  function diagnose(artifactInput, expectedFingerprint, options) {
    var artifact = object(artifactInput) ? sanitize(artifactInput, options && options.apiKey, "") : artifactInput;
    var deterministic = evaluateArtifact(artifact, expectedFingerprint);
    var blocked = !deterministic.valid;
    var semantic = blocked ? Promise.resolve(incomplete("deterministic.blocked", "Semantic diagnosis was not called because the RunArtifact is invalid or stale.", options && options.model)) : semanticRequest(artifact, deterministic, options);
    return semantic.then(function (llmTrack) {
      var semanticHealthy = llmTrack.state === "complete" && llmTrack.report.assessment === "no_additional_concern";
      var healthy = deterministic.healthy && semanticHealthy;
      var summary = llmTrack.state === "complete" ? llmTrack.report.human_summary :
        (deterministic.outcome === "not_runnable" ? "No runnable backend was bound; this artifact records that no training occurred." :
          deterministic.outcome === "invalid_artifact" ? "The RunArtifact schema is invalid; stale or modified snapshots alone are non-blocking warnings." :
          "Deterministic diagnosis completed, but semantic diagnosis is incomplete.");
      return {schema_version: 1, kind: "DiagnosticReport", pair_fingerprint: deterministic.pair_fingerprint,
        artifact_ref: {kind: artifact && artifact.kind || null, status: artifact && artifact.status || null},
        healthy: healthy, deterministic: deterministic, llm_track: llmTrack, human_summary: summary,
        repair_performed: false, disclaimer: DISCLAIMER};
    });
  }
  function htmlEscape(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g,function(character){return ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[character];});
  }
  function formatMetric(value) {
    if (!finite(value)) return "—";
    var magnitude=Math.abs(value);
    return magnitude!==0&&(magnitude>=1e4||magnitude<1e-3)?value.toExponential(3):String(Number(value.toPrecision(5)));
  }
  function reportSuggestions(report, artifact) {
    var deterministic=report&&report.deterministic||{}, summary=deterministic.metrics_summary||{}, findings=deterministic.findings||[];
    var codes=findings.map(function(item){return item.code;}), suggestions=[];
    function add(method,title,action,reason){if(!suggestions.some(function(item){return item.method===method;}))suggestions.push({method:method,title:title,action:action,reason:reason});}
    if(["budget_exhausted","not_converged"].includes(deterministic.outcome)||codes.includes("run.loss_not_decreased")){
      add("Adam → L-BFGS","尝试分阶段优化","先用 Adam 建立稳定下降，再切换 L-BFGS 做全批量精修。","适合损失下降缓慢、预算耗尽或进入平台区的情况。");
      add("RAR / RAD","增加残差自适应采样","在 Sampling 模块选择 RAR 或 RAD，并提高候选池规模。","把新增配点集中到当前 PDE 残差较大的区域，孔边应力集中问题通常尤其受益。");
      add("Gradient balanced / NTK balanced","启用动态损失平衡","在 Loss 模块尝试梯度平衡或 NTK 平衡，并核对边界项权重。","PDE、固定边界和牵引边界梯度尺度差异可能造成单项主导。");
    }
    if(codes.includes("run.loss_nonfinite")||deterministic.outcome==="failed"){
      add("Adam（较小学习率）","先恢复数值稳定性","将学习率降低 5–10 倍，检查坐标、材料量和输出的无量纲化。","非有限损失或运行失败应先排除尺度和梯度爆炸，再增加模型复杂度。");
    }
    var relative=summary.relative_loss_change;
    if(finite(relative)&&relative>-0.9){add("Modified MLP / Fourier features","提高表达能力","先尝试 Modified MLP；若解具有局部高频变化，再添加 Fourier features。","当前损失降幅有限时，门控网络或谱输入表示可能比单纯增加训练步数更有效。");}
    var errors=summary.error_metrics||{}, errorValues=Object.keys(errors).map(function(key){return errors[key];}).filter(finite);
    if(errorValues.some(function(value){return value>0.05;}))add("参考数据 + 自适应采样","针对高误差区域复核","上传独立参考场，定位误差峰值，再增加孔边和载荷边附近采样。","总损失较低也不代表场误差均匀；参考误差图比单一 loss 更能指导下一轮。");
    if(deterministic.outcome==="completed_unverified")add("独立参考场评估","补充可验证的误差指标","在配置生成器上传未参与训练的参考 CSV，比较 relative L2、最大误差和关键区域误差。","训练完成只说明流程结束，不能单独证明解的精度。");
    if(deterministic.outcome==="not_runnable"||deterministic.outcome==="invalid_artifact")add("修复运行契约","先处理阻断项","按错误 findings 修复执行适配器或 RunArtifact 字段，再重新运行诊断。","当前证据不足以比较方法优劣，直接调参可能掩盖配置问题。");
    var semantic=report&&report.llm_track&&report.llm_track.state==="complete"&&report.llm_track.report;
    (semantic&&semantic.repair_hints||[]).forEach(function(hint,index){add("semantic-"+index,"语义诊断建议 "+(index+1),hint.message||"检查语义诊断指出的问题。",hint.rationale||"来自 Diagnostic Agent 的非绑定建议。");});
    if(!suggestions.length)add("基线复核","保持当前方法并增加独立验证","复现实验种子，上传参考场，并比较至少三个随机种子的误差分布。","现有证据未指出明确的单点故障，先补足可重复性证据。");
    return suggestions.slice(0,6);
  }
  function lossChartSvg(artifact) {
    var history=artifact&&artifact.metrics&&Array.isArray(artifact.metrics.loss_history)?artifact.metrics.loss_history.filter(finite):[];
    if(!history.length)return '<div class="diagnostic-empty-chart">RunArtifact 中没有可绘制的有限 loss 历史。</div>';
    var sampled=history;if(history.length>240){sampled=[];for(var s=0;s<240;s+=1)sampled.push(history[Math.round(s*(history.length-1)/239)]);}
    var transformed=sampled.map(function(value){return value>0?Math.log10(value):value;}),minimum=Math.min.apply(Math,transformed),maximum=Math.max.apply(Math,transformed);if(maximum===minimum){maximum+=1;minimum-=1;}
    var left=54,top=20,width=626,height=190,points=transformed.map(function(value,index){var x=left+(sampled.length===1?0:index/(sampled.length-1))*width,y=top+(maximum-value)/(maximum-minimum)*height;return x.toFixed(1)+","+y.toFixed(1);}).join(" ");
    return '<svg class="diagnostic-loss-chart" viewBox="0 0 720 250" role="img" aria-label="训练损失对数曲线"><rect x="0" y="0" width="720" height="250" rx="12" fill="#f8fbfd"/><line x1="54" y1="20" x2="54" y2="210" stroke="#78909c"/><line x1="54" y1="210" x2="680" y2="210" stroke="#78909c"/><polyline fill="none" stroke="#1769d2" stroke-width="3" points="'+points+'"/><circle cx="'+points.split(" ").slice(-1)[0].split(",")[0]+'" cy="'+points.split(" ").slice(-1)[0].split(",")[1]+'" r="4" fill="#d13b2f"/><text x="12" y="28" font-size="12" fill="#526b7d">10^'+formatMetric(maximum)+'</text><text x="12" y="210" font-size="12" fill="#526b7d">10^'+formatMetric(minimum)+'</text><text x="54" y="232" font-size="12" fill="#526b7d">step 1</text><text x="680" y="232" text-anchor="end" font-size="12" fill="#526b7d">step '+history.length+'</text><text x="360" y="246" text-anchor="middle" font-size="12" fill="#526b7d">loss（log10 纵轴）</text></svg>';
  }
  function reportBodyHtml(report,artifact) {
    var deterministic=report&&report.deterministic||{}, summary=deterministic.metrics_summary||{}, semantic=report&&report.llm_track||{};
    var outcomeLabels={completed_unverified:"训练完成 · 精度待验证",budget_exhausted:"预算耗尽",not_converged:"尚未收敛",failed:"运行失败",not_runnable:"不可运行",invalid_artifact:"报告格式无效",concerns:"存在关注项",unverified:"尚未验证"};
    var findings=(deterministic.findings||[]).concat(semantic.state==="complete"?(semantic.report.findings||[]):[]), suggestions=reportSuggestions(report,artifact), errors=summary.error_metrics||{};
    var metrics=[["初始 loss",summary.initial_loss],["最终 loss",summary.final_loss],["最小 loss",summary.minimum_loss],["相对变化",summary.relative_loss_change]].concat(Object.keys(errors).map(function(key){return [key,errors[key]];}));
    return '<div class="diagnostic-report-heading"><div><span class="diagnostic-outcome is-'+htmlEscape(deterministic.outcome||"unknown")+'">'+htmlEscape(outcomeLabels[deterministic.outcome]||deterministic.outcome||"未知")+'</span><h3>Diagnostic Evaluation 报告</h3><p>'+htmlEscape(report&&report.human_summary||"诊断已完成。")+'</p></div><div class="diagnostic-health">'+(report&&report.healthy?"未发现额外关注项":"需要人工复核")+'</div></div>'+
      '<section><h4>关键指标</h4><div class="diagnostic-metric-grid">'+metrics.map(function(item){return '<div><span>'+htmlEscape(item[0])+'</span><strong>'+htmlEscape(formatMetric(item[1]))+'</strong></div>';}).join("")+'</div></section>'+
      '<section><h4>训练过程可视化</h4>'+lossChartSvg(artifact)+'</section>'+
      '<section><h4>诊断发现</h4><div class="diagnostic-findings">'+(findings.length?findings.map(function(item){return '<article class="is-'+htmlEscape(item.severity||"info")+'"><strong>'+htmlEscape((item.severity||"info").toUpperCase())+' · '+htmlEscape(item.code||"finding")+'</strong><p>'+htmlEscape(item.message||"")+'</p><code>'+htmlEscape(item.path||"")+'</code></article>';}).join(""):'<p>没有结构化 finding；这不构成正确性证明。</p>')+'</div></section>'+
      '<section><h4>下一轮建议</h4><div class="diagnostic-suggestions">'+suggestions.map(function(item){return '<article><span>'+htmlEscape(item.method)+'</span><h5>'+htmlEscape(item.title)+'</h5><p>'+htmlEscape(item.action)+'</p><small>'+htmlEscape(item.reason)+'</small></article>';}).join("")+'</div></section>'+
      '<section class="diagnostic-disclaimer"><strong>解释边界</strong><p>'+htmlEscape(report&&report.disclaimer||DISCLAIMER)+'</p><p>这些建议不会自动修改配置，也不保证收敛或精度；请用独立参考数据和重复实验验证。</p></section>';
  }
  function reportHtml(report,artifact) {
    var styles='body{margin:0;background:#eef4f7;color:#173248;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1050px;margin:2rem auto;padding:1.5rem;background:white;border-radius:16px;box-shadow:0 8px 30px #17324820}h1{margin-top:0}.diagnostic-report-heading{display:flex;justify-content:space-between;gap:1rem;border-bottom:1px solid #d7e2e9}.diagnostic-outcome{display:inline-block;padding:.3rem .65rem;border-radius:999px;background:#fff0c7;font-weight:800}.diagnostic-health{font-weight:800}.diagnostic-metric-grid,.diagnostic-suggestions{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.7rem}.diagnostic-metric-grid div,.diagnostic-suggestions article,.diagnostic-findings article{padding:.8rem;border:1px solid #d7e2e9;border-radius:10px}.diagnostic-metric-grid span{display:block;color:#596d7a}.diagnostic-metric-grid strong{font-size:1.35rem}.diagnostic-loss-chart{width:100%;height:auto}.diagnostic-findings{display:grid;gap:.55rem}.diagnostic-findings article{border-left:5px solid #4c8cb5}.diagnostic-findings article.is-warning{border-left-color:#d68a19}.diagnostic-findings article.is-error{border-left-color:#bd352f}.diagnostic-findings p{margin:.3rem 0}.diagnostic-suggestions span{font-size:.78rem;color:#1769d2;font-weight:800}.diagnostic-suggestions h5{margin:.35rem 0;font-size:1rem}.diagnostic-suggestions p{margin:.25rem 0}.diagnostic-disclaimer{margin-top:1rem;padding:.8rem;background:#f5f7fa;border-radius:10px}details{margin-top:1rem}pre{overflow:auto;background:#122433;color:#edf6fb;padding:1rem;border-radius:8px}@media(max-width:600px){main{margin:0;padding:1rem;border-radius:0}.diagnostic-report-heading{display:block}}';
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenPINN Diagnostic Evaluation</title><style>'+styles+'</style></head><body><main><h1>OpenPINN 诊断评估</h1>'+reportBodyHtml(report,artifact)+'<details><summary>查看原始 DiagnosticReport JSON</summary><pre>'+htmlEscape(JSON.stringify(report,null,2))+'</pre></details></main></body></html>';
  }
  function renderReport(container,report,artifact){if(!container)return;container.innerHTML=reportBodyHtml(report,artifact);}
  function uint16(view, offset) { return view.getUint16(offset, true); }
  function uint32(view, offset) { return view.getUint32(offset, true); }
  var MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
  var MAX_ARTIFACT_BYTES = 2 * 1024 * 1024;
  var MAX_ZIP_ENTRIES = 256;
  async function unzipArtifact(buffer) {
    if (buffer.byteLength > MAX_UPLOAD_BYTES) throw new Error("RunArtifact upload exceeds the 10 MB safety limit.");
    var bytes = new Uint8Array(buffer), view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), eocd = -1;
    for (var cursor = bytes.length - 22; cursor >= Math.max(0, bytes.length - 65557); cursor -= 1) {
      if (uint32(view, cursor) === 0x06054b50) { eocd = cursor; break; }
    }
    if (eocd < 0) throw new Error("ZIP end-of-directory record was not found.");
    var entries = uint16(view, eocd + 10), offset = uint32(view, eocd + 16), decoder = new TextDecoder(), selected = null;
    if (entries > MAX_ZIP_ENTRIES) throw new Error("ZIP contains too many entries.");
    for (var index = 0; index < entries; index += 1) {
      if (uint32(view, offset) !== 0x02014b50) throw new Error("ZIP central directory is malformed.");
      var method = uint16(view, offset + 10), compressedSize = uint32(view, offset + 20), uncompressedSize = uint32(view, offset + 24);
      var nameLength = uint16(view, offset + 28), extraLength = uint16(view, offset + 30), commentLength = uint16(view, offset + 32);
      var localOffset = uint32(view, offset + 42), name = decoder.decode(bytes.slice(offset + 46, offset + 46 + nameLength));
      if ((name === "run_artifact.json" || /\/run_artifact\.json$/.test(name)) && !selected) selected = {method: method, size: compressedSize, uncompressedSize: uncompressedSize, offset: localOffset, name: name};
      offset += 46 + nameLength + extraLength + commentLength;
    }
    if (!selected) throw new Error("ZIP does not contain run_artifact.json.");
    if (selected.size > MAX_UPLOAD_BYTES || selected.uncompressedSize > MAX_ARTIFACT_BYTES) throw new Error("run_artifact.json exceeds the 2 MB safety limit.");
    if (uint32(view, selected.offset) !== 0x04034b50) throw new Error("ZIP local entry is malformed.");
    var localName = uint16(view, selected.offset + 26), localExtra = uint16(view, selected.offset + 28);
    var start = selected.offset + 30 + localName + localExtra, compressed = bytes.slice(start, start + selected.size), output;
    if (selected.method === 0) output = compressed;
    else if (selected.method === 8 && typeof DecompressionStream === "function") {
      output = new Uint8Array(await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream("deflate-raw"))).arrayBuffer());
    } else throw new Error("ZIP compression method is unsupported in this browser.");
    if (output.byteLength > MAX_ARTIFACT_BYTES) throw new Error("run_artifact.json exceeds the 2 MB safety limit.");
    return JSON.parse(decoder.decode(output));
  }
  async function readArtifactFile(file) {
    if (!file) throw new Error("Choose a RunArtifact JSON or ZIP file.");
    if (Number(file.size) > MAX_UPLOAD_BYTES) throw new Error("RunArtifact upload exceeds the 10 MB safety limit.");
    var name = String(file.name || "").toLowerCase(), buffer = await file.arrayBuffer();
    if (name.endsWith(".zip")) return unzipArtifact(buffer);
    if (!name.endsWith(".json")) throw new Error("Only .json and .zip uploads are accepted.");
    if (buffer.byteLength > MAX_ARTIFACT_BYTES) throw new Error("run_artifact.json exceeds the 2 MB safety limit.");
    return JSON.parse(new TextDecoder().decode(new Uint8Array(buffer)));
  }
  return {STATUSES: STATUSES, ASSESSMENTS: ASSESSMENTS, DISCLAIMER: DISCLAIMER, endpoint: endpoint,
    evaluateArtifact: evaluateArtifact, semanticPayload: semanticPayload, semanticRequest: semanticRequest,
    diagnose: diagnose, reportSuggestions:reportSuggestions, reportBodyHtml:reportBodyHtml, reportHtml:reportHtml,
    renderReport:renderReport, lossChartSvg:lossChartSvg, unzipArtifact: unzipArtifact, readArtifactFile: readArtifactFile};
}));
