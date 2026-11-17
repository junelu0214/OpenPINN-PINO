(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) module.exports = factory(require("./generic_validation.js"));
  else root.OpenPINNHybridValidation = factory(root.OpenPINNGenericValidation);
}(typeof globalThis !== "undefined" ? globalThis : this, function (generic) {
  "use strict";
  if (!generic) throw new Error("OpenPINNGenericValidation must load before Hybrid Validation");

  var LLM_REQUEST_TIMEOUT_MS = 300000;
  var REVIEW_STATUSES = ["no_concern_identified", "concerns", "insufficient_information"];
  var METHOD_STATUSES = ["suitable", "concerns", "insufficient_information", "unsupported"];
  var DISCLAIMER = "Semantic review is model-generated guidance, not proof of well-posedness, correctness, convergence, trainability, or executable support.";
  function clone(value) { return JSON.parse(JSON.stringify(value)); }
  function object(value) { return value && typeof value === "object" && !Array.isArray(value); }
  function redact(message, key) {
    var text = String(message || "Unknown semantic review error");
    if (key) text = text.split(String(key)).join("[redacted]");
    return text.replace(/Bearer\s+[^\s,;]+/gi, "Bearer [redacted]");
  }
  function endpoint(baseUrl) {
    var url = new URL(String(baseUrl || "").trim());
    if (url.protocol !== "https:" && !(url.protocol === "http:" && ["localhost", "127.0.0.1"].indexOf(url.hostname) >= 0)) {
      throw new Error("Validation API URL must use HTTPS (or localhost HTTP for tests).");
    }
    var path = url.pathname.replace(/\/+$/, "");
    if (!/\/chat\/completions$/.test(path)) path += "/chat/completions";
    url.pathname = path; url.search = ""; url.hash = "";
    return url.toString();
  }
  function resolvePath(problem, method, path, deterministic) {
    var root, tokens = [];
    if (typeof path === "string" && path.charAt(0) === "/") {
      var segments = path.slice(1).split("/");
      if (!segments.length || segments.some(function (segment) { return /~(?:[^01]|$)/.test(segment); })) return {found: false};
      segments = segments.map(function (segment) { return segment.replace(/~1/g, "/").replace(/~0/g, "~"); });
      root = segments.shift();
      tokens = segments.map(function (segment) { return /^\d+$/.test(segment) ? Number(segment) : segment; });
    } else {
      var match = /^(problem|method|deterministic_summary)(.*)$/.exec(path);
      if (!match) return {found: false};
      root = match[1];
      var rest = match[2], tokenPattern = /\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]/g, cursor = 0, found;
      while ((found = tokenPattern.exec(rest))) {
        if (found.index !== cursor) return {found: false};
        tokens.push(found[1] !== undefined ? found[1] : Number(found[2])); cursor = tokenPattern.lastIndex;
      }
      if (cursor !== rest.length) return {found: false};
    }
    if (["problem", "method", "deterministic_summary"].indexOf(root) < 0) return {found: false};
    var value = root === "problem" ? problem : root === "method" ? method : deterministic && deterministic.summary;
    for (var index = 0; index < tokens.length; index += 1) {
      if (value == null || !Object.prototype.hasOwnProperty.call(Object(value), tokens[index])) return {found: false};
      value = value[tokens[index]];
    }
    return {found: true, value: value};
  }
  function string(value, label) {
    if (typeof value !== "string" || !value.trim()) throw new Error(label + " must be a non-empty string.");
    return value.trim();
  }
  function stringList(value, label) {
    if (!Array.isArray(value)) throw new Error(label + " must be an array.");
    return value.map(function (entry, index) {
      if (object(entry)) entry = entry.message || entry.question || entry.information || entry.description || entry.needed || JSON.stringify(entry);
      return string(entry, label + "[" + index + "]");
    });
  }
  function exactKeys(value, allowed, label) {
    Object.keys(value).forEach(function (key) {
      if (allowed.indexOf(key) < 0) throw new Error(label + " contains unsupported field " + key + ".");
    });
  }
  function validateSemanticPayload(payload, context) {
    if (!object(payload)) throw new Error("Semantic response must be a JSON object.");
    exactKeys(payload, ["review_status", "concerns", "unknowns", "information_needed", "assumptions_noted", "method_fit"], "Semantic response");
    if (REVIEW_STATUSES.indexOf(payload.review_status) < 0) throw new Error("Semantic response review_status is invalid.");
    if (!Array.isArray(payload.concerns) || !Array.isArray(payload.unknowns) || !Array.isArray(payload.assumptions_noted)) {
      throw new Error("Semantic concerns, unknowns, and assumptions_noted must be arrays.");
    }
    var concerns = payload.concerns.map(function (concern, index) {
      if (!object(concern)) throw new Error("concerns[" + index + "] must be an object.");
      exactKeys(concern, ["code", "path", "message", "evidence"], "concerns[" + index + "]");
      if (!Array.isArray(concern.evidence) || !concern.evidence.length) throw new Error("Each concern requires input evidence.");
      return {
        code: string(concern.code, "concern code"), path: string(concern.path, "concern path"),
        message: string(concern.message, "concern message"),
        evidence: concern.evidence.map(function (evidence, evidenceIndex) {
          if (!object(evidence)) throw new Error("concern evidence must be an object.");
          exactKeys(evidence, ["path", "quote_or_excerpt", "reasoning_nature"], "concern evidence");
          var path = string(evidence.path, "evidence path"), resolved = resolvePath(context.problem, context.method, path, context.deterministic);
          if (!resolved.found) throw new Error("Evidence path does not exist in the input snapshot: " + path);
          var quote = string(evidence.quote_or_excerpt, "evidence quote_or_excerpt");
          var source = typeof resolved.value === "string" ? resolved.value : JSON.stringify(resolved.value);
          if (source.indexOf(quote) < 0) {
            if (source && quote.indexOf(source) >= 0) quote = source;
            else throw new Error("Evidence excerpt is not present at input path " + path + ".");
          }
          return {path: path, quote_or_excerpt: quote, reasoning_nature: string(evidence.reasoning_nature, "evidence reasoning_nature")};
        }),
      };
    });
    function notes(values, label) {
      return values.map(function (entry, index) {
        if (!object(entry)) throw new Error(label + "[" + index + "] must be an object.");
        entry = clone(entry);
        if (typeof entry.message !== "string" && typeof entry.description === "string") {
          entry.message = entry.description; delete entry.description;
        }
        if (typeof entry.message !== "string" && typeof entry.assumption === "string") {
          entry.message = entry.assumption; delete entry.assumption;
        }
        if (typeof entry.message !== "string" && typeof entry.unknown === "string") {
          entry.message = entry.unknown; delete entry.unknown;
        }
        exactKeys(entry, ["code", "path", "message"], label + "[" + index + "]");
        var path = string(entry.path, label + " path");
        if (!resolvePath(context.problem, context.method, path, context.deterministic).found) throw new Error(label + " path does not exist: " + path);
        var result = {path: path, message: string(entry.message, label + " message")};
        if (label === "unknowns") result.code = typeof entry.code === "string" && entry.code.trim() ? entry.code.trim() : "semantic.unknown";
        return result;
      });
    }
    var methodFit = object(payload.method_fit) ? clone(payload.method_fit) : payload.method_fit;
    if (object(methodFit) && typeof methodFit.message !== "string" && typeof methodFit.rationale === "string") {
      methodFit.message = methodFit.rationale; delete methodFit.rationale;
    }
    if (!object(methodFit) || METHOD_STATUSES.indexOf(methodFit.status) < 0) {
      throw new Error("Semantic method_fit status is invalid.");
    }
    exactKeys(methodFit, ["status", "message"], "method_fit");
    return {
      schema_version: 1, kind: "SemanticValidationReport", review_status: payload.review_status,
      pair_fingerprint: context.pairFingerprint, request_id: context.requestId,
      request_version: "semantic-v1", model: context.model, concerns: concerns,
      unknowns: notes(payload.unknowns, "unknowns"),
      information_needed: stringList(payload.information_needed, "information_needed"),
      assumptions_noted: notes(payload.assumptions_noted, "assumptions_noted"),
      method_fit: {status: methodFit.status, message: string(methodFit.message, "method_fit message")},
      disclaimer: DISCLAIMER,
    };
  }
  function prompt(problem, method, deterministic, pairFingerprint) {
    return [
      "Review only semantic concerns in this final problem-method snapshot.",
      "Return exactly one JSON object with keys review_status, concerns, unknowns, information_needed, assumptions_noted, method_fit.",
      "review_status must be no_concern_identified, concerns, or insufficient_information.",
      "Each concern must have code, path, message, and non-empty evidence. Evidence entries require path, quote_or_excerpt copied verbatim from that input path, and reasoning_nature.",
      "Each unknown is exactly {code, path, message}; each assumption is exactly {path, message}; information_needed is an array of strings.",
      "Paths may use problem.equations[0] dot/bracket syntax or /problem/equations/0 JSON Pointer syntax, and must resolve inside the supplied snapshot.",
      "method_fit must be exactly {status, message}, with no method or other fields. Its status is suitable, concerns, insufficient_information, or unsupported.",
      "Do not claim proof, correctness, convergence, trainability, or executable support. Do not cite external sources. Do not edit the specs. An unregistered method is a capability issue, not invalid physics.",
      JSON.stringify({pair_fingerprint: pairFingerprint, problem: problem, method: method, deterministic_summary: deterministic.summary}),
    ].join("\n");
  }
  function incomplete(code, message, requestId, pairFingerprint, model) {
    return {state: "incomplete", code: code, message: message, request_id: requestId,
      request_version: "semantic-v1", pair_fingerprint: pairFingerprint, model: model || "", disclaimer: DISCLAIMER};
  }
  function semanticRequest(problem, method, deterministic, options, requestId, signal) {
    options = options || {};
    var apiKey = String(options.apiKey || "").trim(), model = String(options.model || "").trim();
    if (!apiKey) return Promise.resolve(incomplete("configuration.missing_key", "Validation API Key is required; no rules fallback was used.", requestId, deterministic.pair_fingerprint, model));
    if (!model) return Promise.resolve(incomplete("configuration.missing_model", "Validation model is required; no rules fallback was used.", requestId, deterministic.pair_fingerprint, model));
    var fetchImpl = options.fetch || (typeof fetch === "function" ? fetch : null);
    if (!fetchImpl) return Promise.resolve(incomplete("transport.unavailable", "This browser does not provide fetch(); semantic review was not run.", requestId, deterministic.pair_fingerprint, model));
    var url;
    try { url = endpoint(options.baseUrl); }
    catch (error) { return Promise.resolve(incomplete("configuration.endpoint", redact(error.message, apiKey), requestId, deterministic.pair_fingerprint, model)); }
    var body = {
      model: model, temperature: 0, response_format: {type: "json_object"},
      messages: [
        {role: "system", content: "You are a bounded semantic reviewer. JSON only. Never mutate specs, fabricate citations, override deterministic failures, or claim trainability/proof."},
        {role: "user", content: prompt(problem, method, deterministic, deterministic.pair_fingerprint)},
      ],
    };
    var scheduleTimeout = typeof options.scheduleTimeout === "function" ? options.scheduleTimeout : setTimeout;
    var cancelTimeout = typeof options.cancelTimeout === "function" ? options.cancelTimeout : clearTimeout;
    var requestController = typeof AbortController === "function" ? new AbortController() : null;
    var timedOut = false, timeoutMs = Number(options.timeoutMs || LLM_REQUEST_TIMEOUT_MS), timer = null;
    if (requestController && signal) {
      if (signal.aborted) requestController.abort();
      else signal.addEventListener("abort", function () { requestController.abort(); }, {once: true});
    }
    if (requestController && Number.isFinite(timeoutMs) && timeoutMs > 0) {
      timer = scheduleTimeout(function () { timedOut = true; requestController.abort(); }, timeoutMs);
    }
    return fetchImpl(url, {method: "POST", headers: {"Content-Type": "application/json", "Accept": "application/json", "Authorization": "Bearer " + apiKey}, body: JSON.stringify(body), signal: requestController ? requestController.signal : signal})
      .then(function (response) {
        if (!response.ok) {
          var code = response.status === 401 || response.status === 403 ? "transport.auth" : "transport.http";
          throw {semanticCode: code, message: "Validation API returned HTTP " + response.status + "."};
        }
        return response.json().catch(function () { throw {semanticCode: "schema.envelope", message: "Validation API response is not JSON."}; });
      }).then(function (envelope) {
        var content;
        try { content = envelope.choices[0].message.content; } catch (_error) { content = null; }
        if (typeof content !== "string") throw {semanticCode: "schema.envelope", message: "Validation API response has an invalid chat envelope."};
        var parsed;
        try { parsed = JSON.parse(content); } catch (_error) { throw {semanticCode: "schema.content", message: "Semantic review content is not JSON."}; }
        try {
          return {state: "complete", report: validateSemanticPayload(parsed, {problem: problem, method: method,
            pairFingerprint: deterministic.pair_fingerprint, requestId: requestId, model: model, deterministic: deterministic})};
        } catch (error) {
          throw {semanticCode: "schema.semantic", message: error.message};
        }
      }).catch(function (error) {
        var code = error && error.name === "AbortError" ? (timedOut ? "transport.timeout" : "transport.cancelled") : error.semanticCode || "transport.network";
        return incomplete(code, redact(error && error.message, apiKey), requestId, deterministic.pair_fingerprint, model);
      }).finally(function () { if (timer) cancelTimeout(timer); });
  }
  function createSession() {
    var generation = 0, controller = null, envelope = null;
    function currentFingerprint(problem, method) {
      return generic.validate(problem, method).pair_fingerprint;
    }
    return {
      run: function (problemInput, methodInput, options) {
        generation += 1;
        var mine = generation;
        if (controller) controller.abort();
        controller = typeof AbortController === "function" ? new AbortController() : null;
        var problem = generic.normalizeProblem(problemInput), method = clone(methodInput);
        var deterministic = generic.validate(problem, method);
        var requestId = options && options.requestId || "semantic-v1-" + mine + "-" + Date.now().toString(36);
        var snapshot = {problem: problem, method: method};
        return semanticRequest(problem, method, deterministic, options, requestId, controller && controller.signal).then(function (semantic) {
          if (mine !== generation) return {state: "stale", ignored: true, request_id: requestId};
          envelope = {schema_version: 1, kind: "HybridValidationReport", pair_fingerprint: deterministic.pair_fingerprint,
            request_id: requestId, request_version: "semantic-v1", snapshots: snapshot,
            deterministic: deterministic, semantic: semantic, training_export_unlocked: false,
            m34_target_gate: "not_run_in_browser"};
          return clone(envelope);
        });
      },
      cancel: function () { generation += 1; if (controller) controller.abort(); controller = null; },
      invalidate: function () { this.cancel(); envelope = null; },
      isCurrent: function (problem, method) {
        try { return !!envelope && envelope.pair_fingerprint === currentFingerprint(problem, method); }
        catch (_error) { return false; }
      },
      exportArtifacts: function (problem, method) {
        if (!this.isCurrent(problem, method)) throw new Error("Hybrid Validation report is stale; revalidate the current final pair.");
        return {label: "NON-TRAINING HYBRID VALIDATION ARTIFACT — no API key and no runnable training code",
          problem: clone(envelope.snapshots.problem), method: clone(envelope.snapshots.method), report: clone(envelope), training_code: null};
      },
      canExportTraining: function () { return false; },
    };
  }
  return {REVIEW_STATUSES: REVIEW_STATUSES, METHOD_STATUSES: METHOD_STATUSES, DISCLAIMER: DISCLAIMER,
    endpoint: endpoint, validateSemanticPayload: validateSemanticPayload, semanticRequest: semanticRequest, createSession: createSession};
}));
