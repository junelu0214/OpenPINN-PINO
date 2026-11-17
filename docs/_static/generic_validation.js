(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.OpenPINNGenericValidation = factory();
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var STATUSES = ["pass", "fail", "unknown", "unsupported", "skipped"];
  function clone(value) { return JSON.parse(JSON.stringify(value)); }
  function object(value) { return value && typeof value === "object" && !Array.isArray(value); }
  function stable(value) {
    if (Array.isArray(value)) return "[" + value.map(stable).join(",") + "]";
    if (object(value)) return "{" + Object.keys(value).sort().map(function (key) {
      return JSON.stringify(key) + ":" + stable(value[key]);
    }).join(",") + "}";
    return JSON.stringify(value);
  }
  function fingerprint(value) {
    var text = stable(value), hash = 2166136261;
    for (var index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index); hash = Math.imul(hash, 16777619);
    }
    return "fnv1a32:" + (hash >>> 0).toString(16).padStart(8, "0");
  }
  function array(value) { return Array.isArray(value) ? clone(value) : (value == null ? [] : [clone(value)]); }

  function fieldsFrom(draft) {
    if (Array.isArray(draft.fields)) return draft.fields.map(function (field, index) {
      var result = clone(field), components = Array.isArray(result.components) ? result.components : [];
      result.name = result.name || (components.length === 1 ? components[0] : "field_" + index);
      result.kind = result.kind || (components.length > 1 ? "vector" : "scalar");
      result.components = components;
      result.shape = result.shape || (components.length ? [components.length] : null);
      result.unit = result.unit || result.units || null;
      delete result.units;
      return result;
    });
    var field = draft.field;
    if (!field) return [];
    if (Array.isArray(field)) return clone(field);
    var components = Array.isArray(field.components) ? field.components.slice() : [];
    return [{
      name: field.name || (components.length === 1 ? components[0] : field.kind || "field"),
      kind: field.kind || (components.length > 1 ? "vector" : "unknown"),
      components: components,
      shape: field.shape || (components.length ? [components.length] : null),
      unit: field.unit || field.units || draft.units && draft.units.field || null,
    }];
  }
  function parameterMapping(draft) {
    var candidates = [], issues = [], normalizations = [];
    function add(name, value, source) {
      if (typeof name !== "string" || !name) return;
      var entry = object(value) ? clone(value) : {value: value};
      entry.name = entry.name || entry.symbol || name;
      if (entry.value === undefined && entry.default !== undefined) entry.value = entry.default;
      if (entry.unit === undefined) entry.unit = entry.units || null;
      delete entry.id; delete entry.symbol; delete entry.units;
      candidates.push({entry: entry, source: source});
    }
    ["parameters", "constants", "coefficients"].forEach(function (key) {
      var source = draft[key];
      if (Array.isArray(source)) source.forEach(function (entry) {
        if (object(entry)) add(entry.name || entry.symbol || entry.id, entry, "draft." + key);
      });
      else if (object(source)) Object.keys(source).forEach(function (name) { add(name, source[name], "draft." + key + "." + name); });
    });
    var pde = object(draft.pde) ? draft.pde : {};
    var equationSources = [draft.expression_source, draft.equation, pde.expression, pde.equation, pde.residual]
      .concat(array(draft.equations)).concat(array(pde.residuals));
    var expression = equationSources.map(function (value) {
      if (typeof value === "string") return value;
      return object(value) ? value.expression || value.residual || value.equation || "" : "";
    }).filter(function (value) { return typeof value === "string"; }).join("\n");
    Object.keys(pde).forEach(function (name) {
      var numeric = typeof pde[name] === "number" || object(pde[name]) && pde[name].value !== undefined;
      var escapedName = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      var referenced = (new RegExp("\\b" + escapedName + "\\b")).test(expression) ||
        name === "diffusivity" && /\b(alpha|diffusivity)\b/.test(expression);
      if (["expression", "equation", "residual", "form", "type", "kind"].indexOf(name) < 0 && numeric && referenced) {
        add(name, pde[name], "draft.pde." + name);
      }
    });
    if (/\balpha\b/.test(expression) && !candidates.some(function (candidate) { return candidate.entry.name === "alpha"; }) && pde.diffusivity !== undefined) {
      add("alpha", pde.diffusivity, "draft.pde.diffusivity");
      normalizations.push({code: "parameter.expression_alias", from: "diffusivity", to: "alpha",
        reason: "The equation references alpha, so pde.diffusivity was exposed under the referenced symbol; an explicit unit is still required."});
    }
    if (/\bf\b/.test(expression) && !candidates.some(function (candidate) { return candidate.entry.name === "f"; }) && pde.source !== undefined) add("f", pde.source, "draft.pde.source");
    var grouped = {};
    candidates.forEach(function (candidate) { (grouped[candidate.entry.name] || (grouped[candidate.entry.name] = [])).push(candidate); });
    var result = [];
    Object.keys(grouped).forEach(function (name) {
      var group = grouped[name], values = group.map(function (candidate) { return stable(candidate.entry.value); });
      var unitful = group.filter(function (candidate) { return typeof candidate.entry.unit === "string" && candidate.entry.unit.trim(); });
      var units = unitful.map(function (candidate) { return candidate.entry.unit; });
      var sameValue = values.every(function (value) { return value === values[0]; });
      var sameUnit = units.every(function (unit) { return unit === units[0]; });
      if (group.length > 1 && sameValue && sameUnit) {
        var chosen = clone((unitful[0] || group[0]).entry);
        result.push(chosen);
        normalizations.push({code: "parameter.duplicate_normalized", from: name, to: name,
          reason: "Equivalent duplicate declarations were normalized to one canonical parameter; the unit-bearing declaration was preferred."});
      } else if (group.length > 1) {
        group.forEach(function (candidate) { result.push(clone(candidate.entry)); });
        issues.push({code: "parameter.duplicate_conflict", path: "problem.parameters", message: "Conflicting declarations exist for parameter " + name + "."});
      } else result.push(clone(group[0].entry));
    });
    function normalizeAlias(referenced, alias) {
      if (!(new RegExp("\\b" + referenced + "\\b")).test(expression)) return;
      var primary = result.filter(function (entry) { return entry.name === referenced; });
      var aliases = result.filter(function (entry) { return entry.name === alias; });
      if (!primary.length || !aliases.length) return;
      var canonical = primary[0], compatible = aliases.every(function (entry) {
        return stable(entry.value) === stable(canonical.value) && (!entry.unit || entry.unit === canonical.unit);
      });
      if (primary.length === 1 && compatible && canonical.unit) {
        result = result.filter(function (entry) { return entry.name !== alias; });
        normalizations.push({code: "parameter.alias_normalized", from: alias, to: referenced,
          reason: "The equation references " + referenced + "; an equivalent alias was removed in favor of the unit-bearing canonical parameter."});
      } else {
        issues.push({code: "parameter.synonym_conflict", path: "problem.parameters", message: "Declarations for " + referenced + " and " + alias + " cannot be normalized unambiguously."});
      }
    }
    normalizeAlias("alpha", "diffusivity"); normalizeAlias("diffusivity", "alpha");
    var hasBoundaryTarget = array(draft.boundaries).some(function (boundary) {
      return object(boundary) && ["value", "expression", "value_expression"].some(function (key) {
        return Object.prototype.hasOwnProperty.call(boundary, key);
      });
    });
    result = result.filter(function (entry) {
      var misplacedBoundaryPlaceholder = entry.name === "value" && typeof entry.value === "string" &&
        !/\bvalue\b/.test(expression) && hasBoundaryTarget;
      if (misplacedBoundaryPlaceholder) normalizations.push({
        code: "parameter.boundary_placeholder_removed", from: "value", to: "problem.boundaries",
        reason: "A symbolic boundary target belongs to its boundary condition, not the numeric PDE parameter table."
      });
      return !misplacedBoundaryPlaceholder;
    });
    return {parameters: result, issues: issues, normalizations: normalizations};
  }
  function equationsFrom(draft) {
    var equations = array(draft.equations);
    var pde = object(draft.pde) ? draft.pde : {};
    if (!equations.length && Array.isArray(pde.residuals)) equations = array(pde.residuals);
    if (!equations.length && draft.pde) equations = [clone(draft.pde)];
    if (!equations.length && draft.equation) equations = [{form: "expression", expression: draft.equation}];
    return equations.map(function (equation, index) {
      if (typeof equation === "string") return {id: "equation_" + index, form: "expression", expression: equation};
      var result = clone(equation);
      result.id = result.id || "equation_" + index;
      result.form = result.form || (result.expression || result.equation ? "expression" : "unknown");
      if (!result.expression && typeof result.equation === "string") result.expression = result.equation;
      return result;
    });
  }
  function boundariesFrom(draft) {
    return array(draft.boundaries).map(function (boundary) {
      if (!object(boundary)) return boundary;
      if (typeof boundary.value_expression === "string" && boundary.value_expression.trim()) {
        boundary.expression = boundary.value_expression;
      }
      return boundary;
    });
  }
  function coordinateBounds(value, name) {
    if (!object(value)) return null;
    if (Array.isArray(value.bounds) && value.bounds.length === 2) return {min: value.bounds[0], max: value.bounds[1]};
    if (object(value.bounds) && value.bounds.min !== undefined && value.bounds.max !== undefined) return {min: value.bounds.min, max: value.bounds.max};
    if (value.min !== undefined && value.max !== undefined) return {min: value.min, max: value.max};
    if (value[name + "_min"] !== undefined && value[name + "_max"] !== undefined) return {min: value[name + "_min"], max: value[name + "_max"]};
    return null;
  }
  function spatialBounds(draft, name) {
    var geometry = object(draft.geometry) ? draft.geometry : {}, outer = object(geometry.outer) ? geometry.outer : geometry;
    if (outer[name + "_min"] !== undefined && outer[name + "_max"] !== undefined) return {min: outer[name + "_min"], max: outer[name + "_max"]};
    if (object(outer.bounds) && Array.isArray(outer.bounds[name])) return {min: outer.bounds[name][0], max: outer.bounds[name][1]};
    if (Array.isArray(outer[name]) && outer[name].length === 2) return {min: outer[name][0], max: outer[name][1]};
    return null;
  }
  function coordinatesFrom(draft) {
    var unitMap = object(draft.units) ? draft.units : {};
    if (Array.isArray(draft.coordinates)) return draft.coordinates.map(function (coordinate) {
      var result = clone(coordinate), name = result.name;
      result.role = result.role || (name === "t" ? "time" : "space");
      result.unit = result.unit || result.units || (result.role === "time" ? unitMap.time : unitMap.space || unitMap.length) || null;
      result.bounds = coordinateBounds(result, name) || (result.role === "time" ? coordinateBounds(draft.time_domain, "t") : spatialBounds(draft, name));
      delete result.units; delete result.min; delete result.max;
      return result;
    });
    var dim = Number.isInteger(draft.dim) && draft.dim > 0 ? draft.dim : null;
    var names = ["x", "y", "z"], coordinates = [];
    if (dim !== null) for (var index = 0; index < dim; index += 1) {
      var name = names[index] || "x" + (index + 1);
      coordinates.push({name: name, role: "space", unit: unitMap.space || unitMap.length || null, bounds: spatialBounds(draft, name)});
    }
    var timeDependent = draft.time_dependent === true || draft.kind === "dynamic" || draft.initial_condition || draft.initial_conditions;
    if (timeDependent) coordinates.push({name: "t", role: "time", unit: unitMap.time || draft.time_domain && draft.time_domain.unit || null,
      bounds: coordinateBounds(draft.time_domain, "t")});
    return coordinates;
  }
  function trainableContract(problem, parameterMeta) {
    var issues = clone(parameterMeta && parameterMeta.issues || []);
    var normalizations = clone(parameterMeta && parameterMeta.normalizations || []);
    function issue(code, path, message) { issues.push({code: code, path: path, message: message}); }
    var coordinateNames = {}, parameterNames = {}, fieldNames = {};
    if (!problem.physics) issue("mapping.physics_missing", "problem.physics", "A physics label is required by the Stage A mapping contract.");
    problem.coordinates.forEach(function (coordinate, index) {
      var path = "problem.coordinates[" + index + "]";
      if (!coordinate.name || coordinateNames[coordinate.name]) issue("mapping.coordinate_name", path + ".name", "Coordinate names must be present and unique.");
      if (coordinate.name) coordinateNames[coordinate.name] = true;
      if (["space", "time"].indexOf(coordinate.role) < 0) issue("mapping.coordinate_role", path + ".role", "Coordinate role must be space or time.");
      if (!coordinate.unit) issue("mapping.coordinate_unit", path + ".unit", "Every coordinate requires an explicit unit for a later training path.");
      if (!object(coordinate.bounds) || typeof coordinate.bounds.min !== "number" || typeof coordinate.bounds.max !== "number" ||
          !Number.isFinite(coordinate.bounds.min) || !Number.isFinite(coordinate.bounds.max) || coordinate.bounds.min >= coordinate.bounds.max) {
        issue("mapping.coordinate_bounds", path + ".bounds", "Every coordinate requires finite ordered top-level bounds.");
      }
    });
    var equationText = problem.equations.map(function (equation) { return equation.expression || equation.residual || ""; }).join("\n");
    var hasTimeCoordinate = problem.coordinates.some(function (coordinate) { return coordinate.role === "time"; });
    var sourceDraft = object(problem.source_draft) ? problem.source_draft : {};
    var timeDependent = hasTimeCoordinate || sourceDraft.time_dependent === true || sourceDraft.kind === "dynamic" ||
      problem.initial_conditions.length > 0 || /\bdt\s*\(/.test(equationText);
    if (timeDependent && !hasTimeCoordinate) issue("mapping.time_coordinate_missing", "problem.coordinates", "A time-dependent problem requires a first-class time coordinate.");
    if (!problem.coordinates.length) issue("mapping.coordinates_missing", "problem.coordinates", "At least one first-class coordinate is required.");
    problem.fields.forEach(function (field, index) {
      var path = "problem.fields[" + index + "]", components = Array.isArray(field.components) ? field.components : [];
      if (!field.name || fieldNames[field.name]) issue("mapping.field_name", path + ".name", "Unknown-field names must be present and unique.");
      if (field.name) fieldNames[field.name] = true;
      if (!components.length) issue("mapping.field_components", path + ".components", "Every unknown field requires at least one component.");
      components.forEach(function (component) { fieldNames[component] = true; });
      if (!field.unit) issue("mapping.field_unit", path + ".unit", "Every unknown field requires an explicit unit.");
    });
    if (!problem.fields.length) issue("mapping.fields_missing", "problem.fields", "At least one first-class unknown field is required.");
    problem.parameters.forEach(function (parameter, index) {
      var path = "problem.parameters[" + index + "]";
      if (!parameter.name || parameterNames[parameter.name]) issue("mapping.parameter_unique", path + ".name", "Parameter names must be present and unique.");
      if (parameter.name) parameterNames[parameter.name] = true;
      if (typeof parameter.value !== "number" || !Number.isFinite(parameter.value)) issue("mapping.parameter_value", path + ".value", "Every executable parameter requires a finite numeric value.");
      if (!parameter.unit) issue("mapping.parameter_unit", path + ".unit", "Every parameter requires an explicit unit, including dimensionless unit 1.");
    });
    if (/\balpha\b/.test(equationText) && parameterNames.alpha && parameterNames.diffusivity) {
      issue("parameter.synonym_conflict", "problem.parameters", "Both alpha and diffusivity remain while the equation references alpha; explicit normalization is required.");
    }
    if (/\bdiffusivity\b/.test(equationText) && parameterNames.alpha && parameterNames.diffusivity) {
      issue("parameter.synonym_conflict", "problem.parameters", "Both diffusivity and alpha remain while the equation references diffusivity; explicit normalization is required.");
    }
    if (!problem.equations.length) issue("mapping.equations_missing", "problem.equations", "At least one first-class PDE equation is required.");
    var allowed = {sin: true, cos: true, exp: true, pi: true};
    problem.equations.forEach(function (equation, index) {
      var expression = equation.expression || equation.residual;
      if (typeof expression !== "string" || !expression.trim()) {
        issue("mapping.equation_expression", "problem.equations[" + index + "]", "A later training path requires a first-class expression or residual."); return;
      }
      (expression.match(/[A-Za-z_][A-Za-z0-9_]*/g) || []).forEach(function (name) {
        var derivative = name.match(/^d([a-z][A-Za-z0-9_]*)$/);
        if (derivative) {
          if (!coordinateNames[derivative[1]]) issue("mapping.derivative_coordinate", "problem.equations[" + index + "].expression", "Derivative " + name + " references an undeclared coordinate.");
          return;
        }
        if (allowed[name] || coordinateNames[name] || fieldNames[name] || parameterNames[name]) return;
        issue("mapping.expression_unbound", "problem.equations[" + index + "].expression", "Expression symbol " + name + " is not bound to a coordinate, field, or canonical parameter.");
      });
    });
    if (!object(problem.geometry)) issue("mapping.geometry_missing", "problem.geometry", "First-class geometry is required.");
    if (!problem.boundaries.length) issue("mapping.boundaries_missing", "problem.boundaries", "At least one first-class boundary condition is required.");
    function checkConditionExpression(value, path) {
      if (typeof value !== "string") return;
      if (/[^A-Za-z0-9_+\-*/().\s]/.test(value)) {
        issue("mapping.boundary_expression_token", path, "Boundary target expression contains a token outside the restricted grammar.");
        return;
      }
      (value.match(/[A-Za-z_][A-Za-z0-9_]*/g) || []).forEach(function (name) {
        var derivative = name.match(/^d([a-z][A-Za-z0-9_]*)$/);
        if (derivative && coordinateNames[derivative[1]]) return;
        if (allowed[name] || coordinateNames[name] || fieldNames[name] || parameterNames[name]) return;
        issue("mapping.boundary_expression_unbound", path, "Boundary target symbol " + name + " is not declared.");
      });
    }
    problem.boundaries.forEach(function (boundary, index) {
      var path = "problem.boundaries[" + index + "]";
      if (!object(boundary)) { issue("mapping.boundary_shape", path, "Every boundary condition must be a structured object."); return; }
      if (typeof boundary.type !== "string" || !boundary.type) issue("mapping.boundary_type", path + ".type", "Every boundary condition requires a type.");
      if (!boundary.location && !boundary.region && !boundary.id) issue("mapping.boundary_location", path, "Every boundary condition requires a location, region, or id.");
      var target = boundary.value_expression !== undefined ? boundary.value_expression :
        boundary.expression !== undefined ? boundary.expression : boundary.value;
      if (!(typeof target === "number" && Number.isFinite(target)) && !(typeof target === "string" && target.trim())) {
        issue("mapping.boundary_value", path + ".value", "Every boundary condition requires a numeric value or restricted target expression.");
      }
      checkConditionExpression(target, path + ".value");
      if (boundary.type === "robin") {
        checkConditionExpression(boundary.alpha, path + ".alpha");
        checkConditionExpression(boundary.beta, path + ".beta");
      }
    });
    if (timeDependent && !problem.initial_conditions.length) issue("mapping.initial_condition_missing", "problem.initial_conditions", "A time-dependent problem requires a first-class initial condition.");
    problem.initial_conditions.forEach(function (initial, index) {
      if (!object(initial) || !(typeof initial.expression === "string" && initial.expression.trim()) && !Object.prototype.hasOwnProperty.call(initial, "value")) {
        issue("mapping.initial_condition_shape", "problem.initial_conditions[" + index + "]", "Initial conditions require an expression or explicit value.");
      }
    });
    problem.opaque_blocks.forEach(function (block, index) {
      var serialized = stable(block);
      if (/"(time_domain|t_min|t_max|boundaries|initial_condition|equation|parameters)"/.test(serialized)) {
        issue("mapping.required_data_opaque", "problem.opaque_blocks[" + index + "]", "Training-required data remains only in an opaque block and was not treated as mapped.");
      }
    });
    return {schema_version: 1, kind: "TrainableProblemContract", mapping_complete: issues.length === 0,
      eligible_for_later_training_path: issues.length === 0, issues: issues, normalizations: normalizations,
      disclaimer: "Mapping completeness is not backend support, trainability, well-posedness, convergence, or physical correctness."};
  }
  function finalizeGeneric(generic, parameterMeta) {
    var prior = generic.trainable_contract;
    delete generic.fingerprint; delete generic.trainable_contract;
    if (prior && !parameterMeta) parameterMeta = {issues: [], normalizations: prior.normalizations || []};
    generic.trainable_contract = trainableContract(generic, parameterMeta || {issues: [], normalizations: []});
    generic.fingerprint = fingerprint(generic);
    return generic;
  }
  function normalizeProblem(draft) {
    if (!object(draft)) throw new TypeError("problem_draft must be an object");
    if (draft.kind === "GenericProblemDescription" && draft.schema_version === 1) {
      var canonical = clone(draft);
      canonical.boundaries = boundariesFrom(canonical);
      return finalizeGeneric(canonical);
    }
    var known = ["schema_version", "physics", "dim", "coordinates", "time_dependent", "time_domain", "kind", "field", "fields", "parameters", "constants", "coefficients", "pde", "equation", "equations", "geometry", "boundaries", "initial_condition", "initial_conditions", "couplings", "opaque_blocks", "units", "constitutive"];
    var extra = {};
    Object.keys(draft).forEach(function (key) { if (known.indexOf(key) < 0) extra[key] = clone(draft[key]); });
    var opaque = array(draft.opaque_blocks);
    if (Object.keys(extra).length) opaque.push({kind: "unmapped_source_fields", value: extra});
    var parameterMeta = parameterMapping(draft);
    var generic = {
      schema_version: 1,
      kind: "GenericProblemDescription",
      physics: typeof draft.physics === "string" ? draft.physics : "",
      dimension: Number.isInteger(draft.dim) && draft.dim > 0 ? draft.dim : null,
      coordinates: coordinatesFrom(draft),
      fields: fieldsFrom(draft),
      parameters: parameterMeta.parameters,
      equations: equationsFrom(draft),
      geometry: draft.geometry == null ? null : clone(draft.geometry),
      boundaries: boundariesFrom(draft),
      initial_conditions: array(draft.initial_conditions || draft.initial_condition),
      couplings: array(draft.couplings),
      opaque_blocks: opaque,
      units: draft.units == null ? null : clone(draft.units),
      source_draft: clone(draft),
    };
    return finalizeGeneric(generic, parameterMeta);
  }
  function normalizeMethod(method) {
    if (!object(method)) throw new TypeError("method_draft must be an object");
    return clone(method);
  }
  function item(status, code, path, message) {
    if (STATUSES.indexOf(status) < 0) throw new Error("invalid validation status");
    return {status: status, code: code, path: path, message: message};
  }
  function layer(id, title, items) {
    var order = ["fail", "unsupported", "unknown", "skipped", "pass"];
    var status = order.find(function (candidate) { return items.some(function (entry) { return entry.status === candidate; }); });
    return {id: id, title: title, status: status || "skipped", items: items};
  }
  function expressionChecks(problem) {
    var results = [], coordinates = problem.coordinates.map(function (entry) { return entry.name; });
    var symbols = new Set(coordinates.concat(["pi"]));
    problem.parameters.forEach(function (entry) { if (entry.name || entry.symbol || entry.id) symbols.add(entry.name || entry.symbol || entry.id); });
    problem.fields.forEach(function (field) {
      if (field.name) symbols.add(field.name);
      (field.components || []).forEach(function (name) { symbols.add(name); });
    });
    var allowedFunctions = new Set(["sin", "cos", "exp"]);
    problem.equations.forEach(function (equation, index) {
      var path = "problem.equations[" + index + "]";
      if (equation.form !== "expression" && equation.form !== "residual") {
        results.push(item("unsupported", "expression.form_opaque", path + ".form", "This mathematical form is retained but is outside the restricted expression checker.")); return;
      }
      var expression = equation.expression || equation.residual;
      if (typeof expression !== "string" || !expression.trim()) {
        results.push(item("unknown", "expression.missing", path, "No checkable expression was supplied.")); return;
      }
      if (/[^A-Za-z0-9_+\-*/().=\s]/.test(expression)) {
        results.push(item("fail", "expression.illegal_token", path + ".expression", "Expression contains a token outside the restricted grammar.")); return;
      }
      if ((expression.match(/=/g) || []).length > 1 || /^\s*=|=\s*$/.test(expression)) {
        results.push(item("fail", "expression.equality", path + ".expression", "Restricted expressions allow at most one equality with both sides present.")); return;
      }
      var depth = 0;
      for (var cursor = 0; cursor < expression.length; cursor += 1) {
        if (expression[cursor] === "(") depth += 1;
        if (expression[cursor] === ")") depth -= 1;
        if (depth < 0) break;
      }
      if (depth !== 0) results.push(item("fail", "expression.parentheses", path + ".expression", "Expression parentheses are unbalanced."));
      var identifiers = expression.match(/[A-Za-z_][A-Za-z0-9_]*/g) || [];
      identifiers.forEach(function (name) {
        var derivative = name.match(/^d([a-z][A-Za-z0-9_]*)$/);
        if (derivative) {
          if (coordinates.indexOf(derivative[1]) < 0) results.push(item("fail", "expression.derivative_coordinate", path + ".expression", "Derivative " + name + " uses an undeclared coordinate."));
        } else if (!symbols.has(name) && !allowedFunctions.has(name)) {
          results.push(item("fail", "expression.unknown_symbol", path + ".expression", "Symbol " + name + " is not declared as a field, coordinate, or parameter."));
        }
      });
      allowedFunctions.forEach(function (name) {
        if (new RegExp("\\b" + name + "\\b(?!\\s*\\()", "g").test(expression)) {
          results.push(item("fail", "expression.function_call", path + ".expression", "Function " + name + " must be called with parentheses."));
        }
      });
      if (!results.some(function (entry) { return entry.path.indexOf(path) === 0 && entry.status === "fail"; })) {
        results.push(item("pass", "expression.restricted_syntax", path + ".expression", "Restricted syntax and symbol scope checks passed without executing the expression."));
      }
    });
    return results.length ? results : [item("unknown", "expression.none", "problem.equations", "No equation was available for syntax checking.")];
  }
  function capabilityText(value) { return String(value == null ? "" : value).replace(/\s+/g, "").toLowerCase(); }
  function capabilityBounds(problem, name) {
    var coordinate = problem.coordinates.find(function (entry) { return entry.name === name; });
    return coordinate && object(coordinate.bounds) ? [coordinate.bounds.min, coordinate.bounds.max] : null;
  }
  function executionCapability(problem, method) {
    var incomplete = !problem || !problem.trainable_contract || problem.trainable_contract.mapping_complete !== true;
    if (incomplete) return {status: "incomplete", backend_id: null, reason: "Stage A mapping_complete is required.", smoke_requires_python: true, training_complete: false};
    var loss = object(method.loss) ? method.loss : {}, differentiator = object(method.differentiator) ? method.differentiator : {};
    if (method.path !== "pinn" || (loss.formulation || "strong") !== "strong" || (differentiator.mode || "autodiff") !== "autodiff") {
      return {status: "unsupported", backend_id: null, reason: "Only strong-form PINN with autodiff is registered.", smoke_requires_python: true, training_complete: false};
    }
    var fields = problem.fields || [], components = fields.length === 1 && Array.isArray(fields[0].components) ? fields[0].components : [];
    if (components.length !== 1 || typeof components[0] !== "string" || !components[0]) return {status: "unsupported", backend_id: null, reason: "A single scalar field is required.", smoke_requires_python: true, training_complete: false};
    if (!Array.isArray(problem.equations) || !problem.equations.length) return {status: "unsupported", backend_id: null, reason: "At least one strong-form equation is required.", smoke_requires_python: true, training_complete: false};
    var expression = problem.equations.length === 1 ? capabilityText(problem.equations[0].expression || problem.equations[0].residual) : "";
    var names = problem.coordinates.map(function (entry) { return entry.name; });
    var roles = problem.coordinates.map(function (entry) { return entry.role; });
    var interval = object(problem.geometry) && capabilityText(problem.geometry.type) === "interval";
    var xBounds = capabilityBounds(problem, "x"), tBounds = capabilityBounds(problem, "t");
    var boundaries = problem.boundaries || [];
    function locations(type) { return boundaries.filter(function (entry) { return capabilityText(entry.type) === type && Number(entry.value) === 0; }).map(function (entry) { return capabilityText(entry.location); }).sort(); }
    var dirichlet = locations("dirichlet"), neumann = locations("neumann");
    if (components[0] === "u" && JSON.stringify(names) === JSON.stringify(["x"]) && JSON.stringify(roles) === JSON.stringify(["space"]) && interval && JSON.stringify(xBounds) === JSON.stringify([0, 1]) &&
        expression.indexOf("dx(dx(u))") >= 0 && expression.indexOf("pi*pi*sin(pi*x)") >= 0 && expression.indexOf("dt(") < 0 &&
        JSON.stringify(dirichlet) === JSON.stringify(["x=0", "x=1"]) && neumann.length === 0 && !(problem.initial_conditions || []).length) {
      return {status: "supported", backend_id: "poisson_1d", reason: "Registered 1D interval/Dirichlet strong-form structure.", smoke_requires_python: true, training_complete: false};
    }
    var parameters = problem.parameters || [], alpha = parameters.find(function (entry) { return entry.name === "alpha"; });
    var initials = problem.initial_conditions || [], initial = initials.length === 1 ? capabilityText(initials[0].expression) : "";
    if (components[0] === "u" && JSON.stringify(names) === JSON.stringify(["x", "t"]) && JSON.stringify(roles) === JSON.stringify(["space", "time"]) && interval && JSON.stringify(xBounds) === JSON.stringify([0, 1]) &&
        JSON.stringify(tBounds) === JSON.stringify([0, 1]) && expression.indexOf("dt(u)") >= 0 && expression.indexOf("dx(dx(u))") >= 0 &&
        expression.indexOf("alpha") >= 0 && parameters.length === 1 && alpha && Number(alpha.value) === 0.01 &&
        JSON.stringify(dirichlet) === JSON.stringify(["x=0"]) && JSON.stringify(neumann) === JSON.stringify(["x=1"]) && initial === "sin(pi*x/2)") {
      return {status: "supported", backend_id: "heat_1d_unsteady", reason: "Registered 1D interval+time mixed-BC strong-form structure.", smoke_requires_python: true, training_complete: false};
    }
    var spatial = problem.coordinates.filter(function (entry) { return entry.role === "space"; });
    var temporal = problem.coordinates.filter(function (entry) { return entry.role === "time"; });
    var spatialNames = spatial.map(function (entry) { return entry.name; });
    var genericCoordinates = (JSON.stringify(spatialNames) === JSON.stringify(["x"]) ||
      JSON.stringify(spatialNames) === JSON.stringify(["x", "y"])) && temporal.length <= 1 &&
      spatial.length + temporal.length === problem.coordinates.length &&
      (!temporal.length || temporal[0].name === "t") && problem.coordinates.every(function (entry) {
        return object(entry.bounds) && Number.isFinite(entry.bounds.min) && Number.isFinite(entry.bounds.max) && entry.bounds.min < entry.bounds.max;
      });
    var genericGeometry = spatial.length === 1 && interval || spatial.length === 2 && object(problem.geometry) &&
      ["rectangle", "rectangular"].indexOf(capabilityText(problem.geometry.type)) >= 0;
    function conditionValue(value) { return typeof value === "number" && Number.isFinite(value) || typeof value === "string" && Boolean(value.trim()); }
    function boundaryLocation(location) {
      var match = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*$/.exec(String(location || ""));
      if (!match) return false;
      var coordinate = problem.coordinates.find(function (entry) { return entry.name === match[1]; });
      var value = Number(match[2]);
      return coordinate && coordinate.role === "space" && object(coordinate.bounds) &&
        (value === coordinate.bounds.min || value === coordinate.bounds.max);
    }
    var genericBoundaries = boundaries.length > 0 && boundaries.every(function (entry) {
      var targetSource = entry.value_expression !== undefined ? entry.value_expression :
        entry.expression !== undefined ? entry.expression : entry.value;
      var kind = capabilityText(entry.type), target = conditionValue(targetSource);
      return ["dirichlet", "neumann", "robin"].indexOf(kind) >= 0 && boundaryLocation(entry.location) && target &&
        (kind !== "robin" || conditionValue(entry.alpha) && conditionValue(entry.beta));
    });
    var genericInitials = !temporal.length && !initials.length || temporal.length === 1 && initials.length > 0;
    if (genericCoordinates && genericGeometry && problem.equations.length > 0 && genericBoundaries && genericInitials) {
      return {status: "supported", backend_id: "scalar_strong_form_v1", reason: "Registered scalar strong-form capabilities: 1D/2D, optional time, multiple compiled residuals, Dirichlet/Neumann/Robin.", smoke_requires_python: true, training_complete: false};
    }
    return {status: "unsupported", backend_id: null, reason: "No registered executable backend matches the declared operators and conditions.", smoke_requires_python: true, training_complete: false};
  }
  function adapter(problem, method) {
    var draft = problem.source_draft;
    var allowed = ["dim", "physics", "geometry", "field", "constitutive", "boundaries", "units"];
    var extra = Object.keys(draft).filter(function (key) { return allowed.indexOf(key) < 0; });
    if (draft.physics !== "linear_elasticity") return {status: "unsupported", reason: "Only an exact linear_elasticity Executable ProblemSpec mapping is known.", path: "problem.physics"};
    if (extra.length || problem.opaque_blocks.length || problem.couplings.length || problem.initial_conditions.length) {
      return {status: "unsupported", reason: "Generic fields cannot be represented losslessly by the current Executable ProblemSpec: " + extra.join(", "), path: "problem"};
    }
    var requiredProblem = ["dim", "physics", "geometry", "field", "constitutive", "boundaries"];
    var requiredMethod = ["method", "path", "sampling", "loss", "optimizer", "network", "differentiator", "training"];
    var missing = requiredProblem.filter(function (key) { return draft[key] == null; }).map(function (key) { return "problem." + key; })
      .concat(requiredMethod.filter(function (key) { return method[key] == null; }).map(function (key) { return "method." + key; }));
    if (missing.length) return {status: "fail", reason: "Executable pair is incomplete: " + missing.join(", "), path: missing[0]};
    return {status: "pass", problem_spec: clone(draft), method_spec: clone(method), target_gate: "python_m34_required"};
  }
  function validate(problemInput, methodInput) {
    var problem = problemInput && problemInput.kind === "GenericProblemDescription" ? clone(problemInput) : normalizeProblem(problemInput);
    var method = normalizeMethod(methodInput);
    var l0 = [];
    l0.push(problem.physics ? item("pass", "description.physics", "problem.physics", "A free-form physics label is present.") : item("fail", "description.physics_missing", "problem.physics", "A physics label is required."));
    l0.push(problem.fields.length ? item("pass", "description.fields", "problem.fields", "At least one field is described.") : item("unknown", "description.fields_unknown", "problem.fields", "Field information is not yet known."));
    l0.push(problem.equations.length ? item("pass", "description.equations", "problem.equations", "Equation material is retained.") : item("unknown", "description.equations_unknown", "problem.equations", "Equation material is not yet known."));
    l0.push(problem.trainable_contract.mapping_complete
      ? item("pass", "mapping.complete", "problem.trainable_contract", "First-class fields satisfy the Stage A mapping contract for a later training path.")
      : item("unknown", "mapping.incomplete", "problem.trainable_contract", "The Stage A mapping contract is incomplete; this problem is not eligible for a later training path."));
    var l1 = [];
    problem.fields.forEach(function (field, index) {
      if (field.shape && field.components && field.components.length && field.shape[0] !== field.components.length) {
        l1.push(item("fail", "structure.field_shape", "problem.fields[" + index + "].shape", "Field shape does not match its component count."));
      }
    });
    if (!l1.length) l1.push(item("pass", "structure.consistent", "problem", "Declared field shapes are structurally consistent."));
    var unitItems = problem.parameters.length && problem.parameters.every(function (entry) { return entry.unit; })
      ? [item("pass", "units.declared", "problem.parameters", "All declared parameters include units.")]
      : [item("unknown", "units.incomplete", "problem.parameters", "Units are incomplete, so dimensional consistency is unknown.")];
    var knownPaths = {pinn: ["baseline", "adam_lbfgs", "adaptive_activation", "adaptive_sampling", "adaptive_weights", "domain_decomposition", "fourier", "hard_boundary", "modified_mlp", "time_window", "transfer"], operator: ["deeponet", "fno", "pino"]};
    var methodStatus = knownPaths[method.path] && knownPaths[method.path].indexOf(method.method) >= 0;
    var l5 = [methodStatus
      ? item("pass", "method.registered", "method.method", "The browser recognizes this method/path capability label.")
      : item("unsupported", "method.unsupported", "method.method", "The method or backend is unsupported; this does not make the problem description invalid.")];
    var adapted = adapter(problem, method), capability = executionCapability(problem, method);
    var l6 = [item(adapted.status, "adapter." + adapted.status, adapted.path || "pair", adapted.reason || "Lossless Executable pair produced; Python M34 validation is still required.")];
    var layers = [
      layer("L0", "description completeness", l0), layer("L1", "structural consistency", l1),
      layer("L2", "restricted expression and symbol checks", expressionChecks(problem)), layer("L3", "units", unitItems),
      layer("L4", "well-posedness and physical correctness", [item("unknown", "physics.not_proven", "pair", "The browser does not prove well-posedness, stability, convergence, trainability, or physical correctness.")]),
      layer("L5", "method and backend capability", l5), layer("L6", "Executable adapter", l6),
    ];
    var all = [].concat.apply([], layers.map(function (entry) { return entry.items; }));
    var pairFingerprint = fingerprint({problem: problem, method: method});
    return {
      schema_version: 1, kind: "LayeredValidationReport", problem_fingerprint: problem.fingerprint,
      method_fingerprint: fingerprint(method), pair_fingerprint: pairFingerprint, layers: layers,
      trainable_contract: clone(problem.trainable_contract),
      execution_capability: capability,
      summary: {
        description_ready_for_review: !all.some(function (entry) { return entry.status === "fail" && entry.path.indexOf("problem") === 0; }),
        mapping_complete_for_later_training: problem.trainable_contract.mapping_complete,
        has_fail: all.some(function (entry) { return entry.status === "fail"; }),
        has_unknown: all.some(function (entry) { return entry.status === "unknown"; }),
        has_unsupported: all.some(function (entry) { return entry.status === "unsupported"; }),
        executable_adapter: adapted.status,
        target_gate: "not_run_in_browser",
        training_export_unlocked: false,
        execution_capability: capability.status,
      },
      adapter: adapted,
    };
  }
  function createSession() {
    var accepted = null, report = null;
    return {
      validate: function (problem, method) { report = validate(problem, method); accepted = report.pair_fingerprint; return clone(report); },
      invalidate: function () { accepted = null; report = null; },
      isCurrent: function (problem, method) {
        try { return accepted !== null && accepted === fingerprint({problem: normalizeProblem(problem), method: normalizeMethod(method)}); }
        catch (_error) { return false; }
      },
      exportArtifacts: function (problem, method) {
        if (!report || !this.isCurrent(problem, method)) throw new Error("Layered Validation report is stale; validate the current pair again.");
        return {problem: normalizeProblem(problem), method: normalizeMethod(method), report: clone(report), training_code: null,
          label: "NON-TRAINING ARTIFACTS — generic specification, planning draft, and scoped validation report only"};
      },
      canExportTraining: function () { return false; },
    };
  }
  return {STATUSES: STATUSES, fingerprint: fingerprint, normalizeProblem: normalizeProblem, validate: validate,
    executionCapability: executionCapability, createSession: createSession};
}));
