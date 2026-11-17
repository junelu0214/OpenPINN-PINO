/* Browser Understanding trial. Readiness follows the physics family, not final Specification Validation. */
(function (global, factory) {
  "use strict";
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (global) global.OpenPINNUnderstandingTrial = api;
  if (global && global.document) {
    if (global.document.readyState === "loading") {
      global.document.addEventListener("DOMContentLoaded", api.initialize);
    } else {
      api.initialize();
    }
  }
}(typeof window !== "undefined" ? window : null, function () {
  "use strict";

  var LLM_REQUEST_TIMEOUT_MS = 300000;
  var NUMBER = "[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?";
  var REQUIRED = ["dim", "physics", "geometry", "field", "constitutive", "boundaries"];
  var QUESTIONS = {
    dim: "问题是二维的吗？请说明空间维数。",
    physics: "要描述哪一种物理方程或本构问题？",
    "geometry.outer": "请给出矩形区域的 x/y 范围。",
    "geometry.holes": "请检查孔洞的中心、半径及其是否严格位于外边界内。",
    field: "未知场是什么？例如位移分量 ux、uy。",
    "constitutive.E": "材料的杨氏模量 E 是多少？请同时注明单位。",
    "constitutive.nu": "材料的泊松比 nu 是多少？",
    boundaries: "请说明每条外边界上的 Dirichlet/Neumann 条件。",
    "boundaries.left": "左边界条件是什么？",
    "boundaries.right": "右边界条件是什么？",
    "boundaries.top": "上边界条件是什么？",
    "boundaries.bottom": "下边界条件是什么？",
    "boundaries.hole_0": "圆孔边界条件是什么？",
    geometry: "请给出求解区域的范围。",
    "pde.diffusivity": "热传导系数或扩散系数是多少？",
    "pde.source": "方程的源项 f 是什么？",
    pde: "请给出该物理问题的控制方程或 PDE 描述。",
    initial_condition: "初始条件是什么？",
    L: "L 的数值是多少？"
  };
  var EN_QUESTIONS = {
    dim: "Is the problem two-dimensional? Specify its spatial dimension.",
    physics: "Which governing physics or constitutive problem should be described?",
    "geometry.outer": "Give the rectangle's x and y ranges.",
    "geometry.holes": "Check each hole's center, radius, and strict containment inside the outer boundary.",
    field: "What is the unknown field, for example displacement components ux and uy?",
    "constitutive.E": "What is Young's modulus E? Include its unit.",
    "constitutive.nu": "What is Poisson's ratio nu?",
    boundaries: "Describe the Dirichlet or Neumann condition on every outer edge.",
    "boundaries.left": "What is the left-edge boundary condition?",
    "boundaries.right": "What is the right-edge boundary condition?",
    "boundaries.top": "What is the top-edge boundary condition?",
    "boundaries.bottom": "What is the bottom-edge boundary condition?",
    "boundaries.hole_0": "What is the circular-hole boundary condition?",
    geometry: "Give the domain bounds.",
    "pde.diffusivity": "What is the thermal diffusivity or diffusion coefficient?",
    "pde.source": "What is the source term f?",
    pde: "What is the governing equation or PDE for this physics problem?",
    initial_condition: "What is the initial condition?",
    L: "What is the numeric value of L?"
  };

  function normalized(text) {
    return String(text || "").toLowerCase().replace(/\s+/g, " ");
  }

  function has(text) {
    var value = normalized(text);
    return Array.prototype.slice.call(arguments, 1).some(function (phrase) {
      return value.includes(normalized(phrase));
    });
  }

  function firstPair(patterns, text) {
    for (var index = 0; index < patterns.length; index += 1) {
      var match = new RegExp(patterns[index], "i").exec(text);
      if (match) return [Number(match[1]), Number(match[2])];
    }
    return null;
  }

  function firstNumber(patterns, text) {
    for (var index = 0; index < patterns.length; index += 1) {
      var match = new RegExp(patterns[index], "i").exec(text);
      if (match) return Number(match[1]);
    }
    return null;
  }

  function extractGeometry(text) {
    var xBounds = firstPair([
      "x\\s*(?:=|范围(?:为)?|从)?\\s*(" + NUMBER + ")\\s*(?:到|至|to|~|～|,|，)\\s*(" + NUMBER + ")",
      "(" + NUMBER + ")\\s*(?:<=|≤)\\s*x\\s*(?:<=|≤)\\s*(" + NUMBER + ")"
    ], text);
    var yBounds = firstPair([
      "y\\s*(?:=|范围(?:为)?|从)?\\s*(" + NUMBER + ")\\s*(?:到|至|to|~|～|,|，)\\s*(" + NUMBER + ")",
      "(" + NUMBER + ")\\s*(?:<=|≤)\\s*y\\s*(?:<=|≤)\\s*(" + NUMBER + ")"
    ], text);
    if (!xBounds || !yBounds || !has(text, "矩形", "rectangle")) return null;
    var geometry = {
      outer: {
        type: "rectangle", x_min: xBounds[0], x_max: xBounds[1],
        y_min: yBounds[0], y_max: yBounds[1]
      },
      holes: []
    };
    var center = firstPair([
      "(?:圆孔|孔|hole).*?(?:中心|center)\\s*(?:为|at)?\\s*[（(]\\s*(" +
        NUMBER + ")\\s*[,，]\\s*(" + NUMBER + ")\\s*[)）]"
    ], text);
    var radius = firstNumber([
      "(?:圆孔|孔|hole).*?(?:半径|radius)\\s*(?:为|=|is)?\\s*(" + NUMBER + ")"
    ], text);
    if (center && radius !== null) {
      geometry.holes.push({type: "circle", center: center, radius: radius});
    }
    return geometry;
  }

  function freeBoundary(region) {
    return {
      id: "free_" + (region === "hole_0" ? "hole" : region),
      region: region, type: "neumann", dofs: ["ux", "uy"], value: [0, 0]
    };
  }

  function extractBoundaries(text, hasHole) {
    var boundaries = [];
    if (has(
      text, "左边固定", "左边固支", "左边界固定", "left edge fixed",
      "left edge is fixed", "fixed left"
    )) {
      boundaries.push({
        id: "fix_left", region: "left", type: "dirichlet",
        dofs: ["ux", "uy"], value: [0, 0]
      });
    }
    var traction = firstNumber([
      "(?:右边界|右边|right(?:\\s+edge)?).*?(?:x\\s*方向)?" +
        "(?:牵引|traction|载荷|load).*?(?:为|=|is)?\\s*(" + NUMBER + ")"
    ], text);
    if (traction !== null) {
      boundaries.push({
        id: "traction_right", region: "right", type: "neumann",
        dofs: ["ux"], value: traction
      });
      boundaries.push({
        id: "shear_right", region: "right", type: "neumann",
        dofs: ["uy"], value: 0
      });
    }
    var freeAll = has(
      text, "上下边和孔边自由", "上、下边和孔边自由", "上边、下边和孔边自由",
      "top, bottom, and hole boundaries are traction-free",
      "top, bottom and hole boundaries are traction-free"
    );
    if (freeAll || has(
      text, "上下边自由", "上、下边自由", "top and bottom are traction-free",
      "top and bottom edges are traction-free"
    )) {
      boundaries.push(freeBoundary("top"), freeBoundary("bottom"));
    }
    if (hasHole && (freeAll || has(
      text, "孔边自由", "圆孔边界自由", "hole boundary is traction-free"
    ))) boundaries.push(freeBoundary("hole_0"));
    return boundaries;
  }

  function ruleExtract(text) {
    var draft = {};
    if (has(text, "二维", "2d", "two-dimensional")) draft.dim = 2;
    if (has(text, "线弹性", "linear elasticity", "linear elastic")) {
      draft.physics = "linear_elasticity";
    }
    var geometry = extractGeometry(text);
    if (geometry) draft.geometry = geometry;
    if (has(text, "位移", "displacement", "ux", "uy")) {
      draft.field = {kind: "displacement", components: ["ux", "uy"]};
    }
    var youngs = firstNumber([
      "(?:杨氏模量|young['’]?s modulus|\\bE)\\s*(?:为|=|is)?\\s*(" + NUMBER + ")"
    ], text);
    var poisson = firstNumber([
      "(?:泊松比|poisson['’]?s ratio|\\bnu|ν)\\s*(?:为|=|is)?\\s*(" + NUMBER + ")"
    ], text);
    var constitutive = {};
    if (draft.physics === "linear_elasticity") {
      constitutive.model = "isotropic_linear";
      constitutive.stress_voigt = true;
    }
    if (youngs !== null) constitutive.E = youngs;
    if (poisson !== null) constitutive.nu = poisson;
    if (Object.keys(constitutive).length) draft.constitutive = constitutive;
    var boundaries = extractBoundaries(text, !!(geometry && geometry.holes.length));
    if (boundaries.length) draft.boundaries = boundaries;
    return draft;
  }

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function deepMerge(target, additions) {
    Object.keys(additions || {}).forEach(function (key) {
      var value = additions[key];
      if (key.includes(".")) {
        var parts = key.split("."), cursor = target;
        parts.slice(0, -1).forEach(function (part, index) {
          var nextIsIndex = /^\d+$/.test(parts[index + 1]);
          if (!cursor[part] || typeof cursor[part] !== "object") {
            cursor[part] = nextIsIndex ? [] : {};
          }
          cursor = cursor[part];
        });
        cursor[parts[parts.length - 1]] = clone(value);
      } else if (
        value && typeof value === "object" && !Array.isArray(value) &&
        target[key] && typeof target[key] === "object" && !Array.isArray(target[key])
      ) {
        deepMerge(target[key], value);
      } else {
        target[key] = clone(value);
      }
    });
  }

  function missingFields(draft) {
    var missing = REQUIRED.filter(function (key) { return !(key in draft); });
    if (draft.geometry && !("outer" in draft.geometry)) missing.push("geometry.outer");
    if (draft.constitutive) {
      ["E", "nu"].forEach(function (key) {
        if (!(key in draft.constitutive)) missing.push("constitutive." + key);
      });
    }
    if (Array.isArray(draft.boundaries)) {
      var regions = new Set(draft.boundaries.map(function (bc) { return bc.region; }));
      var required = ["left", "right", "top", "bottom"];
      var holes = draft.geometry && Array.isArray(draft.geometry.holes) ? draft.geometry.holes : [];
      holes.forEach(function (_hole, index) { required.push("hole_" + index); });
      required.sort().forEach(function (region) {
        if (!regions.has(region)) missing.push("boundaries." + region);
      });
    }
    return missing;
  }

  var KNOWN_SYMBOLS = {
    x: 1, y: 1, z: 1, t: 1, pi: 1, sin: 1, cos: 1, tan: 1, exp: 1, log: 1,
    ln: 1, sqrt: 1, abs: 1, pow: 1, sinh: 1, cosh: 1, tanh: 1
  };

  function finiteNumber(value) {
    return typeof value === "number" && isFinite(value);
  }

  function orderedPair(low, high) {
    return finiteNumber(low) && finiteNumber(high) && high > low;
  }

  function definedNames(draft) {
    var names = {};
    ["parameters", "constants", "coefficients"].forEach(function (key) {
      var value = draft[key];
      if (Array.isArray(value)) {
        value.forEach(function (item) {
          if (!item || typeof item !== "object") return;
          var name = item.name || item.symbol || item.id;
          if (typeof name === "string" && (finiteNumber(item.value) || typeof item.value === "string")) names[name] = 1;
        });
        return;
      }
      if (!value || typeof value !== "object" || Array.isArray(value)) return;
      Object.keys(value).forEach(function (name) {
        var item = value[name];
        if (finiteNumber(item) || typeof item === "string") names[name] = 1;
      });
    });
    Object.keys(draft).forEach(function (key) {
      if (finiteNumber(draft[key])) names[key] = 1;
    });
    var field = draft.field;
    if (field && Array.isArray(field.components)) {
      field.components.forEach(function (component) {
        if (typeof component === "string") names[component] = 1;
      });
    }
    return names;
  }

  function freeSymbols(expression, defined) {
    if (typeof expression !== "string") return [];
    var found = [];
    var tokens = expression.match(/[A-Za-z_][A-Za-z0-9_]*|π/g) || [];
    tokens.forEach(function (token) {
      if (token === "π" || KNOWN_SYMBOLS[token.toLowerCase()] || defined[token]) return;
      if (found.indexOf(token) < 0) found.push(token);
    });
    return found;
  }

  function concreteValue(value, defined) {
    if (finiteNumber(value)) return true;
    if (Array.isArray(value) && value.length && value.every(finiteNumber)) return true;
    return typeof value === "string" && value.trim() && !freeSymbols(value, defined).length;
  }

  function geometryOk(geometry) {
    if (!geometry || typeof geometry !== "object" || Array.isArray(geometry)) return false;
    if (orderedPair(geometry.x_min, geometry.x_max)) {
      if ("y_min" in geometry || "y_max" in geometry) {
        return orderedPair(geometry.y_min, geometry.y_max);
      }
      return true;
    }
    var outer = geometry.outer;
    if (outer && orderedPair(outer.x_min, outer.x_max)) {
      if ("y_min" in outer || "y_max" in outer) return orderedPair(outer.y_min, outer.y_max);
      return true;
    }
    var bounds = geometry.bounds;
    if (bounds && typeof bounds === "object" && !Array.isArray(bounds) && Object.keys(bounds).length) {
      return Object.keys(bounds).every(function (key) {
        var pair = bounds[key];
        return Array.isArray(pair) && pair.length === 2 && orderedPair(pair[0], pair[1]);
      });
    }
    var domain = geometry.domain;
    return Array.isArray(domain) && domain.length === 2 && orderedPair(domain[0], domain[1]);
  }

  function lookupDiffusivity(draft) {
    var pde = draft.pde && typeof draft.pde === "object" && !Array.isArray(draft.pde) ? draft.pde : {};
    var material = draft.constitutive && typeof draft.constitutive === "object" &&
      !Array.isArray(draft.constitutive) ? draft.constitutive : {};
    var pairs = [
      [pde, "diffusivity"], [pde, "alpha"], [pde, "coefficient"],
      [draft, "diffusivity"], [draft, "alpha"], [draft, "thermal_diffusivity"],
      [material, "alpha"], [material, "diffusivity"]
    ];
    for (var index = 0; index < pairs.length; index += 1) {
      if (Object.prototype.hasOwnProperty.call(pairs[index][0], pairs[index][1])) {
        return pairs[index][0][pairs[index][1]];
      }
    }
    return undefined;
  }

  function sourceValue(draft) {
    var pde = draft.pde && typeof draft.pde === "object" && !Array.isArray(draft.pde) ? draft.pde : {};
    if (Object.prototype.hasOwnProperty.call(pde, "source")) return pde.source;
    if (Object.prototype.hasOwnProperty.call(pde, "f")) return pde.f;
    if (Object.prototype.hasOwnProperty.call(draft, "source")) return draft.source;
    return undefined;
  }

  function sourceOk(draft) {
    if (!draft.pde && !Object.prototype.hasOwnProperty.call(draft, "source")) return false;
    var source = sourceValue(draft);
    if (source === undefined) return false;
    var defined = definedNames(draft);
    if (source && typeof source === "object" && !Array.isArray(source)) {
      if (Object.prototype.hasOwnProperty.call(source, "value") && finiteNumber(source.value)) return true;
      return typeof source.expression === "string" && !freeSymbols(source.expression, defined).length;
    }
    if (finiteNumber(source)) return true;
    return typeof source === "string" && source.trim() && !freeSymbols(source, defined).length;
  }

  function listedBoundaryGaps(boundaries, defined) {
    if (!Array.isArray(boundaries) || !boundaries.length) return ["boundaries"];
    var gaps = [];
    boundaries.forEach(function (bc, index) {
      if (!bc || typeof bc !== "object" || Array.isArray(bc)) {
        gaps.push("boundaries[" + index + "]");
        return;
      }
      if (!["dirichlet", "neumann", "robin"].includes(bc.type)) {
        gaps.push("boundaries[" + index + "].type");
      }
      var target = bc.value_expression !== undefined ? bc.value_expression :
        bc.expression !== undefined && !/=/.test(bc.expression) ? bc.expression : bc.value;
      if (!concreteValue(target, defined)) gaps.push("boundaries[" + index + "].value");
    });
    return gaps;
  }

  function sideGaps(boundaries) {
    if (!Array.isArray(boundaries)) return [];
    var regions = {};
    boundaries.forEach(function (bc) {
      if (bc && typeof bc.region === "string") regions[bc.region.toLowerCase()] = 1;
    });
    if (regions.all || regions.boundary || regions.domain_boundary || regions.outer) return [];
    return ["left", "right", "top", "bottom"].filter(function (name) {
      return !regions[name];
    }).map(function (name) { return "boundaries." + name; });
  }

  function isSteady(draft) {
    if (draft.steady === true) return true;
    return !!(draft.pde && draft.pde.steady === true);
  }

  function initialConditionGaps(draft) {
    if (isSteady(draft)) return [];
    var initial = draft.initial_condition;
    if (initial == null) return ["initial_condition"];
    var defined = definedNames(draft);
    if (typeof initial === "string") return freeSymbols(initial, defined);
    if (typeof initial === "object" && !Array.isArray(initial)) {
      if (typeof initial.expression === "string") return freeSymbols(initial.expression, defined);
      if (Object.prototype.hasOwnProperty.call(initial, "value") && finiteNumber(initial.value)) return [];
    }
    return ["initial_condition"];
  }

  function uniqueList(items) {
    var seen = {};
    var result = [];
    items.forEach(function (item) {
      if (!seen[item]) {
        seen[item] = 1;
        result.push(item);
      }
    });
    return result;
  }

  function physicsFamily(physics) {
    if (typeof physics !== "string" || !physics.trim()) return "unknown";
    var key = physics.trim().toLowerCase().replace(/[\s-]+/g, "_");
    if (["linear_elasticity", "elasticity", "linear_elastic"].includes(key)) return "linear_elasticity";
    if (["heat", "heat_equation", "heat_conduction", "heat_transfer", "diffusion", "thermal"].includes(key)) {
      return "heat";
    }
    if (["poisson", "poisson_equation"].includes(key)) return "poisson";
    if (["unknown", "unknown_physics", "unspecified", "other", "其他", "未知"].includes(key)) {
      return "unknown";
    }
    return key;
  }

  function elasticityGaps(draft) {
    var missing = missingFields(draft);
    var material = draft.constitutive;
    if (material && typeof material === "object" && !Array.isArray(material)) {
      if ("E" in material && !(finiteNumber(material.E) && material.E > 0)) missing.push("constitutive.E");
      if ("nu" in material && !(finiteNumber(material.nu) && material.nu >= -1 && material.nu < 0.5)) {
        missing.push("constitutive.nu");
      }
    }
    var outer = draft.geometry && draft.geometry.outer;
    if (outer && ["x_min", "x_max", "y_min", "y_max"].some(function (key) { return key in outer; })) {
      if (!(orderedPair(outer.x_min, outer.x_max) && orderedPair(outer.y_min, outer.y_max))) {
        missing.push("geometry.outer");
      }
    }
    var holes = draft.geometry && draft.geometry.holes;
    if (outer && Array.isArray(holes) && holes.some(function (hole) {
      var center = hole && hole.center, radius = hole && hole.radius;
      return !(hole && hole.type === "circle" && Array.isArray(center) && center.length === 2 &&
        center.every(finiteNumber) && finiteNumber(radius) && radius > 0 &&
        finiteNumber(outer.x_min) && finiteNumber(outer.x_max) &&
        finiteNumber(outer.y_min) && finiteNumber(outer.y_max) &&
        outer.x_min < center[0] - radius && center[0] + radius < outer.x_max &&
        outer.y_min < center[1] - radius && center[1] + radius < outer.y_max);
    })) missing.push("geometry.holes");
    if (Array.isArray(draft.boundaries)) {
      var defined = definedNames(draft);
      draft.boundaries.forEach(function (bc, index) {
        if (!bc || typeof bc !== "object" || Array.isArray(bc)) {
          missing.push("boundaries[" + index + "]");
          return;
        }
        if (!concreteValue(bc.value, defined)) {
          missing.push(typeof bc.region === "string" ? "boundaries." + bc.region : "boundaries[" + index + "].value");
        }
      });
    }
    return missing;
  }

  function heatGaps(draft) {
    var missing = [];
    if (![1, 2, 3].includes(draft.dim)) missing.push("dim");
    if (!geometryOk(draft.geometry)) missing.push("geometry");
    var diffusivity = lookupDiffusivity(draft);
    if (!(finiteNumber(diffusivity) && diffusivity >= 0)) missing.push("pde.diffusivity");
    var defined = definedNames(draft);
    missing = missing.concat(listedBoundaryGaps(draft.boundaries, defined), initialConditionGaps(draft));
    return missing;
  }

  function poissonGaps(draft) {
    var missing = [];
    if (![1, 2, 3].includes(draft.dim)) missing.push("dim");
    var domainOk = geometryOk(draft.geometry);
    if (!domainOk) missing.push("geometry");
    if (!sourceOk(draft)) missing.push("pde.source");
    missing = missing.concat(listedBoundaryGaps(draft.boundaries, definedNames(draft)));
    if (draft.dim === 2 && domainOk) missing = missing.concat(sideGaps(draft.boundaries));
    return missing;
  }

  function fieldOk(field) {
    if (typeof field === "string") return !!field.trim();
    if (!field || typeof field !== "object" || Array.isArray(field)) return false;
    return (typeof field.name === "string" && !!field.name.trim()) ||
      (Array.isArray(field.components) && field.components.length && field.components.every(function (item) {
        return typeof item === "string" && !!item.trim();
      })) || (typeof field.kind === "string" && !!field.kind.trim());
  }

  function pdeOk(draft) {
    if (["expression_source", "equation", "governing_equation"].some(function (key) {
      return typeof draft[key] === "string" && !!draft[key].trim();
    })) return true;
    if (typeof draft.pde === "string") return !!draft.pde.trim();
    return !!(draft.pde && typeof draft.pde === "object" &&
      !Array.isArray(draft.pde) && Object.keys(draft.pde).length);
  }

  function genericDynamic(draft, family) {
    var pde = draft.pde && typeof draft.pde === "object" && !Array.isArray(draft.pde) ? draft.pde : {};
    return draft.kind === "dynamic" || draft.time_dependent === true ||
      pde.kind === "dynamic" || pde.time_dependent === true ||
      ["wave", "transient", "evolution", "dynamic"].some(function (token) {
        return family.indexOf(token) >= 0;
      });
  }

  function genericInitialOk(value, defined) {
    if (finiteNumber(value)) return true;
    if (typeof value === "string") return !!value.trim() && !freeSymbols(value, defined).length;
    if (Array.isArray(value)) return value.length > 0 && value.every(function (item) {
      return genericInitialOk(item, defined);
    });
    if (value && typeof value === "object") {
      var keys = Object.keys(value);
      if (!keys.length) return false;
      if (Object.prototype.hasOwnProperty.call(value, "expression")) {
        return genericInitialOk(value.expression, defined);
      }
      if (Object.prototype.hasOwnProperty.call(value, "value")) {
        return genericInitialOk(value.value, defined);
      }
      return keys.every(function (key) { return genericInitialOk(value[key], defined); });
    }
    return false;
  }

  function genericGaps(draft, family) {
    var missing = [];
    if (![1, 2, 3].includes(draft.dim)) missing.push("dim");
    if (!geometryOk(draft.geometry)) missing.push("geometry");
    if (!fieldOk(draft.field)) missing.push("field");
    if (!pdeOk(draft)) missing.push("pde");
    missing = missing.concat(listedBoundaryGaps(draft.boundaries, definedNames(draft)));
    if (genericDynamic(draft, family)) {
      var initial = draft.initial_condition !== undefined ? draft.initial_condition : draft.initial_conditions;
      if (!genericInitialOk(initial, definedNames(draft))) missing.push("initial_condition");
    }
    return missing;
  }

  function assessFamily(draft) {
    if (!draft || typeof draft !== "object" || Array.isArray(draft)) {
      return {family: "unknown", missing: ["physics"], ready: false};
    }
    var family = physicsFamily(draft.physics);
    var missing = family === "linear_elasticity" ? elasticityGaps(draft) :
      family === "heat" ? heatGaps(draft) :
      family === "poisson" ? poissonGaps(draft) :
      family === "unknown" ? ["physics"] : genericGaps(draft, family);
    missing = uniqueList(missing);
    return {family: family, missing: missing, ready: missing.length === 0};
  }

  function questionsFor(missing, language) {
    var map = language === "en" ? EN_QUESTIONS : QUESTIONS;
    return missing.map(function (item) {
      if (map[item]) return map[item];
      if (/^boundaries\[\d+\]\.value$/.test(item)) {
        return language === "en" ?
          "What is the numeric value or coordinate expression for this boundary condition?" :
          "请给出该边界条件的数值或坐标表达式。";
      }
      if (/^boundaries\[\d+\]\.type$/.test(item)) {
        return language === "en" ?
          "Is this boundary Dirichlet, Neumann, or Robin?" :
          "请说明该边界是 Dirichlet、Neumann 还是 Robin。";
      }
      return language === "en" ? "Please provide " + item + "." : "请补充 " + item + "。";
    });
  }

  function isSchemaNoise(item) {
    return /(^problem\.|agent a|尚不能验证|only linear_elasticity|only dim\s*=\s*2|extra fields are not permitted)/i
      .test(String(item));
  }

  function isElasticityPollution(item) {
    return /constitutive(?:\.(?:e|nu))?\b|young(?:['’]s)?(?:\s+modulus)?|poisson(?:['’]s)?\s+ratio|杨氏模量|泊松比|isotropic[_ -]?linear|stress[_ -]?voigt|\bnu\b|ν|\bux\b|\buy\b|displacement|位移/i
      .test(String(item));
  }

  function filterLlmItems(items, family) {
    return items.filter(function (item) {
      if (isSchemaNoise(item)) return false;
      return family === "linear_elasticity" || !isElasticityPollution(item);
    });
  }

  function reconcileLlm(draft, llmMissing, llmQuestions, language) {
    var assessment = assessFamily(draft);
    if (assessment.ready) return {missing: [], questions: [], ready: true};
    if (assessment.family === "unknown") {
      return {missing: ["physics"], questions: questionsFor(["physics"], language), ready: false};
    }
    var missing = assessment.missing.slice();
    filterLlmItems(llmMissing, assessment.family).forEach(function (item) {
      if (missing.indexOf(item) < 0) missing.push(item);
    });
    var questionsIn = filterLlmItems(llmQuestions, assessment.family);
    return {
      missing: missing,
      questions: questionsIn.length ? uniqueList(questionsIn) : questionsFor(assessment.missing, language),
      ready: false
    };
  }

  var CONFIGURATOR_HANDOFF_KEY = "openpinn-understanding-configurator-v1";
  var METHOD_HANDOFF_KEY = "openpinn-planning-configurator-v1";

  function firstString(values) {
    for (var index = 0; index < values.length; index += 1) {
      if (typeof values[index] === "string" && values[index].trim()) return values[index].trim();
    }
    return null;
  }

  function draftBounds(geometry) {
    if (!geometry || typeof geometry !== "object" || Array.isArray(geometry)) return null;
    var source = geometry.outer && typeof geometry.outer === "object" ? geometry.outer : geometry;
    if (orderedPair(source.x_min, source.x_max)) {
      return {
        xMin: source.x_min, xMax: source.x_max,
        yMin: orderedPair(source.y_min, source.y_max) ? source.y_min : null,
        yMax: orderedPair(source.y_min, source.y_max) ? source.y_max : null
      };
    }
    if (geometry.bounds && typeof geometry.bounds === "object") {
      var x = geometry.bounds.x, y = geometry.bounds.y;
      if (Array.isArray(x) && x.length === 2 && orderedPair(x[0], x[1])) {
        return {
          xMin: x[0], xMax: x[1],
          yMin: Array.isArray(y) && y.length === 2 && orderedPair(y[0], y[1]) ? y[0] : null,
          yMax: Array.isArray(y) && y.length === 2 && orderedPair(y[0], y[1]) ? y[1] : null
        };
      }
    }
    if (Array.isArray(geometry.domain) && geometry.domain.length === 2 &&
        orderedPair(geometry.domain[0], geometry.domain[1])) {
      return {xMin: geometry.domain[0], xMax: geometry.domain[1], yMin: null, yMax: null};
    }
    return null;
  }

  function draftFieldName(field) {
    if (typeof field === "string") return field.trim();
    if (!field || typeof field !== "object" || Array.isArray(field)) return null;
    return firstString([
      field.name,
      Array.isArray(field.components) ? field.components[0] : null,
      field.symbol
    ]);
  }

  function laplacianSource(field, dimension) {
    var source = "dx(dx(" + field + "))";
    if (dimension === 2) source += "+dy(dy(" + field + "))";
    return source;
  }

  function negativeLaplacianSource(field, dimension) {
    var source = "-dx(dx(" + field + "))";
    if (dimension === 2) source += "-dy(dy(" + field + "))";
    return source;
  }

  function normalizeUnit(unit) {
    if (typeof unit !== "string") return null;
    return unit.trim().replace(/²/g, "^2").replace(/·/g, "*") || null;
  }

  function normalizeDraftExpression(source, field, dimension) {
    if (typeof source !== "string" || !source.trim()) return null;
    var escaped = field.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    var laplacian = laplacianSource(field, dimension);
    var normalizedSource = source.trim()
      .replace(/[−–—]/g, "-")
      .replace(/[×·]/g, "*")
      .replace(/÷/g, "/")
      .replace(/π/g, "pi")
      .replace(/α/g, "alpha")
      .replace(/ν/g, "nu")
      .replace(/²/g, "**2");
    normalizedSource = normalizedSource
      .replace(new RegExp("(?:laplacian|Laplacian)\\s*\\(\\s*" + escaped + "\\s*\\)", "g"), "(" + laplacian + ")")
      .replace(new RegExp("(?:Δ|∇\\s*(?:\\*\\*2|2))\\s*\\(?\\s*" + escaped + "\\s*\\)?", "g"), "(" + laplacian + ")")
      .replace(new RegExp("\\b" + escaped + "_xx\\b", "g"), "dx(dx(" + field + "))")
      .replace(new RegExp("\\b" + escaped + "_yy\\b", "g"), "dy(dy(" + field + "))")
      .replace(new RegExp("\\b" + escaped + "_t\\b", "g"), "dt(" + field + ")")
      .replace(/\^/g, "**");
    if (!/^[A-Za-z0-9_+\-*/().,=\s]+$/.test(normalizedSource) || /\*\*\*/.test(normalizedSource)) return null;
    return normalizedSource;
  }

  function numericParameterValue(item) {
    if (finiteNumber(item)) return item;
    if (item && typeof item === "object" && !Array.isArray(item) && finiteNumber(item.value)) return item.value;
    return null;
  }

  function draftExpression(draft, family, field, dimension, warnings) {
    var pde = draft.pde && typeof draft.pde === "object" && !Array.isArray(draft.pde) ? draft.pde : {};
    var components = draft.field && Array.isArray(draft.field.components) ? draft.field.components : [];
    if (components.length > 1) {
      warnings.push("当前公式编辑器只支持单场方程；多分量场未合成为单场公式。");
      return null;
    }
    var raw = firstString([
      draft.expression_source, draft.equation, draft.governing_equation,
      pde.expression_source, pde.expression, pde.equation
    ]);
    if (raw) {
      var normalized = normalizeDraftExpression(raw, field, dimension);
      if (normalized) {
        if (normalized !== raw) warnings.push("控制方程已规范化为编辑器支持的 ASCII 导数语法。");
        if (family === "poisson" && /[Δ∇]/.test(raw) && numericParameterValue(sourceValue(draft)) !== null) {
          return negativeLaplacianSource(field, dimension) + "=f";
        }
        return normalized;
      }
      warnings.push("草稿中的控制方程含编辑器不支持的符号，未覆盖当前公式。");
      return null;
    }
    if (family === "heat" && numericParameterValue(lookupDiffusivity(draft)) !== null) {
      var heatLeft = isSteady(draft) ? "" : "dt(" + field + ")";
      var diffusion = "alpha*(" + laplacianSource(field, dimension) + ")";
      warnings.push("已从热/扩散结构字段合成控制方程，并使用 alpha 表示扩散系数。");
      return (heatLeft ? heatLeft + "-" + diffusion : "-" + diffusion) + "=0";
    }
    if (family === "poisson" && sourceValue(draft) !== undefined) {
      warnings.push("已从 Poisson 源项合成控制方程，并使用 f 表示源项。");
      return negativeLaplacianSource(field, dimension) + "=f";
    }
    warnings.push("草稿没有可合成的单场控制方程，已保留编辑器当前公式并标记为部分预填。");
    return null;
  }

  function explicitParameters(draft, family, warnings) {
    var result = [], symbols = {};
    var units = draft.units && typeof draft.units === "object" && !Array.isArray(draft.units) ? draft.units : {};
    function add(symbol, item, defaultUnit, label) {
      var value = numericParameterValue(item);
      var unit = item && typeof item === "object" && !Array.isArray(item) ? normalizeUnit(item.unit) : null;
      if (!/^[A-Za-z][A-Za-z0-9_]{0,31}$/.test(symbol) || value === null || symbols[symbol]) return;
      unit = unit || normalizeUnit(units[symbol]) || defaultUnit || "1";
      if (!(item && typeof item === "object" && item.unit) && !units[symbol]) {
        warnings.push("参数 " + symbol + " 未声明单位，已采用 " + unit + " 作为预填单位；请核对。");
      }
      symbols[symbol] = true;
      result.push({
        id: "understanding." + symbol, symbol: symbol, value: value, unit: unit,
        label: label || symbol, source: "Understanding draft"
      });
    }
    ["parameters", "constants", "coefficients"].forEach(function (key) {
      var value = draft[key];
      if (Array.isArray(value)) {
        value.forEach(function (item) {
          if (item && typeof item === "object" && typeof item.symbol === "string") add(item.symbol, item, "1");
        });
      } else if (value && typeof value === "object") {
        Object.keys(value).forEach(function (symbol) { add(symbol, value[symbol], "1"); });
      }
    });
    if (family === "heat") add("alpha", lookupDiffusivity(draft), "m^2/s", "扩散系数");
    if (family === "poisson") add("f", sourceValue(draft), "1/m^2", "源项");
    var material = draft.constitutive && typeof draft.constitutive === "object" && !Array.isArray(draft.constitutive) ? draft.constitutive : {};
    add("E", material.E, "Pa", "杨氏模量");
    add("nu", material.nu, "1", "泊松比");
    return result;
  }

  function boundaryRegion(boundary, bounds) {
    if (boundary && typeof boundary.region === "string") return boundary.region.toLowerCase();
    var location = boundary && typeof boundary.location === "string" ? boundary.location.replace(/\s+/g, "") : "";
    if (!bounds) return null;
    if (location === "x=" + bounds.xMin) return "left";
    if (location === "x=" + bounds.xMax) return "right";
    if (bounds.yMin !== null && location === "y=" + bounds.yMin) return "bottom";
    if (bounds.yMax !== null && location === "y=" + bounds.yMax) return "top";
    return null;
  }

  function scalarBoundaryValue(value) {
    if (finiteNumber(value)) return value;
    if (value && typeof value === "object" && !Array.isArray(value) && finiteNumber(value.value)) {
      return value.value;
    }
    return null;
  }

  function draftToPhysicsProject(draft) {
    draft = draft && typeof draft === "object" && !Array.isArray(draft) ? clone(draft) : {};
    var warnings = [], bounds = draftBounds(draft.geometry), family = physicsFamily(draft.physics);
    var dimension = [1, 2].includes(draft.dim) ? draft.dim : null;
    if (draft.dim === 3) warnings.push("当前网页配置器只支持一维或二维；三维维数未自动写入。");
    var kind = genericDynamic(draft, family) || (family === "heat" && !isSteady(draft)) ||
      draft.initial_condition != null ? "dynamic" : "static";
    var field = draftFieldName(draft.field);
    if (field && !/^[A-Za-z][A-Za-z0-9_]{0,31}$/.test(field)) {
      warnings.push("未知场名称不是编辑器支持的英文标识符，已保留当前场名。");
      field = null;
    }
    var expression = draftExpression(draft, family, field || "u", dimension || 1, warnings);
    var units = draft.units && typeof draft.units === "object" && !Array.isArray(draft.units) ? draft.units : {};
    var defaultFieldUnit = family === "heat" ? "K" : family === "poisson" ? "1" : null;
    var spec = {
      field: field,
      expression_source: expression,
      parameters: explicitParameters(draft, family, warnings),
      field_unit: firstString([units.field, units.unknown, units.value, defaultFieldUnit]),
      coordinate_units: {
        x: firstString([units.x, units.length, "m"]),
        y: firstString([units.y, units.length, "m"]),
        t: firstString([units.t, units.time, "s"])
      }
    };
    var boundaries = [];
    (Array.isArray(draft.boundaries) ? draft.boundaries : []).forEach(function (boundary) {
      if (!boundary || typeof boundary !== "object" || Array.isArray(boundary)) return;
      if (Array.isArray(boundary.dofs) && boundary.dofs.length) {
        warnings.push("分量边界条件不能无损映射到当前单场边界控件，已跳过并保留在草稿中。");
        return;
      }
      var region = boundaryRegion(boundary, bounds), value = scalarBoundaryValue(boundary.value);
      if (["left", "right", "top", "bottom"].indexOf(region) < 0 ||
          ["dirichlet", "neumann", "robin"].indexOf(boundary.type) < 0 || value === null) {
        warnings.push("一条边界条件不能无损映射到当前单场边界控件，已跳过并保留在草稿中。");
        return;
      }
      if (boundary.type === "robin" && !(finiteNumber(boundary.alpha) && finiteNumber(boundary.beta))) {
        warnings.push("Robin 边界缺少明确的 alpha/beta，已跳过并保留在草稿中。");
        return;
      }
      boundaries.push({
        region: region, type: boundary.type, value: value,
        alpha: boundary.type === "robin" ? boundary.alpha : null,
        beta: boundary.type === "robin" ? boundary.beta : null
      });
    });
    var holes = draft.geometry && Array.isArray(draft.geometry.holes) ? draft.geometry.holes : [];
    if (holes.length && family !== "linear_elasticity") warnings.push("草稿中的孔洞不会改变求解路径；请在物理问题设置中核对几何。");
    if (draft.initial_conditions && draft.initial_condition == null) {
      warnings.push("多分量 initial_conditions 已保留在草稿中；请在当前单场配置器中手动核对初值表达。");
    }
    return {
      context: {dimension: dimension, kind: kind},
      spec: spec,
      geometry: bounds ? {type: dimension === 2 ? "rectangle" : "interval", bounds: bounds} : null,
      boundaries: boundaries,
      initial_value: draft.initial_condition && finiteNumber(draft.initial_condition.value) ?
        draft.initial_condition.value : (finiteNumber(draft.initial_condition) ? draft.initial_condition : null),
      warnings: uniqueList(warnings)
    };
  }

  function createConfiguratorHandoff(report, options) {
    options = options || {};
    if (!report || typeof report !== "object" || !report.draft || typeof report.draft !== "object") {
      throw new Error("Understanding report is required.");
    }
    return {
      schema_version: 1,
      source: "understanding_trial",
      ready_for_validation: report.ready_for_validation === true,
      missing: Array.isArray(report.missing) ? report.missing.slice() : [],
      draft: clone(report.draft),
      mapping: draftToPhysicsProject(report.draft),
      path: options.path === "operator" ? "operator" : "pinn",
      language: options.language === "en" ? "en" : "zh"
    };
  }

  function issue(code, path, message, severity) {
    return {code: code, path: path, message: message, severity: severity || "error"};
  }

  // Elasticity ProblemSpec preflight kept for callers outside Understanding.
  // missing / questions / ready_for_validation must use assessFamily instead.
  function validateDraft(draft) {
    var errors = [], warnings = [];
    function rejectExtra(object, allowed, path) {
      if (!object || typeof object !== "object" || Array.isArray(object)) return;
      Object.keys(object).forEach(function (key) {
        if (!allowed.includes(key)) {
          errors.push(issue("schema.extra_forbidden", path + "." + key, "Extra fields are not permitted."));
        }
      });
    }
    rejectExtra(draft, ["dim", "physics", "geometry", "field", "constitutive", "boundaries", "units"], "problem");
    if (draft.dim !== 2) errors.push(issue("schema.literal_error", "problem.dim", "Only dim=2 is supported."));
    if (draft.physics !== "linear_elasticity") {
      errors.push(issue("schema.enum", "problem.physics", "Only linear_elasticity is supported."));
    }
    var outer = draft.geometry && draft.geometry.outer;
    rejectExtra(draft.geometry, ["outer", "holes"], "problem.geometry");
    rejectExtra(outer, ["type", "x_min", "x_max", "y_min", "y_max"], "problem.geometry.outer");
    if (!outer || outer.type !== "rectangle" || !(outer.x_max > outer.x_min) || !(outer.y_max > outer.y_min)) {
      errors.push(issue("schema.geometry", "problem.geometry.outer", "Rectangle bounds must be ordered."));
    }
    var holes = draft.geometry && draft.geometry.holes;
    if (!Array.isArray(holes)) {
      errors.push(issue("schema.geometry", "problem.geometry.holes", "Holes must be a list."));
    } else {
      holes.forEach(function (hole, index) {
        rejectExtra(hole, ["type", "center", "radius"], "problem.geometry.holes." + index);
        var center = hole && hole.center;
        if (!hole || hole.type !== "circle" || !Array.isArray(center) || center.length !== 2 ||
            !center.every(Number.isFinite) || !(hole.radius > 0)) {
          errors.push(issue(
            "schema.geometry", "problem.geometry.holes." + index,
            "A circular hole needs a center and positive radius."
          ));
        } else if (outer && !(
          outer.x_min < center[0] - hole.radius && center[0] + hole.radius < outer.x_max &&
          outer.y_min < center[1] - hole.radius && center[1] + hole.radius < outer.y_max
        )) {
          errors.push(issue(
            "schema.geometry", "problem.geometry.holes." + index,
            "The hole must lie strictly inside the rectangle."
          ));
        }
        if (hole && Array.isArray(center) && center.length === 2 && hole.radius > 0) {
          holes.slice(index + 1).forEach(function (other) {
            if (other && Array.isArray(other.center) &&
                other.center.length === 2 && other.radius > 0) {
              var distance = Math.hypot(
                center[0] - other.center[0], center[1] - other.center[1]
              );
              if (distance <= hole.radius + other.radius) {
                errors.push(issue(
                  "schema.geometry", "problem.geometry.holes",
                  "Circular holes must not touch or overlap."
                ));
              }
            }
          });
        }
      });
    }
    var field = draft.field;
    rejectExtra(field, ["kind", "components"], "problem.field");
    if (!field || field.kind !== "displacement" || !Array.isArray(field.components) ||
        !field.components.includes("ux") || !field.components.includes("uy") ||
        new Set(field.components).size !== field.components.length) {
      errors.push(issue("schema.field", "problem.field", "A displacement field with ux and uy is required."));
    }
    var material = draft.constitutive;
    rejectExtra(material, ["model", "E", "nu", "stress_voigt"], "problem.constitutive");
    if (!material || material.model !== "isotropic_linear" || !(material.E > 0) ||
        !(material.nu >= -1 && material.nu < 0.5)) {
      errors.push(issue("schema.constitutive", "problem.constitutive", "Material E and nu are invalid."));
    }
    var boundaries = draft.boundaries;
    if (!Array.isArray(boundaries) || !boundaries.length) {
      errors.push(issue("schema.boundaries", "problem.boundaries", "At least one boundary condition is required."));
    } else {
      var ids = new Set(), assignments = new Map();
      boundaries.forEach(function (bc) {
        rejectExtra(bc, ["id", "region", "type", "dofs", "value"], "problem.boundaries");
        if (!bc || typeof bc !== "object" || Array.isArray(bc)) {
          errors.push(issue("schema.boundary", "problem.boundaries", "A boundary condition is invalid."));
          return;
        }
        if (ids.has(bc.id)) {
          errors.push(issue(
            "boundary.duplicate_id", "problem.boundaries",
            "Boundary ids must be unique."
          ));
        }
        ids.add(bc.id);
        if (typeof bc.id !== "string" || typeof bc.region !== "string" ||
            !["dirichlet", "neumann"].includes(bc.type) || !Array.isArray(bc.dofs) ||
            !bc.dofs.length || new Set(bc.dofs).size !== bc.dofs.length ||
            bc.dofs.some(function (dof) {
              return !field || !Array.isArray(field.components) || !field.components.includes(dof);
            }) ||
            (Array.isArray(bc.value) && bc.value.length !== bc.dofs.length) ||
            (!Array.isArray(bc.value) && !Number.isFinite(bc.value))) {
          errors.push(issue("schema.boundary", "problem.boundaries", "A boundary condition is invalid."));
        }
        (bc.dofs || []).forEach(function (dof) {
          var key = bc.region + ":" + dof, previous = assignments.get(key);
          if (previous && previous !== bc.type) {
            errors.push(issue(
              "schema.boundary_conflict", "problem.boundaries",
              "Dirichlet and Neumann conditions conflict."
            ));
          }
          assignments.set(key, bc.type);
        });
        if (bc.type === "neumann" && Array.isArray(bc.value) &&
            bc.value.every(function (value) { return value === 0; }) &&
            Array.isArray(bc.dofs) && bc.dofs.includes("ux") && bc.dofs.includes("uy")) {
          warnings.push(issue(
            "boundary.traction_free_explicit", "problem.boundaries[id=" + bc.id + "]",
            "Explicit zero traction is equivalent to a natural boundary condition.", "warning"
          ));
        }
      });
    }
    if (draft.units) rejectExtra(draft.units, ["length", "force", "stress"], "problem.units");
    return {ok: errors.length === 0, errors: errors, warnings: warnings};
  }

  function understandProblem(text, options) {
    options = options || {};
    if (typeof text !== "string" || !text.trim()) throw new Error("text must be non-empty");
    if (![undefined, null, "", "pinn", "operator"].includes(options.path)) {
      throw new Error("path must be pinn, operator, or unspecified");
    }
    var draft = ruleExtract(text);
    if (options.answers) deepMerge(draft, options.answers);
    var assessment = assessFamily(draft);
    var warnings = [];
    if (options.path === "operator") {
      warnings.push(issue(
        "understanding.operator_planning_deferred", "path",
        "Operator method planning is outside Problem Understanding.", "warning"
      ));
    }
    return {
      draft: draft,
      missing: assessment.missing,
      questions: questionsFor(assessment.missing, options.language),
      warnings: warnings,
      ready_for_validation: assessment.ready
    };
  }

  function llmEndpoint(baseUrl) {
    var value = String(baseUrl || "").trim().replace(/\/+$/, "");
    if (!value) throw new Error("API base URL is required.");
    if (value.endsWith("/chat/completions")) return value;
    if (value.endsWith("/v1")) return value + "/chat/completions";
    return value + "/chat/completions";
  }

  function validateLlmExtraction(value) {
    function invalid(message) {
      var error = new Error(message);
      error.invalidPayload = true;
      throw error;
    }
    function inspect(item, depth) {
      if (depth > 12) invalid("API draft nesting is too deep.");
      if (item === null || ["string", "number", "boolean"].includes(typeof item)) return;
      if (Array.isArray(item)) {
        item.forEach(function (child) { inspect(child, depth + 1); });
        return;
      }
      if (!item || typeof item !== "object") invalid("API draft is not JSON data.");
      Object.keys(item).forEach(function (key) {
        if (["code", "method", "network", "optimizer", "script", "training", "training_script"].includes(key.toLowerCase())) {
          invalid("API returned an out-of-scope field: " + key);
        }
        inspect(item[key], depth + 1);
      });
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      invalid("API output must be a JSON object.");
    }
    var expected = ["draft", "missing", "questions", "warnings"];
    if (Object.keys(value).sort().join("|") !== expected.slice().sort().join("|")) {
      invalid("API output must contain draft, missing, questions, and warnings.");
    }
    if (!value.draft || typeof value.draft !== "object" || Array.isArray(value.draft)) {
      invalid("API draft must be an object.");
    }
    ["missing", "questions"].forEach(function (name) {
      if (!Array.isArray(value[name]) || !value[name].every(function (item) { return typeof item === "string"; })) {
        invalid("API " + name + " must be an array of strings.");
      }
    });
    if (!Array.isArray(value.warnings)) invalid("API warnings must be an array.");
    var warnings = value.warnings.map(function (warning) {
      if (typeof warning === "string") {
        return issue("understanding.llm_warning", "draft", warning, "warning");
      }
      if (!warning || typeof warning !== "object" || Array.isArray(warning) ||
          Object.keys(warning).sort().join("|") !== "code|message|path" ||
          !["code", "path", "message"].every(function (key) {
            return typeof warning[key] === "string" && warning[key];
          })) invalid("API warning objects require code, path, and message.");
      return issue(warning.code, warning.path, warning.message, "warning");
    });
    inspect(value.draft, 0);
    return {
      draft: clone(value.draft), missing: value.missing.slice(),
      questions: value.questions.slice(), warnings: warnings
    };
  }

  function llmReport(extraction, options) {
    options = options || {};
    var draft = extraction.draft;
    if (typeof options.sourceText === "string" && options.sourceText.trim()) {
      deepMerge(draft, ruleExtract(options.sourceText));
    }
    if (physicsFamily(draft.physics) === "linear_elasticity") {
      if (!draft.constitutive && draft.material && typeof draft.material === "object") {
        draft.constitutive = {
          model: "isotropic_linear",
          E: draft.material.E !== undefined ? draft.material.E : draft.material.youngs_modulus,
          nu: draft.material.nu !== undefined ? draft.material.nu : draft.material.poisson_ratio,
          stress_voigt: true
        };
      }
      if (draft.geometry && draft.geometry.bounds && !draft.geometry.outer) {
        var bounds = draft.geometry.bounds;
        if (Array.isArray(bounds.x) && Array.isArray(bounds.y)) {
          draft.geometry.outer = {
            type: "rectangle", x_min: bounds.x[0], x_max: bounds.x[1],
            y_min: bounds.y[0], y_max: bounds.y[1]
          };
        }
      }
      if (draft.geometry && draft.geometry.hole && !draft.geometry.holes) {
        draft.geometry.holes = [{
          type: "circle", center: clone(draft.geometry.hole.center), radius: draft.geometry.hole.radius
        }];
      }
      if (Array.isArray(draft.boundaries)) {
        draft.boundaries.forEach(function (boundary, index) {
          if (!boundary || typeof boundary !== "object") return;
          if (boundary.region === "hole" || boundary.region === "hole_boundary") boundary.region = "hole_0";
          if (boundary.type === "traction" || boundary.type === "free") boundary.type = "neumann";
          if (boundary.value && typeof boundary.value === "object" && !Array.isArray(boundary.value)) {
            if (finiteNumber(boundary.value.ux) || finiteNumber(boundary.value.uy)) {
              boundary.value = [Number(boundary.value.ux || 0), Number(boundary.value.uy || 0)];
            } else if (finiteNumber(boundary.value.x) || finiteNumber(boundary.value.y)) {
              boundary.value = [Number(boundary.value.x || 0), Number(boundary.value.y || 0)];
            }
          }
          if (!boundary.id) boundary.id = "llm_boundary_" + index;
          if (!boundary.dofs) {
            boundary.dofs = Array.isArray(boundary.value) && boundary.value.length === 1 ? ["ux"] : ["ux", "uy"];
          }
        });
      }
    }
    if (options.answers) deepMerge(draft, options.answers);
    var reconciled = reconcileLlm(
      draft, extraction.missing, extraction.questions, options.language
    );
    var warnings = extraction.warnings.slice();
    if (options.path === "operator") {
      warnings.push(issue(
        "understanding.operator_planning_deferred", "path",
        "Operator method planning is outside Problem Understanding.", "warning"
      ));
    }
    return {
      draft: draft,
      missing: reconciled.missing,
      questions: reconciled.questions,
      warnings: warnings,
      ready_for_validation: reconciled.ready
    };
  }

  function llmPrompt(text, options) {
    var heatExample = {
      draft: {
        dim: 1,
        physics: "heat_equation",
        coordinates: [
          {name: "x", role: "space", unit: "m", bounds: {min: 0, max: 1}},
          {name: "t", role: "time", unit: "s", bounds: {min: 0, max: 1}}
        ],
        time_domain: {t_min: 0, t_max: 1, unit: "s"},
        units: {space: "m", time: "s"},
        geometry: {type: "interval", x_min: 0, x_max: 1},
        field: {name: "u", kind: "temperature", components: ["u"], unit: "K"},
        parameters: [{name: "alpha", value: 0.01, unit: "m^2/s"}],
        pde: {
          type: "diffusion", diffusivity: 0.01,
          expression: "dt(u)-alpha*dx(dx(u))=0"
        },
        boundaries: [
          {location: "x=0", type: "dirichlet", value: null},
          {location: "x=1", type: "neumann", value: null}
        ],
        initial_condition: {expression: "sin(pi*x/L)"}
      },
      missing: ["boundaries[0].value", "boundaries[1].value", "L"],
      questions: [
        "x=0 的 Dirichlet 值是多少？",
        "x=1 的 Neumann 通量是多少？",
        "L 的数值是多少？"
      ],
      warnings: [
        {
          code: "understanding.inferred",
          path: "geometry",
          message: "Interval [0, 1] was inferred from the stated endpoints x=0 and x=1."
        },
        {
          code: "understanding.inferred",
          path: "initial_condition.expression",
          message: "Normalized pai to pi; L remains unknown."
        }
      ]
    };
    var poissonExample = {
      draft: {
        dim: 2,
        physics: "poisson",
        geometry: {type: "quadrilateral", bounds: {x: [0, 1], y: [0, 1]}},
        field: {kind: "scalar", components: ["u"]},
        pde: {
          expression: "-dx(dx(u))-dy(dy(u))=f",
          source: {expression: "1", value: 1}
        },
        boundaries: ["left", "right", "top", "bottom"].map(function (side) {
          return {region: side, type: "dirichlet", value: 0};
        })
      },
      missing: [],
      questions: [],
      warnings: []
    };
    return "Extract problem-side PDE facts only. Do not recommend a solver, model, " +
      "sampler, loss, optimizer, training plan, or Python code. You may reasonably " +
      "extend obvious facts. Normalize spelling such as pai to pi. If endpoints such " +
      "as x=0 and x=1 are stated and no other domain is given, the interval [0, 1] is " +
      "a reasonable inference. Mark every inferred fact in warnings as an object with " +
      "code \"understanding.inferred\", a path, and a message. Do not invent numeric " +
      "values the user did not state, including Dirichlet values, Neumann fluxes, " +
      "symbolic lengths, and material constants; put those true gaps in missing and questions. " +
      "Preserve every explicitly stated coordinate interval and unit as first-class " +
      "draft.coordinates entries with name, role, unit, and bounds {min,max}. For a " +
      "time interval also emit draft.time_domain {t_min,t_max,unit}. Preserve the " +
      "unknown field name, components, and unit. Emit every stated equation parameter " +
      "in draft.parameters with name, value, and unit. If the equation references a " +
      "parameter such as alpha and the text states its value and unit, parameters must " +
      "contain that exact unit-bearing symbol (for example alpha=0.01 m^2/s), even when " +
      "a descriptive pde.diffusivity alias is also emitted. Do not move these " +
      "training-required facts only into warnings, opaque text, or an unmapped block. " +
      "For a boundary target that depends on coordinates or constants, preserve it as " +
      "boundary.value_expression (for example y, y+2, or -pi); do not invent a numeric " +
      "boundary value and do not add a PDE parameter named value. Numeric boundary targets " +
      "remain boundary.value. Preserve multiple stated residuals as pde.residuals or equations. " +
      "Return exactly one JSON object with keys draft, missing, questions, " +
      "warnings. draft must be a JSON object of problem facts for the declared physics " +
      "family; missing and questions must be arrays of strings and may mention only " +
      "fields that family needs. If the physics family is unknown, ask which physics " +
      "it is and do not assume linear elasticity or ask for E, nu, ux, or uy. " +
      "Whenever a single-field PDE can be expressed, include pde.expression or " +
      "top-level expression_source using ASCII editor syntax: dx(u), dy(u), dt(u), " +
      "nested calls for second derivatives, explicit *, and ASCII parameter names. " +
      "Never emit Delta, nabla, unicode subscripts, or implicit multiplication in " +
      "that expression. warnings " +
      "must be an array of strings or objects with code, path, and message strings. " +
      "Do not include ready_for_validation: the application decides readiness from " +
      "physics-family consistency, not from an elasticity schema.\n" +
      "Example 1 input: 一维非定常热传导，x单位m且x∈[0,1]，t单位s且t∈[0,1]，未知温度u单位K；x=0处为Dirichlet边界，x=1处为Neumann边界（数值未给），初值sin(pai*x/L)，alpha=0.01 m^2/s。\n" +
      "Example 1 output:\n" + JSON.stringify(heatExample) + "\n" +
      "Example 2 input: 二维泊松方程，方程定义域为[0, 1]*[0, 1]的四边形，边界条件均为0的Dirichlet边界条件，方程的源项f(x, y)=1。\n" +
      "Example 2 output:\n" + JSON.stringify(poissonExample) + "\n" +
      "Input:\n" + JSON.stringify({
        description: text,
        path_context: options.path || null,
        confirmed_answers: options.answers || {}
      });
  }

  function understandWithCloud(text, options) {
    options = options || {};
    var apiKey = String(options.apiKey || "").trim();
    if (!apiKey) return Promise.reject(new Error("API Key is required."));
    var fetchImpl = options.fetch || (typeof fetch === "function" ? fetch : null);
    if (!fetchImpl) return Promise.reject(new Error("This browser does not provide fetch()."));
    var endpoint;
    try { endpoint = llmEndpoint(options.baseUrl); }
    catch (error) { return Promise.reject(error); }
    var scheduleTimeout = typeof options.scheduleTimeout === "function" ? options.scheduleTimeout : setTimeout;
    var cancelTimeout = typeof options.cancelTimeout === "function" ? options.cancelTimeout : clearTimeout;
    var controller = typeof AbortController === "function" ? new AbortController() : null;
    var timeoutId = controller ? scheduleTimeout(function () { controller.abort(); }, options.timeout || LLM_REQUEST_TIMEOUT_MS) : null;
    var body = {
      model: options.model,
      messages: [
        {role: "system", content: "You extract PDE problem facts as one JSON object. You may normalize obvious notation and infer a domain only from stated endpoints, and you must mark every inferred fact in warnings with code understanding.inferred. Never inject linear-elasticity fields into another declared physics family. Include an editor-legal ASCII pde.expression for every representable single-field PDE. If physics is unknown, ask for its type. Output JSON only."},
        {role: "user", content: llmPrompt(text, options)}
      ],
      temperature: 0,
      response_format: {type: "json_object"}
    };
    return fetchImpl(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer " + apiKey
      },
      body: JSON.stringify(body),
      signal: controller ? controller.signal : undefined
    }).then(function (response) {
      if (!response.ok) throw new Error("API returned HTTP " + response.status + ".");
      return response.json().catch(function () {
        var error = new Error("API response is not valid JSON.");
        error.invalidPayload = true;
        throw error;
      });
    }).then(function (envelope) {
      var content;
      try { content = envelope.choices[0].message.content; }
      catch (_error) { content = null; }
      if (typeof content !== "string") {
        var envelopeError = new Error("API response has an invalid chat envelope.");
        envelopeError.invalidPayload = true;
        throw envelopeError;
      }
      var parsed;
      try { parsed = JSON.parse(content); }
      catch (_error) {
        var contentError = new Error("API message is not valid JSON.");
        contentError.invalidPayload = true;
        throw contentError;
      }
      var reportOptions = Object.assign({}, options, {sourceText: text});
      return llmReport(validateLlmExtraction(parsed), reportOptions);
    }).finally(function () {
      if (timeoutId) cancelTimeout(timeoutId);
    });
  }

  function validatePlan(value, options) {
    options = options || {};
    function invalid(message) {
      var error = new Error(message); error.invalidPayload = true; throw error;
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) invalid("Planning output must be a JSON object.");
    if (!value.method_draft) invalid("Planning output must contain method_draft.");
    var draft = value.method_draft;
    if (!draft || typeof draft !== "object" || Array.isArray(draft)) invalid("Planning method_draft must be an object.");
    var normalizationWarnings = [];
    function warning(path, message) {
      normalizationWarnings.push({code: "planning.browser_normalized", path: path, message: message});
    }
    var normalizedDraft = clone(draft);
    if (typeof normalizedDraft.method === "string" && normalizedDraft.method.indexOf("/") > 0) {
      var methodParts = normalizedDraft.method.split("/");
      normalizedDraft.method = methodParts[0];
      if (!normalizedDraft.variant && methodParts[1]) normalizedDraft.variant = methodParts[1];
      warning("method_draft.method", "Split the combined method/variant identifier.");
    }
    if (normalizedDraft.model !== undefined && normalizedDraft.network === undefined) {
      normalizedDraft.network = normalizedDraft.model; warning("method_draft.network", "Mapped model to network.");
    }
    if (normalizedDraft.sampler !== undefined && normalizedDraft.sampling === undefined) {
      normalizedDraft.sampling = normalizedDraft.sampler; warning("method_draft.sampling", "Mapped sampler to sampling.");
    }
    if (normalizedDraft.derivative !== undefined && normalizedDraft.differentiator === undefined) {
      normalizedDraft.differentiator = normalizedDraft.derivative; warning("method_draft.differentiator", "Mapped derivative to differentiator.");
    }
    ["sampling", "optimizer", "network"].forEach(function (name) {
      if (typeof normalizedDraft[name] === "string") normalizedDraft[name] = {name: normalizedDraft[name]};
    });
    if (typeof normalizedDraft.loss === "string") normalizedDraft.loss = {formulation: normalizedDraft.loss};
    if (typeof normalizedDraft.differentiator === "string") normalizedDraft.differentiator = {mode: normalizedDraft.differentiator};
    if (typeof normalizedDraft.training === "number") normalizedDraft.training = {steps: normalizedDraft.training};
    if (typeof normalizedDraft.training === "string") {
      if (/^\d+$/.test(normalizedDraft.training.trim())) normalizedDraft.training = {steps: Number(normalizedDraft.training)};
      else {
        normalizedDraft.training = undefined;
        warning("method_draft.training", "Replaced an unsupported training label with the browser default.");
      }
    }
    if (normalizedDraft.training && normalizedDraft.training.steps === undefined &&
        Number.isInteger(normalizedDraft.training.epochs)) {
      normalizedDraft.training.steps = normalizedDraft.training.epochs;
      warning("method_draft.training.steps", "Mapped training epochs to steps.");
    }
    var defaults = {
      method: "baseline", variant: "plain_pinn", path: options.path || "pinn",
      sampling: {name: "uniform", interior: 512, boundary: 128},
      loss: {formulation: "strong", weighting: "fixed"},
      optimizer: {name: "adam", lr: 0.001},
      network: {name: "mlp", hidden_layers: [64, 64, 64], activation: "tanh"},
      differentiator: {mode: "autodiff"}, training: {steps: 5000}
    };
    Object.keys(defaults).forEach(function (name) {
      if (normalizedDraft[name] === undefined || normalizedDraft[name] === null) {
        normalizedDraft[name] = clone(defaults[name]);
        warning("method_draft." + name, "Filled the registered browser default for " + name + ".");
      } else if (defaults[name] && typeof defaults[name] === "object" && !Array.isArray(defaults[name]) &&
          normalizedDraft[name] && typeof normalizedDraft[name] === "object" && !Array.isArray(normalizedDraft[name])) {
        Object.keys(defaults[name]).forEach(function (key) {
          if (normalizedDraft[name][key] === undefined) normalizedDraft[name][key] = clone(defaults[name][key]);
        });
      }
    });
    var allowedDraftKeys = ["method", "variant", "path", "sampling", "loss", "optimizer", "network", "differentiator", "training"];
    Object.keys(normalizedDraft).forEach(function (key) {
      if (allowedDraftKeys.indexOf(key) < 0) {
        delete normalizedDraft[key];
        if (["model", "sampler", "derivative"].indexOf(key) < 0) warning("method_draft." + key, "Ignored an unsupported planning field.");
      }
    });
    function rejectExecutableKeys(item) {
      if (Array.isArray(item)) { item.forEach(rejectExecutableKeys); return; }
      if (!item || typeof item !== "object") return;
      Object.keys(item).forEach(function (key) {
        if (["code", "python", "script", "training_script"].indexOf(key.toLowerCase()) >= 0) invalid("Planning method_draft contains executable content.");
        rejectExecutableKeys(item[key]);
      });
    }
    rejectExecutableKeys(draft);
    var rationale = typeof value.rationale === "string" ? [value.rationale] : value.rationale;
    if (!Array.isArray(rationale) || !rationale.length ||
        !rationale.every(function (item) { return typeof item === "string" && item.trim(); })) {
      invalid("Planning rationale must be a non-empty string array.");
    }
    var sourceWarnings = value.warnings === undefined ? [] : value.warnings;
    if (!Array.isArray(sourceWarnings)) invalid("Planning warnings must be an array.");
    var path = String(normalizedDraft.path || "").toLowerCase();
    normalizedDraft.path = path;
    var pinnMethods = ["baseline", "adam_lbfgs", "adaptive_activation", "adaptive_sampling", "adaptive_weights", "domain_decomposition", "fourier", "hard_boundary", "modified_mlp", "time_window", "transfer"];
    var operatorMethods = ["deeponet", "fno", "pino"];
    var ready = path === "pinn" || path === "operator";
    if (!ready) warning("method_draft.path", "The planning path is not registered.");
    if (typeof normalizedDraft.method !== "string" ||
        (path === "pinn" ? pinnMethods : operatorMethods).indexOf(normalizedDraft.method) < 0) {
      ready = false; warning("method_draft.method", "The recommended method is not registered for this path.");
    }
    function componentKey(value) {
      return typeof value === "string" ? value.toLowerCase().replace(/[_ -]+/g, "") : "";
    }
    [
      ["sampling.name", normalizedDraft.sampling && normalizedDraft.sampling.name, ["uniform", "adaptivemovement", "importancesampling", "residualadaptive", "timewindow"]],
      ["loss.formulation", normalizedDraft.loss && normalizedDraft.loss.formulation, ["strong", "interfacestrong"]],
      ["loss.weighting", normalizedDraft.loss && normalizedDraft.loss.weighting, ["fixed", "magnitudepower", "pointwiseselfadaptive", "relobralo", "selfadaptive", "uncertainty"]],
      ["optimizer.name", normalizedDraft.optimizer && normalizedDraft.optimizer.name, ["adam", "adamlbfgs", "adamw", "lbfgs"]],
      ["network.name", normalizedDraft.network && normalizedDraft.network.name, ["mlp", "modifiedmlp", "domaindecomposition"]],
      ["differentiator.mode", normalizedDraft.differentiator && normalizedDraft.differentiator.mode, ["autodiff", "grid", "finitedifference"]]
    ].forEach(function (entry) {
      if (entry[2].indexOf(componentKey(entry[1])) < 0) {
        ready = false;
        warning("method_draft." + entry[0], "The recommended component is not registered.");
      }
    });
    if (!Number.isInteger(normalizedDraft.training.steps) || normalizedDraft.training.steps <= 0) {
      ready = false; warning("method_draft.training.steps", "Training steps must be a positive integer.");
    }
    var warnings = sourceWarnings.map(function (warning) {
      if (typeof warning === "string" && warning.trim()) return warning.trim();
      if (warning && typeof warning === "object" && !Array.isArray(warning) &&
          Object.keys(warning).sort().join("|") === "code|message|path" &&
          ["code", "path", "message"].every(function (key) { return typeof warning[key] === "string" && warning[key]; })) return clone(warning);
      invalid("Planning warning objects require code, path, and message.");
    });
    return {method_draft: normalizedDraft, rationale: rationale.map(function (item) { return item.trim(); }), warnings: warnings.concat(normalizationWarnings), ready_for_validation: ready};
  }

  function planningPrompt(problem, path) {
    return "Plan a solver for this structured PDE problem. Return exactly one JSON object with keys method_draft, rationale, warnings. method_draft must contain registered OpenPINN MethodSpec fields: method, optional variant, path, sampling, loss, optimizer, network, differentiator, training. For a PINN default use baseline/plain_pinn, uniform, strong, fixed, adam, mlp, autodiff. rationale must be short Chinese strings. Do not return Python, scripts, executable code, training results, or a ready flag. Planning does not authorize code generation; Python Specification Validation Agent validate_specs(problem, method_draft) must still pass. Input:\n" + JSON.stringify({problem: problem, path: path || "pinn"});
  }

  function planWithCloud(problem, options) {
    options = options || {};
    var apiKey = String(options.apiKey || "").trim();
    if (!apiKey) return Promise.reject(new Error("API Key is required."));
    var fetchImpl = options.fetch || (typeof fetch === "function" ? fetch : null);
    if (!fetchImpl) return Promise.reject(new Error("This browser does not provide fetch()."));
    var endpoint;
    try { endpoint = llmEndpoint(options.baseUrl); } catch (error) { return Promise.reject(error); }
    var scheduleTimeout = typeof options.scheduleTimeout === "function" ? options.scheduleTimeout : setTimeout;
    var cancelTimeout = typeof options.cancelTimeout === "function" ? options.cancelTimeout : clearTimeout;
    var controller = typeof AbortController === "function" ? new AbortController() : null;
    var timeoutId = controller ? scheduleTimeout(function () { controller.abort(); }, options.timeout || LLM_REQUEST_TIMEOUT_MS) : null;
    var body = {
      model: options.model,
      messages: [
        {role: "system", content: "You recommend a registered OpenPINN MethodSpec as strict JSON only. Never output executable code or claim Specification Validation Agent validation succeeded."},
        {role: "user", content: planningPrompt(problem, options.path)}
      ],
      temperature: 0,
      response_format: {type: "json_object"}
    };
    return fetchImpl(endpoint, {
      method: "POST",
      headers: {"Content-Type": "application/json", "Accept": "application/json", "Authorization": "Bearer " + apiKey},
      body: JSON.stringify(body),
      signal: controller ? controller.signal : undefined
    }).then(function (response) {
      if (!response.ok) throw new Error("API returned HTTP " + response.status + ".");
      return response.json().catch(function () { throw new Error("Planning API response is not valid JSON."); });
    }).then(function (envelope) {
      var content;
      try { content = envelope.choices[0].message.content; } catch (_error) { content = null; }
      if (typeof content !== "string") throw new Error("Planning API response has an invalid chat envelope.");
      var parsed;
      try { parsed = JSON.parse(content); } catch (_error) { throw new Error("Planning API message is not valid JSON."); }
      return validatePlan(parsed, {path: options.path});
    }).finally(function () {
      if (timeoutId) cancelTimeout(timeoutId);
    });
  }

  function planningErrorMessage(error, apiKey, language) {
    var message = error && error.message ? String(error.message) : "Unknown planning error.";
    if (apiKey) message = message.split(String(apiKey)).join("[redacted]");
    message = message.replace(/Bearer\s+[^\s,;]+/gi, "Bearer [redacted]");
    var prefix = language === "en" ? "Planning failed: " : "方法规划失败：";
    if (/failed to fetch|network|cors|load failed/i.test(message)) {
      prefix += language === "en" ? "network or CORS error. " : "网络或 CORS 错误。";
    } else if (/json/i.test(message)) {
      prefix += language === "en" ? "JSON error. " : "JSON 错误。";
    } else if (/schema|planning output|method_draft|rationale|warning/i.test(message)) {
      prefix += language === "en" ? "schema error. " : "结构错误。";
    }
    return prefix + message;
  }

  function understandingErrorMessage(error, apiKey, language) {
    var message = error && error.message ? String(error.message) : "Unknown understanding error.";
    if (apiKey) message = message.split(String(apiKey)).join("[redacted]");
    message = message.replace(/Bearer\s+[^\s,;]+/gi, "Bearer [redacted]");
    var prefix = language === "en" ? "Understanding failed: " : "问题理解失败：";
    if (error && error.name === "AbortError") {
      prefix += language === "en" ? "request timed out or was cancelled. " : "请求超时或已取消。";
    } else if (/failed to fetch|network|cors|load failed/i.test(message)) {
      prefix += language === "en" ? "network or CORS error. " : "网络或 CORS 错误。";
    } else if (/json/i.test(message)) {
      prefix += language === "en" ? "JSON error. " : "JSON 错误。";
    }
    return prefix + message;
  }

  var UI = {
    zh: {
      emptyText: "请先输入问题描述。",
      invalidAnswers: "补充答案必须是 JSON 对象。",
      missingKey: "请先填写 API Key。",
      running: "正在请求用户配置的 API…",
      requestFailed: "API 请求失败：请检查 Key、接口地址、模型、网络与 CORS。",
      keyCleared: "API Key 已从本浏览器清除。",
      ready: "是（当前物理族草稿已自洽）",
      notReady: "否（仍需补充或修正）",
      applyReady: "填入配置器",
      applyPartial: "仍填已有字段",
      partialWarning: "草稿尚未就绪：仅预填已有字段，不代表代码可运行；配置器会保留问题并导出显式 NotImplementedError 脚本。",
      handoffFailed: "无法写入当前标签页的 sessionStorage，请检查浏览器隐私设置。",
      planningRunning: "正在请求方法规划…",
      planningFailed: "方法规划失败：请检查 Key、接口、CORS 与返回 JSON。",
      planningReady: "方法草稿仅通过浏览器结构预检；生码前仍须 Python Specification Validation Agent 校验最终问题—方法组合。",
      empty: "无"
    },
    en: {
      emptyText: "Enter a problem description first.",
      invalidAnswers: "Additional answers must be a JSON object.",
      missingKey: "Enter an API Key first.",
      running: "Requesting the configured API…",
      requestFailed: "API request failed. Check the Key, base URL, model, network, and CORS.",
      keyCleared: "The API Key was cleared from this browser.",
      ready: "Yes (draft is consistent for this physics family)",
      notReady: "No (more information or corrections required)",
      applyReady: "Apply to configurator",
      applyPartial: "Apply available fields",
      partialWarning: "This draft is incomplete. Available fields will be prefilled, but code generation is not ready; failed configurator validation keeps copy and download disabled.",
      handoffFailed: "The handoff could not be saved to this tab's sessionStorage. Check the browser privacy settings.",
      planningRunning: "Requesting a method recommendation…",
      planningFailed: "Planning failed. Check the Key, endpoint, CORS, and returned JSON.",
      planningReady: "The method draft passed only the browser shape check; Python Specification Validation Agent must validate the final problem-method pair before codegen.",
      empty: "None"
    }
  };

  function initialize() {
    var root = document.querySelector("[data-openpinn-understanding-trial]");
    if (!root || root.dataset.ready) return;
    root.dataset.ready = "true";
    var language = /^en/i.test(document.documentElement.lang || "") ? "en" : "zh";
    var labels = UI[language];
    var currentReport = null;
    var currentPlan = null;
    var genericApi = window.OpenPINNGenericValidation || null;
    var hybridApi = window.OpenPINNHybridValidation || null;
    var codebuilderApi = window.OpenPINNCodeBuilderAgent || null;
    var diagnosticApi = window.OpenPINNDiagnosticAgent || null;
    var validationSession = hybridApi ? hybridApi.createSession() : null;
    var buildResult = null;
    var diagnosticArtifact = null;
    var diagnosticReport = null;
    var diagnosticGeneration = 0;
    var lastValidationApiOptions = null;
    if (language === "en") {
      root.querySelectorAll("[data-en]").forEach(function (node) {
        node.textContent = node.dataset.en;
      });
      root.querySelectorAll("[data-en-value]").forEach(function (node) {
        node.value = node.dataset.enValue;
      });
      root.querySelectorAll("[data-en-placeholder]").forEach(function (node) {
        node.placeholder = node.dataset.enPlaceholder;
      });
    }
    function role(name) { return root.querySelector('[data-role="' + name + '"]'); }
    function list(name, values, format) {
      var node = role(name);
      node.replaceChildren();
      if (!values.length) {
        var empty = document.createElement("li");
        empty.textContent = labels.empty;
        node.appendChild(empty);
        return;
      }
      values.forEach(function (value) {
        var item = document.createElement("li");
        item.textContent = format ? format(value) : value;
        node.appendChild(item);
      });
    }
    function invalidateBuild(message) {
      buildResult = null;
      role("build-code").disabled = true;
      role("copy-code").disabled = true;
      role("download-codebuilder").disabled = true;
      role("build-report").textContent = "";
      role("build-source").textContent = "";
      role("build-status").textContent = message || (language === "en" ?
        "Run Hybrid Validation for the current final pair before code generation." :
        "生码前，请先对当前最终组合运行 Hybrid Validation。");
      invalidateDiagnostic(language === "en" ? "Generate current code, run it locally, then choose its RunArtifact." : "生成当前代码并在本地运行后，再选择其 RunArtifact。");
    }
    function invalidateDiagnostic(message) {
      diagnosticGeneration += 1;
      diagnosticArtifact = null;
      diagnosticReport = null;
      role("diagnostic-file").disabled = !diagnosticApi;
      role("diagnostic-file").value = "";
      role("run-diagnostic").disabled = true;
      role("download-diagnostic").disabled = true;
      role("diagnostic-deterministic").textContent = "";
      role("diagnostic-semantic").textContent = "";
      role("diagnostic-report").textContent = "";
      role("diagnostic-summary").textContent = "";
      role("diagnostic-file-status").textContent = message || (language === "en" ?
        "Generate current code, run it locally, then choose its RunArtifact." : "生成当前代码并在本地运行后，再选择其 RunArtifact。");
      role("diagnostic-status").textContent = language === "en" ?
        "Diagnosis is guidance, not proof or Repair." : "诊断只是建议，不是证明，也不是 Repair。";
    }
    function render(report) {
      currentReport = report;
      if (validationSession) validationSession.invalidate();
      role("validate-layered").disabled = true;
      role("download-generic").disabled = true;
      role("cancel-validation").disabled = true;
      role("validation-report").textContent = "";
      role("semantic-report").textContent = "";
      role("execution-capability").textContent = language === "en" ? "Execution capability: not checked." : "执行能力：尚未检查。";
      role("validation-status").textContent = language === "en" ?
        "Hybrid Validation is stale; revalidate the current problem-method pair." :
        "Hybrid Validation 报告已过期；请对当前问题—方法组合重新检查。";
      invalidateBuild(language === "en" ?
        "The previous build is stale. Re-run Planning and Hybrid Validation for this problem." :
        "旧构建已过期；请为当前问题重新运行 Planning 和 Hybrid Validation。");
      role("draft").textContent = JSON.stringify(report.draft, null, 2);
      list("missing", report.missing);
      list("questions", report.questions);
      list("warnings", report.warnings, function (warning) {
        return "[" + warning.code + "] " + warning.message;
      });
      role("ready").textContent = report.ready_for_validation ? labels.ready : labels.notReady;
      role("ready").dataset.value = String(report.ready_for_validation);
      role("apply").textContent = report.ready_for_validation ? labels.applyReady : labels.applyPartial;
      role("apply").setAttribute("aria-disabled", "false");
      role("apply").classList.toggle("is-partial", !report.ready_for_validation);
      role("apply-warning").textContent = report.ready_for_validation ? "" : labels.partialWarning;
      role("results").hidden = false;
      role("plan").disabled = !report.ready_for_validation;
      role("plan-results").hidden = true;
      role("plan-ready").dataset.value = "false";
      role("apply-method").setAttribute("aria-disabled", "true");
      role("apply-joint").setAttribute("aria-disabled", "true");
      currentPlan = null;
    }
    function run() {
      role("input-error").textContent = "";
      if (!role("text").value.trim()) {
        role("input-error").textContent = labels.emptyText;
        if (!currentReport) role("results").hidden = true;
        return;
      }
      var answers = null, rawAnswers = role("answers").value.trim();
      if (rawAnswers) {
        try {
          answers = JSON.parse(rawAnswers);
          if (!answers || Array.isArray(answers) || typeof answers !== "object") throw new Error();
        } catch (_error) {
          role("input-error").textContent = labels.invalidAnswers;
          if (!currentReport) role("results").hidden = true;
          return;
        }
      }
      var options = {
        path: role("path").value || null,
        answers: answers,
        language: language,
        baseUrl: role("base-url").value,
        model: role("model").value,
        apiKey: role("api-key").value
      };
      if (!options.apiKey.trim()) {
        role("input-error").textContent = labels.missingKey;
        if (!currentReport) {
          role("results").hidden = true;
          role("ready").dataset.value = "false";
        }
        return;
      }
      try {
        localStorage.setItem("openpinn-understanding-base-url", options.baseUrl);
        localStorage.setItem("openpinn-understanding-model", options.model);
        localStorage.setItem("openpinn-understanding-api-key", options.apiKey);
      } catch (_error) { /* Storage is optional. */ }
      role("run").disabled = true;
      role("input-error").textContent = labels.running;
      if (!currentReport) {
        role("results").hidden = true;
        role("plan").disabled = true;
        role("ready").dataset.value = "false";
        role("ready").textContent = labels.notReady;
      }
      understandWithCloud(role("text").value, options).then(function (report) {
        role("input-error").textContent = "";
        render(report);
      }).catch(function (error) {
        role("input-error").textContent = labels.requestFailed + " " + understandingErrorMessage(error, options.apiKey, language);
        if (!currentReport) {
          role("results").hidden = true;
          role("ready").dataset.value = "false";
          role("ready").textContent = labels.notReady;
        }
      }).finally(function () {
        role("run").disabled = false;
      });
    }
    try {
      role("base-url").value = localStorage.getItem("openpinn-understanding-base-url") || role("base-url").value;
      role("model").value = localStorage.getItem("openpinn-understanding-model") || role("model").value;
      role("api-key").value = localStorage.getItem("openpinn-understanding-api-key") || "";
      role("plan-base-url").value = localStorage.getItem("openpinn-planning-base-url") || role("base-url").value;
      role("plan-model").value = localStorage.getItem("openpinn-planning-model") || role("model").value;
      role("plan-api-key").value = localStorage.getItem("openpinn-planning-api-key") || role("api-key").value;
      role("validation-base-url").value = localStorage.getItem("openpinn-validation-base-url") || role("plan-base-url").value;
      role("validation-model").value = localStorage.getItem("openpinn-validation-model") || role("plan-model").value;
      role("validation-api-key").value = localStorage.getItem("openpinn-validation-api-key") || "";
      role("diagnostic-base-url").value = localStorage.getItem("openpinn-diagnostic-base-url") || role("validation-base-url").value;
      role("diagnostic-model").value = localStorage.getItem("openpinn-diagnostic-model") || role("validation-model").value;
      role("diagnostic-api-key").value = localStorage.getItem("openpinn-diagnostic-api-key") || "";
    } catch (_error) { /* Storage is optional. */ }
    role("clear-key").addEventListener("click", function () {
      role("api-key").value = "";
      try { localStorage.removeItem("openpinn-understanding-api-key"); }
      catch (_error) { /* Storage is optional. */ }
      role("input-error").textContent = labels.keyCleared;
    });
    role("plan-clear-key").addEventListener("click", function () {
      role("plan-api-key").value = "";
      lastValidationApiOptions = null;
      try { localStorage.removeItem("openpinn-planning-api-key"); }
      catch (_error) { /* Storage is optional. */ }
      role("plan-status").textContent = labels.keyCleared;
    });
    role("validation-clear-key").addEventListener("click", function () {
      role("validation-api-key").value = "";
      lastValidationApiOptions = null;
      try { localStorage.removeItem("openpinn-validation-api-key"); }
      catch (_error) { /* Storage is optional. */ }
      role("validation-status").textContent = "Validation API Key 已从当前浏览器清除。";
    });
    role("diagnostic-clear-key").addEventListener("click", function () {
      role("diagnostic-api-key").value = "";
      try { localStorage.removeItem("openpinn-diagnostic-api-key"); }
      catch (_error) { /* Storage is optional. */ }
      role("diagnostic-status").textContent = language === "en" ?
        "The Diagnostic API Key was cleared from this browser." : "Diagnostic API Key 已从当前浏览器清除。";
    });
    role("apply").addEventListener("click", function (event) {
      if (!currentReport || role("apply").getAttribute("aria-disabled") === "true") {
        event.preventDefault();
        return;
      }
      try {
        var handoff = createConfiguratorHandoff(currentReport, {
          path: role("path").value,
          language: language
        });
        sessionStorage.setItem(CONFIGURATOR_HANDOFF_KEY, JSON.stringify(handoff));
      } catch (_error) {
        event.preventDefault();
        role("apply-warning").textContent = labels.handoffFailed;
      }
    });
    role("plan").addEventListener("click", function () {
      if (!currentReport || !currentReport.ready_for_validation) return;
      invalidateBuild(language === "en" ? "Planning changed; Hybrid Validation must run again." : "规划正在更新；必须重新运行 Hybrid Validation。");
      role("plan").disabled = true;
      role("plan-status").textContent = labels.planningRunning;
      role("plan-results").hidden = true;
      var planOptions = {
        path: role("path").value || "pinn", baseUrl: role("plan-base-url").value,
        model: role("plan-model").value, apiKey: role("plan-api-key").value
      };
      try {
        localStorage.setItem("openpinn-planning-base-url", planOptions.baseUrl);
        localStorage.setItem("openpinn-planning-model", planOptions.model);
        localStorage.setItem("openpinn-planning-api-key", planOptions.apiKey);
      } catch (_error) { /* Storage is optional. */ }
      planWithCloud(currentReport.draft, planOptions).then(function (report) {
        currentPlan = report;
        role("method-draft").textContent = JSON.stringify(report.method_draft, null, 2);
        list("plan-rationale", report.rationale);
        list("plan-warnings", report.warnings, function (warning) {
          return typeof warning === "string" ? warning : "[" + warning.code + "] " + warning.message;
        });
        role("plan-ready").dataset.value = String(report.ready_for_validation);
        role("plan-ready").textContent = labels.planningReady;
        role("plan-status").textContent = labels.planningReady;
        role("plan-results").hidden = false;
        role("apply-method").setAttribute("aria-disabled", "false");
        role("apply-joint").setAttribute("aria-disabled", "false");
        role("validate-layered").disabled = !genericApi || !hybridApi;
      }).catch(function (error) {
        currentPlan = null;
        if (validationSession) validationSession.invalidate();
        role("validate-layered").disabled = true;
        role("download-generic").disabled = true;
        invalidateBuild(language === "en" ? "Planning is incomplete, so skeleton generation is blocked." : "Planning 未完成，骨架生成已阻断。");
        role("plan-ready").dataset.value = "false";
        role("plan-status").textContent = planningErrorMessage(error, planOptions.apiKey, language);
        role("plan-results").hidden = true;
        role("apply-method").setAttribute("aria-disabled", "true");
        role("apply-joint").setAttribute("aria-disabled", "true");
      }).finally(function () { role("plan").disabled = false; });
    });
    role("validate-layered").addEventListener("click", function () {
      if (!validationSession || !currentReport || !currentPlan) return;
      invalidateBuild(language === "en" ? "Hybrid Validation is running; no skeleton is available yet." : "Hybrid Validation 正在运行；尚不能生成骨架。");
      var deterministic = genericApi.validate(currentReport.draft, currentPlan.method_draft);
      role("validation-report").textContent = JSON.stringify(deterministic, null, 2);
      var initialCapability = deterministic.execution_capability || {status: "incomplete", backend_id: null};
      role("execution-capability").textContent = (language === "en" ? "Execution capability: " : "执行能力：") +
        initialCapability.status + (initialCapability.backend_id ? " / " + initialCapability.backend_id : "");
      role("semantic-report").textContent = "Semantic review in progress…";
      role("validation-status").textContent = language === "en" ?
        "Deterministic checks completed; semantic review is running. The browser is not training." :
        "确定性结构检查已完成；正在请求真实 LLM 语义审查。网页不会训练。";
      role("validate-layered").disabled = true;
      role("cancel-validation").disabled = false;
      role("download-generic").disabled = true;
      var reuse = role("validation-reuse-planning").checked;
      var options = {
        baseUrl: reuse ? role("plan-base-url").value : role("validation-base-url").value,
        model: reuse ? role("plan-model").value : role("validation-model").value,
        apiKey: reuse ? role("plan-api-key").value : role("validation-api-key").value,
        timeoutMs: LLM_REQUEST_TIMEOUT_MS,
      };
      lastValidationApiOptions = {
        baseUrl: options.baseUrl, model: options.model, apiKey: options.apiKey,
        timeoutMs: options.timeoutMs,
      };
      try {
        if (!reuse) {
          localStorage.setItem("openpinn-validation-base-url", options.baseUrl);
          localStorage.setItem("openpinn-validation-model", options.model);
          localStorage.setItem("openpinn-validation-api-key", options.apiKey);
        }
      } catch (_error) { /* Storage is optional. */ }
      validationSession.run(currentReport.draft, currentPlan.method_draft, options).then(function (hybrid) {
        if (hybrid.state === "stale") return;
        role("semantic-report").textContent = JSON.stringify(hybrid.semantic, null, 2);
        role("download-generic").disabled = false;
        role("build-code").disabled = !(codebuilderApi && hybrid.semantic.state === "complete" &&
          hybrid.deterministic && !hybrid.deterministic.summary.has_fail);
        role("build-status").textContent = role("build-code").disabled ?
          (language === "en" ? "Code generation is blocked because Hybrid Validation is incomplete or has deterministic failures." : "Hybrid Validation 未完成或存在确定性失败，生码已阻断。") :
          (language === "en" ? "The current validated pair can be passed to deterministic CodeBuilder capability matching." : "当前已验证组合可进入确定性 CodeBuilder 能力匹配。");
        var capability = hybrid.deterministic.execution_capability || {status: "incomplete", backend_id: null};
        role("execution-capability").textContent = (language === "en" ? "Execution capability: " : "执行能力：") +
          capability.status + (capability.backend_id ? " / " + capability.backend_id : "");
        if (hybrid.semantic.state === "complete" && capability.status === "supported") {
          role("validation-status").textContent = language === "en" ?
            "Hybrid Validation is complete. Continue to CodeBuilder for a local training script; the browser does not train." :
            "Hybrid Validation 完成；可进入 CodeBuilder 生成本地训练脚本（网页不训练）。";
        } else if (hybrid.semantic.state === "complete") {
          role("validation-status").textContent = language === "en" ?
            "Hybrid Validation is complete, but this structure has no registered runnable backend; CodeBuilder will keep the non-runnable skeleton." :
            "Hybrid Validation 完成，但当前结构没有已注册的可运行后端；CodeBuilder 将保留不可运行骨架。";
        } else {
          role("validation-status").textContent = (language === "en" ? "Semantic review is incomplete (" : "LLM 语义审查未完成（") +
            hybrid.semantic.code + (language === "en" ? "); no fallback claimed success." : "）；未使用规则伪装成功。");
        }
      }).finally(function () {
        role("validate-layered").disabled = false;
        role("cancel-validation").disabled = true;
      });
    });
    role("cancel-validation").addEventListener("click", function () {
      if (!validationSession) return;
      validationSession.cancel();
      role("semantic-report").textContent = JSON.stringify({state: "incomplete", code: "transport.cancelled", message: "Semantic review cancelled by user; no fallback was used."}, null, 2);
      role("validation-status").textContent = "LLM 语义审查已取消；报告未完成，训练导出仍锁定。";
      role("cancel-validation").disabled = true;
      role("validate-layered").disabled = false;
      role("download-generic").disabled = true;
      invalidateBuild(language === "en" ? "Semantic review was cancelled; skeleton generation is blocked." : "语义审查已取消，骨架生成已阻断。");
    });
    role("build-code").addEventListener("click", function () {
      if (!codebuilderApi || !validationSession || !currentReport || !currentPlan || role("build-code").disabled) return;
      try {
        var validationArtifact = validationSession.exportArtifacts(currentReport.draft, currentPlan.method_draft);
        buildResult = codebuilderApi.build(validationArtifact.report, currentReport.draft, currentPlan.method_draft);
        role("build-report").textContent = JSON.stringify(buildResult.report, null, 2);
        role("build-source").textContent = buildResult.source || "";
        role("copy-code").disabled = !buildResult.report.ok;
        role("download-codebuilder").disabled = !buildResult.report.ok;
        role("diagnostic-file").disabled = !diagnosticApi;
        role("diagnostic-file-status").textContent = buildResult.report.ok ?
          (language === "en" ? "Run the generated Python locally to create openpinn_run/run_artifact.json, then upload JSON or ZIP." : "在本地运行生成的 Python 以产生 openpinn_run/run_artifact.json，再上传 JSON 或 ZIP。") :
          (language === "en" ? "Build failed; Diagnostic upload remains locked." : "构建失败，Diagnostic 上传仍锁定。");
        var concernCount = buildResult.report.semantic_concerns_visible.length;
        role("build-status").textContent = buildResult.report.ok ?
          (buildResult.report.backend_runnable ?
            (language === "en" ? "Runnable local script generated; use explicit budgets or --smoke. Training was not started in the browser." : "已生成本地可运行脚本；请使用显式预算或 --smoke。网页未启动训练。") :
            (language === "en" ? "Non-runnable skeleton generated; backend_runnable=false and no training was started." : "已生成不可运行骨架；backend_runnable=false，未启动训练。")) +
            (concernCount ? (language === "en" ? " Semantic concerns remain visible and require review." : " 语义 concerns 保持可见，仍需人工复核。") : "") :
          (language === "en" ? "Skeleton generation was blocked; inspect BuildReport errors." : "骨架生成已阻断；请检查 BuildReport errors。");
      } catch (error) {
        invalidateBuild(error.message);
      }
    });
    role("copy-code").addEventListener("click", function () {
      if (!buildResult || !buildResult.source || role("copy-code").disabled) return;
      var copied = navigator.clipboard && navigator.clipboard.writeText ? navigator.clipboard.writeText(buildResult.source) : Promise.reject(new Error("Clipboard API unavailable"));
      copied.then(function () {
        role("build-status").textContent = buildResult.report.backend_runnable ?
          (language === "en" ? "Runnable local script copied." : "本地可运行脚本已复制。") :
          (language === "en" ? "Non-runnable skeleton copied." : "不可运行骨架已复制。");
      }).catch(function () {
        role("build-status").textContent = language === "en" ? "Clipboard access failed; use the visible skeleton text." : "剪贴板访问失败；请使用页面中显示的骨架文本。";
      });
    });
    role("download-codebuilder").addEventListener("click", function () {
      if (!buildResult || !buildResult.source || role("download-codebuilder").disabled) return;
      var blob = new Blob([buildResult.source], {type: "text/x-python;charset=utf-8"});
      var url = URL.createObjectURL(blob), anchor = document.createElement("a");
      anchor.href = url; anchor.download = buildResult.filename;
      document.body.appendChild(anchor); anchor.click(); anchor.remove();
      window.setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    });
    role("diagnostic-file").addEventListener("change", function () {
      diagnosticGeneration += 1;
      var fileGeneration = diagnosticGeneration;
      var file = role("diagnostic-file").files && role("diagnostic-file").files[0];
      diagnosticArtifact = null; diagnosticReport = null;
      role("run-diagnostic").disabled = true; role("download-diagnostic").disabled = true;
      role("diagnostic-report").textContent = ""; role("diagnostic-semantic").textContent = ""; role("diagnostic-summary").textContent = "";
      if (!diagnosticApi || !file) return;
      role("diagnostic-file-status").textContent = language === "en" ? "Reading RunArtifact locally…" : "正在本地读取 RunArtifact…";
      diagnosticApi.readArtifactFile(file).then(function (artifact) {
        if (fileGeneration !== diagnosticGeneration) return;
        diagnosticArtifact = artifact;
        var expectedFingerprint = buildResult && buildResult.report && buildResult.report.pair_fingerprint || null;
        var deterministic = diagnosticApi.evaluateArtifact(artifact, expectedFingerprint);
        role("diagnostic-deterministic").textContent = JSON.stringify(deterministic, null, 2);
        role("run-diagnostic").disabled = !deterministic.valid;
        role("diagnostic-file-status").textContent = deterministic.valid ?
          (language === "en" ? "RunArtifact loaded locally; ready for dual-track diagnosis." : "RunArtifact 已在本地载入，可运行双轨诊断。") :
          (language === "en" ? "RunArtifact schema is invalid; stale or modified code alone is only a warning." : "RunArtifact schema 无效；仅指纹过期或代码修改只会产生提示。");
      }).catch(function (error) {
        if (fileGeneration !== diagnosticGeneration) return;
        role("diagnostic-deterministic").textContent = "";
        role("diagnostic-file-status").textContent = error.message;
      });
    });
    role("run-diagnostic").addEventListener("click", function () {
      if (!diagnosticApi || !diagnosticArtifact || role("run-diagnostic").disabled) return;
      diagnosticGeneration += 1;
      var runGeneration = diagnosticGeneration;
      var reuse = role("diagnostic-reuse-validation").checked;
      var reusedValidation = lastValidationApiOptions || {
        baseUrl: role("validation-base-url").value,
        model: role("validation-model").value,
        apiKey: role("validation-api-key").value,
      };
      var options = {
        baseUrl: reuse ? reusedValidation.baseUrl : role("diagnostic-base-url").value,
        model: reuse ? reusedValidation.model : role("diagnostic-model").value,
        apiKey: reuse ? reusedValidation.apiKey : role("diagnostic-api-key").value,
        timeoutMs: LLM_REQUEST_TIMEOUT_MS,
      };
      try {
        if (!reuse) {
          localStorage.setItem("openpinn-diagnostic-base-url", options.baseUrl);
          localStorage.setItem("openpinn-diagnostic-model", options.model);
          localStorage.setItem("openpinn-diagnostic-api-key", options.apiKey);
        }
      } catch (_error) { /* Storage is optional. */ }
      role("run-diagnostic").disabled = true;
      role("diagnostic-status").textContent = language === "en" ?
        "Deterministic checks complete; requesting real LLM diagnosis…" : "确定性检查已完成；正在请求真实 LLM 诊断…";
      var expectedFingerprint = buildResult && buildResult.report && buildResult.report.pair_fingerprint || null;
      diagnosticApi.diagnose(diagnosticArtifact, expectedFingerprint, options).then(function (report) {
        if (runGeneration !== diagnosticGeneration) return;
        diagnosticReport = report;
        role("diagnostic-deterministic").textContent = JSON.stringify(report.deterministic, null, 2);
        role("diagnostic-semantic").textContent = JSON.stringify(report.llm_track, null, 2);
        role("diagnostic-report").textContent = JSON.stringify(report, null, 2);
        role("diagnostic-summary").textContent = report.human_summary;
        role("download-diagnostic").disabled = false;
        role("diagnostic-status").textContent = report.llm_track.state === "complete" ?
          (language === "en" ? "Diagnosis complete; no inputs were modified and repair_performed=false." : "诊断完成；未修改任何输入，repair_performed=false。") :
          (language === "en" ? "LLM diagnosis is incomplete; deterministic findings remain separate and no fallback claimed success." : "LLM 诊断未完成；确定性 findings 保持独立，未用 fallback 冒充成功。");
      }).finally(function () { if (runGeneration === diagnosticGeneration) role("run-diagnostic").disabled = false; });
    });
    role("download-diagnostic").addEventListener("click", function () {
      if (!diagnosticReport || role("download-diagnostic").disabled) return;
      var blob = new Blob([JSON.stringify(diagnosticReport, null, 2)], {type: "application/json;charset=utf-8"});
      var url = URL.createObjectURL(blob), anchor = document.createElement("a");
      anchor.href = url; anchor.download = "openpinn-diagnostic-report.json";
      document.body.appendChild(anchor); anchor.click(); anchor.remove();
      window.setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    });
    role("download-generic").addEventListener("click", function () {
      if (!validationSession || !currentReport || !currentPlan || role("download-generic").disabled) return;
      try {
        var artifact = validationSession.exportArtifacts(currentReport.draft, currentPlan.method_draft);
        var blob = new Blob([JSON.stringify(artifact, null, 2)], {type: "application/json;charset=utf-8"});
        var url = URL.createObjectURL(blob), anchor = document.createElement("a");
        anchor.href = url; anchor.download = "openpinn-non-training-hybrid-validation.json";
        document.body.appendChild(anchor); anchor.click(); anchor.remove();
        window.setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
      } catch (error) {
        role("download-generic").disabled = true;
        role("validation-status").textContent = error.message;
      }
    });
    function consumeHybridRevalidation() {
      var raw;
      try { raw = sessionStorage.getItem("openpinn-hybrid-revalidate-v1"); }
      catch (_error) { return; }
      if (!raw) return;
      try {
        var handoff = JSON.parse(raw);
        if (!handoff || handoff.schema_version !== 1 || handoff.source !== "configurator_current_pair" ||
            !handoff.problem_draft || !handoff.method_draft) throw new Error("Hybrid revalidation handoff is invalid.");
        var assessment = assessFamily(handoff.problem_draft);
        render({draft: handoff.problem_draft, missing: assessment.missing, questions: [], warnings: [],
          ready_for_validation: assessment.ready});
        currentPlan = {method_draft: handoff.method_draft, rationale: ["从配置器当前编辑快照返回。"], warnings: [], ready_for_validation: true};
        role("method-draft").textContent = JSON.stringify(currentPlan.method_draft, null, 2);
        list("plan-rationale", currentPlan.rationale);
        list("plan-warnings", []);
        role("plan-ready").dataset.value = "true";
        role("plan-ready").textContent = "当前配置器快照已导入；必须重新运行 Hybrid Validation。";
        role("plan-results").hidden = false;
        role("validate-layered").disabled = !hybridApi;
        role("validation-status").textContent = "已导入配置器当前 Problem/Method 快照；旧报告无效，请重新运行 Hybrid Validation。";
        sessionStorage.removeItem("openpinn-hybrid-revalidate-v1");
      } catch (error) {
        role("validation-status").textContent = "Hybrid 重新检查导入失败：" + error.message;
      }
    }
    role("apply-method").addEventListener("click", function (event) {
      if (!currentPlan || !currentReport || role("apply-method").getAttribute("aria-disabled") === "true") {
        event.preventDefault(); return;
      }
      try {
        sessionStorage.setItem(METHOD_HANDOFF_KEY, JSON.stringify({
          schema_version: 1, source: "solver_planning", language: language,
          path: currentPlan.method_draft.path, problem_draft: currentReport.draft,
          method_draft: currentPlan.method_draft,
          ready_for_validation: currentPlan.ready_for_validation,
          warnings: currentPlan.warnings
        }));
      } catch (_error) {
        event.preventDefault(); role("plan-status").textContent = labels.handoffFailed;
      }
    });
    var applyJoint = role("apply-joint");
    if (applyJoint) applyJoint.addEventListener("click", function (event) {
      if (!window.OpenPINNJointHandoff) { event.preventDefault(); role("plan-status").textContent = "联合 handoff 模块未加载。"; return; }
      try {
        var envelope = window.OpenPINNJointHandoff.build(currentReport, currentPlan, {
          inputText: role("text").value,
          understandingModel: role("model").value,
          planningModel: role("plan-model").value,
        });
        sessionStorage.setItem(window.OpenPINNJointHandoff.STORAGE_KEY, JSON.stringify(envelope));
      } catch (error) {
        event.preventDefault(); role("plan-status").textContent = "联合导入失败：" + error.message;
      }
    });
    consumeHybridRevalidation();
    role("run").addEventListener("click", run);
  }

  return {
    understandProblem: understandProblem,
    understandWithCloud: understandWithCloud,
    planWithCloud: planWithCloud,
    validatePlan: validatePlan,
    validateDraft: validateDraft,
    assessFamily: assessFamily,
    draftToPhysicsProject: draftToPhysicsProject,
    createConfiguratorHandoff: createConfiguratorHandoff,
    configuratorHandoffKey: CONFIGURATOR_HANDOFF_KEY,
    methodHandoffKey: METHOD_HANDOFF_KEY,
    initialize: initialize
  };
}));
