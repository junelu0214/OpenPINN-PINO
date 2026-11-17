(function (root, factory) {
  "use strict";
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.OpenPINNConfiguratorPaths = api;
})(typeof window !== "undefined" ? window : null, function () {
  "use strict";

  var SAFE_TEXT = /^[A-Za-z0-9_./-]+$/;

  function pythonString(value) {
    var text = String(value);
    if (!SAFE_TEXT.test(text)) {
      throw new Error(
        "生成脚本中的字符串只能包含英文字母、数字、下划线、点、斜杠和连字符。被拒绝的值：" +
        JSON.stringify(text) + "。"
      );
    }
    return "'" + text + "'";
  }

  function formatFloat(value) {
    if (!Number.isFinite(value)) {
      throw new Error("生成脚本中不能写入非有限数值。");
    }
    var text = Object.is(value, -0) ? "-0.0" : String(value);
    if (text.indexOf("e") === -1 && text.indexOf("E") === -1 && text.indexOf(".") === -1) {
      text += ".0";
    }
    return text;
  }

  function formatInt(value) {
    if (!Number.isFinite(value) || Math.trunc(value) !== value) {
      throw new Error("应为整数，实际为 " + value + "。");
    }
    return String(value);
  }

  function isWrappedFloat(value) {
    return Boolean(value) && typeof value === "object" &&
      Object.prototype.hasOwnProperty.call(value, "__float") &&
      Object.keys(value).length === 1;
  }

  function pythonLiteral(value, indent) {
    indent = indent || 0;
    if (typeof value === "boolean") return value ? "True" : "False";
    if (typeof value === "number") return Number.isInteger(value) ? formatInt(value) : formatFloat(value);
    if (typeof value === "string") return pythonString(value);
    if (isWrappedFloat(value)) return formatFloat(value.__float);
    if (Array.isArray(value)) {
      if (!value.length) return "[]";
      return "[" + value.map(function (item) { return pythonLiteral(item, indent); }).join(", ") + "]";
    }
    if (value && typeof value === "object") {
      var keys = Object.keys(value);
      if (!keys.length) return "{}";
      var pad = " ".repeat(indent);
      var inner = " ".repeat(indent + 4);
      var lines = ["{"];
      keys.forEach(function (key) {
        lines.push(inner + pythonString(key) + ": " + pythonLiteral(value[key], indent + 4) + ",");
      });
      lines.push(pad + "}");
      return lines.join("\n");
    }
    throw new Error("无法将此值写入生成脚本。");
  }

  function header(title, filename) {
    return [
      '"""' + title,
      "",
      "Save this file as examples/" + filename + " and run it from the repository root:",
      "",
      "    python examples/" + filename,
      "",
      "Training or solving runs in this local Python process.",
      "The documentation page does not train or solve in the browser.",
      '"""',
      "",
      "from __future__ import annotations",
      "",
      "",
    ].join("\n");
  }

  function downloadStem(name) {
    var cleaned = String(name).replace(/[^A-Za-z0-9_-]/g, "_").replace(/^_+|_+$/g, "");
    if (!cleaned) {
      throw new Error("清理后的脚本名称为空，未替换为默认名称。");
    }
    return cleaned + "_config";
  }

  function copyDefault(value) {
    return Array.isArray(value) ? value.slice() : value;
  }

  function hookByCanonical(payload, canonical) {
    return (payload.operator_hooks || []).find(function (hook) {
      return hook.name === canonical;
    }) || null;
  }

  function choiceByValue(payload, value) {
    return (payload.operator_choices || []).find(function (choice) {
      return choice.value === value;
    }) || null;
  }

  function coerceOption(name, raw, schema) {
    var kind = schema.kind;
    if (kind === "integer") {
      if (typeof raw === "boolean" || typeof raw === "string" && raw.trim() === "") {
        return {error: name + " 必须是整数。"};
      }
      var integer = typeof raw === "number" ? raw : Number(raw);
      if (!Number.isFinite(integer) || Math.trunc(integer) !== integer) {
        return {error: name + " 必须是整数。"};
      }
      if (schema.minimum !== null && schema.minimum !== undefined && integer < Number(schema.minimum)) {
        return {error: name + " 必须大于或等于 " + schema.minimum + "。"};
      }
      return {value: integer};
    }
    if (kind === "number") {
      if (typeof raw === "boolean" || typeof raw === "string" && raw.trim() === "") {
        return {error: name + " 必须是数值。"};
      }
      var number = typeof raw === "number" ? raw : Number(raw);
      if (!Number.isFinite(number)) return {error: name + " 必须是有限数值。"};
      if (schema.minimum !== null && schema.minimum !== undefined && number < Number(schema.minimum)) {
        return {error: name + " 必须大于或等于 " + schema.minimum + "。"};
      }
      return {value: number};
    }
    if (kind === "boolean") {
      if (typeof raw === "boolean") return {value: raw};
      if (raw === "true" || raw === "True") return {value: true};
      if (raw === "false" || raw === "False") return {value: false};
      return {error: name + " 必须是布尔值。"};
    }
    if (kind === "choice") {
      var selected = String(raw).toLowerCase();
      var choices = schema.choices || [];
      if (choices.indexOf(selected) === -1) {
        return {error: name + " 必须是 " + JSON.stringify(choices) + " 之一；该值未替换为默认值。"};
      }
      return {value: selected};
    }
    if (kind === "integer_list") {
      var items = Array.isArray(raw) ? raw : String(raw).split(",");
      var widths = [];
      if (Array.isArray(raw) && !raw.length) return {value: []};
      if (!Array.isArray(raw) && String(raw).trim() === "") return {value: []};
      for (var index = 0; index < items.length; index += 1) {
        var part = typeof items[index] === "string" ? items[index].trim() : items[index];
        if (part === "") continue;
        var parsed = typeof part === "number" ? part : Number(part);
        if (!Number.isFinite(parsed) || Math.trunc(parsed) !== parsed || parsed <= 0) {
          return {error: name + " 中的各项必须是正整数。"};
        }
        widths.push(parsed);
      }
      return {value: widths};
    }
    return {error: "未知 Operator 参数类型：" + kind + "。"};
  }

  function resolveOperatorOptions(hook, supplied) {
    var schema = hook.parameters || {};
    var errors = [];
    var unknown = Object.keys(supplied || {}).filter(function (key) { return !Object.prototype.hasOwnProperty.call(schema, key); });
    if (unknown.length) {
      errors.push(
        "Operator“" + hook.name + "”含未知选项：" + unknown.join(", ") +
        "；可用选项：" + Object.keys(schema).join(", ") + "。未知选项未被丢弃。"
      );
      return {errors: errors, options: null};
    }
    var resolved = {};
    Object.keys(schema).forEach(function (key) {
      var has = supplied && Object.prototype.hasOwnProperty.call(supplied, key) && supplied[key] !== undefined && supplied[key] !== null;
      if (!has) {
        resolved[key] = copyDefault(schema[key].default);
        return;
      }
      var coerced = coerceOption(key, supplied[key], schema[key]);
      if (coerced.error) errors.push(coerced.error);
      else resolved[key] = coerced.value;
    });
    return {errors: errors, options: errors.length ? null : resolved};
  }

  function trainerErrors(hook, options) {
    var limits = hook.trainer_limits;
    if (!limits) {
      return ["Operator“" + hook.name + "”没有已注册的内置训练器，也未选择其他训练器。"];
    }
    var errors = [];
    var mismatched = [];
    Object.keys(limits.equals || {}).forEach(function (key) {
      if (options[key] !== limits.equals[key]) {
        mismatched.push(key + "=" + JSON.stringify(options[key]) + "（内置训练器要求 " + JSON.stringify(limits.equals[key]) + "）");
      }
    });
    if (mismatched.length) errors.push("当前选项不符合内置训练器要求：" + mismatched.join("；"));
    if (limits.grid_size_at_least_twice_modes && Number(options.grid_size) < 2 * Number(options.n_modes)) {
      errors.push("grid_size 必须至少为 n_modes 的两倍；该值未被自动修改。");
    }
    if (limits.sensor_grid_at_least_twice_modes && options.training_mode !== "data_only" && Number(options.n_sensors) < 2 * Number(options.n_modes)) {
      errors.push("n_sensors 必须至少为 n_modes 的两倍；该值未被自动修改。");
    }
    if (limits.weight_rules) {
      if (options.training_mode === "hybrid" && Number(options.data_weight) === 0 && Number(options.physics_weight) === 0) {
        errors.push("hybrid 模式下 data_weight 与 physics_weight 不能同时为 0；训练模式未被修改。");
      } else if (options.training_mode === "physics_only" && Number(options.physics_weight) === 0) {
        errors.push("physics_only 模式下 physics_weight 必须大于 0；训练模式未被修改。");
      } else if (["data_only", "physics_only", "hybrid"].indexOf(options.training_mode) === -1) {
        errors.push("training_mode 必须是 data_only、physics_only 或 hybrid，实际为“" + options.training_mode + "”；训练模式未被修改。");
      }
    }
    return errors;
  }

  function resolveExtensions(payload, supplied) {
    var contract = payload.operator_extensions || {choices: {}, defaults: {}};
    var choices = contract.choices || {};
    var selected = Object.assign({}, contract.defaults || {});
    var incoming = supplied || {};
    var unknown = Object.keys(incoming).filter(function (kind) {
      return !Object.prototype.hasOwnProperty.call(choices, kind);
    }).sort();
    if (unknown.length) {
      return {errors: ["未知 Operator 扩展类型：" + unknown.join(", ") + "。"]};
    }
    var errors = [];
    Object.keys(incoming).forEach(function (kind) {
      var name = String(incoming[kind]);
      if ((choices[kind] || []).indexOf(name) === -1) {
        errors.push(
          "Operator 扩展 " + kind + "='" + name + "' 未注册；可用选项：" +
          (choices[kind] || []).join(", ") + "。未选择备用项。"
        );
      } else {
        selected[kind] = name;
      }
    });
    return {errors: errors, extensions: errors.length ? null : selected};
  }

  function typedOptions(hook, options) {
    var schema = hook.parameters || {};
    var typed = {};
    Object.keys(schema).forEach(function (key) {
      var kind = schema[key].kind;
      var value = options[key];
      if (kind === "number") typed[key] = {__float: value};
      else typed[key] = value;
    });
    return typed;
  }

  function emitCsvVariables(prefix, entry) {
    return [
      prefix + "_FILE = Path(" + pythonLiteral(entry.path) + ")",
      "if not " + prefix + "_FILE.is_absolute():",
      "    " + prefix + "_FILE = PROJECT_ROOT / " + prefix + "_FILE",
      prefix + "_COLUMNS = " + pythonLiteral(entry.columns),
      prefix + "_COORDINATE_SCALES = " + pythonLiteral(entry.scales),
      prefix + "_VALUE_COLUMN = " + pythonLiteral(entry.valueColumn),
      prefix + "_VALUE_SCALE = " + formatFloat(entry.valueScale),
      prefix + "_PREDICTION_SCALE = " + formatFloat(entry.predictionScale),
      prefix + "_FIELD_INDEX = " + formatInt(entry.fieldIndex),
      prefix + "_FIXED_TIME = " + (entry.fixedTime === null ? "None" : formatFloat(entry.fixedTime)),
    ];
  }

  function renderOperatorScript(selectedName, options, hook, designedProblem, extensions, observation, reference) {
    var filename = downloadStem(selectedName) + ".py";
    var lines = [];
    // Observation and reference stay independent. Never copy one path into the other role.
    if (designedProblem || observation || reference) {
      lines.push("from pathlib import Path");
      lines.push("");
    }
    if (designedProblem) {
      lines.push("import math");
      lines.push("");
    }
    lines.push("import openpinn as op");
    if (designedProblem) lines.push("import torch");
    lines.push("");
    if (designedProblem) {
      lines.push(designedProblem.source);
      lines.push("");
      lines.push("DESIGNED_PROBLEM = PROBLEM");
      lines.push("# This PDE and its boundary/initial conditions are the training objective below.");
      lines.push("");
    }
    if (["physics_only", "hybrid"].indexOf(options.training_mode) !== -1) {
      if (hook.name === "deeponet") {
        lines.push("# DeepONet physics differentiates predictions at query coordinates with autodiff.");
      } else {
        lines.push("# FNO physics reuses GridPDEProblem and structured-grid derivatives.");
      }
      lines.push("# The requested training_mode is not rewritten to another mode.");
      lines.push("");
    }
    lines.push("OPERATOR = " + pythonLiteral(String(selectedName)));
    var scriptOptions = typedOptions(hook, options);
    if (designedProblem) {
      if (hook.name === "deeponet") {
        scriptOptions.n_dims = designedProblem.coordinateDimension;
        scriptOptions.n_fields = 1;
      } else {
        scriptOptions.n_dims = designedProblem.coordinateDimension;
        scriptOptions.in_channels = 1;
        scriptOptions.out_channels = 1;
      }
    }
    lines.push("OPTIONS = " + pythonLiteral(scriptOptions));
    lines.push("EXTENSIONS = " + pythonLiteral(extensions));
    lines.push("# EXTENSIONS records registry selections for local trainer customization.");
    lines.push("");
    if (observation || reference) {
      lines.push("PROJECT_ROOT = Path(__file__).resolve().parents[1]");
    }
    if (observation) {
      lines.push.apply(lines, emitCsvVariables("OBSERVATION", observation));
      lines.push("# OBSERVATION_* is the training data term only. It is not reused as the post-train reference.");
      lines.push("");
    }
    if (reference) {
      lines.push.apply(lines, emitCsvVariables("REFERENCE", reference));
      lines.push("# REFERENCE_* is post-train evaluation only. It is not reused as the training observation.");
      lines.push("");
    }
    lines.push("CONFIG = op.build_operator_config(OPERATOR, **OPTIONS)");
    if (designedProblem) {
      lines.push("RESULT = op.train_operator_problem(");
      lines.push("    CONFIG, DESIGNED_PROBLEM,");
      lines.push("    observation=" + (observation ? "OBSERVATION_FILE" : "None") + ",");
      if (observation) {
        lines.push("    coordinate_columns=OBSERVATION_COLUMNS,");
        lines.push("    coordinate_scales=OBSERVATION_COORDINATE_SCALES,");
        lines.push("    value_column=OBSERVATION_VALUE_COLUMN,");
        lines.push("    value_scale=OBSERVATION_VALUE_SCALE,");
        lines.push("    fixed_time=OBSERVATION_FIXED_TIME,");
      }
      lines.push("    input_function=example_field,");
      lines.push("    device=" + pythonLiteral("cpu") + ",");
      lines.push(")");
    } else {
      lines.push("RESULT = op.train_operator(CONFIG, device=" + pythonLiteral("cpu") + ")");
    }
    if (reference) {
      lines.push("REFERENCE_METRICS = op.evaluate_operator_reference_csv(");
      lines.push("    RESULT, REFERENCE_FILE,");
      lines.push("    coordinate_columns=REFERENCE_COLUMNS,");
      lines.push("    coordinate_scales=REFERENCE_COORDINATE_SCALES,");
      lines.push("    value_column=REFERENCE_VALUE_COLUMN,");
      lines.push("    value_scale=REFERENCE_VALUE_SCALE,");
      lines.push("    prediction_scale=REFERENCE_PREDICTION_SCALE,");
      lines.push("    field_index=REFERENCE_FIELD_INDEX,");
      lines.push("    fixed_time=REFERENCE_FIXED_TIME,");
      lines.push(")");
    }
    lines.push("print(");
    lines.push("    {");
    lines.push('        "operator": RESULT.operator,');
    if (designedProblem) {
      lines.push('        "designed_problem": DESIGNED_PROBLEM.name,');
      lines.push('        "designed_dimension": ' + formatInt(designedProblem.dimension) + ',');
      lines.push('        "training_mode": CONFIG["training_mode"],');
      lines.push('        "trainer_scope": "designed_problem",');
    }
    lines.push('        "steps": RESULT.steps,');
    lines.push('        "initial_loss": RESULT.initial_loss,');
    lines.push('        "final_loss": RESULT.final_loss,');
    lines.push("    }");
    lines.push(")");
    lines.push("");
    return header("Local " + hook.title + " training script.", filename) + lines.join("\n");
  }

  function csvEntryErrors(entry, designedProblem, label) {
    var errors = [];
    if (entry.fixedTime !== null && !Number.isFinite(entry.fixedTime)) {
      errors.push(label + "的固定时间必须是有限数值。");
    }
    var expectedColumns = Number(designedProblem.coordinateDimension) - (entry.fixedTime === null ? 0 : 1);
    if (entry.columns.length !== expectedColumns) {
      errors.push(label + " CSV 需要按模型输入顺序填写 " + expectedColumns + " 个坐标列。");
    }
    if (entry.scales.length !== Number(designedProblem.coordinateDimension)) {
      errors.push(label + "的坐标换算系数必须与模型坐标逐一对应。");
    }
    return errors;
  }

  function operatorState(payload, selectedName, supplied, designedProblem, suppliedExtensions, observation, reference) {
    var choice = choiceByValue(payload, selectedName);
    var hook = choice ? hookByCanonical(payload, choice.canonical) : hookByCanonical(payload, selectedName);
    if (!hook) {
      var known = (payload.operator_hooks || []).map(function (item) { return item.name; }).join(", ") || "无";
      return {
        ok: false,
        errors: ["未知 Operator hook“" + selectedName + "”；可用 hook：" + known + "。未选择备用 Operator。"],
      };
    }
    if (!hook.selectable || !hook.implemented || (choice && choice.selectable === false)) {
      return {
        ok: false,
        errors: ["Operator“" + hook.name + "”不可选。请选择已实现的 Operator；页面不会替换为其他项。"],
      };
    }
    var resolved = resolveOperatorOptions(hook, supplied || {});
    if (resolved.errors.length) return {ok: false, errors: resolved.errors};
    var trainer = designedProblem ? [] : trainerErrors(hook, resolved.options);
    if (trainer.length) return {ok: false, errors: trainer};
    var extensionState = resolveExtensions(payload, suppliedExtensions);
    if (extensionState.errors.length) return {ok: false, errors: extensionState.errors};
    // physics_only has no training observation. A filled reference must not take its place.
    if (resolved.options.training_mode === "physics_only") observation = null;
    if (designedProblem) {
      if ([1, 2].indexOf(Number(designedProblem.dimension)) === -1) {
        return {
          ok: false,
          errors: ["Operator PDE 设计需要一维或二维问题，未替换为备用问题。"],
        };
      }
      if (typeof designedProblem.source !== "string" || !designedProblem.source.trim()) {
        return {ok: false, errors: ["设计的问题未生成 Python 源码，复制和下载保持禁用。"]};
      }
      if (![1, 2].includes(Number(designedProblem.coordinateDimension))) {
        return {ok: false, errors: ["Operator 训练支持一或两个模型坐标；尚未启用三维坐标。"]};
      }
      var csvErrors = [];
      if (["data_only", "hybrid"].indexOf(resolved.options.training_mode) !== -1 && !observation) {
        csvErrors.push("training_mode=" + resolved.options.training_mode + " 需要启用观测 CSV 路径。复制和下载保持禁用。");
      }
      if (observation) csvErrors = csvErrors.concat(csvEntryErrors(observation, designedProblem, "观测数据"));
      if (reference) csvErrors = csvErrors.concat(csvEntryErrors(reference, designedProblem, "训练后参考数据"));
      if (csvErrors.length) return {ok: false, errors: csvErrors};
    }
    return {
      ok: true,
      errors: [],
      source: renderOperatorScript(selectedName, resolved.options, hook, designedProblem, extensionState.extensions, observation, reference),
      downloadName: downloadStem(selectedName),
      message: "本地 " + hook.title + " 脚本已就绪。请复制或下载后用 Python 运行；本页面不会执行训练。",
    };
  }

  function wrapFloat(value) {
    return {__float: Number(value)};
  }

  function wrapPlateProblem(problem) {
    return {
      dim: problem.dim,
      physics: problem.physics,
      geometry: {
        outer: {
          type: problem.geometry.outer.type,
          x_min: wrapFloat(problem.geometry.outer.x_min),
          x_max: wrapFloat(problem.geometry.outer.x_max),
          y_min: wrapFloat(problem.geometry.outer.y_min),
          y_max: wrapFloat(problem.geometry.outer.y_max),
        },
        holes: (problem.geometry.holes || []).map(function (hole) {
          return {
            type: hole.type,
            center: [wrapFloat(hole.center[0]), wrapFloat(hole.center[1])],
            radius: wrapFloat(hole.radius),
          };
        }),
      },
      field: {
        kind: problem.field.kind,
        components: problem.field.components.slice(),
      },
      constitutive: {
        model: problem.constitutive.model,
        assumption: problem.constitutive.assumption || "plane_stress",
        E: wrapFloat(problem.constitutive.E),
        nu: wrapFloat(problem.constitutive.nu),
        stress_voigt: Boolean(problem.constitutive.stress_voigt),
      },
      boundaries: problem.boundaries.map(function (bc) {
        return {
          id: bc.id,
          region: bc.region,
          type: bc.type,
          dofs: bc.dofs.slice(),
          value: Array.isArray(bc.value) ? bc.value.map(wrapFloat) : wrapFloat(bc.value),
        };
      }),
      units: {
        length: problem.units.length,
        force: problem.units.force,
        stress: problem.units.stress,
      },
    };
  }

  function wrapPlateMethod(method) {
    var wrapped = {
      method: method.method,
      path: method.path,
      sampling: {
        name: method.sampling.name,
        interior: method.sampling.interior,
        boundary: method.sampling.boundary,
      },
      loss: {
        formulation: method.loss.formulation,
        weights: {
          pde: wrapFloat(method.loss.weights.pde),
          boundary: wrapFloat(method.loss.weights.boundary),
        },
      },
      optimizer: {
        name: method.optimizer.name,
        lr: wrapFloat(method.optimizer.lr),
      },
      network: {
        name: method.network.name,
        hidden_layers: method.network.hidden_layers.slice(),
        activation: method.network.activation,
      },
      differentiator: {mode: method.differentiator.mode},
      training: {steps: method.training.steps},
    };
    if (method.variant) {
      wrapped = {
        method: method.method,
        variant: method.variant,
        path: method.path,
        sampling: wrapped.sampling,
        loss: wrapped.loss,
        optimizer: wrapped.optimizer,
        network: wrapped.network,
        differentiator: wrapped.differentiator,
        training: wrapped.training,
      };
    }
    return wrapped;
  }

  function renderPlateScript(problem, method, settings) {
    var filename = "plate_with_hole_config.py";
    var lines = [
      "from datetime import datetime, timezone",
      "from pathlib import Path",
      "",
      "import openpinn as op",
      "from openpinn.agents import validate_specs",
      "from openpinn.codegen import from_specs",
      "",
      "PROBLEM = " + pythonLiteral(wrapPlateProblem(problem)),
      "METHOD = " + pythonLiteral(wrapPlateMethod(method)),
      "STEPS = " + pythonLiteral(settings.steps),
      "SEED = " + pythonLiteral(settings.seed),
      "DEVICE = " + pythonLiteral(settings.device),
      "OUTPUT_ROOT = Path(" + pythonLiteral(settings.outputDir) + ")",
      'OUTPUT_DIR = OUTPUT_ROOT / ("plate_with_hole_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))',
      "",
      "REPORT = validate_specs(PROBLEM, METHOD)",
      "if not REPORT.ok:",
      "    details = '; '.join(",
      '        f"[{item.code}] {item.path}: {item.message}" for item in REPORT.errors',
      "    )",
      "    raise SystemExit(",
      '        "Specification Validation Agent failed with "',
      '        f"{len(REPORT.errors)} error(s): {details} "',
      '        "No fallback spec was generated."',
      "    )",
      "",
      "PROBLEM_OBJ, KWARGS = from_specs(",
      "    PROBLEM,",
      "    METHOD,",
      "    steps=STEPS,",
      "    device=DEVICE,",
      "    pre_validate=True,",
      ")",
      "try:",
      "    RESULT = op.solve(",
      "        PROBLEM_OBJ,",
      "        **KWARGS,",
      '        run_name="plate_with_hole",',
      "        output_dir=OUTPUT_DIR,",
      "        seed=SEED,",
      "        plot=False,",
      "    )",
      "except BaseException as ERROR:",
      "    FAILED_ARTIFACT_PATH = op.write_diagnostic_run_artifact(",
      "        OUTPUT_DIR, status=\"interrupted\" if isinstance(ERROR, KeyboardInterrupt) else \"failed\",",
      '        exported_pair_snapshot={"problem": PROBLEM, "method": METHOD},',
      "        script_path=__file__, code_origin=\"exported\", source=\"configurator\",",
      '        generator_id="openpinn.configurator.v1", budgets={"max_steps": STEPS, "max_wall_seconds": None},',
      "        steps_completed=0, device=DEVICE, failure=ERROR,",
      "    )",
      '    print("RunArtifact:", FAILED_ARTIFACT_PATH)',
      "    raise",
      "ARTIFACT_PATH = op.write_diagnostic_run_artifact(",
      "    RESULT.output_dir,",
      "    result=RESULT,",
      '    status="budget_exhausted",',
      "    exported_pair_snapshot={\"problem\": PROBLEM, \"method\": METHOD},",
      "    script_path=__file__,",
      '    code_origin="exported",',
      '    source="configurator",',
      '    generator_id="openpinn.configurator.v1",',
      "    budgets={\"max_steps\": STEPS, \"max_wall_seconds\": None},",
      "    steps_completed=len(RESULT.history),",
      "    device=DEVICE,",
      ")",
      'print(RESULT.metrics.get("final_total_loss"))',
      'print("RunArtifact:", ARTIFACT_PATH)',
      "",
    ];
    return header("Local 2D linear-elastic plate-with-hole script.", filename) + lines.join("\n");
  }

  function finiteNumber(value, label, errors) {
    if (typeof value === "string" && value.trim() === "") {
      errors.push(label + " is empty. No default was substituted.");
      return NaN;
    }
    var number = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(number)) {
      errors.push(label + " must be a finite number. The previous value was not kept.");
      return NaN;
    }
    return number;
  }

  function positiveInt(value, label, errors) {
    var number = finiteNumber(value, label, errors);
    if (!Number.isFinite(number)) return NaN;
    if (Math.trunc(number) !== number || number < 1) {
      errors.push(label + " must be a positive integer, got " + value + ". The previous value was not kept.");
      return NaN;
    }
    return number;
  }

  function plateErrors(problem, method, settings, contract) {
    var errors = [];
    var limits = contract.limits || {};
    if ((limits.physics || []).indexOf(problem.physics) === -1) {
      errors.push(
        "[codegen.unsupported_physics] problem.physics: M3 codegen supports physics " +
        JSON.stringify(limits.physics || []) + ", got " + JSON.stringify(problem.physics) +
        ". No other physics was substituted."
      );
    }
    if (method.path === "operator") {
      errors.push(contract.operator_path_error || "[codegen.operator_path] method.path: operator path is not valid for plate codegen.");
    } else if ((limits.solver_paths || ["pinn"]).indexOf(method.path) === -1) {
      errors.push(
        "[codegen.unsupported_path] method.path: M3 codegen supports solver paths " +
        JSON.stringify(limits.solver_paths || []) + ", got " + JSON.stringify(method.path) +
        ". No other path was substituted."
      );
    }
    if (method.path === "pinn") {
      var known = (contract.pinn_methods || []).filter(function (item) {
        return item.method === method.method && item.variant === method.variant;
      });
      if (!known.length) {
        errors.push(
          "Unknown OpenPINN method '" + method.method + "'" +
          (method.variant ? " / variant '" + method.variant + "'" : "") +
          ". No fallback to baseline was selected."
        );
      }
    }
    var outer = problem.geometry.outer;
    if (!(outer.x_max > outer.x_min)) {
      errors.push("x_max (" + outer.x_max + ") must be greater than x_min (" + outer.x_min + ").");
    }
    if (!(outer.y_max > outer.y_min)) {
      errors.push("y_max (" + outer.y_max + ") must be greater than y_min (" + outer.y_min + ").");
    }
    var hole = problem.geometry.holes[0];
    if (!(hole.radius > 0)) errors.push("Hole radius must be > 0. The radius was not replaced.");
    var inside = outer.x_min < hole.center[0] - hole.radius &&
      hole.center[0] + hole.radius < outer.x_max &&
      outer.y_min < hole.center[1] - hole.radius &&
      hole.center[1] + hole.radius < outer.y_max;
    if (!inside) {
      errors.push(
        "Hole 0 (center=(" + hole.center[0] + ", " + hole.center[1] + "), radius=" + hole.radius +
        ") does not lie strictly inside the outer rectangle [" + outer.x_min + ", " + outer.x_max +
        "] x [" + outer.y_min + ", " + outer.y_max + "]. No geometry was clipped or replaced."
      );
    }
    if (!(problem.constitutive.E > 0)) errors.push("Young's modulus E must be positive. It was not replaced.");
    if (!(problem.constitutive.nu >= -1 && problem.constitutive.nu < 0.5)) {
      errors.push("Poisson's ratio nu must be in [-1, 0.5). It was not replaced.");
    }
    if ((limits.structural_assumptions || ["plane_stress"]).indexOf(problem.constitutive.assumption) === -1) {
      errors.push("[codegen.unsupported_structural_assumption] problem.constitutive.assumption: M3 codegen supports " +
        JSON.stringify(limits.structural_assumptions || ["plane_stress"]) + ", got " + JSON.stringify(problem.constitutive.assumption) + ".");
    }
    if (!Array.isArray(method.network.hidden_layers) || !method.network.hidden_layers.length ||
        method.network.hidden_layers.some(function (width) { return !(width > 0) || Math.trunc(width) !== width; })) {
      errors.push("Hidden layer widths must be positive integers. They were not replaced with a default network.");
    }
    if (!(method.optimizer.lr > 0)) errors.push("Learning rate must be positive. It was not replaced.");
    if (!(method.sampling.interior >= 1) || !(method.sampling.boundary >= 1)) {
      errors.push("Interior and boundary sample counts must be positive integers.");
    }
    if (!(settings.steps >= 1) || Math.trunc(settings.steps) !== settings.steps) {
      errors.push("steps must be a positive integer. The previous value was not kept.");
    }
    if (!(settings.seed >= 0) || Math.trunc(settings.seed) !== settings.seed) {
      errors.push("seed must be a non-negative integer. It was not replaced with 0.");
    }
    if (!settings.outputDir || !SAFE_TEXT.test(settings.outputDir)) {
      errors.push("output_dir " + JSON.stringify(settings.outputDir || "") + " is empty or contains unsupported characters. It was not replaced.");
    }
    if (!settings.device || !SAFE_TEXT.test(settings.device)) {
      errors.push("device is not a usable script token. It was not replaced with 'cpu'.");
    }
    return errors;
  }

  function plateState(problem, method, settings, contract) {
    var errors = plateErrors(problem, method, settings, contract);
    if (errors.length) return {ok: false, errors: errors};
    return {
      ok: true,
      errors: [],
      source: renderPlateScript(problem, method, settings),
      downloadName: contract.download_name || "plate_with_hole_config",
      message: "Browser contract precheck passed. The exported script still calls Python Specification Validation Agent validate_specs, then M3 from_specs and solve. This page does not claim full Python validation or solve.",
    };
  }

  function plateFingerprint(problem, method) {
    return JSON.stringify({problem: problem, method: method});
  }

  function createPlateValidationSession() {
    var acceptedFingerprint = null;
    var acceptedSource = null;
    return {
      validate: function (problem, method, settings, contract) {
        var state = plateState(problem, method, settings, contract);
        acceptedFingerprint = state.ok ? plateFingerprint(problem, method) : null;
        acceptedSource = state.ok ? state.source : null;
        state.validation = "browser_contract_precheck";
        state.fullPythonValidation = false;
        return state;
      },
      invalidate: function () {
        acceptedFingerprint = null;
        acceptedSource = null;
      },
      isCurrent: function (problem, method) {
        return acceptedFingerprint !== null &&
          acceptedFingerprint === plateFingerprint(problem, method);
      },
      authorizeExport: function (problem, method, settings, contract) {
        if (!this.isCurrent(problem, method)) {
          return {
            ok: false,
            errors: ["Final ProblemSpec/MethodSpec changed after the browser contract precheck. Revalidate before export."],
          };
        }
        var current = plateState(problem, method, settings, contract);
        if (!current.ok || current.source !== acceptedSource) {
          return {
            ok: false,
            errors: (current.errors || []).concat([
              "The export payload differs from the validated payload. Export remains blocked."
            ]),
          };
        }
        current.validation = "browser_contract_precheck";
        current.fullPythonValidation = false;
        return current;
      },
    };
  }

  function readControl(root, role) {
    return root.querySelector('[data-role="' + role + '"]');
  }

  function readOperatorSupplied(root) {
    var supplied = {};
    root.querySelectorAll("[data-operator-parameter]").forEach(function (input) {
      var name = input.dataset.operatorParameter;
      if (input.type === "checkbox") supplied[name] = input.checked;
      else supplied[name] = input.value;
    });
    var trainingMode = readControl(root, "operator-training-mode");
    if (trainingMode) supplied.training_mode = trainingMode.value;
    return supplied;
  }

  function readOperatorExtensions(root) {
    var selected = {};
    root.querySelectorAll("[data-operator-extension]").forEach(function (input) {
      selected[input.dataset.operatorExtension] = input.value;
    });
    return selected;
  }

  function readCsvEntry(root, prefix, label) {
    var enabled = readControl(root, prefix + "-enable");
    if (!enabled || !enabled.checked || enabled.disabled) return null;
    var pathInput = readControl(root, prefix + "-path");
    var path = pathInput ? pathInput.value.trim() : "";
    if (!path) throw new Error("启用" + label + "时必须填写 CSV 运行时路径。");
    var columns = readControl(root, prefix + "-coordinates").value.split(",").map(function (item) { return item.trim(); });
    var scales = readControl(root, prefix + "-coordinate-scales").value.split(",").map(function (item) { return Number(item.trim()); });
    var valueColumn = readControl(root, prefix + "-value").value.trim();
    var valueScale = Number(readControl(root, prefix + "-value-scale").value);
    var predictionScale = Number(readControl(root, prefix + "-prediction-scale").value);
    var fieldIndex = Number(readControl(root, prefix + "-field-index").value);
    var fixedText = readControl(root, prefix + "-fixed-time").value.trim();
    if (!columns.length || columns.some(function (item) { return !item; })) throw new Error(label + "的坐标列不能为空。");
    if (scales.some(function (item) { return !Number.isFinite(item) || item <= 0; })) throw new Error(label + "的坐标换算系数必须是正数。");
    if (!valueColumn || columns.indexOf(valueColumn) !== -1) throw new Error(label + "的场值列必须与坐标列不同。");
    if (!Number.isFinite(valueScale) || valueScale === 0) throw new Error(label + "的场值换算系数必须是非零有限数值。");
    if (!Number.isFinite(predictionScale) || predictionScale === 0) throw new Error(label + "的预测值换算系数必须是非零有限数值。");
    if (fieldIndex !== 0) throw new Error(label + "目前只支持标量场分量 0。");
    return {path: path, columns: columns, scales: scales, valueColumn: valueColumn, valueScale: valueScale, predictionScale: predictionScale, fieldIndex: fieldIndex, fixedTime: fixedText === "" ? null : Number(fixedText)};
  }

  function readOperatorObservation(root) {
    var panel = root.querySelector(".configurator-observation-panel");
    if (panel && panel.hidden) return null;
    return readCsvEntry(root, "observation", "观测数据");
  }

  function readOperatorReference(root) {
    return readCsvEntry(root, "reference", "训练后参考数据");
  }

  function selectedOperator(root) {
    var select = readControl(root, "operator-model");
    if (select) return select.value;
    var local = readControl(root, "local-script");
    return local ? local.value : "";
  }

  function renderOperatorFromDom(root, payload, designedProblem) {
    return operatorState(
      payload,
      selectedOperator(root),
      readOperatorSupplied(root),
      designedProblem,
      readOperatorExtensions(root),
      readOperatorObservation(root),
      readOperatorReference(root)
    );
  }

  function nonNegativeInt(value, label, errors) {
    var number = finiteNumber(value, label, errors);
    if (!Number.isFinite(number)) return NaN;
    if (Math.trunc(number) !== number || number < 0) {
      errors.push(label + " must be a non-negative integer, got " + value + ". It was not replaced with 0.");
      return NaN;
    }
    return number;
  }

  function readPlate(root, contract) {
    var errors = [];
    var base = JSON.parse(JSON.stringify(contract.problem));
    var method = JSON.parse(JSON.stringify(contract.method));
    base.physics = readControl(root, "plate-physics").value;
    base.geometry.outer.x_min = finiteNumber(readControl(root, "plate-x-min").value, "x_min", errors);
    base.geometry.outer.x_max = finiteNumber(readControl(root, "plate-x-max").value, "x_max", errors);
    base.geometry.outer.y_min = finiteNumber(readControl(root, "plate-y-min").value, "y_min", errors);
    base.geometry.outer.y_max = finiteNumber(readControl(root, "plate-y-max").value, "y_max", errors);
    base.geometry.holes[0].center = [
      finiteNumber(readControl(root, "plate-hole-x").value, "hole center x", errors),
      finiteNumber(readControl(root, "plate-hole-y").value, "hole center y", errors),
    ];
    base.geometry.holes[0].radius = finiteNumber(readControl(root, "plate-hole-radius").value, "hole radius", errors);
    base.constitutive.E = finiteNumber(readControl(root, "plate-young").value, "Young's modulus E", errors);
    base.constitutive.nu = finiteNumber(readControl(root, "plate-nu").value, "Poisson's ratio nu", errors);
    var assumption = readControl(root, "plate-assumption");
    base.constitutive.assumption = assumption ? assumption.value : (base.constitutive.assumption || "plane_stress");
    base.boundaries.forEach(function (bc) {
      if (bc.id === "traction_right") {
        bc.value = finiteNumber(readControl(root, "plate-traction").value, "right-edge traction", errors);
      }
    });
    var solverPath = readControl(root, "plate-solver-path").value;
    method.path = solverPath;
    if (solverPath === "operator") {
      method.method = readControl(root, "plate-operator-method").value;
      delete method.variant;
    } else {
      var selected = readControl(root, "plate-method");
      var parsed = String(selected.value || "").split("/");
      method.method = parsed[0];
      method.variant = parsed[1];
    }
    method.sampling.interior = positiveInt(readControl(root, "plate-interior").value, "interior samples", errors);
    method.sampling.boundary = positiveInt(readControl(root, "plate-boundary").value, "boundary samples", errors);
    var networkName = readControl(root, "plate-network-name");
    var samplerName = readControl(root, "plate-sampler-name");
    var lossWeighting = readControl(root, "plate-loss-weighting");
    var optimizerName = readControl(root, "plate-optimizer-name");
    var differentiatorMode = readControl(root, "plate-differentiator-mode");
    if (networkName && networkName.value) method.network.name = networkName.value;
    if (samplerName && samplerName.value) method.sampling.name = samplerName.value;
    if (lossWeighting && lossWeighting.value) method.loss.weighting = lossWeighting.value;
    if (optimizerName && optimizerName.value) method.optimizer.name = optimizerName.value;
    if (differentiatorMode && differentiatorMode.value) method.differentiator.mode = differentiatorMode.value;
    method.optimizer.lr = finiteNumber(readControl(root, "plate-lr").value, "learning rate", errors);
    method.network.activation = readControl(root, "plate-activation").value;
    var hidden = String(readControl(root, "plate-hidden").value || "");
    method.network.hidden_layers = hidden.trim() === "" ? [] : hidden.split(",").filter(function (part) {
      return part.trim() !== "";
    }).map(function (part) { return Number(part.trim()); });
    method.training.steps = positiveInt(readControl(root, "plate-steps").value, "steps", errors);
    var settings = {
      steps: method.training.steps,
      seed: nonNegativeInt(readControl(root, "plate-seed").value, "seed", errors),
      device: "cpu",
      outputDir: readControl(root, "plate-output").value.trim(),
    };
    return {problem: base, method: method, settings: settings, readErrors: errors};
  }

  function renderPlateFromDom(root, payload) {
    var contract = payload.plate_contract;
    if (!contract) {
      return {ok: false, errors: ["Plate contract is missing from the page payload. No script was generated."]};
    }
    var read = readPlate(root, contract);
    var errors = read.readErrors.concat(plateErrors(read.problem, read.method, read.settings, contract));
    if (read.readErrors.length) return {ok: false, errors: read.readErrors};
    return plateState(read.problem, read.method, read.settings, contract);
  }

  function renderState(root, payload, designedProblem) {
    if (root.dataset.structuredProblem === "linear_elasticity") return renderPlateFromDom(root, payload);
    if (readControl(root, "operator-model")) return renderOperatorFromDom(root, payload, designedProblem);
    var local = readControl(root, "local-script");
    if (!local) return {ok: false, errors: ["未找到 Operator 控件，未生成脚本。"]};
    return renderOperatorFromDom(root, payload);
  }

  function applyState(root, state, nodes) {
    var errors = state.errors || ["校验失败。"];
    if (!state.ok) {
      var message = errors.join(" | ");
      nodes.code.textContent = [
        '"""OpenPINN warned raw export.',
        '',
        'This script is intentionally incomplete. Review ISSUES and replace the explicit',
        'NotImplementedError only after supplying the missing or unsupported details.',
        'No training result or loss is fabricated.',
        '"""',
        '',
        'ISSUES = ' + JSON.stringify(errors, null, 2),
        '',
        'raise NotImplementedError(' + JSON.stringify(message) + ')',
        '',
      ].join("\n");
      nodes.copyButton.disabled = false;
      nodes.downloadButton.disabled = false;
      if (nodes.status) {
        nodes.status.textContent = "存在问题，已生成带显式 NotImplementedError 的诚实原始脚本；仍可复制或下载。";
      }
      root.dataset.scriptName = "openpinn_incomplete_config";
      return;
    }
    nodes.code.textContent = state.source;
    nodes.copyButton.disabled = false;
    nodes.downloadButton.disabled = false;
    root.dataset.scriptName = state.downloadName;
    if (nodes.status) nodes.status.textContent = "";
  }

  function fillSelect(select, options, selected) {
    var owner = select.ownerDocument;
    select.innerHTML = "";
    options.forEach(function (option) {
      var node = owner.createElement("option");
      node.value = option.value;
      node.textContent = option.label;
      if (option.canonical) node.dataset.canonical = option.canonical;
      select.appendChild(node);
    });
    if (selected) select.value = selected;
  }

  function parameterInput(owner, name, schema) {
    var label = owner.createElement("label");
    label.textContent = (schema.label || name) + " ";
    var input;
    if (schema.kind === "choice") {
      input = owner.createElement("select");
      (schema.choices || []).forEach(function (choice) {
        var option = owner.createElement("option");
        option.value = choice;
        option.textContent = choice;
        input.appendChild(option);
      });
      input.value = schema.default;
    } else if (schema.kind === "boolean") {
      input = owner.createElement("input");
      input.type = "checkbox";
      input.checked = Boolean(schema.default);
    } else if (schema.kind === "integer_list") {
      input = owner.createElement("input");
      input.type = "text";
      input.value = (schema.default || []).join(", ");
      input.spellcheck = false;
    } else {
      input = owner.createElement("input");
      input.type = "number";
      input.step = schema.kind === "integer" ? "1" : "any";
      if (schema.minimum !== null && schema.minimum !== undefined) input.min = String(schema.minimum);
      input.value = schema.default;
    }
    input.dataset.operatorParameter = name;
    input.dataset.kind = schema.kind;
    label.appendChild(input);
    if (schema.description) {
      var help = owner.createElement("span");
      help.textContent = schema.description;
      label.appendChild(help);
    }
    return label;
  }

  function renderOperatorParameters(root, payload) {
    var container = readControl(root, "operator-parameters");
    if (!container) return;
    var selected = selectedOperator(root);
    var choice = choiceByValue(payload, selected);
    var hook = choice ? hookByCanonical(payload, choice.canonical) : null;
    container.innerHTML = "";
    var summary = readControl(root, "operator-summary");
    if (summary) summary.textContent = hook ? hook.summary : "";
    if (!hook) return;
    Object.keys(hook.parameters || {}).forEach(function (name) {
      var schema = hook.parameters[name];
      if (name === "training_mode" || schema.ui_hidden) return;
      container.appendChild(parameterInput(container.ownerDocument, name, schema));
    });
  }

  function renderOperatorExtensions(root, payload) {
    var container = readControl(root, "operator-extension-fields");
    if (!container) return;
    var contract = payload.operator_extensions || {choices: {}, defaults: {}};
    container.innerHTML = "";
    Object.keys(contract.choices || {}).forEach(function (kind) {
      var label = container.ownerDocument.createElement("label");
      label.textContent = kind + " ";
      var select = container.ownerDocument.createElement("select");
      select.dataset.operatorExtension = kind;
      (contract.choices[kind] || []).forEach(function (name) {
        var option = container.ownerDocument.createElement("option");
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
      });
      select.value = (contract.defaults || {})[kind] || "";
      label.appendChild(select);
      container.appendChild(label);
    });
  }

  function setControl(root, role, value) {
    var node = readControl(root, role);
    if (node) node.value = value;
  }

  function fillPlateDefaults(root, contract) {
    if (!contract || root.dataset.plateDefaults === "true") return;
    var problem = contract.problem;
    var method = contract.method;
    var outer = problem.geometry.outer;
    var hole = problem.geometry.holes[0];
    setControl(root, "plate-x-min", outer.x_min);
    setControl(root, "plate-x-max", outer.x_max);
    setControl(root, "plate-y-min", outer.y_min);
    setControl(root, "plate-y-max", outer.y_max);
    setControl(root, "plate-hole-x", hole.center[0]);
    setControl(root, "plate-hole-y", hole.center[1]);
    setControl(root, "plate-hole-radius", hole.radius);
    setControl(root, "plate-young", problem.constitutive.E);
    setControl(root, "plate-nu", problem.constitutive.nu);
    setControl(root, "plate-assumption", problem.constitutive.assumption || "plane_stress");
    var traction = problem.boundaries.find(function (bc) { return bc.id === "traction_right"; });
    setControl(root, "plate-traction", traction ? traction.value : 0);
    setControl(root, "plate-interior", method.sampling.interior);
    setControl(root, "plate-boundary", method.sampling.boundary);
    setControl(root, "plate-lr", method.optimizer.lr);
    setControl(root, "plate-steps", method.training.steps);
    setControl(root, "plate-seed", 0);
    setControl(root, "plate-hidden", method.network.hidden_layers.join(", "));
    setControl(root, "plate-output", "runs");
    var activation = readControl(root, "plate-activation");
    if (activation) activation.value = method.network.activation;
    var physics = readControl(root, "plate-physics");
    if (physics && !physics.options.length) {
      fillSelect(physics, [{value: problem.physics, label: problem.physics}].concat(
        (contract.rejected_physics || []).map(function (name) {
          return {value: name, label: name + " (not supported by M3 codegen)"};
        })
      ), problem.physics);
    }
    var solver = readControl(root, "plate-solver-path");
    if (solver && !solver.options.length) {
      fillSelect(solver, [
        {value: "pinn", label: "pinn (solve)"},
        {value: "operator", label: "operator (rejected for this solid path)"},
      ], "pinn");
    }
    var methodSelect = readControl(root, "plate-method");
    if (methodSelect && !methodSelect.options.length) {
      var options = (contract.pinn_methods || []).map(function (item) {
        return {value: item.method + "/" + item.variant, label: item.title + "  ·  " + item.method + "/" + item.variant};
      });
      options.push({value: "not_a_method/plain_pinn", label: "not_a_method (not registered)"});
      fillSelect(methodSelect, options, method.method + "/" + method.variant);
    }
    var operatorMethod = readControl(root, "plate-operator-method");
    if (operatorMethod && !operatorMethod.options.length) {
      fillSelect(operatorMethod, (contract.rejected_solver_paths || []).length ? [
        {value: "deeponet", label: "deeponet"},
        {value: "fno", label: "fno"},
      ] : [], "deeponet");
    }
    root.dataset.plateDefaults = "true";
  }

  function togglePlateOperator(root) {
    var solver = readControl(root, "plate-solver-path");
    var operatorField = readControl(root, "plate-operator-field");
    var methodField = readControl(root, "plate-method-field");
    if (!solver) return;
    var operatorPath = solver.value === "operator";
    if (operatorField) operatorField.hidden = !operatorPath;
    if (methodField) methodField.hidden = operatorPath;
  }

  function prepareConfigurator(root, payload, onChange) {
    var operatorSelect = readControl(root, "operator-model");
    if (operatorSelect && !operatorSelect.options.length) {
      var choices = (payload.operator_choices || []).map(function (choice) {
        var label = choice.label;
        if (!choice.selectable) label += "（不可选）";
        return {value: choice.value, label: label, canonical: choice.canonical};
      });
      fillSelect(operatorSelect, choices, "deeponet");
    }
    var trainingMode = readControl(root, "operator-training-mode");
    if (trainingMode && !trainingMode.options.length) {
      fillSelect(trainingMode, (payload.training_modes || []).map(function (mode) {
        return {value: mode, label: mode};
      }), "data_only");
    }
    fillPlateDefaults(root, payload.plate_contract);
    renderOperatorParameters(root, payload);
    renderOperatorExtensions(root, payload);
    togglePlateOperator(root);
    if (operatorSelect) {
      operatorSelect.addEventListener("change", function () {
        renderOperatorParameters(root, payload);
        onChange();
      });
    }
    if (trainingMode) trainingMode.addEventListener("change", onChange);
    var solver = readControl(root, "plate-solver-path");
    if (solver) solver.addEventListener("change", function () { togglePlateOperator(root); onChange(); });
    root.addEventListener("input", function (event) {
      var target = event.target;
      if (!target || !target.dataset) return;
      if (target.dataset.operatorParameter || target.dataset.operatorExtension || (target.dataset.role && target.dataset.role.indexOf("plate-") === 0)) onChange();
    });
    root.addEventListener("change", function (event) {
      var target = event.target;
      if (!target || !target.dataset) return;
      if (target.dataset.operatorParameter || target.dataset.operatorExtension || (target.dataset.role && target.dataset.role.indexOf("plate-") === 0)) onChange();
    });
  }

  function refreshDesigner(root, payload) {
    var local = readControl(root, "local-script");
    var operatorBox = readControl(root, "designer-operator");
    if (operatorBox) operatorBox.hidden = false;
    var state = renderState(root, payload);
    var code = readControl(root, "local-code");
    var copyButton = readControl(root, "local-copy");
    var downloadButton = readControl(root, "local-download");
    var status = readControl(root, "local-message");
    if (!code || !copyButton || !downloadButton) return;
    applyState(root, state, {
      code: code,
      copyButton: copyButton,
      downloadButton: downloadButton,
      status: status,
    });
    if (state.ok && status) status.textContent = state.message;
  }

  function copyText(text, button, status) {
    if (button.disabled) return;
    var view = button.ownerDocument.defaultView;
    function copied() {
      if (status) status.textContent = "已复制。请保存到 examples/ 下并在本地运行；本页面未执行训练。";
    }
    function fallback() {
      var area = button.ownerDocument.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      button.parentElement.appendChild(area);
      area.select();
      var ok = false;
      try { ok = button.ownerDocument.execCommand("copy"); } catch (error) { ok = false; }
      area.remove();
      if (ok) copied();
      else if (status) status.textContent = "复制失败，请改为下载 .py 文件。";
    }
    if (view.navigator.clipboard && view.isSecureContext) {
      view.navigator.clipboard.writeText(text).then(copied).catch(fallback);
      return;
    }
    fallback();
  }

  function downloadText(text, filename, button, status) {
    if (button.disabled) return;
    var view = button.ownerDocument.defaultView;
    var blob = new view.Blob([text], {type: "text/x-python;charset=utf-8"});
    var url = view.URL.createObjectURL(blob);
    var anchor = button.ownerDocument.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    button.ownerDocument.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    view.setTimeout(function () { view.URL.revokeObjectURL(url); }, 1000);
    if (status) {
      status.textContent = "已开始下载 " + filename + "。请在仓库根目录运行 python examples/" + filename + "；本页面未执行训练。";
    }
  }

  function prepareDesigner(root, payload) {
    var local = readControl(root, "local-script");
    if (local && !local.options.length) {
      var options = [];
      (payload.operator_choices || []).forEach(function (choice) {
        var label = choice.label;
        if (!choice.selectable) label += "（不可选）";
        options.push({value: choice.value, label: label, canonical: choice.canonical});
      });
      fillSelect(local, options, "deeponet");
    }
    var trainingMode = readControl(root, "operator-training-mode");
    if (trainingMode && !trainingMode.options.length) {
      fillSelect(trainingMode, (payload.training_modes || []).map(function (mode) {
        return {value: mode, label: mode};
      }), "data_only");
    }
    renderOperatorParameters(root, payload);
    renderOperatorExtensions(root, payload);
    var refresh = function () { refreshDesigner(root, payload); };
    if (local) {
      local.addEventListener("change", function () {
        renderOperatorParameters(root, payload);
        refresh();
      });
    }
    root.addEventListener("input", function (event) {
      var target = event.target;
      if (!target || !target.dataset) return;
      if (target.dataset.operatorParameter || target.dataset.operatorExtension) refresh();
    });
    root.addEventListener("change", function (event) {
      var target = event.target;
      if (!target || !target.dataset) return;
      if (target.dataset.role === "local-script") return;
      if (target.dataset.operatorParameter || target.dataset.operatorExtension || target.dataset.role === "operator-training-mode") refresh();
    });
    var copyButton = readControl(root, "local-copy");
    var downloadButton = readControl(root, "local-download");
    var code = readControl(root, "local-code");
    var status = readControl(root, "local-message");
    if (copyButton) copyButton.addEventListener("click", function () { copyText(code.textContent, copyButton, status); });
    if (downloadButton) {
      downloadButton.addEventListener("click", function () {
        if (downloadButton.disabled) return;
        downloadText(code.textContent, root.dataset.scriptName + ".py", downloadButton, status);
      });
    }
    refresh();
  }

  function prepare(root, payload, onChange) {
    if (readControl(root, "operator-model") || readControl(root, "plate-fields")) prepareConfigurator(root, payload, onChange || function () {});
    if (readControl(root, "local-script")) prepareDesigner(root, payload);
  }

  return {
    prepare: prepare,
    renderState: renderState,
    applyState: applyState,
    operatorState: operatorState,
    plateState: plateState,
    createPlateValidationSession: createPlateValidationSession,
    renderOperatorScript: renderOperatorScript,
    renderPlateScript: renderPlateScript,
    pythonLiteral: pythonLiteral,
    downloadStem: downloadStem,
  };
});
