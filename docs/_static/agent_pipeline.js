(function () {
  "use strict";
  function initialize(root) {
    var joint = window.OpenPINNJointHandoff;
    var trial = window.OpenPINNUnderstandingTrial;
    var diagnostic = window.OpenPINNDiagnosticAgent;
    if (!joint || !trial) return;
    function role(name) { return root.querySelector('[data-agent-role="' + name + '"]'); }
    var understanding = null, planning = null, artifact = null, diagnosticReport = null;
    function options(prefix) {
      var modelControl = root.querySelector('[data-ecosystem-control="model"]');
      var path = modelControl && ["deeponet", "fno"].indexOf(modelControl.value) !== -1 ? "operator" : "pinn";
      return {baseUrl: role(prefix + "-base-url").value, model: role(prefix + "-model").value,
        apiKey: role(prefix + "-key").value, path: path};
    }
    function setStatus(message, error) {
      role("status").textContent = message;
      role("status").classList.toggle("is-error", Boolean(error));
    }
    function updateRaw() {
      role("understanding-json").value = understanding ? JSON.stringify(understanding.draft, null, 2) : "{}";
      role("planning-json").value = planning ? JSON.stringify(planning.method_draft, null, 2) : "{}";
    }
    function showUnderstandingSummary(report) {
      var box = role("understanding-summary");
      if (!box) return;
      function escape(value) { return String(value).replace(/[&<>]/g, function (char) {
        return ({"&": "&amp;", "<": "&lt;", ">": "&gt;"})[char];
      }); }
      var missing = Array.isArray(report && report.missing) ? report.missing : [];
      var questions = Array.isArray(report && report.questions) ? report.questions : [];
      box.hidden = false;
      box.innerHTML = missing.length ?
        "<strong>Understanding 尚缺字段：</strong><code>" + missing.map(escape).join("</code>、<code>") + "</code>" +
        (questions.length ? "<br><strong>需要确认：</strong>" + questions.map(escape).join("；") : "") :
        "<strong>Understanding 完整性检查：</strong>未报告缺失字段。";
    }
    role("understand").addEventListener("click", function () {
      setStatus("Understanding 正在生成问题草稿…");
      trial.understandWithCloud(role("description").value, options("understanding")).then(function (report) {
        understanding = report; updateRaw(); showUnderstandingSummary(report);
        var count = Array.isArray(report.missing) ? report.missing.length : 0;
        setStatus(count ? "Understanding 完成，但仍有 " + count + " 个缺失字段；请查看下方清单，仍可继续 Planning 和编辑。" :
          "Understanding 完成且未报告缺失字段；可继续 Planning。");
      }).catch(function (error) { setStatus("Understanding 失败：" + error.message, true); });
    });
    role("plan").addEventListener("click", function () {
      try {
        var draft = JSON.parse(role("understanding-json").value || "{}");
        if (!Object.keys(draft).length) throw new Error("请先生成或填写 Understanding 问题 JSON。");
        setStatus("Planning 正在生成方法草稿…");
        trial.planWithCloud(draft, options("planning")).then(function (report) {
          planning = report; updateRaw();
          setStatus("Planning 完成；现在可联合导入配置器。未通过 Validation 仍允许导出并显示问题。");
        }).catch(function (error) { setStatus("Planning 失败：" + error.message, true); });
      } catch (error) { setStatus(error.message, true); }
    });
    role("build-joint").addEventListener("click", function () {
      try {
        var problem = JSON.parse(role("understanding-json").value || "{}");
        var method = JSON.parse(role("planning-json").value || "{}");
        var u = Object.keys(problem).length ? Object.assign({}, understanding || {}, {draft: problem}) : null;
        var p = Object.keys(method).length ? Object.assign({}, planning || {}, {method_draft: method}) : null;
        var envelope = joint.build(u, p, {inputText: role("description").value,
          understandingModel: role("understanding-model").value, planningModel: role("planning-model").value});
        role("joint-json").value = JSON.stringify(envelope, null, 2);
        window.sessionStorage.setItem(joint.STORAGE_KEY, JSON.stringify(envelope));
        root.dispatchEvent(new CustomEvent("openpinn:joint-handoff", {detail: envelope}));
        setStatus("联合 handoff 已导入。问题来自 Understanding，方法来自 Planning；警告不锁定导出。");
      } catch (error) { setStatus("联合导入失败：" + error.message, true); }
    });
    role("apply-joint-json").addEventListener("click", function () {
      try {
        var envelope = joint.parse(role("joint-json").value);
        root.dispatchEvent(new CustomEvent("openpinn:joint-handoff", {detail: envelope}));
        setStatus("编辑后的 JointAgentHandoff 已应用。");
      } catch (error) { setStatus(error.message, true); }
    });
    role("diagnostic-file").addEventListener("change", function () {
      var file = role("diagnostic-file").files && role("diagnostic-file").files[0];
      artifact = null; diagnosticReport=null; role("diagnose").disabled = true;
      role("download-diagnostic-html").disabled=true;role("download-diagnostic-json").disabled=true;
      role("diagnostic-report-view").hidden=true;role("diagnostic-report-view").replaceChildren();
      if (!file || !diagnostic) return;
      diagnostic.readArtifactFile(file).then(function (value) {
        artifact = value; role("diagnose").disabled = false;
        role("diagnostic-output").textContent = JSON.stringify(diagnostic.evaluateArtifact(value, null), null, 2);
        role("diagnostic-status").textContent="RunArtifact 已读取；可运行诊断生成可视化报告。";
      }).catch(function (error) { role("diagnostic-output").textContent = error.message;role("diagnostic-status").textContent="读取失败："+error.message; });
    });
    role("diagnose").addEventListener("click", function () {
      if (!artifact || !diagnostic) return;
      role("diagnose").disabled=true;role("diagnostic-status").textContent="正在生成确定性分析、语义诊断与可视化报告…";
      diagnostic.diagnose(artifact, null, {baseUrl: role("diagnostic-base-url").value,
        model: role("diagnostic-model").value, apiKey: role("diagnostic-key").value}).then(function (report) {
          diagnosticReport=report;
          role("diagnostic-output").textContent = JSON.stringify(report, null, 2);
          diagnostic.renderReport(role("diagnostic-report-view"),report,artifact);
          role("diagnostic-report-view").hidden=false;
          role("download-diagnostic-html").disabled=false;role("download-diagnostic-json").disabled=false;
          role("diagnostic-status").textContent=report.llm_track&&report.llm_track.state==="complete"?"诊断报告已生成；可下载包含图表与建议的离线 HTML。":"语义诊断未完整返回；已保留确定性结论、图表和可下载报告。";
      }).catch(function(error){role("diagnostic-status").textContent="诊断失败："+error.message;}).finally(function(){role("diagnose").disabled=!artifact;});
    });
    function download(content,type,name){var blob=new Blob([content],{type:type});var url=URL.createObjectURL(blob),anchor=document.createElement("a");anchor.href=url;anchor.download=name;document.body.appendChild(anchor);anchor.click();anchor.remove();window.setTimeout(function(){URL.revokeObjectURL(url);},1000);}
    role("download-diagnostic-html").addEventListener("click",function(){if(!diagnosticReport||!artifact)return;download(diagnostic.reportHtml(diagnosticReport,artifact),"text/html;charset=utf-8","openpinn-diagnostic-report.html");});
    role("download-diagnostic-json").addEventListener("click",function(){if(!diagnosticReport)return;download(JSON.stringify(diagnosticReport,null,2),"application/json;charset=utf-8","openpinn-diagnostic-report.json");});
    updateRaw();
  }
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-openpinn-configurator]").forEach(initialize);
  });
}());
