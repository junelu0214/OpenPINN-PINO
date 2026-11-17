(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.OpenPINNMultiphysicsEditor = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
  "use strict";

  var CLASSES = [
    {id: "solid_mechanics", label: "固体力学", description: "位移、应力、应变、本构与载荷"},
    {id: "fluid_mechanics", label: "流体力学", description: "速度、压力、连续性与动量守恒"},
    {id: "heat_transfer", label: "传热学", description: "导热、对流与热源"},
    {id: "mass_transport", label: "质量传递", description: "扩散、对流与反应"},
    {id: "electromagnetics", label: "电磁学", description: "电势、电场、磁场与介质"},
    {id: "waves_acoustics", label: "波动与声学", description: "波动、Helmholtz 与声压"},
    {id: "multiphysics", label: "多物理耦合", description: "热-固、流-热与渗流-固体耦合"},
    {id: "general_pde", label: "通用方程", description: "标量 PDE 或自定义方程组"}
  ];

  function field(id, label, components, unit, role) {
    return {id: id, label: label, components: components, unit: unit, role: role || "dependent"};
  }
  function equation(id, label, target, expression) {
    return {id: id, label: label, target: target, expression: expression};
  }
  function property(id, label, value, unit, inverse) {
    return Object.assign({id: id, label: label, value: value, unit: unit, role:"known"}, inverse || {});
  }
  function boundary(id, region, type, fieldId, components, value) {
    return {id: id, region: region, type: type, field: fieldId, components: components, value: value};
  }
  function profile(spec) {
    spec.dimension = spec.dimension || 2;
    spec.analysis = spec.analysis || "stationary";
    spec.couplings = spec.couplings || [];
    spec.geometry_features = spec.geometry_features || [];
    spec.execution = spec.execution || {status: "editing_only", adapter: null};
    return spec;
  }

  var PROFILES = [
    profile({id:"linear_elasticity", physics_class:"solid_mechanics", label:"线弹性", description:"小变形位移法；支持各向同性材料、平面应力/平面应变和分量载荷。", geometry_bounds:{x:[0,2],y:[0,2]}, fields:[field("displacement","位移",["ux","uy"],"m")], equations:[equation("momentum_x","x 动量平衡","ux","dx(sxx)+dy(sxy)+fx=0"),equation("momentum_y","y 动量平衡","uy","dx(sxy)+dy(syy)+fy=0")], constitutive:{model:"isotropic_linear", label:"各向同性线弹性", assumption:"plane_stress", properties:[property("E","杨氏模量",210e9,"Pa"),property("nu","泊松比",0.3,"1")]}, boundaries:[boundary("fixed_left","left","fixed","displacement",["ux","uy"],["0","0"]),boundary("traction_right","right","traction","displacement",["ux","uy"],["1e6","0"]),boundary("free_top","top","traction","displacement",["ux","uy"],["0","0"]),boundary("free_bottom","bottom","traction","displacement",["ux","uy"],["0","0"]),boundary("free_hole","hole_0","traction","displacement",["ux","uy"],["0","0"])], geometry_features:[{id:"hole_0",type:"circle",center:[1,1],radius:0.25}], execution:{status:"runnable",adapter:"elasticity_codegen",notes:"当前执行适配器支持二维各向同性平面应力矩形单圆孔。"}}),
    profile({id:"hyperelasticity", physics_class:"solid_mechanics", label:"超弹性", description:"大变形固体；可编辑应变能模型，执行适配待接入。", fields:[field("displacement","位移",["ux","uy"],"m")], equations:[equation("momentum","动量平衡","displacement","Div(P)+f=0")], constitutive:{model:"neo_hookean",label:"Neo-Hookean",assumption:"plane_strain",properties:[property("mu","剪切模量",1e6,"Pa"),property("kappa","体积模量",1e8,"Pa")]}, boundaries:[boundary("fixed","left","fixed","displacement",["ux","uy"],["0","0"])]}),
    profile({id:"incompressible_navier_stokes", physics_class:"fluid_mechanics", label:"不可压 Navier–Stokes", description:"速度-压力混合形式，包含连续性与动量守恒。", analysis:"stationary", fields:[field("velocity","速度",["u","v"],"m/s"),field("pressure","压力",["p"],"Pa")], equations:[equation("continuity","连续性","pressure","dx(u)+dy(v)=0"),equation("momentum_x","x 动量","u","rho*(u*dx(u)+v*dy(u))+dx(p)-mu*(dx(dx(u))+dy(dy(u)))-fx=0"),equation("momentum_y","y 动量","v","rho*(u*dx(v)+v*dy(v))+dy(p)-mu*(dx(dx(v))+dy(dy(v)))-fy=0")], constitutive:{model:"newtonian",label:"牛顿流体",properties:[property("rho","密度",1000,"kg/m^3"),property("mu","动力黏度",0.001,"Pa*s")]}, boundaries:[boundary("inlet","left","velocity_inlet","velocity",["u","v"],["1","0"]),boundary("outlet","right","pressure_outlet","pressure",["p"],["0"]),boundary("wall","top,bottom","no_slip","velocity",["u","v"],["0","0"])]}),
    profile({id:"stokes_flow", physics_class:"fluid_mechanics", label:"Stokes 流", description:"低雷诺数不可压流动。", fields:[field("velocity","速度",["u","v"],"m/s"),field("pressure","压力",["p"],"Pa")], equations:[equation("continuity","连续性","pressure","dx(u)+dy(v)=0"),equation("momentum_x","x 动量","u","dx(p)-mu*(dx(dx(u))+dy(dy(u)))-fx=0"),equation("momentum_y","y 动量","v","dy(p)-mu*(dx(dx(v))+dy(dy(v)))-fy=0")], constitutive:{model:"newtonian",label:"牛顿流体",properties:[property("mu","动力黏度",0.001,"Pa*s")]}, boundaries:[boundary("inlet","left","velocity_inlet","velocity",["u","v"],["1","0"]),boundary("outlet","right","pressure_outlet","pressure",["p"],["0"])]}),
    profile({id:"darcy_flow", physics_class:"fluid_mechanics", label:"Darcy 多孔介质流", description:"压力势驱动的多孔介质渗流。", fields:[field("pressure","孔隙压力",["p"],"Pa")], equations:[equation("mass","质量守恒","p","dt(p)-div(k/mu*grad(p))-q=0")], constitutive:{model:"darcy",label:"Darcy 定律",properties:[property("k","渗透率",1e-12,"m^2"),property("mu","动力黏度",0.001,"Pa*s"),property("storage","储水系数",1e-9,"1/Pa")]}, boundaries:[boundary("pressure_left","left","dirichlet","pressure",["p"],["1e5"]),boundary("flux_right","right","flux","pressure",["p"],["0"])]}),
    profile({id:"darcy_inverse_seepage", physics_class:"fluid_mechanics", label:"渗流反分析（PINN）", description:"一维稳态 Darcy 渗流单实例；PINN 由 x、p 稀疏观测联合识别压力场与渗透率 k。", dimension:1, analysis:"stationary", geometry_bounds:{x:[0,1],y:[0,1]}, fields:[field("pressure","孔隙压力",["p"],"1")], equations:[equation("darcy","Darcy 平衡","p","-dx(k*dx(p))-q=0")], constitutive:{model:"darcy",label:"Darcy 定律",properties:[property("k","待反演渗透率",1.0,"1",{role:"inverse",lower:0.2,upper:5.0}),property("q","已知源项",1.0,"1",{role:"known"})]}, boundaries:[boundary("pressure_left","left","dirichlet","pressure",["p"],["0"]),boundary("pressure_right","right","dirichlet","pressure",["p"],["0"])], execution:{status:"runnable",adapter:"darcy_inverse_pinn",notes:"PINN 对单个稳态渗流实例联合反演压力场与渗透率。"}}),
    profile({id:"heat_conduction", physics_class:"heat_transfer", label:"固体传热", description:"瞬态或稳态热传导与体热源。", analysis:"transient", fields:[field("temperature","温度",["T"],"K")], equations:[equation("energy","能量守恒","T","rho*cp*dt(T)-dx(k*dx(T))-dy(k*dy(T))-Q=0")], constitutive:{model:"fourier",label:"Fourier 导热",properties:[property("rho","密度",7800,"kg/m^3"),property("cp","定压比热",500,"J/(kg*K)"),property("k","导热系数",45,"W/(m*K)"),property("Q","体热源",0,"W/m^3")]}, boundaries:[boundary("temperature_left","left","dirichlet","temperature",["T"],["300"]),boundary("insulated_right","right","neumann","temperature",["T"],["0"]),boundary("insulated_bottom","bottom","neumann","temperature",["T"],["0"]),boundary("insulated_top","top","neumann","temperature",["T"],["0"])], execution:{status:"runnable",adapter:"scalar_expression",notes:"映射到现有单场公式代码生成器。"}}),
    profile({id:"convection_diffusion", physics_class:"mass_transport", label:"稀物质传递", description:"浓度的扩散、对流和反应。", analysis:"transient", fields:[field("concentration","浓度",["c"],"mol/m^3")], equations:[equation("species","组分守恒","c","dt(c)+vx*dx(c)+vy*dy(c)-dx(D*dx(c))-dy(D*dy(c))-R=0")], constitutive:{model:"fick",label:"Fick 扩散",properties:[property("D","扩散系数",1e-9,"m^2/s"),property("vx","x 速度",0,"m/s"),property("vy","y 速度",0,"m/s"),property("R","反应速率",0,"mol/(m^3*s)")]}, boundaries:[boundary("concentration_left","left","dirichlet","concentration",["c"],["1"]),boundary("no_flux","right","flux","concentration",["c"],["0"])]}),
    profile({id:"electrostatics", physics_class:"electromagnetics", label:"静电场", description:"电势与介电材料中的 Gauss 定律。", fields:[field("potential","电势",["V"],"V")], equations:[equation("gauss","Gauss 定律","V","dx(epsilon*dx(V))+dy(epsilon*dy(V))+rhoq=0")], constitutive:{model:"linear_dielectric",label:"线性介电材料",properties:[property("epsilon","介电常数",8.854e-12,"F/m"),property("rhoq","电荷密度",0,"C/m^3")]}, boundaries:[boundary("ground","left","ground","potential",["V"],["0"]),boundary("potential","right","potential","potential",["V"],["1"])]}),
    profile({id:"helmholtz_acoustics", physics_class:"waves_acoustics", label:"频域声学 / Helmholtz", description:"频域标量波动与阻抗边界。", fields:[field("acoustic_pressure","声压",["p"],"Pa")], equations:[equation("helmholtz","Helmholtz 方程","p","dx(dx(p))+dy(dy(p))+k**2*p-q=0")], constitutive:{model:"linear_acoustics",label:"线性声学",properties:[property("rho","密度",1.225,"kg/m^3"),property("c0","声速",343,"m/s"),property("k","波数",18.318,"1/m")]}, boundaries:[boundary("source","left","pressure","acoustic_pressure",["p"],["1"]),boundary("radiation","right","radiation","acoustic_pressure",["p"],["0"])]}),
    profile({id:"thermoelasticity", physics_class:"multiphysics", label:"热弹性", description:"温度场与热膨胀固体的双向/单向耦合。", analysis:"transient", fields:[field("displacement","位移",["ux","uy"],"m"),field("temperature","温度",["T"],"K")], equations:[equation("momentum","固体动量平衡","displacement","div(sigma(T))+f=0"),equation("energy","能量守恒","T","rho*cp*dt(T)-div(k*grad(T))-Q=0")], constitutive:{model:"linear_thermoelastic",label:"各向同性热弹性",assumption:"plane_stress",properties:[property("E","杨氏模量",210e9,"Pa"),property("nu","泊松比",0.3,"1"),property("alpha","热膨胀系数",1.2e-5,"1/K"),property("Tref","参考温度",293.15,"K"),property("k","导热系数",45,"W/(m*K)")]}, boundaries:[boundary("fixed","left","fixed","displacement",["ux","uy"],["0","0"]),boundary("hot","right","temperature","temperature",["T"],["400"])], couplings:[{id:"thermal_strain",type:"domain",source:"temperature",target:"displacement",expression:"epsilon_th=alpha*(T-Tref)*I"}]}),
    profile({id:"consolidation_inverse_family", physics_class:"multiphysics", label:"固结问题族反分析（DeepONet）", description:"一维瞬态 Terzaghi 固结问题族；DeepONet 学习 (cv,U0) 到 u(z,t) 的映射，再由 z、t、u 观测反演新成员的固结系数 cv。", dimension:1, analysis:"transient", geometry_bounds:{x:[0,1],y:[0,1]}, fields:[field("pore_pressure","超孔隙水压力",["u"],"1")], equations:[equation("consolidation","Terzaghi 固结","u","dt(u)-cv*dx(dx(u))=0")], constitutive:{model:"terzaghi",label:"Terzaghi 一维固结",properties:[property("cv","待反演固结系数",0.2,"1",{role:"inverse",lower:0.05,upper:0.8}),property("U0","已知初始压力幅值",1.2,"1",{role:"known"})]}, boundaries:[boundary("drained_top","left","dirichlet","pore_pressure",["u"],["0"]),boundary("drained_bottom","right","dirichlet","pore_pressure",["u"],["0"])], execution:{status:"runnable",adapter:"consolidation_inverse_operator",notes:"DeepONet 学习连续 (cv,U0) 固结问题族，冻结算子后反演新成员的 cv。"}}),
    profile({id:"conjugate_heat_transfer", physics_class:"multiphysics", label:"共轭传热", description:"流体动量与固体/流体能量方程耦合。", fields:[field("velocity","速度",["u","v"],"m/s"),field("pressure","压力",["p"],"Pa"),field("temperature","温度",["T"],"K")], equations:[equation("continuity","连续性","p","dx(u)+dy(v)=0"),equation("momentum","流体动量","velocity","rho*(v·grad(v))+grad(p)-mu*laplace(v)=0"),equation("energy","能量守恒","T","rho*cp*(dt(T)+v·grad(T))-div(k*grad(T))-Q=0")], constitutive:{model:"newtonian_fourier",label:"牛顿流体 + Fourier 导热",properties:[property("rho","密度",1000,"kg/m^3"),property("mu","动力黏度",0.001,"Pa*s"),property("cp","比热",4180,"J/(kg*K)"),property("k","导热系数",0.6,"W/(m*K)")]}, boundaries:[], couplings:[{id:"advection_heat",type:"domain",source:"velocity",target:"temperature",expression:"v·grad(T)"}]}),
    profile({id:"scalar_pde", physics_class:"general_pde", label:"通用标量 PDE", description:"一个因变量和一个可安全解析的残差表达式。", fields:[field("field","标量场",["u"],"1")], equations:[equation("pde","控制方程","u","dx(dx(u))+dy(dy(u))=0")], constitutive:{model:"custom",label:"自定义系数",properties:[]}, boundaries:[boundary("left","left","dirichlet","field",["u"],["0"]),boundary("right","right","dirichlet","field",["u"],["1"]),boundary("bottom","bottom","neumann","field",["u"],["0"]),boundary("top","top","neumann","field",["u"],["0"])], execution:{status:"runnable",adapter:"scalar_expression",notes:"由现有公式编辑器验证量纲与生成代码。"}}),
    profile({id:"equation_system", physics_class:"general_pde", label:"自定义方程组", description:"自由定义多场、多个残差、本构参数与场间耦合。", fields:[field("field_1","场 1",["u"],"1"),field("field_2","场 2",["v"],"1")], equations:[equation("equation_1","方程 1","u","R1(u,v)=0"),equation("equation_2","方程 2","v","R2(u,v)=0")], constitutive:{model:"custom",label:"自定义本构",properties:[]}, boundaries:[]})
  ];

  var UNITS = ["1","m","m/s","m^2/s","Pa","Pa*s","K","V","F/m","C/m^3","kg/m^3","J/(kg*K)","W/(m*K)","W/m^3","mol/m^3","mol/(m^3*s)","1/m","1/Pa","1/K"];
  var FIELD_ROLES = [["dependent","因变量"],["auxiliary","辅助变量"],["multiplier","拉格朗日乘子"]];
  var CONSTITUTIVE_MODELS = [["isotropic_linear","各向同性线弹性"],["orthotropic_linear","正交各向异性线弹性"],["neo_hookean","Neo-Hookean"],["mooney_rivlin","Mooney–Rivlin"],["custom","自定义本构"]];
  var BOUNDARY_TYPES = {
    solid_mechanics:[["fixed","固定约束"],["displacement","指定位移"],["traction","表面力"],["symmetry","对称"],["roller","滚支/法向约束"]],
    fluid_mechanics:[["velocity_inlet","速度入口"],["pressure_outlet","压力出口"],["no_slip","无滑移壁面"],["slip","滑移壁面"],["flux","质量通量"]],
    heat_transfer:[["dirichlet","指定温度"],["neumann","热流/绝热"],["robin","对流换热"],["radiation","表面辐射"]],
    mass_transport:[["dirichlet","指定浓度"],["neumann","通量"],["robin","传质交换"],["reaction","表面反应"]],
    electromagnetics:[["ground","接地"],["potential","指定电势"],["charge","表面电荷"],["insulation","电绝缘"]],
    waves_acoustics:[["pressure","指定声压"],["radiation","辐射边界"],["impedance","阻抗"],["sound_hard","硬声场"]],
    multiphysics:[["dirichlet","指定值"],["neumann","指定通量"],["continuity","连续性"],["interface","耦合界面"]],
    general_pde:[["dirichlet","Dirichlet"],["neumann","Neumann"],["robin","Robin"]]
  };

  function clone(value) { return JSON.parse(JSON.stringify(value)); }
  function byId(id) { return PROFILES.find(function (item) { return item.id === id; }); }
  function cleanText(value, label, max) {
    value = String(value == null ? "" : value).trim();
    if (!value) throw new Error(label + "不能为空。");
    if (value.length > (max || 256)) throw new Error(label + "过长。");
    if (/[<>\0]/.test(value)) throw new Error(label + "包含禁止字符。");
    return value;
  }
  function validateModel(model) {
    if (!model || model.schema_version !== 2) throw new Error("多物理模型必须使用 schema_version=2。");
    if (!CLASSES.some(function (item) { return item.id === model.physics_class; })) throw new Error("未知物理分类。");
    if (![1,2,3].includes(Number(model.dimension))) throw new Error("空间维度必须为 1、2 或 3。");
    if (!["stationary","transient","frequency","eigenvalue"].includes(model.analysis)) throw new Error("未知分析类型。");
    if (!Array.isArray(model.fields) || !model.fields.length || model.fields.length > 12) throw new Error("因变量数量必须为 1–12。");
    var symbols = [];
    model.fields.forEach(function (item, index) {
      item.id = cleanText(item.id, "场 ID", 64); item.label = cleanText(item.label, "场名称", 128);
      if (!Array.isArray(item.components) || !item.components.length) throw new Error("第 " + (index+1) + " 个场缺少分量。");
      item.components = item.components.map(function (symbol) {
        symbol = cleanText(symbol, "场分量", 32);
        if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(symbol)) throw new Error("场分量必须是字母开头的标识符：" + symbol);
        if (symbols.includes(symbol)) throw new Error("场分量不能重复：" + symbol);
        symbols.push(symbol); return symbol;
      });
      item.unit = cleanText(item.unit || "1", "场单位", 64);
    });
    if (!Array.isArray(model.equations) || !model.equations.length || model.equations.length > 24) throw new Error("方程数量必须为 1–24。");
    model.equations.forEach(function (item) { item.id=cleanText(item.id,"方程 ID",64); item.label=cleanText(item.label,"方程名称",128); item.target=cleanText(item.target,"方程目标",64); item.expression=cleanText(item.expression,"残差表达式",4096); });
    if (!model.constitutive || !Array.isArray(model.constitutive.properties)) throw new Error("本构/材料设置无效。");
    model.constitutive.properties.forEach(function (item) { item.id=cleanText(item.id,"材料参数",64); item.label=cleanText(item.label,"参数名称",128); item.unit=cleanText(item.unit||"1","参数单位",64); if (!Number.isFinite(Number(item.value))) throw new Error(item.label+"必须是有限数值。"); item.value=Number(item.value);item.role=item.role||"known";if(!["known","inverse"].includes(item.role))throw new Error(item.label+" 的参数角色无效。");if(item.role==="inverse"){item.lower=Number(item.lower);item.upper=Number(item.upper);if(!(item.lower>0&&item.upper>item.lower&&item.value>item.lower&&item.value<item.upper))throw new Error(item.label+" 必须满足 0 < 下界 < 初值 < 上界。");} });
    if (!Array.isArray(model.boundaries) || model.boundaries.length > 48) throw new Error("边界条件数量无效。");
    model.boundaries.forEach(function (item) { item.id=cleanText(item.id,"边界 ID",64); item.region=cleanText(item.region,"边界区域",128); item.type=cleanText(item.type,"边界类型",64); item.field=cleanText(item.field,"边界作用场",64); item.components=Array.isArray(item.components)?item.components.map(String):[]; item.value=Array.isArray(item.value)?item.value.map(String):[String(item.value == null?"0":item.value)]; });
    if (!Array.isArray(model.geometry_features)) model.geometry_features=[];
    model.geometry_features.forEach(function(item){item.id=cleanText(item.id,"几何特征 ID",64);if(item.type!=="circle")throw new Error("当前解析几何特征仅支持圆孔。");if(!Array.isArray(item.center)||item.center.length!==2||!item.center.every(function(value){return Number.isFinite(Number(value));}))throw new Error(item.id+" 的圆心必须包含两个有限数值。");item.center=item.center.map(Number);item.radius=Number(item.radius);if(!Number.isFinite(item.radius)||item.radius<=0)throw new Error(item.id+" 的半径必须大于 0。");});
    model.execution = model.execution || {status:"editing_only",adapter:null};
    return model;
  }
  function effectiveExecution(model) {
    var definition = byId(model.interface);
    var result = clone(definition && definition.execution || model.execution || {status:"editing_only",adapter:null});
    if (result.adapter === "elasticity_codegen" && (Number(model.dimension) !== 2 || model.constitutive.assumption !== "plane_stress")) {
      return {status:"editing_only",adapter:null,notes:"当前线弹性执行适配器仅支持二维平面应力；此配置仍可建模和导出。"};
    }
    if (result.adapter === "scalar_expression" && ![1,2].includes(Number(model.dimension))) {
      return {status:"editing_only",adapter:null,notes:"当前标量公式执行适配器仅支持一维或二维。"};
    }
    if (result.adapter === "scalar_expression") {
      var components = model.fields.reduce(function (all,item) { return all.concat(item.components || []); }, []);
      var compatibleBoundaries = (model.boundaries || []).every(function (item) {
        return ["dirichlet","neumann"].includes(item.type) && item.value && item.value.every(function(value){return Number.isFinite(Number(value));});
      });
      var regions=[];(model.boundaries||[]).forEach(function(item){String(item.region).split(",").forEach(function(region){regions.push(region.trim());});});
      var required=Number(model.dimension)===1?["left","right"]:["left","right","bottom","top"];
      if (components.length !== 1 || model.equations.length !== 1 || !compatibleBoundaries || !required.every(function(region){return regions.includes(region);})) {
        return {status:"editing_only",adapter:null,notes:"当前标量执行适配器要求一个场分量、一条方程，以及覆盖全部外边界的数值型 Dirichlet/Neumann 条件。"};
      }
    }
    return result;
  }

  function compiledExpression(source) {
    return String(source || "")
      .replace(/\bdt\(/g,"Dₜ(").replace(/\bdx\(/g,"Dₓ(").replace(/\bdy\(/g,"Dᵧ(")
      .replace(/\bgrad\(/gi,"∇(").replace(/\bdiv\(/gi,"∇·(").replace(/\blaplace\(/gi,"∇²(")
      .replace(/\bDiv\(/g,"DIV(").replace(/\bsxx\b/g,"σₓₓ").replace(/\bsxy\b/g,"σₓᵧ")
      .replace(/\bsyy\b/g,"σᵧᵧ").replace(/\bux\b/g,"uₓ").replace(/\buy\b/g,"uᵧ")
      .replace(/\bnu\b/g,"ν").replace(/\bmu\b/g,"μ").replace(/\brho\b/g,"ρ")
      .replace(/\balpha\b/g,"α").replace(/\bepsilon\b/g,"ε").replace(/\bkappa\b/g,"κ")
      .replace(/\*\*/g,"^").replace(/\*/g,"·");
  }

  function compiledPreview(model) {
    return (model.equations || []).map(function(item) {
      return "R[" + item.target + "]  ←  " + compiledExpression(item.expression);
    }).join("\n");
  }

  function latexSymbol(name) {
    var greek={alpha:"\\alpha",beta:"\\beta",gamma:"\\gamma",delta:"\\delta",epsilon:"\\varepsilon",kappa:"\\kappa",lambda:"\\lambda",mu:"\\mu",nu:"\\nu",rho:"\\rho",sigma:"\\sigma",tau:"\\tau",omega:"\\omega"};
    var engineering={sxx:"\\sigma_{xx}",sxy:"\\sigma_{xy}",syy:"\\sigma_{yy}",ux:"u_x",uy:"u_y",fx:"f_x",fy:"f_y",rhoq:"\\rho_q",Tref:"T_{\\mathrm{ref}}",c0:"c_0"};
    if(engineering[name])return engineering[name];
    if(greek[name])return greek[name];
    var parts=String(name).split("_");
    var base=greek[parts[0]]||parts[0].replace(/([A-Za-z]+)/g,"\\mathrm{$1}");
    return parts.length>1?base+"_{"+parts.slice(1).join("\\_")+"}":base;
  }

  function expressionToLatex(source) {
    var text=String(source||""),output="",index=0;
    while(index<text.length){
      var rest=text.slice(index),match=rest.match(/^[A-Za-z][A-Za-z0-9_]*/);
      if(match){
        var name=match[0],after=index+name.length,isCall=text[after]==="(";
        if(isCall){
          var operators={dx:"\\partial_x",dy:"\\partial_y",dt:"\\partial_t",grad:"\\nabla",div:"\\nabla\\!\\cdot",Div:"\\operatorname{Div}",laplace:"\\nabla^2"};
          output+=(operators[name]||("\\operatorname{"+name+"}"))+"\\left(";index=after+1;continue;
        }
        output+=latexSymbol(name);index=after;continue;
      }
      if(text.slice(index,index+2)==="**"){
        var exponent=text.slice(index+2).match(/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)/);
        if(exponent){output+="^{"+exponent[0]+"}";index+=2+exponent[0].length;continue;}
        output+="^{\\,}";index+=2;continue;
      }
      var character=text[index];
      if(character==="(")output+="\\left(";
      else if(character===")")output+="\\right)";
      else if(character==="*")output+="\\,";
      else if(character==="·")output+="\\cdot ";
      else output+=character;
      index+=1;
    }
    return output;
  }

  function compiledLatex(model) {
    var rows=(model.equations||[]).map(function(item){var source=String(item.expression||""),parts=source.split("=");var residual=parts.length===2?(parts[1].trim()==="0"?parts[0]:("("+parts[0]+")-("+parts[1]+")")):source;return "R_{"+latexSymbol(item.target)+"} & = "+expressionToLatex(residual);});
    return "\\begin{aligned}"+rows.join("\\\\[0.45em]")+"\\end{aligned}";
  }

  function element(doc, name, className, text) { var node=doc.createElement(name); if(className)node.className=className; if(text!==undefined)node.textContent=text; return node; }
  function input(doc, value, role) { var node=element(doc,"input"); node.value=value==null?"":String(value); if(role)node.dataset.role=role; return node; }
  function labeled(doc, label, control) { var node=element(doc,"label"); node.append(label, control); return node; }
  function option(doc, value, label) { var node=element(doc,"option",null,label); node.value=value; return node; }

  function create(container, options) {
    options = options || {}; var doc=container.ownerDocument; var state=clone(byId(options.profile || "scalar_pde")); state.schema_version=2;
    container.classList.add("multiphysics-editor"); container.replaceChildren();
    var heading=element(doc,"div","multiphysics-heading"); var headingText=element(doc,"div"); headingText.append(element(doc,"h3",null,"通用物理问题编辑器"),element(doc,"p",null,"在一个模型中组织几何、数据、因变量、守恒方程、本构/材料、边界条件与耦合。配置可导出；执行能力单独、明确标注。"));
    var status=element(doc,"span","multiphysics-capability"); heading.append(headingText,status); container.append(heading);
    var selectors=element(doc,"div","multiphysics-selectors"); var classSelect=element(doc,"select"); classSelect.dataset.role="multiphysics-class"; CLASSES.forEach(function(item){classSelect.append(option(doc,item.id,item.label));});
    var profileSelect=element(doc,"select"); profileSelect.dataset.role="multiphysics-profile";
    var dimensionSelect=element(doc,"select"); dimensionSelect.dataset.role="multiphysics-dimension"; [1,2,3].forEach(function(n){dimensionSelect.append(option(doc,String(n),n+"D"));});
    var analysisSelect=element(doc,"select"); analysisSelect.dataset.role="multiphysics-analysis"; [["stationary","稳态"],["transient","瞬态"],["frequency","频域"],["eigenvalue","特征值"]].forEach(function(item){analysisSelect.append(option(doc,item[0],item[1]));});
    selectors.append(labeled(doc,"物理分类",classSelect),labeled(doc,"物理接口",profileSelect),labeled(doc,"空间维度",dimensionSelect),labeled(doc,"分析类型",analysisSelect)); container.append(selectors);
    var description=element(doc,"p","multiphysics-description"); container.append(description);
    var geometrySection=element(doc,"section","multiphysics-section multiphysics-geometry-section");
    var geometryHeading=element(doc,"div","multiphysics-section-heading"),geometryCopy=element(doc,"div");geometryCopy.append(element(doc,"strong",null,"计算域与几何"),element(doc,"p",null,"编辑解析几何或导入图片 / DXF 图纸；孔洞同步到实时图例与支持的代码生成器。"));geometryHeading.append(geometryCopy);geometrySection.append(geometryHeading);
    var geometrySlot=element(doc,"div","multiphysics-workspace-slot");geometrySlot.dataset.role="multiphysics-geometry-slot";var geometryAvailability=element(doc,"p","multiphysics-geometry-availability","当前外部图纸与交互几何支持 1D/2D；3D 物理模型可编辑和导出，但尚不提供浏览器三维 CAD 建模。");geometryAvailability.hidden=true;geometrySection.append(geometrySlot,geometryAvailability);
    var geometryFeatures=element(doc,"div","multiphysics-geometry-features"),featureHeading=element(doc,"div","multiphysics-section-heading"),featureCopy=element(doc,"div");featureCopy.append(element(doc,"strong",null,"解析几何特征 / 孔洞"),element(doc,"p",null,"圆孔用圆心与半径定义，区域 ID 可直接用于边界条件。"));var addFeature=element(doc,"button",null,"+ 添加圆孔");addFeature.type="button";addFeature.dataset.action="add-geometry-feature";featureHeading.append(featureCopy,addFeature);var featureList=element(doc,"div","multiphysics-list");featureList.dataset.role="multiphysics-geometry-features";geometryFeatures.append(featureHeading,featureList);geometrySection.append(geometryFeatures);container.append(geometrySection);
    var dataSection=element(doc,"section","multiphysics-section multiphysics-data-section"),dataHeading=element(doc,"div","multiphysics-section-heading"),dataCopy=element(doc,"div");dataCopy.append(element(doc,"strong",null,"训练与参考数据"),element(doc,"p",null,"上传 CSV 检查列名，并分别配置训练观测与训练后独立参考场；文件只由本地脚本读取。"));dataHeading.append(dataCopy);var dataSlot=element(doc,"div","multiphysics-workspace-slot");dataSlot.dataset.role="multiphysics-data-slot";dataSection.append(dataHeading,dataSlot);container.append(dataSection);
    function section(title, hint, role) { var box=element(doc,"section","multiphysics-section"); var head=element(doc,"div","multiphysics-section-heading"); var copy=element(doc,"div"); copy.append(element(doc,"strong",null,title),element(doc,"p",null,hint)); var add=element(doc,"button",null,"+ 添加"); add.type="button"; add.dataset.action="add-"+role; head.append(copy,add); var list=element(doc,"div","multiphysics-list"); list.dataset.role="multiphysics-"+role+"-list"; box.append(head,list); container.append(box); return {box:box,head:head,list:list,add:add}; }
    var fieldsSection=section("因变量 / 场变量","一个场可含多个分量，例如位移 (ux, uy) 或速度 (u, v)。","fields");
    var equationsSection=section("控制方程组","每行表示一个残差；系统方程无需挤入单个标量表达式。","equations");
    var compiledSection=element(doc,"section","multiphysics-section multiphysics-compiled"); compiledSection.append(element(doc,"strong",null,"控制方程编译预览")); var compiledHint=element(doc,"p",null,"以数学公式显示归一化后的残差算子；生成代码使用同一模型快照。"); var compiledOutput=element(doc,"div","multiphysics-compiled-formula"); compiledOutput.dataset.role="multiphysics-compiled-preview"; compiledSection.append(compiledHint,compiledOutput); container.append(compiledSection);
    var constitutiveSection=element(doc,"section","multiphysics-section multiphysics-constitutive-section"); var constitutiveHeading=element(doc,"div","multiphysics-section-heading"); var constitutiveCopy=element(doc,"div"); constitutiveCopy.append(element(doc,"strong",null,"本构与结构假设"),element(doc,"p",null,"仅固体力学显示；结构假设只在二维时出现。")); constitutiveHeading.append(constitutiveCopy); constitutiveSection.append(constitutiveHeading); container.append(constitutiveSection);
    var constitutiveTop=element(doc,"div","multiphysics-constitutive"); var modelInput=element(doc,"select"); modelInput.dataset.role="multiphysics-constitutive-model"; CONSTITUTIVE_MODELS.forEach(function(item){modelInput.append(option(doc,item[0],item[1]));}); var assumptionSelect=element(doc,"select"); assumptionSelect.dataset.role="multiphysics-assumption"; [["plane_stress","平面应力"],["plane_strain","平面应变"]].forEach(function(item){assumptionSelect.append(option(doc,item[0],item[1]));}); var modelLabel=labeled(doc,"本构模型",modelInput); var assumptionLabel=labeled(doc,"结构假设",assumptionSelect); assumptionLabel.dataset.role="multiphysics-assumption-field"; constitutiveTop.append(modelLabel,assumptionLabel); constitutiveSection.append(constitutiveTop);
    var materialSection=section("材料与物理参数","修改方程中使用的物理参数、数值和单位。","properties"); materialSection.add.hidden=true;
    var boundariesSection=section("边界与载荷","区域名称与几何边界对应；可指定作用场、分量和向量/标量目标。","boundaries");
    var couplingsSection=section("多物理耦合","显式记录源场、目标场及耦合项。","couplings");
    var actions=element(doc,"div","multiphysics-actions"); var exportButton=element(doc,"button",null,"导出模型 JSON"); exportButton.type="button"; var importButton=element(doc,"button",null,"导入模型 JSON"); importButton.type="button"; var jsonArea=element(doc,"textarea"); jsonArea.rows=5; jsonArea.placeholder="在这里查看、复制或粘贴 schema v2 模型 JSON"; actions.append(exportButton,importButton,jsonArea); container.append(actions);
    var feedback=element(doc,"p","multiphysics-feedback"); feedback.setAttribute("role","status"); container.append(feedback);

    function removeButton(item, collection) { var button=element(doc,"button","multiphysics-remove","删除"); button.type="button"; button.addEventListener("click",function(){var index=collection.indexOf(item);if(index>=0)collection.splice(index,1);renderRows();changed();}); return button; }
    function rowInput(item,key,labelText,collection,wide) { var control=input(doc,Array.isArray(item[key])?item[key].join(", "):item[key]); control.addEventListener("input",function(){item[key]=(key==="components"||key==="value")?control.value.split(",").map(function(v){return v.trim();}).filter(Boolean):control.value;changed();}); var label=labeled(doc,labelText,control); if(wide)label.className="is-wide"; return label; }
    function rowSelect(item,key,labelText,items,wide) { var control=element(doc,"select"); var values=items.map(function(entry){return String(entry[0]);}); if(!values.includes(String(item[key])))items=items.concat([[item[key],item[key]]]); items.forEach(function(entry){control.append(option(doc,String(entry[0]),String(entry[1])));}); control.value=String(item[key]); control.addEventListener("change",function(){item[key]=control.value;renderRows();changed();}); var label=labeled(doc,labelText,control);if(wide)label.className="is-wide";return label; }
    function unitSelect(item) { return rowSelect(item,"unit","单位",UNITS.map(function(unit){return [unit,unit];})); }
    function componentChoices() { var result=[];state.fields.forEach(function(item){result.push([item.id,item.label+"（整场）"]);(item.components||[]).forEach(function(symbol){result.push([symbol,symbol]);});});return result; }
    function regionChoices(current) { var base=state.dimension===1?["left","right"]:["left","right","bottom","top","boundary"].concat((state.geometry_features||[]).map(function(item){return item.id;}));return base.concat(String(current||"").split(",")).filter(function(value,index,all){return value&&all.indexOf(value)===index;}).map(function(value){return [value,value];}); }
    function renderCompiledFormula(){var latex=compiledLatex(state);compiledOutput.dataset.latex=latex;compiledOutput.classList.remove("is-fallback");if(doc.defaultView&&doc.defaultView.katex){try{doc.defaultView.katex.render(latex,compiledOutput,{throwOnError:true,displayMode:true});return;}catch(_error){}}compiledOutput.textContent=compiledPreview(state);compiledOutput.classList.add("is-fallback");}
    function renderRows() {
      featureList.replaceChildren();(state.geometry_features||[]).forEach(function(item,index){var row=element(doc,"div","multiphysics-row multiphysics-geometry-row");var idInput=input(doc,item.id||("hole_"+index));idInput.addEventListener("input",function(){item.id=idInput.value;changed();});var type=element(doc,"select");type.append(option(doc,"circle","圆孔"));var cx=input(doc,item.center&&item.center[0]);cx.type="number";cx.step="any";var cy=input(doc,item.center&&item.center[1]);cy.type="number";cy.step="any";var radius=input(doc,item.radius);radius.type="number";radius.step="any";radius.min="0";function updateFeature(){item.type="circle";item.center=[Number(cx.value),Number(cy.value)];item.radius=Number(radius.value);changed();}cx.addEventListener("input",updateFeature);cy.addEventListener("input",updateFeature);radius.addEventListener("input",updateFeature);row.append(labeled(doc,"区域 ID",idInput),labeled(doc,"类型",type),labeled(doc,"圆心 x",cx),labeled(doc,"圆心 y",cy),labeled(doc,"半径",radius),removeButton(item,state.geometry_features));featureList.append(row);});
      fieldsSection.list.replaceChildren(); state.fields.forEach(function(item){var row=element(doc,"div","multiphysics-row");row.append(rowInput(item,"label","名称",state.fields),rowInput(item,"id","ID",state.fields),rowSelect(item,"role","角色",FIELD_ROLES),rowInput(item,"components","分量（逗号分隔）",state.fields,true),unitSelect(item),removeButton(item,state.fields));fieldsSection.list.append(row);});
      equationsSection.list.replaceChildren(); state.equations.forEach(function(item){var row=element(doc,"div","multiphysics-row");row.append(rowInput(item,"label","方程",state.equations),rowSelect(item,"target","目标分量/场",componentChoices()),rowInput(item,"expression","残差 = 0",state.equations,true),removeButton(item,state.equations));equationsSection.list.append(row);});
      materialSection.list.replaceChildren(); state.constitutive.properties.forEach(function(item){var row=element(doc,"div","multiphysics-row");var numeric=input(doc,item.value);numeric.type="number";numeric.step="any";numeric.addEventListener("input",function(){item.value=numeric.value;changed();});var role=rowSelect(item,"role","参数角色",[["known","已知参数"],["inverse","待反演参数"]]);var lower=input(doc,item.lower===undefined?"":item.lower);lower.type="number";lower.step="any";var upper=input(doc,item.upper===undefined?"":item.upper);upper.type="number";upper.step="any";function updateBounds(){item.lower=lower.value===""?undefined:Number(lower.value);item.upper=upper.value===""?undefined:Number(upper.value);changed();}lower.addEventListener("input",updateBounds);upper.addEventListener("input",updateBounds);lower.disabled=item.role!=="inverse";upper.disabled=item.role!=="inverse";row.append(rowInput(item,"label","属性",state.constitutive.properties),rowInput(item,"id","符号",state.constitutive.properties),role,labeled(doc,item.role==="inverse"?"反演初值":"数值",numeric),labeled(doc,"下界",lower),labeled(doc,"上界",upper),unitSelect(item),removeButton(item,state.constitutive.properties));materialSection.list.append(row);});
      boundariesSection.list.replaceChildren(); state.boundaries.forEach(function(item){var row=element(doc,"div","multiphysics-row");var boundaryTypes=BOUNDARY_TYPES[state.physics_class]||[["dirichlet","Dirichlet"],["neumann","Neumann"],["robin","Robin"]];row.append(rowSelect(item,"region","区域",regionChoices(item.region)),rowSelect(item,"type","类型",boundaryTypes),rowSelect(item,"field","作用场",state.fields.map(function(fieldItem){return [fieldItem.id,fieldItem.label];})),rowInput(item,"components","分量",state.boundaries),rowInput(item,"value","值/表达式",state.boundaries,true),removeButton(item,state.boundaries));boundariesSection.list.append(row);});
      couplingsSection.list.replaceChildren(); state.couplings.forEach(function(item){var row=element(doc,"div","multiphysics-row");var fieldChoices=state.fields.map(function(fieldItem){return [fieldItem.id,fieldItem.label];});row.append(rowSelect(item,"type","类型",[["domain","域耦合"],["boundary","边界耦合"],["source","源项耦合"]]),rowSelect(item,"source","源场",fieldChoices),rowSelect(item,"target","目标场",fieldChoices),rowInput(item,"expression","耦合表达式",state.couplings,true),removeButton(item,state.couplings));couplingsSection.list.append(row);});
      renderCompiledFormula();
    }
    function syncSelectors() { classSelect.value=state.physics_class; profileSelect.replaceChildren(); PROFILES.filter(function(item){return item.physics_class===state.physics_class;}).forEach(function(item){profileSelect.append(option(doc,item.id,item.label));}); if(Array.prototype.some.call(profileSelect.options,function(item){return item.value===state.interface;}))profileSelect.value=state.interface; dimensionSelect.value=String(state.dimension); analysisSelect.value=state.analysis; var modelValue=state.constitutive.model||"custom";if(!Array.prototype.some.call(modelInput.options,function(item){return item.value===modelValue;}))modelInput.append(option(doc,modelValue,state.constitutive.label||modelValue));modelInput.value=modelValue; assumptionSelect.value=state.constitutive.assumption||"plane_stress"; var solidConstitutive=state.physics_class==="solid_mechanics"||state.interface==="thermoelasticity";constitutiveSection.hidden=!solidConstitutive; assumptionLabel.hidden=!solidConstitutive||Number(state.dimension)!==2; couplingsSection.box.hidden=state.fields.length<2&&state.physics_class!=="multiphysics"; geometryFeatures.hidden=Number(state.dimension)!==2;geometrySlot.hidden=Number(state.dimension)===3;geometryAvailability.hidden=Number(state.dimension)!==3; var definition=byId(state.interface); description.textContent=definition?definition.description:"自定义导入的物理接口。"; var capability=effectiveExecution(state); var runnable=capability.status==="runnable"; status.textContent=runnable?"可生成运行代码":"可完整建模 · 执行适配待接入"; status.title=capability.notes||""; status.classList.toggle("is-runnable",runnable); status.classList.toggle("is-editing-only",!runnable); renderCompiledFormula(); }
    function changed() { state.execution=effectiveExecution(state); syncSelectors(); try { validateModel(clone(state)); feedback.textContent="配置有效，代码与编译预览已同步。"; feedback.classList.remove("is-error"); } catch(error){feedback.textContent=error.message;feedback.classList.add("is-error");} if(typeof options.onChange==="function")options.onChange({model:clone(state),valid:!feedback.classList.contains("is-error"),compiled:compiledPreview(state),latex:compiledLatex(state)}); }
    function loadProfile(id) { var next=byId(id); if(!next)return; state=clone(next); state.schema_version=2; state.interface=state.id; delete state.id; syncSelectors();renderRows();changed(); }
    classSelect.addEventListener("change",function(){var first=PROFILES.find(function(item){return item.physics_class===classSelect.value;});if(first)loadProfile(first.id);}); profileSelect.addEventListener("change",function(){loadProfile(profileSelect.value);}); dimensionSelect.addEventListener("change",function(){state.dimension=Number(dimensionSelect.value);renderRows();changed();});analysisSelect.addEventListener("change",function(){state.analysis=analysisSelect.value;changed();});modelInput.addEventListener("change",function(){state.constitutive.model=modelInput.value;state.constitutive.label=modelInput.options[modelInput.selectedIndex].textContent;changed();});assumptionSelect.addEventListener("change",function(){state.constitutive.assumption=assumptionSelect.value||undefined;changed();});
    fieldsSection.add.addEventListener("click",function(){state.fields.push(field("field_"+(state.fields.length+1),"新场",["u"+(state.fields.length+1)],"1"));renderRows();changed();}); equationsSection.add.addEventListener("click",function(){state.equations.push(equation("equation_"+(state.equations.length+1),"新方程","u","R=0"));renderRows();changed();});boundariesSection.add.addEventListener("click",function(){state.boundaries.push(boundary("boundary_"+(state.boundaries.length+1),"boundary","dirichlet",state.fields[0].id,state.fields[0].components,["0"]));renderRows();changed();});couplingsSection.add.addEventListener("click",function(){state.couplings.push({id:"coupling_"+(state.couplings.length+1),type:"domain",source:state.fields[0].id,target:state.fields[0].id,expression:"source"});renderRows();changed();});
    addFeature.addEventListener("click",function(){var bounds=state.geometry_bounds||{x:[0,1],y:[0,1]};state.geometry_features.push({id:"hole_"+state.geometry_features.length,type:"circle",center:[(bounds.x[0]+bounds.x[1])/2,(bounds.y[0]+bounds.y[1])/2],radius:Math.min(bounds.x[1]-bounds.x[0],bounds.y[1]-bounds.y[0])/8});renderRows();changed();});
    exportButton.addEventListener("click",function(){jsonArea.value=JSON.stringify(validateModel(clone(state)),null,2);feedback.textContent="模型 JSON 已生成。";}); importButton.addEventListener("click",function(){try{var parsed=JSON.parse(jsonArea.value);validateModel(parsed);state=clone(parsed);syncSelectors();renderRows();changed();feedback.textContent="模型 JSON 已导入。";}catch(error){feedback.textContent="导入失败："+error.message;feedback.classList.add("is-error");}});
    function applyDraft(draft) {
      if (!draft || draft.physics !== "linear_elasticity") return clone(state);
      loadProfile("linear_elasticity");
      var constitutive = draft.constitutive || {};
      state.constitutive.assumption = constitutive.assumption || draft.formulation || "plane_stress";
      state.constitutive.properties.forEach(function (item) {
        if (constitutive[item.id] !== undefined) item.value = constitutive[item.id];
      });
      var geometry = draft.geometry || {};
      var outer = geometry.outer || {};
      if (geometry.bounds) outer = {x_min:geometry.bounds.x&&geometry.bounds.x[0],x_max:geometry.bounds.x&&geometry.bounds.x[1],y_min:geometry.bounds.y&&geometry.bounds.y[0],y_max:geometry.bounds.y&&geometry.bounds.y[1]};
      if ([outer.x_min,outer.x_max,outer.y_min,outer.y_max].every(function(value){return Number.isFinite(Number(value));})) state.geometry_bounds={x:[Number(outer.x_min),Number(outer.x_max)],y:[Number(outer.y_min),Number(outer.y_max)]};
      var hole = Array.isArray(geometry.holes) ? geometry.holes[0] : geometry.hole;
      if (hole) state.geometry_features = [{id:"hole_0",type:"circle",center:clone(hole.center || [1,1]),radius:hole.radius}];
      var traction = (draft.boundaries || []).find(function (item) { return item.region === "right" && item.type === "neumann"; });
      if (traction) {
        var target = state.boundaries.find(function (item) { return item.type === "traction"; });
        if (target) target.value = Array.isArray(traction.value) ? traction.value.map(String) : [String(traction.value),"0"];
      }
      renderRows(); changed(); return clone(state);
    }
    function applyScalarSpec(spec, context, boundaries) {
      if (!spec || !spec.expression_source) return clone(state);
      loadProfile("scalar_pde");
      state.dimension=Number(context&&context.dimension)||2;
      state.analysis=context&&context.kind==="dynamic"?"transient":"stationary";
      var symbol=spec.field||"u";
      state.fields=[field("field",context&&context.field_label||"标量场",[symbol],spec.field_unit||"1")];
      state.equations=[equation("pde","控制方程",symbol,spec.expression_source)];
      state.constitutive.properties=(spec.parameters||[]).map(function(item){return property(item.symbol||item.id,item.description||item.label||item.symbol||item.id,item.value,item.unit||"1");});
      state.boundaries.forEach(function(item){item.field="field";item.components=[symbol];});
      (boundaries||[]).forEach(function(item,index){var region=item.region||item.location||item.side||("boundary_"+index);var replacement=boundary(item.id||("boundary_"+index),region,item.type||"dirichlet","field",[symbol],[String(item.value===undefined?0:item.value)]);var existing=state.boundaries.findIndex(function(candidate){return candidate.region===region;});if(existing>=0)state.boundaries[existing]=replacement;else state.boundaries.push(replacement);});
      renderRows();changed();return clone(state);
    }
    state.interface=state.id; delete state.id; syncSelectors();renderRows();changed();
    if(doc.defaultView)doc.defaultView.addEventListener("load",renderCompiledFormula,{once:true});
    return {read:function(){var result=clone(state);result.execution=effectiveExecution(result);return validateModel(result);},loadProfile:loadProfile,applyDraft:applyDraft,applyScalarSpec:applyScalarSpec,exportModel:function(){return JSON.stringify(validateModel(clone(state)),null,2);},importModel:function(raw){jsonArea.value=typeof raw==="string"?raw:JSON.stringify(raw);importButton.click();return clone(state);},getCapability:function(){return effectiveExecution(state);},getCompiledPreview:function(){return compiledPreview(state);},getCompiledLatex:function(){return compiledLatex(state);},destroy:function(){container.replaceChildren();}};
  }

  return {classes:clone(CLASSES),profiles:clone(PROFILES),validateModel:validateModel,compiledPreview:compiledPreview,compiledLatex:compiledLatex,create:create};
});
