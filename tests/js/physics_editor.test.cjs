const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM} = require('jsdom');
const editor = require('../../docs/_static/physics_editor.js');
const catalog = JSON.parse(fs.readFileSync(path.join(__dirname, '../../src/openpinn/physics_catalog.json'), 'utf8'));
const copy = value => JSON.parse(JSON.stringify(value));
function spec(source, extra={}) {
  const s = Object.assign({schema_version:1, dimension:1, kind:'static', field:'p', field_unit:'1', coordinate_units:{x:'m',y:'m',t:'s'}, parameters:[], expression_source:source}, extra);
  s.expression = editor.parseExpression(source, s);
  return editor.validateSpec(s, catalog);
}
function view(context={dimension:1,kind:'static'}) {
  const dom = new JSDOM('<div id="host"></div>');
  const host = dom.window.document.getElementById('host');
  const instance = editor.create(host, {catalog,...context});
  return {dom,host,instance};
}
function controls(view) {
  const node = role => {
    const result = view.host.querySelector('[data-role="physics-'+role+'"]');
    assert(result, 'missing physics control: '+role);
    return result;
  };
  const set = (role,value,event='input') => {
    const input=node(role); input.value=String(value);
    input.dispatchEvent(new view.dom.window.Event(event,{bubbles:true}));
    return input;
  };
  return {node,set};
}
test('all bundled templates and definitions agree with frontend validation', () => {
  for (const t of catalog.templates) {
    const validated = editor.validateSpec(t.spec,catalog);
    assert.equal(editor.toSource(editor.parseExpression(t.expression_text,validated)), editor.toSource(validated.expression));
  }
  const {dom,instance}=view(); assert.equal(instance.read().dimension,1); dom.window.close();
});
test('consolidation templates expose pressure as the unknown and cv as a separate coefficient',()=>{
  for (const id of ['consolidation_1d','consolidation_2d_isotropic','pressure_diffusion']) {
    const entry=catalog.templates.find(item=>item.id===id);
    assert(entry,id);assert.equal(entry.spec.field,'p');assert.equal(entry.spec.field_unit,'Pa');
    assert.match(entry.label,/固结/);
    assert.equal(editor.validateSpec(entry.spec,catalog).kind,'dynamic');
  }
  const {dom,instance,host}=view({dimension:1,kind:'dynamic'});
  try {
    instance.loadTemplate('consolidation_1d');
    const field=host.querySelector('[data-role="physics-field"]');
    const preset=host.querySelector('[data-role="physics-field-preset"]');
    const unit=host.querySelector('[data-role="physics-field-unit"]');
    assert.equal(field.value,'p');assert.equal(unit.value,'Pa');
    assert.equal(instance.read().parameters.find(p=>p.symbol==='cv').unit,'m^2/s');
    field.value='u';field.dispatchEvent(new dom.window.Event('change',{bubbles:true}));
    preset.value='p';host.querySelector('[data-role="physics-use-field-preset"]').click();
    assert.equal(field.value,'p');assert.equal(unit.value,'Pa');
    assert.match(instance.read().expression_source,/dt\(p\)/);
  } finally {dom.window.close();}
});
test('distributed extension example validates and applies material values',()=>{
  const pack=JSON.parse(fs.readFileSync(path.join(__dirname,'../../examples/physics/site_diffusion_parameter_pack.json'),'utf8'));
  const {dom,host,instance}=view(); instance.importParameterPack(pack);
  const select=host.querySelector('[data-role="physics-material"]');select.value='site.demo_only';
  host.querySelector('[data-role="physics-add-material"]').click();
  assert.equal(instance.read().parameters.find(p=>p.symbol==='cx_site').value,.2);
  assert.equal(instance.read().parameters.find(p=>p.symbol==='cy_site').unit,'m^2/year');
  dom.window.close();
});
test('precedence, equation conversion, powers and pi', () => {
  assert.equal(editor.toSource(spec('-p**2=0').expression),'((-(p ** 2)) - 0)');
  assert.equal(editor.toSource(spec('p**(1+1)').expression),'(p ** (1 + 1))');
  assert(editor.toLatex(spec('sin(pi*p)').expression).includes('sin'));
});
test('unsupported syntax, undefined symbols, resource limits, order and dimensions are rejected', () => {
  for (const source of ['p.foo','p[0]','__import__(p)','fetch(p)','p;alert(1)','unknown+p','sin(p,2)','p=1=2','p/(2-2)','p**17','p+sqrt(-1)','p**(0/0)', 'dx(dx(dx(p)))','dy(p)','dt(p)','x', 'p + y']) {
    assert.throws(()=>spec(source), source);
  }
  assert.throws(()=>spec('dt(dt(p))',{kind:'dynamic'}));
  assert.throws(()=>spec('p+x',{field_unit:'Pa'}), /量纲/);
  assert.throws(()=>spec('sin(p)',{field_unit:'Pa'}), /无量纲/);
  assert.throws(()=>spec('('.repeat(40)+'p'+')'.repeat(40)),/嵌套/);
  assert.throws(()=>spec('p',{parameters:[{id:'test',symbol:'pi',value:1,unit:'1'}]}),/保留/);
});
test('known coefficients and continuous spatial variation use dimensionally valid formulas', () => {
  const parameters=[{id:'a',symbol:'a',value:2,unit:'m^2/s'},{id:'L',symbol:'L',value:3,unit:'m'}];
  assert(spec('dt(p)-dx(a*(1+x/L)*dx(p))',{kind:'dynamic',field_unit:'Pa',parameters}));
  assert.throws(()=>spec('dt(p)-a*dx(dx(p))',{kind:'dynamic',field_unit:'Pa',parameters:[{...parameters[0],unit:'s'}]}));
});
test('spec roundtrip and failed import preserve current state',()=>{
  const {dom,instance}=view(), before=instance.exportProject();
  instance.importProject(before); assert.equal(instance.exportProject(),before);
  const bad=JSON.parse(before); bad.expression_source='p';
  assert.throws(()=>instance.importProject(bad), /不一致/); assert.equal(instance.exportProject(),before);
  assert.throws(()=>instance.importProject({...JSON.parse(before),dimension:2}),/维度/);
  assert.equal(instance.exportProject(),before);
  assert.throws(()=>instance.importProject('{"__proto__":{}}'), /禁止/);
  dom.window.close();
});
test('prefill is transactional and the slim editor hides preset controls',()=>{
  const v=view({dimension:1,kind:'dynamic'});
  try {
    assert.equal(v.host.querySelector('[data-role="physics-template-controls"]').hidden,true);
    assert.equal(v.host.querySelector('[data-role="physics-field-preset-controls"]').hidden,true);
    const before=v.instance.exportProject();
    for (const expression_source of [null,'Delta u = 0']) {
      const result=v.instance.applyPrefill({
        field:'u',field_unit:'K',coordinate_units:{x:'m',t:'s'},parameters:[],expression_source,
      });
      assert.equal(result.valid,false);
      assert.equal(result.applied,false);
      assert.equal(v.instance.exportProject(),before,'failed prefill must preserve the current formula and parameters');
    }
  } finally { v.dom.window.close(); }
});
test('parameter packs separate definitions and material values, reimport without overwriting',()=>{
  const a=view(), b=view();
  const source=a.instance.exportParameterPack(), pack=JSON.parse(source);
  assert(pack.parameters.every(p=>p.value===null)); assert.equal(pack.materials[0].parameters[0].value,1);
  b.instance.importParameterPack(source);
  const select=b.host.querySelector('[data-role="physics-parameter-definition"]');
  assert([...select.options].some(o=>o.value==='demo_cx'));
  const oldOptions=select.innerHTML;
  assert.throws(()=>b.instance.importParameterPack(source),/材料 id 重复/); assert.equal(select.innerHTML,oldOptions);
  const invalid=copy(pack); invalid.parameters.push({...invalid.parameters[0],id:'bad',unit:'bad'});
  assert.throws(()=>b.instance.importParameterPack(invalid)); assert.equal(select.innerHTML,oldOptions);
  assert.throws(()=>b.instance.importParameterPack({...pack,units:{}}),/不支持/);
  a.dom.window.close();b.dom.window.close();
});
test('minimum and maximum, SI overflow, and reserved names are checked',()=>{
  for (const p of [
    {value:-1,minimum:0}, {value:2,maximum:1}, {value:null}, {value:Infinity}, {value:1,minimum:5,maximum:0}, {value:1,symbol:'a__b'},
  ]) assert.throws(()=>spec('p',{parameters:[Object.assign({id:'a',symbol:'a',unit:'Pa',value:1},p)]}));
  assert.throws(()=>spec('p',{parameters:[{id:'a',symbol:'a',value:1e308,unit:'GPa'}]}));
});
test('context changes preserve the typed formula and flag incompatibility',()=>{
  const {dom,instance}=view(); instance.setContext({dimension:2,kind:'dynamic'});instance.loadTemplate('pressure_diffusion');
  const source=instance.read().expression_source;
  instance.setContext({dimension:1,kind:'static'}); assert.throws(()=>instance.read());
  instance.setContext({dimension:2,kind:'dynamic'});assert.equal(instance.read().expression_source,source);
  dom.window.close();
});
test('formula is first and routine parameter editing does not expose internal identifiers',()=>{
  const v=view({dimension:2,kind:'dynamic'}), {node}=controls(v);
  try {
    const follows=v.dom.window.Node.DOCUMENT_POSITION_FOLLOWING;
    for (const role of ['parameters','search','material','import-pack']) {
      assert(node('formula').compareDocumentPosition(node(role)) & follows, role+' must follow the formula');
    }
    for (const role of ['insert-dx','insert-dy','insert-dt','insert-dxx','insert-dyy','insert-dxy']) {
      assert(node(role).compareDocumentPosition(node('parameters')) & follows);
    }
    for (const key of ['id','description','source']) {
      const details=node('parameter-0-'+key).closest('details');
      assert(details, key+' belongs in advanced details');
      assert.equal(details.open,false, key+' should be collapsed initially');
    }
    for (const key of ['label','symbol','value','unit']) {
      const input=node('parameter-0-'+key);
      assert(!input.closest('details'), key+' is part of routine editing');
    }
    const idDetails=node('new-id').closest('details');
    assert(idDetails); assert.equal(idDetails.open,false);
    for (const role of ['new-label','new-symbol','new-value','new-unit']) {
      assert(!idDetails.contains(node(role)),role+' must not require opening advanced identity settings');
    }
  } finally { v.dom.window.close(); }
});
test('visible derivative controls construct the requested two-dimensional pressure equation',()=>{
  const v=view({dimension:2,kind:'dynamic'}), {node,set}=controls(v);
  try {
    set('template','pressure_diffusion','change'); node('load-template').click();
    set('field','u'); set('formula','');
    for (const role of ['insert-dx','insert-dy','insert-dt','insert-dxx','insert-dyy','insert-dxy']) {
      assert.equal(node(role).disabled,false);
      assert.match(node(role).textContent,/u/,'labels follow the currently selected field');
      assert.match(node(role).textContent,/∂/,'derivatives have mathematical labels');
    }
    for (const role of ['insert-dxx','insert-dyy','insert-dxy']) assert.match(node(role).textContent,/²|2/);
    node('insert-dt').click();
    const formula=node('formula');
    function append(text) { formula.setRangeText(text,formula.value.length,formula.value.length,'end'); }
    append(' = cx * '); node('insert-dxx').click();
    append(' + cy * '); node('insert-dyy').click();
    assert.equal(formula.value,'dt(u) = cx * dx(dx(u)) + cy * dy(dy(u))');
    const actual=v.instance.read();
    assert.equal(actual.field,'u'); assert.equal(actual.dimension,2); assert.equal(actual.kind,'dynamic');
    const expected=editor.parseExpression('dt(u) - cx*dx(dx(u)) - cy*dy(dy(u))',actual);
    // The equation syntax converts the right-hand side as one residual subtree.
    assert.equal(editor.toSource(actual.expression),'(dt(u) - ((cx * dx(dx(u))) + (cy * dy(dy(u)))))');
    assert.equal(editor.validateAST(expected,actual).type,'binary');
    assert.equal(v.instance.getStatus().valid,true);
    assert.match(node('latex').getAttribute('data-latex'),/partial/);
    assert(!node('latex').getAttribute('data-latex').endsWith('= 0'),'the preview preserves left = right instead of turning everything into a residual');
  } finally { v.dom.window.close(); }
});
test('derivative insertion wraps selections while parameter insertion replaces selections',()=>{
  const v=view({dimension:2,kind:'dynamic'}), {node,set}=controls(v);
  try {
    set('template','pressure_diffusion','change'); node('load-template').click();
    set('field','u');
    const formula=set('formula','2*u'); formula.setSelectionRange(0,3);
    node('insert-dxx').click(); assert.equal(formula.value,'dx(dx(2*u))');
    assert.equal(formula.selectionStart,formula.value.length);
    set('formula','u'); formula.setSelectionRange(0,1);
    node('insert-dxy').click(); assert.equal(formula.value,'dx(dy(u))');
    set('formula','u + 2*u'); formula.setSelectionRange(4,7);
    set('insert-parameter','cy','change'); node('insert-symbol').click();
    assert.equal(formula.value,'u + cy','inserted coefficients replace the highlighted text');
    set('field','head');
    assert.match(node('insert-dyy').textContent,/head/);
    set('formula',''); node('insert-dyy').click(); assert.equal(formula.value,'dy(dy(head))');
  } finally { v.dom.window.close(); }
});
test('derivative availability follows context and incompatible templates preserve user work',()=>{
  const v=view(), {node,set}=controls(v);
  try {
    assert.equal(node('insert-dx').disabled,false);
    assert.equal(node('insert-dxx').disabled,false);
    for (const role of ['insert-dy','insert-dyy','insert-dxy','insert-dt']) assert.equal(node(role).disabled,true);
    const before=v.instance.exportProject();
    assert.throws(()=>v.instance.loadTemplate('pressure_diffusion'),/维度|类型/);
    assert.equal(v.instance.exportProject(),before);
    const original=node('formula').value;
    v.instance.setContext({dimension:2,kind:'dynamic'});
    for (const role of ['insert-dy','insert-dyy','insert-dxy','insert-dt']) assert.equal(node(role).disabled,false);
    assert.equal(node('formula').value,original,'changing context must not overwrite a typed formula');
    set('formula','dt(p)');
    v.instance.setContext({dimension:2,kind:'static'});
    assert.equal(node('insert-dt').disabled,true); assert.equal(node('insert-dyy').disabled,false);
    assert.equal(node('formula').value,'dt(p)'); assert.equal(v.instance.getStatus().valid,false);
    v.instance.setContext({dimension:1,kind:'dynamic'});
    assert.equal(node('insert-dt').disabled,false); assert.equal(node('insert-dxy').disabled,true);
  } finally { v.dom.window.close(); }
});
test('committed field renames replace whole identifiers without rewriting parameter or function names',()=>{
  const v=view(), {node,set}=controls(v);
  try {
    set('field-unit','1');
    set('new-label','偏置'); set('new-symbol','pp'); set('new-value',1); set('new-unit','1'); node('add-custom').click();
    set('formula','exp(p) + pp');
    set('field','u','input');
    assert.equal(node('formula').value,'exp(p) + pp','typing a field name must not rewrite the formula prematurely');
    assert.match(node('insert-dxx').textContent,/u/);
    set('field','u','change');
    assert.equal(node('formula').value,'exp(u) + pp'); assert.equal(v.instance.read().field,'u');
    const source=node('formula').value;
    set('field','not valid','change'); assert.equal(node('formula').value,source);
    assert.equal(v.instance.getStatus().valid,false);
    set('field','pp','change'); assert.equal(node('formula').value,source,'a parameter collision cannot rename the field');
    set('field','u','change'); assert.equal(v.instance.read().expression_source,source);
  } finally { v.dom.window.close(); }
});
test('custom parameters need no explicit id and automatic ids remain unique',()=>{
  const v=view(), {node,set}=controls(v);
  try {
    set('new-label','渗流系数'); set('new-symbol','a'); set('new-value',2); set('new-unit','m^2/s');
    node('add-custom').click();
    let parameters=v.instance.read().parameters;
    const first=parameters.find(p=>p.symbol==='a');
    assert(first); assert.equal(first.id,'custom.a'); assert.equal(first.label,'渗流系数');
    assert.equal(first.value,2); assert.equal(first.unit,'m^2/s');
    // Rename only the formula symbol: the first stable id remains reserved.
    set('parameter-1-symbol','a_old');
    set('new-id',''); set('new-label','第二个系数'); set('new-symbol','a'); set('new-value',3); set('new-unit','m^2/s');
    node('add-custom').click();
    parameters=v.instance.read().parameters;
    const second=parameters.find(p=>p.symbol==='a');
    assert(second); assert.notEqual(second.id,first.id); assert.match(second.id,/^custom\.a/);
    assert.equal(parameters.length,3); assert.equal(new Set(parameters.map(p=>p.id)).size,3);
    set('parameter-2-label','修改后的名称');
    assert.equal(v.instance.read().parameters[2].label,'修改后的名称');
    const count=parameters.length;
    set('new-id','explicit.duplicate'); set('new-symbol','a'); node('add-custom').click();
    assert.equal(v.instance.read().parameters.length,count,'automatic ids must not permit duplicate active symbols');
    assert.match(node('status').textContent,/重复/);
  } finally { v.dom.window.close(); }
});
test('automatic ids avoid imported library definitions and advanced explicit ids still roundtrip',()=>{
  const v=view(), {node,set}=controls(v);
  try {
    v.instance.importParameterPack({schema_version:1,parameters:[{id:'custom.k',label:'库参数',symbol:'other_k',value:null,unit:'1',category:'custom'}],materials:[]});
    set('new-label','自定义系数'); set('new-symbol','k'); set('new-value',4); set('new-unit','1'); node('add-custom').click();
    const generated=v.instance.read().parameters.find(p=>p.symbol==='k');
    assert(generated); assert.notEqual(generated.id,'custom.k');
    node('new-id').closest('details').open=true;
    set('new-id','project.source'); set('new-label','来源项'); set('new-symbol','q'); set('new-value',5); set('new-unit','Pa'); node('add-custom').click();
    const source=v.instance.exportProject();
    assert.equal(JSON.parse(source).parameters.find(p=>p.symbol==='q').id,'project.source');
    v.instance.importProject(source); assert.equal(v.instance.exportProject(),source);
  } finally { v.dom.window.close(); }
});
