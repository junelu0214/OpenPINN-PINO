/* Exercise the real generated page, not a handwritten code-generation substitute. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {JSDOM}=require('jsdom');
const root=path.resolve(__dirname,'../..');
async function main() {
  const dom=new JSDOM(fs.readFileSync(path.join(root,'docs/_build/html/method_configurator.html'),'utf8'),{runScripts:'outside-only'});
  const w=dom.window,d=w.document;
  await new Promise(resolve=>w.addEventListener('DOMContentLoaded',resolve,{once:true}));
  const errors=[];w.addEventListener('error',event=>errors.push(event.message));
  for(const file of ['polygon_image_trace','dxf_geometry_import','physics_editor','method_configurator']) w.eval(fs.readFileSync(path.join(root,'docs/_static',file+'.js'),'utf8'));
  d.dispatchEvent(new w.Event('DOMContentLoaded'));
  const node=role=>d.querySelector('[data-role="'+role+'"]');
  function change(element,value,event='change'){assert(element);element.value=String(value);element.dispatchEvent(new w.Event(event,{bubbles:true}));}
  const set=(role,value)=>change(node(role),value);
  const code=()=>node('code').textContent;
  const programs=[];
  function capture(name){assert.equal(node('copy').disabled,false,node('message').textContent); assert.match(code(),/equation = op.make_physics_residual/);programs.push({name,code:code()});}
  set('problem','custom_equation_template');set('custom-equation-mode','expression');set('steps',2);
  capture('default_1d');
  set('custom-dimension',2);set('custom-kind','dynamic');
  set('physics-template','pressure_diffusion');node('physics-load-template').click();
  set('custom-initial-preset','constant');set('custom-initial-value',2);
  capture('pressure');
  // Use the visible derivative controls to enter the user's equation, not a preset string.
  set('physics-field','u');
  assert(!/\bp\b/.test(node('physics-formula').value),'committing a field rename updates its expression tokens');
  set('physics-formula','');
  const userFormula=node('physics-formula');
  const append=text=>userFormula.setRangeText(text,userFormula.value.length,userFormula.value.length,'end');
  node('physics-insert-dt').click();append(' = cx * ');node('physics-insert-dxx').click();
  append(' + cy * ');node('physics-insert-dyy').click();
  assert.equal(userFormula.value,'dt(u) = cx * dx(dx(u)) + cy * dy(dy(u))');
  capture('pressure_user_equation');
  assert.match(code(),/dt\(u\) = cx \* dx\(dx\(u\)\) \+ cy \* dy\(dy\(u\)\)/);
  change(node('physics-parameter-0-value'),0.7,'input');capture('pressure_changed');
  assert.notEqual(programs.at(-1).code,programs.at(-2).code);
  // Unit changes must appear in the executable spec, not just in labels.
  set('physics-length-unit','cm');set('physics-time-unit','day');set('physics-field-unit','kPa');
  capture('converted_units');
  const formula=node('physics-formula').value;
  set('physics-formula','dt(u)+dx(u)');assert.equal(node('copy').disabled,false);assert.equal(node('download-code').disabled,false);assert.match(code(),/NotImplementedError/);
  assert.match(node('physics-status').textContent,/量纲/);
  set('physics-formula',formula);assert.equal(node('copy').disabled,false);
  set('custom-kind','static');assert.equal(node('copy').disabled,false);assert.match(code(),/NotImplementedError/);
  set('physics-template','steady_darcy');node('physics-load-template').click();capture('darcy');
  set('custom-kind','dynamic');set('physics-template','heat_diffusion');node('physics-load-template').click();capture('heat');
  // Add a custom thermal source with units, then exercise delete and invalid values.
  set('physics-new-label','体积热源');set('physics-new-symbol','Q');set('physics-new-value',2);set('physics-new-unit','W/m^3');node('physics-add-custom').click();
  assert.equal(node('physics-parameter-3-id').value,'custom.Q');
  assert.equal(node('physics-parameter-3-id').closest('details').open,false);
  const base=node('physics-formula').value;set('physics-formula',base+' - Q');capture('heat_source');
  assert.match(code(),/custom\.Q/);assert.match(code(),/体积热源/);
  change(node('physics-parameter-3-value'),'','input');assert.equal(node('copy').disabled,false);assert.match(code(),/NotImplementedError/);
  change(node('physics-parameter-3-value'),3,'input');capture('heat_source_changed');
  const beforeLabel=code();change(node('physics-parameter-3-label'),'实验热源','input');
  assert.notEqual(code(),beforeLabel);assert.match(code(),/实验热源/);
  // Actual file-input import, including state preservation on a rejected file.
  async function upload(role,data) {
    const input=node(role);Object.defineProperty(input,'files',{configurable:true,value:[new w.File([JSON.stringify(data)],'physics.json')]});
    input.dispatchEvent(new w.Event('change',{bubbles:true}));
    await new Promise(resolve=>w.setTimeout(resolve,100));
  }
  const pack={schema_version:1,parameters:[{id:'local.extra_source',symbol:'Qs',value:null,unit:'W/m^3',description:'演示附加源',label:'附加源',category:'custom'}],materials:[{id:'local.demo',label:'演示参数组',source:'测试，无实测含义',parameters:[{id:'local.extra_source',value:4,unit:'W/m^3'}]}]};
  await upload('physics-import-pack',pack);set('physics-material','local.demo');node('physics-add-material').click();
  set('physics-formula',node('physics-formula').value+' - Qs');capture('material');
  const before=code();await upload('physics-import-pack',{...pack,units:{}});assert.equal(code(),before);assert.match(node('physics-status').textContent,/不支持/);
  set('custom-x-max',5);capture('geometry_changed');assert.match(code(),/X_MAX = 5\b/);
  change(d.querySelector('[data-role="custom-boundary-side"][data-side="left"] [data-role="boundary-value"]'),9);
  capture('boundary_changed');assert.match(code(),/LEFT_BOUNDARY_VALUE = 9\b/);
  // Old modes still generate their native residuals.
  set('custom-equation-mode','free');assert.match(code(),/def equation\(x, u\):/);assert(!code().includes('make_physics_residual'));
  assert.deepEqual(errors,[]);
  dom.window.close();return programs;
}
if(process.argv.includes('--json')) main().then(result=>process.stdout.write(JSON.stringify(result))).catch(error=>{console.error(error);process.exitCode=1;});
else require('node:test')('physical editor -> live configurator -> generated Python', async()=>{await main();});
