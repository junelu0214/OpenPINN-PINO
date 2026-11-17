const test = require('node:test');
const assert = require('node:assert/strict');
const {JSDOM} = require('jsdom');
const editor = require('../../docs/_static/multiphysics_editor.js');

function view(profile='scalar_pde') {
  const dom = new JSDOM('<div id="host"></div>');
  const host = dom.window.document.getElementById('host');
  return {dom, host, instance: editor.create(host,{profile})};
}

test('catalog covers general-purpose physics classes and specialist interfaces', () => {
  assert.deepEqual(editor.classes.map(item=>item.id), [
    'solid_mechanics','fluid_mechanics','heat_transfer','mass_transport',
    'electromagnetics','waves_acoustics','multiphysics','general_pde'
  ]);
  for (const id of ['linear_elasticity','incompressible_navier_stokes','heat_conduction',
    'convection_diffusion','electrostatics','helmholtz_acoustics','thermoelasticity','equation_system']) {
    assert(editor.profiles.some(item=>item.id===id), id);
  }
});

test('linear elasticity exposes vector fields, constitutive properties and executable adapter', () => {
  const {dom,host,instance}=view('linear_elasticity');
  try {
    const model=instance.read();
    assert.equal(model.physics_class,'solid_mechanics');
    assert.deepEqual(model.fields[0].components,['ux','uy']);
    assert.equal(model.constitutive.assumption,'plane_stress');
    assert(model.constitutive.properties.some(item=>item.id==='E'));
    assert.equal(model.execution.adapter,'elasticity_codegen');
    assert.equal(model.execution.status,'runnable');
    assert.equal(host.querySelector('.multiphysics-constitutive-section').hidden,false);
    assert.equal(host.querySelector('[data-role="multiphysics-assumption-field"]').hidden,false);
    assert.equal(host.querySelector('[data-role="multiphysics-constitutive-model"]').tagName,'SELECT');
    assert.match(host.querySelector('.multiphysics-heading h3').textContent,/通用物理问题编辑器/);
    assert.equal(host.querySelectorAll('[data-role="multiphysics-geometry-features"] .multiphysics-geometry-row').length,1);
    assert.match(host.querySelector('[data-role="multiphysics-compiled-preview"]').dataset.latex,/\\partial_x/);
    assert.match(host.querySelector('[data-role="multiphysics-compiled-preview"]').dataset.latex,/\\sigma_\{xx\}/);
    assert.match(host.querySelector('[data-role="multiphysics-compiled-preview"]').textContent,/σₓₓ/);
    assert.match(instance.getCompiledPreview(),/R\[ux\].*Dₓ/);
    const dimension=host.querySelector('[data-role="multiphysics-dimension"]');dimension.value='3';dimension.dispatchEvent(new dom.window.Event('change',{bubbles:true}));
    assert.equal(host.querySelector('[data-role="multiphysics-assumption-field"]').hidden,true);
    assert.equal(host.querySelector('[data-role="multiphysics-geometry-slot"]').hidden,true);
    assert.equal(host.querySelector('.multiphysics-geometry-availability').hidden,false);
    assert.equal(instance.getCapability().status,'editing_only');
  } finally { dom.window.close(); }
});

test('Darcy inverse profile exposes bounded inferred permeability and runnable adapter',()=>{
  const {dom,host,instance}=view('darcy_inverse_seepage');
  try{
    const model=instance.read();
    assert.equal(model.dimension,1);
    assert.equal(model.execution.adapter,'darcy_inverse_pinn');
    const permeability=model.constitutive.properties.find(item=>item.id==='k');
    assert.deepEqual({role:permeability.role,value:permeability.value,lower:permeability.lower,upper:permeability.upper},{role:'inverse',value:1,lower:0.2,upper:5});
    assert(host.textContent.includes('待反演参数'));
  }finally{dom.window.close();}
});

test('consolidation inverse profile exposes bounded cv and operator adapter',()=>{
  const {dom,instance}=view('consolidation_inverse_family');
  try{
    const model=instance.read();
    assert.equal(model.analysis,'transient');
    assert.equal(model.execution.adapter,'consolidation_inverse_operator');
    const coefficient=model.constitutive.properties.find(item=>item.id==='cv');
    assert.deepEqual({role:coefficient.role,value:coefficient.value,lower:coefficient.lower,upper:coefficient.upper},{role:'inverse',value:0.2,lower:0.05,upper:0.8});
  }finally{dom.window.close();}
});

test('multi-field fluid and coupled profiles are editable but do not claim solver support', () => {
  for (const id of ['incompressible_navier_stokes','thermoelasticity','conjugate_heat_transfer']) {
    const {dom,instance}=view(id);
    try {
      const model=instance.read();
      assert(model.fields.length>1,id);
      assert.equal(model.execution.status,'editing_only');
      assert.equal(instance.getCapability().adapter,null);
    } finally { dom.window.close(); }
  }
});

test('non-solid interfaces hide constitutive controls and use option controls where choices are bounded',()=>{
  const {dom,host}=view('incompressible_navier_stokes');
  try {
    assert.equal(host.querySelector('.multiphysics-constitutive-section').hidden,true);
    assert(host.querySelector('[data-role="multiphysics-fields-list"] select'));
    assert(host.querySelector('[data-role="multiphysics-equations-list"] select'));
    assert(host.querySelector('[data-role="multiphysics-boundaries-list"] select'));
    assert.match(host.querySelector('[data-role="multiphysics-compiled-preview"]').dataset.latex,/\\partial_x/);
  } finally {dom.window.close();}
});

test('field, equation, boundary and coupling rows can be added and exported', () => {
  const {dom,host,instance}=view('equation_system');
  try {
    host.querySelector('[data-action="add-fields"]').click();
    host.querySelector('[data-action="add-equations"]').click();
    host.querySelector('[data-action="add-boundaries"]').click();
    host.querySelector('[data-action="add-couplings"]').click();
    const model=instance.read();
    assert.equal(model.fields.length,3);
    assert.equal(model.equations.length,3);
    assert.equal(model.boundaries.length,1);
    assert.equal(model.couplings.length,1);
    assert.deepEqual(JSON.parse(instance.exportModel()),model);
  } finally { dom.window.close(); }
});

test('analytic holes can be added and edited in the general geometry section',()=>{
  const {dom,host,instance}=view('scalar_pde');
  try {
    host.querySelector('[data-action="add-geometry-feature"]').click();
    const row=host.querySelector('[data-role="multiphysics-geometry-features"] .multiphysics-geometry-row');
    const numbers=row.querySelectorAll('input[type="number"]');
    numbers[2].value='0.2';numbers[2].dispatchEvent(new dom.window.Event('input',{bubbles:true}));
    assert.equal(instance.read().geometry_features[0].radius,0.2);
  } finally {dom.window.close();}
});

test('model validation rejects duplicate components and unsafe text', () => {
  const model=JSON.parse(JSON.stringify(editor.profiles.find(item=>item.id==='equation_system')));
  model.schema_version=2; model.interface=model.id; delete model.id;
  model.fields[1].components=['u'];
  assert.throws(()=>editor.validateModel(model),/不能重复/);
  model.fields[1].components=['v']; model.equations[0].expression='<script>';
  assert.throws(()=>editor.validateModel(model),/禁止字符/);
});
