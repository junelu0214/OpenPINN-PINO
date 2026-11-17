/* Workflow nodes navigate the current workbench, never a second page or tab. */
const test=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const {pathToFileURL}=require('node:url');
const {JSDOM}=require('jsdom');
const project=path.resolve(__dirname,'../..'),root=path.join(project,'docs/_build/html');
const page=path.join(root,'method_configurator.html');
const destinations={
  start:'configurator-workbench',direct_problem:'configurator-problem',
  direct_modules:'configurator-geometry',agent_understanding:'configurator-agent-pipeline',
  agent_planning:'configurator-agent-pipeline',agent_handoff:'configurator-problem',
  method:'configurator-method-title',code:'configurator-code-title',
  run:'configurator-run-help',diagnostic:'configurator-diagnostic'
};
const helpIds=['configurator-run-help','configurator-results-help','configurator-troubleshooting-help'];

test('all workflow and help links point to existing anchors in this configurator page',()=>{
  const dom=new JSDOM(fs.readFileSync(page,'utf8'));
  const d=dom.window.document,guide=d.querySelector('.configurator-guide');
  try {
    assert(guide);assert(guide.querySelector('svg title'));assert(guide.querySelector('svg desc'));
    const nodes=[...guide.querySelectorAll('.framework-node')],keys=new Set(nodes.map(a=>a.dataset.node));
    assert.deepEqual([...keys],Object.keys(destinations));
    for (const node of nodes) assert.equal(node.getAttribute('href'),'#'+destinations[node.dataset.node]);
    for (const edge of guide.querySelectorAll('.framework-edge')) {
      assert(keys.has(edge.dataset.from));assert(keys.has(edge.dataset.to));
      assert(edge.querySelector('path[marker-end]'));
    }
    assert.equal(guide.querySelectorAll('a').length,12);
    for (const link of guide.querySelectorAll('a')) {
      assert.equal(link.getAttribute('target'),null);
      const href=link.getAttribute('href');assert.match(href,/^#[A-Za-z][\w-]*$/);
      assert(d.getElementById(href.slice(1)),href);
      assert.equal(d.querySelectorAll('[id="'+href.slice(1)+'"]').length,1,'anchor must be unique: '+href);
    }
    assert(!guide.textContent.includes('新标签页'));
    for (const id of helpIds) {
      const details=d.getElementById(id);assert.equal(details.tagName,'DETAILS');assert.equal(details.open,false);
      assert(details.querySelector('summary'));assert(guide.querySelector('a[href="#'+id+'"]'));
    }
    const details=d.querySelector('.configurator-usage-details');assert(details);assert.equal(details.open,false);
    assert(details.textContent.includes('MultiPolygon'));assert(details.textContent.includes('op.make_physics_residual'));
    const workbench=d.querySelector('.openpinn-configurator');assert(workbench);assert(!workbench.closest('details'));
    assert(guide.compareDocumentPosition(workbench)&dom.window.Node.DOCUMENT_POSITION_FOLLOWING);
    assert(d.getElementById('configurator-workbench'));
    assert.equal(d.querySelector('.configurator-guide-actions a').getAttribute('href'),'#configurator-workbench');
    // Native anchors preserve keyboard navigation; visited links are not marked as selected.
    for (const a of nodes) {assert.equal(a.getAttribute('tabindex'),'0');assert(a.getAttribute('aria-label'));}
    const css=fs.readFileSync(path.join(root,'_static/manual_diagrams.css'),'utf8');
    assert(css.includes(':focus-visible'));assert(css.includes(':visited'));assert(css.includes('overflow-x: auto'));
  } finally {dom.window.close();}
});

async function workbench(hash='') {
  const dom=new JSDOM(fs.readFileSync(page,'utf8'),{
    runScripts:'outside-only',url:pathToFileURL(page).href+hash
  });
  const w=dom.window,d=w.document,errors=[],scrolls=[],opens=[];
  w.addEventListener('error',event=>errors.push(event.message));
  w.Element.prototype.scrollIntoView=function(options){scrolls.push({element:this,options});};
  w.open=(...args)=>{opens.push(args);return null;};
  await new Promise(resolve=>w.addEventListener('DOMContentLoaded',resolve,{once:true}));
  for (const file of ['polygon_image_trace','dxf_geometry_import','physics_editor','multiphysics_editor','method_configurator','configurator_navigation']) {
    w.eval(fs.readFileSync(path.join(project,'docs/_static',file+'.js'),'utf8'));
  }
  d.dispatchEvent(new w.Event('DOMContentLoaded'));
  const node=role=>d.querySelector('[data-role="'+role+'"]');
  const link=key=>d.querySelector('.configurator-guide [data-node="'+key+'"]');
  const set=(role,value)=>{
    const control=node(role);assert(control,role);control.value=String(value);
    control.dispatchEvent(new w.Event('change',{bubbles:true}));
  };
  const snapshot=()=>JSON.stringify({
    controls:[...d.querySelectorAll('.openpinn-configurator input, .openpinn-configurator select, .openpinn-configurator textarea')]
      .map(el=>[el.dataset.role,el.value,el.checked,el.disabled]),
    code:node('code').textContent
  });
  const click=element=>{
    assert(element);
    // Dispatch on a child of the SVG anchor to exercise event delegation as a real node click does.
    const target=element.querySelector('rect')||element;
    const event=new w.MouseEvent('click',{bubbles:true,cancelable:true,button:0});
    target.dispatchEvent(event);
    assert(event.defaultPrevented,'same-page guide handles navigation without a native second-page load');
  };
  const assertLocal=()=>{
    assert.equal(w.location.pathname,new URL(pathToFileURL(page)).pathname);
    assert.equal(w.location.search,'');assert.deepEqual(opens,[]);assert.deepEqual(errors,[]);
  };
  const assertAt=target=>{
    assert(target);
    assert(scrolls.length,'navigation scrolls to its visible destination');
    const scrolled=scrolls.at(-1).element;
    assert(scrolled===target||scrolled.contains(target)||target.contains(scrolled),'scrolled destination matches the requested section');
    assert(target===d.activeElement||target.contains(d.activeElement),'keyboard focus follows the destination');
    assert(!d.activeElement.closest('[hidden]'),'never focus a hidden control');
  };
  return {dom,w,d,node,link,set,snapshot,click,scrolls,assertAt,assertLocal};
}

test('custom-only selection is preserved when workflow nodes are clicked',async()=>{
  const h=await workbench();
  try {
    const {d,node,link,set,click,snapshot,assertAt,assertLocal}=h;
    assert.equal(node('problem').options.length,1);
    assert.equal(node('problem').value,'custom_equation_template');
    assert(!node('custom-problem-panel').hidden);
    set('steps',137);set('seed',7);set('output-root','runs/guide-preserve');
    const before=snapshot();
    for (const key of ['direct_modules']) {
      click(link(key));
      assert.equal(snapshot(),before,'a navigation click must not switch problem or reset settings');
      assert(!node('custom-problem-panel').hidden);
      assertAt(node('custom-geometry-field'));
      assertLocal();
    }
    for (const key of ['start','direct_problem','agent_understanding','agent_planning','agent_handoff','method','code','diagnostic']) {
      click(link(key));assertAt(d.getElementById(destinations[key]));assert.equal(snapshot(),before);assertLocal();
    }
  } finally {h.dom.window.close();}
});

test('direct-route nodes navigate the shared problem editor without changing inputs or code',async()=>{
  const h=await workbench();
  try {
    const {d,node,link,set,click,snapshot,assertAt,assertLocal}=h;
    set('problem','custom_equation_template');
    let before=snapshot();click(link('direct_modules'));assertAt(node('custom-geometry-field'));assert.equal(snapshot(),before);
    set('multiphysics-dimension',2);set('multiphysics-analysis','transient');
    set('custom-x-max',5);set('steps',41);
    const boundaryRow=Array.from(node('multiphysics-boundaries-list').children).find(row=>row.querySelector('select')&&row.querySelector('select').value==='left');
    const boundary=Array.from(boundaryRow.querySelectorAll('input')).find(input=>input.value==='0');
    boundary.value='9';boundary.dispatchEvent(new h.w.Event('input',{bubbles:true}));
    before=snapshot();
    click(link('direct_modules'));assertAt(node('custom-geometry-field'));assert.equal(snapshot(),before);
    click(link('direct_problem'));assertAt(d.getElementById('configurator-problem'));assert.equal(snapshot(),before);
    // A navigation operation must preserve even a temporarily invalid expression being edited.
    const equation=Array.from(node('multiphysics-equations-list').querySelectorAll('input')).find(input=>input.value.includes('dx('));
    equation.value=equation.value+' + unknown_parameter';equation.dispatchEvent(new h.w.Event('input',{bubbles:true}));
    assert(!node('copy').disabled);assert.match(node('code').textContent,/NotImplementedError/);before=snapshot();
    click(link('direct_modules'));assertAt(node('custom-geometry-field'));assert.equal(snapshot(),before);assert(!node('copy').disabled);
    assertLocal();
  } finally {h.dom.window.close();}
});

test('run, results and troubleshooting help open in place; hash history restores the correct destination',async()=>{
  const h=await workbench();
  try {
    const {w,d,node,click,snapshot,assertAt,assertLocal}=h,before=snapshot();
    for (const id of helpIds) {
      const details=d.getElementById(id);details.open=false;
      click(d.querySelector('.configurator-guide a[href="#'+id+'"]'));
      assert(details.open);assert.equal(w.location.hash,'#'+id);assertAt(details);assert.equal(snapshot(),before);
      await new Promise(resolve=>w.setTimeout(resolve,0));
    }
    // A copied URL/hashchange and browser back/forward must also expand collapsed destinations.
    const run=d.getElementById(helpIds[0]);run.open=false;
    const changed=new Promise(resolve=>w.addEventListener('hashchange',resolve,{once:true}));
    w.location.hash=helpIds[0];await changed;
    assert(run.open);assertAt(run);assert.equal(snapshot(),before);
    async function traverse(direction,id) {
      const details=d.getElementById(id);details.open=false;
      await new Promise((resolve,reject)=>{
        const timeout=w.setTimeout(()=>reject(new Error('history '+direction+' did not produce hashchange')),1000);
        w.addEventListener('hashchange',()=>{w.clearTimeout(timeout);resolve();},{once:true});
        w.history[direction]();
      });
      assert.equal(w.location.hash,'#'+id);assert(details.open);assertAt(details);assert.equal(snapshot(),before);
    }
    await traverse('back',helpIds[2]);await traverse('forward',helpIds[0]);
    assertLocal();assert.equal(node('guide-navigation-status').getAttribute('role'),'status');
  } finally {h.dom.window.close();}
});

test('opening a configurator deep link expands its local help without opening another page',async()=>{
  const h=await workbench('#configurator-results-help');
  try {
    const target=h.d.getElementById('configurator-results-help');
    assert(target.open);h.assertAt(target);h.assertLocal();
  } finally {h.dom.window.close();}
});
