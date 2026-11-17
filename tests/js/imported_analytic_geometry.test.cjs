/* Requires jsdom; uses real file-input events and the current configurator code. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {JSDOM} = require('jsdom');
const root = path.resolve(__dirname, '../..');
const pause = () => new Promise(resolve => setTimeout(resolve, 10));
function drawing(outer, holes = [[10, 20, 4], [30, 20, 4]]) {
  const values = [0, 'SECTION', 2, 'ENTITIES', 0, 'LWPOLYLINE', 70, 1,
    ...outer.flatMap(([x, y]) => [10, x, 20, y]),
    ...holes.flatMap(([x, y, r]) => [0, 'CIRCLE', 10, x, 20, y, 40, r]),
    0, 'ENDSEC', 0, 'EOF'];
  return values.join('\n') + '\n';
}
async function main() {
  const dom = new JSDOM(fs.readFileSync(path.join(root, 'docs/_build/html/method_configurator.html'), 'utf8'), {runScripts: 'outside-only'});
  const {window} = dom, d = window.document;
  await new Promise(resolve => window.addEventListener('DOMContentLoaded', resolve, {once:true}));
  for (const file of ['polygon_image_trace.js', 'dxf_geometry_import.js', 'physics_editor.js', 'method_configurator.js']) {
    window.eval(fs.readFileSync(path.join(root, 'docs/_static', file), 'utf8'));
  }
  d.dispatchEvent(new window.Event('DOMContentLoaded'));
  const node = role => d.querySelector('[data-role="' + role + '"]');
  const change = (element, value) => { element.value = String(value); element.dispatchEvent(new window.Event('change', {bubbles:true})); };
  const set = (role, value) => change(node(role), value);
  const code = () => node('code').textContent;
  const results = [];
  set('problem', 'custom_equation_template'); set('custom-dimension', '2'); set('custom-geometry-type', 'dxf'); set('steps', 1);
  async function load(text, name = 'circles.dxf') {
    const input = node('custom-dxf-file');
    Object.defineProperty(input, 'files', {configurable:true, value:[new window.File([text], name)]});
    input.dispatchEvent(new window.Event('change', {bubbles:true}));
    for (let i=0; i<100; i++) {
      await pause();
      if (node('custom-polygon-status').textContent.includes(name)) break;
    }
    assert.match(node('custom-polygon-status').textContent, /已载入/);
    assert.match(code(), /PROBLEM = op.PDEProblem/);
  }
  const rectangle = [[0, 0], [40, 0], [40, 40], [0, 40]];
  await load(fs.readFileSync(path.join(root, 'examples/geometries/analytic_two_holes.dxf'), 'utf8'), 'analytic_two_holes.dxf');
  assert.match(code(), /holes=CIRCULAR_HOLES, stratified_boundary=True/);
  assert(!code().includes('SPACE.rings'));
  assert(!code().includes('make_circle_hole'));
  assert.equal(d.querySelectorAll('[data-polygon-boundary-edge]').length, 6);
  assert.match(node('custom-polygon-status').textContent, /训练几何：解析/);
  // Distinct BCs must survive edits and keep hole 1/2 and outer edge numbering.
  let rows = [...d.querySelectorAll('[data-polygon-boundary-edge]')];
  change(rows[0].querySelector('select'), 'neumann');
  change(rows[4].querySelector('select'), 'robin');
  change(rows[4].querySelector('[data-boundary-key="value"]'), 3);
  change(rows[4].querySelector('[data-boundary-key="alpha"]'), 2);
  change(rows[4].querySelector('[data-boundary-key="beta"]'), 0.5);
  results.push({name:'static', code:code()});
  set('custom-polygon-ring', 1);
  set('custom-dxf-circle-diameter', 10);
  set('custom-dxf-circle-x', 11);
  set('custom-dxf-circle-y', 19);
  assert.match(code(), /HOLE_1_RADIUS = 5\b/);
  assert.match(code(), /HOLE_2_RADIUS = 4\b/);
  assert.match(code(), /HOLE_1_CENTER_X = 11\b/);
  assert.match(code(), /HOLE_1_CENTER_Y = 19\b/);
  assert.match(code(), /POLYGON_R2_BOUNDARY_VALUE = 3\b/);
  assert.match(code(), /POLYGON_R2_ROBIN_ALPHA = 2\b/);
  results.push({name:'edited', code:code()});
  const before = code();
  set('custom-dxf-circle-radius', 30);
  assert.match(node('custom-polygon-status').textContent, /失败/);
  assert.equal(code(), before); // invalid edit rejected, previous valid geometry remains
  assert.equal(node('custom-dxf-circle-radius').value, '5');
  set('custom-kind', 'dynamic');
  assert.match(code(), /SpaceTimeDomain/);
  results.push({name:'dynamic', code:code()});
  set('custom-kind', 'static');
  // Editing one outer vertex switches representation without losing BCs.
  set('custom-polygon-ring', 0);
  const originalX = d.querySelector('[aria-label="顶点 3 x"]').value;
  change(d.querySelector('[aria-label="顶点 3 x"]'), Number(originalX) - 1);
  assert.match(code(), /SPACE = op.Polygon/);
  assert.match(code(), /POLYGON_R2_ROBIN_ALPHA = 2\b/);
  change(d.querySelector('[aria-label="顶点 3 x"]'), originalX);
  assert.match(code(), /stratified_boundary=True/);
  assert.match(code(), /POLYGON_R2_BOUNDARY_VALUE = 3\b/);
  // Corner order may be reversed/rotated; do not assign BCs by assumed order.
  await load(drawing([[40, 40], [40, 0], [0, 0], [0, 40]]), 'reversed.dxf');
  assert.match(code(), /stratified_boundary=True/);
  results.push({name:'reversed', code:code()});
  // A trapezoid is not a rectangle; retain Polygon and quantify circle chord error.
  await load(drawing([[0, 0], [40, 0], [39, 40], [0, 40]]), 'trapezoid.dxf');
  assert.match(code(), /SPACE = op.Polygon/);
  assert.match(node('custom-polygon-status').textContent, /最大弦高/);
  results.push({name:'fallback', code:code()});
  // A non-circular hole must never silently become an analytic disk.
  const squareHole = [0, 'LWPOLYLINE', 70, 1, 10, 8, 20, 18, 10, 12, 20, 18, 10, 12, 20, 22, 10, 8, 20, 22].join('\n');
  await load(drawing(rectangle, []).replace('0\nENDSEC', squareHole + '\n0\nENDSEC'), 'square_hole.dxf');
  assert.match(code(), /SPACE = op.Polygon/);
  assert(!code().includes('CIRCULAR_HOLES'));
  const external = process.argv.find(arg => arg.endsWith('.dxf'));
  if (external) {
    await load(fs.readFileSync(external, 'utf8'), path.basename(external));
    assert.match(code(), /stratified_boundary=True/);
    assert.equal(d.querySelectorAll('[data-polygon-boundary-edge]').length, 6);
    results.push({name:'user_dxf', code:code()});
  }
  dom.window.close();
  if (process.argv.includes('--json')) process.stdout.write(JSON.stringify(results));
  else console.log('ANALYTIC_IMPORT_DOM_OK: ' + results.map(r => r.name).join(', '));
}
main().catch(error => {console.error(error); process.exitCode=1;});
