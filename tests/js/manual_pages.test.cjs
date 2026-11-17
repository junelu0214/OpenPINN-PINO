/* Role-based reading paths plus exhaustive coverage/link checks of built pages. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {JSDOM} = require('jsdom');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../../docs/_build/html');
const cache = new Map();
function document(file) {
  if (!cache.has(file)) cache.set(file, new JSDOM(fs.readFileSync(path.join(root, file), 'utf8')));
  return cache.get(file).window.document;
}
function verifyLink(file, href) {
  if (/^(https?:|mailto:)/.test(href)) return;
  const [target, anchor] = href.split('#');
  const targetFile = target || file;
  assert(fs.existsSync(path.join(root, targetFile)), `${file}: missing ${href}`);
  if (anchor) assert(document(targetFile).getElementById(decodeURIComponent(anchor)), `${file}: broken ${href}`);
}
function hasLink(file, target) {
  const links = [...document(file).querySelectorAll('[role="main"] a')];
  assert(links.some(a => a.getAttribute('href') === target), `${file}: no route to ${target}`);
}
for (const name of ['user', 'developer']) {
  const index = `${name}_guide.html`, overview = document(index);
  assert(!overview.querySelector('.openpinn-manual'), 'overview must not embed all content');
  const childLinks = [...overview.querySelectorAll('[role="main"] .manual-overview-links a')];
  assert.equal(childLinks.length, 6);
  const pages = childLinks.map(a => a.getAttribute('href'));
  assert.equal(new Set(pages).size, 6);
  assert(pages.every(p => p.startsWith(`${name}_guide_`) && !p.includes('#')));
  const full = document(`${name}_manual.html`);
  const expected = [...full.querySelectorAll('main > section')];
  const seen = new Set();
  for (const page of pages) {
    const d = document(page);
    const current = [...d.querySelectorAll('.wy-menu-vertical .toctree-l2.current > a')];
    assert.equal(current.length, 1);
    assert.equal(current[0].textContent, childLinks[pages.indexOf(page)].textContent);
    const sections = [...d.querySelectorAll('.openpinn-manual > section')];
    for (const section of sections) {
      assert(!seen.has(section.id), `duplicate section ${section.id}`);
      seen.add(section.id);
      assert.equal(section.textContent, full.getElementById(section.id).textContent, `changed/missing text ${section.id}`);
    }
    for (const a of d.querySelectorAll('[role="main"] a[href]')) verifyLink(page, a.getAttribute('href'));
    hasLink(page, index);
  }
  assert.deepEqual([...seen].sort(), expected.map(s => s.id).sort());
  const routes = JSON.parse(overview.querySelector('[data-manual-routes]').textContent);
  for (const [id, page] of Object.entries(routes)) verifyLink(index, page + '#' + id);
  let redirected = null;
  const listeners = {};
  const firstId = Object.keys(routes)[0];
  const window = {location:{hash:'#'+firstId, replace:target => { redirected = target; }}, addEventListener:(event, fn) => { listeners[event] = fn; }};
  const documentStub = {readyState:'complete', querySelector:selector => selector === '[data-manual-routes]' ? {textContent:JSON.stringify(routes)} : null};
  vm.runInNewContext(fs.readFileSync(path.join(root, '_static/manuals.js'), 'utf8'), {window, document:documentStub});
  assert.equal(redirected, routes[firstId]+'#'+firstId);
  redirected = null;
  window.location.hash = '#%E0%A4%A'; listeners.hashchange();
  assert.equal(redirected, null);
  assert.equal(full.querySelectorAll('aside nav > details').length, 6);
  console.log(`${name}: 6 real pages, sidebar current state, complete unique content, internal links and legacy routes PASS`);
}
// User: start -> build custom model -> import geometry -> return for PDE/BC -> run -> results.
hasLink('user_guide.html', 'user_guide_start.html');
hasLink('user_guide.html', 'user_guide_modeling.html');
hasLink('user_guide_modeling.html', 'user_guide_geometry.html');
hasLink('user_guide_geometry.html', 'user_guide_modeling.html');
hasLink('user_guide_modeling.html', 'user_guide_training.html');
hasLink('user_guide_modeling.html', 'user_guide_results.html');
// Developer: onboarding -> configurator -> backend contracts -> verification / lookup.
for (const target of ['start', 'modeling', 'runtime', 'delivery', 'reference']) {
  hasLink('developer_guide_configurator.html', `developer_guide_${target}.html`);
}
hasLink('developer_guide_runtime.html', 'developer_guide_configurator.html#extension-method');
assert(document('developer_guide_runtime.html').getElementById('registries').textContent.includes('方法配置与计算组件'));
assert(!document('developer_guide_start.html').getElementById('python-index'));
assert(document('developer_guide_reference.html').getElementById('python-index'));
console.log('USER_AND_DEVELOPER_READING_PATHS_PASS');
for (const dom of cache.values()) dom.window.close();
