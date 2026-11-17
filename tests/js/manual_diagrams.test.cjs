/* Check all clickable diagram targets, offline parity, and native-link semantics.
   DOM tests do not assert real-browser visual layout or keyboard rendering. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {JSDOM} = require('jsdom');
const root = path.resolve(__dirname, '../../docs/_build/html');
const cache = new Map();
function doc(file) {
  if (!cache.has(file)) cache.set(file, new JSDOM(fs.readFileSync(path.join(root, file), 'utf8')));
  return cache.get(file).window.document;
}
let count = 0, links = 0;
for (const name of ['user', 'developer']) {
  const overview = doc(`${name}_guide.html`);
  const routes = [...overview.querySelectorAll('[role="main"] .manual-overview-links a')];
  assert.equal(routes.length, 6);
  const files = [`${name}_guide.html`, `${name}_manual.html`, ...routes.map(a => a.getAttribute('href'))];
  const fullMaps = doc(`${name}_manual.html`).querySelectorAll('main .manual-map');
  assert.equal(fullMaps.length, 4, 'retain only four purposeful body diagrams per manual');
  let pageMaps = 0;
  for (const file of files) {
    const document = doc(file);
    const maps = [...document.querySelectorAll('.manual-map')];
    if (!file.includes('_guide_')) {
      const svg = document.querySelector('.manual-framework svg');
      assert(svg, `${file}: missing actual framework flowchart`);
      assert(svg.querySelector('title'));
      assert(svg.querySelector('desc'));
      const nodes = [...svg.querySelectorAll('.framework-node')];
      assert.equal(nodes.length, 12);
      const keys = new Set(nodes.map(n => n.dataset.node));
      assert.equal(keys.size, nodes.length);
      for (const edge of svg.querySelectorAll('.framework-edge')) {
        assert(keys.has(edge.dataset.from));
        assert(keys.has(edge.dataset.to));
        assert(edge.querySelector('path').getAttribute('marker-end'));
      }
      for (const a of nodes) {
        const [target, hash] = a.getAttribute('href').split('#');
        assert(doc(target || file).getElementById(hash), `${file}: broken framework link`);
        assert.equal(a.getAttribute('tabindex'), '0');
        if (file.endsWith('_manual.html')) assert(!target, 'offline flowchart must stay local');
      }
      if (name === 'user') {
        assert(svg.querySelector('[data-from="revise"][data-to="custom"]'));
        assert(svg.querySelector('.framework-decision'));
      } else {
        for (const id of ['feedback', 'plan', 'baseline', 'algorithm', 'math', 'fix', 'implement', 'tests', 'pr', 'review', 'revise', 'done']) assert(keys.has(id));
        assert(svg.querySelector('[data-from="tests"][data-to="implement"]'));
        assert(svg.querySelector('[data-from="review"][data-to="revise"]'));
        assert(svg.querySelector('[data-from="revise"][data-to="implement"]'));
        assert(!keys.has('trainer'), 'developer landing page must explain contribution, not runtime');
      }
    }
    if (file.includes('_guide_')) pageMaps += maps.length;
    if (file.endsWith('_manual.html')) {
      assert(document.querySelector('style').textContent.includes('.manual-map-tree'));
    } else {
      assert(document.querySelector('link[href*="manual_diagrams.css"]'));
    }
    for (const map of maps) {
      count++;
      assert(map.querySelector('figcaption')?.textContent.trim());
      assert(map.getAttribute('aria-label'));
      assert.equal(map.querySelectorAll('button,[onclick],[tabindex="-1"]').length, 0);
      const cards = [...map.querySelectorAll('a.manual-map-node')];
      assert(cards.length >= 3);
      for (const a of cards) {
        links++;
        assert(a.querySelector('.manual-map-icon'));
        assert(a.querySelector('strong')?.textContent.trim());
        assert(a.querySelector('small')?.textContent.trim());
        const href = a.getAttribute('href');
        assert(!/^(https?:|javascript:)/.test(href), `${file}: external diagram destination`);
        const [target, hash] = href.split('#');
        assert(fs.existsSync(path.join(root, target || file)), `${file}: missing ${href}`);
        if (hash) assert(doc(target || file).getElementById(hash), `${file}: invalid anchor ${href}`);
        if (file.endsWith('_manual.html')) assert(href.startsWith('#'), 'offline diagrams must use local anchors');
      }
    }
  }
  assert.equal(pageMaps, fullMaps.length, 'website/standalone diagram parity');
}
const css = fs.readFileSync(path.join(root, '_static/manual_diagrams.css'), 'utf8');
assert(css.includes('max-width: 650px'));
assert(css.includes(':focus-visible'));
assert(css.includes(':visited'));
assert(css.includes(':active'));
assert(css.includes('@media print'));
console.log(`MANUAL_DIAGRAMS_PASS: ${count} rendered diagrams / ${links} valid native links (includes offline copies)`);
for (const dom of cache.values()) dom.window.close();
