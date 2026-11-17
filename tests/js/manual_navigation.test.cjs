/* Run after building manuals and Sphinx; requires jsdom (optionally via NODE_PATH). */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {JSDOM} = require('jsdom');
const root = path.resolve(__dirname, '../..');

async function check(name, standalone) {
  const file = path.join(root, 'docs/_build/html', name + '.html');
  const dom = new JSDOM(fs.readFileSync(file, 'utf8'), {
    url: pathToFileURL(file).href, runScripts: 'outside-only',
  });
  const {window} = dom;
  const document = window.document;
  await new Promise(resolve => window.addEventListener('DOMContentLoaded', resolve, {once: true}));
  window.HTMLElement.prototype.scrollIntoView = function () {};
  if (standalone) document.querySelectorAll('script').forEach(s => window.eval(s.textContent));
  else window.eval(fs.readFileSync(path.join(root, 'docs/_static/manuals.js'), 'utf8'));
  const nav = document.querySelector(standalone ? 'aside nav' : '.manual-contents');
  const links = [...nav.querySelectorAll('a')];
  assert(links.length >= (standalone ? 20 : 2));
  if (standalone) assert.equal(nav.querySelectorAll(':scope > details').length, 6);
  else assert.equal(document.querySelector('.manual-local-nav').open, false);
  const current = () => [...nav.querySelectorAll('[aria-current="location"]')];
  const go = hash => {
    window.location.hash = hash;
    window.dispatchEvent(new window.HashChangeEvent('hashchange'));
  };
  assert.equal(current().length, 0);
  go(links[0].hash);
  assert.deepEqual(current(), [links[0]]);
  go(links[1].hash);
  assert.deepEqual(current(), [links[1]]);
  assert.equal(links[0].hasAttribute('aria-current'), false);
  // A deep link into a subsection selects its parent chapter, not two entries.
  const sub = document.querySelector('h3[id]') || document.getElementById(links[0].hash.slice(1));
  go('#' + sub.id);
  assert.deepEqual(current(), [links.find(a => a.hash === '#' + sub.closest('section').id)]);
  go('#missing-section'); assert.equal(current().length, 0);
  go('#%E0%A4%A'); assert.equal(current().length, 0); // malformed URL escape
  go(''); assert.equal(current().length, 0);
  // History navigation dispatches hashchange and restores the sole active chapter.
  go(links[0].hash); go(links[1].hash);
  await new Promise(resolve => setTimeout(resolve, 10));
  const back = new Promise(resolve => window.addEventListener('popstate', resolve, {once: true}));
  window.history.back(); await back;
  await new Promise(resolve => setTimeout(resolve, 10));
  assert.deepEqual(current(), [links[0]]);
  const ids = [...document.querySelectorAll('[id]')].map(n => n.id);
  assert.equal(ids.length, new Set(ids).size);
  for (const a of document.querySelectorAll('main a[href^="#"], .openpinn-manual a[href^="#"]')) {
    assert(document.getElementById(a.hash.slice(1)), 'broken anchor: ' + a.hash);
  }
  const search = document.querySelector(standalone ? '#chapter-search' : '[data-manual-search]');
  search.value = 'nonexistent_chapter_123'; search.dispatchEvent(new window.Event('input'));
  assert(links.every(a => standalone ? a.hidden : a.parentElement.hidden));
  search.value = ''; search.dispatchEvent(new window.Event('input'));
  assert(links.every(a => standalone ? !a.hidden : !a.parentElement.hidden));
  const css = standalone ? document.querySelector('style').textContent : fs.readFileSync(path.join(root, 'docs/_static/manuals.css'), 'utf8');
  assert.match(css, /a:visited\s*\{\s*color:\s*#2980b9/);
  assert.match(css, /aria-current="location"/);
  console.log(name + ': chapter switching, history, deep links, search, anchors and style rules PASS');
  dom.window.close();
}
(async () => {
  const files = fs.readdirSync(path.join(root, 'docs/_build/html'));
  for (const name of files.filter(name => /^(user|developer)_guide_.+\.html$/.test(name))) {
    // The operator protocol is a standalone Sphinx reference page, not one
    // of the split manual chapters that receives .manual-contents navigation.
    if (name === 'developer_guide_operators.html') {
      const source = fs.readFileSync(path.join(root, 'docs/_build/html', name), 'utf8');
      assert.match(source, /Operator evaluation and model-native derivatives/);
      assert.doesNotMatch(source, /class="manual-contents"/);
      continue;
    }
    await check(name.slice(0, -5), false);
  }
  for (const name of ['user_manual', 'developer_manual']) await check(name, true);
})().catch(error => { console.error(error); process.exitCode = 1; });
