/* Browsers hide visited history from scripts. Model pseudo-states with equally
   specific classes to test the real CSS cascade; this is not visual browser QA. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {JSDOM} = require('jsdom');
const root = path.resolve(__dirname, '../../docs/_build/html');
function simulatedStates(css) {
  return css.replace(/:visited\b/g, '.test-visited').replace(/:active\b/g, '.test-active');
}
for (const file of ['user_guide.html', 'developer_guide.html', 'user_guide_modeling.html', 'developer_guide_runtime.html', 'user_manual.html', 'developer_manual.html']) {
  const dom = new JSDOM(fs.readFileSync(path.join(root, file), 'utf8'));
  const d = dom.window.document;
  for (const link of d.querySelectorAll('link[rel="stylesheet"]')) {
    const source = path.join(root, link.getAttribute('href').split('?')[0]);
    const style = d.createElement('style');
    style.textContent = simulatedStates(fs.readFileSync(source, 'utf8'));
    link.replaceWith(style);
  }
  for (const style of d.querySelectorAll('style')) style.textContent = simulatedStates(style.textContent);
  const standalone = file.endsWith('_manual.html');
  const links = [...d.querySelectorAll(standalone ? 'main a[href], aside a[href]' : '.wy-nav-content a[href]:not(.btn):not(.headerlink)')];
  assert(links.length > 0);
  for (const link of links) {
    const color = () => dom.window.getComputedStyle(link).color;
    assert.equal(color(), 'rgb(41, 128, 185)', `${file}: normal ${link.textContent}`);
    link.classList.add('test-visited');
    assert.equal(color(), 'rgb(41, 128, 185)', `${file}: visited ${link.textContent}`);
    link.classList.add('test-active');
    assert.equal(color(), 'rgb(155, 89, 182)', `${file}: pressed ${link.textContent}`);
    link.classList.remove('test-active');
    link.focus();
    assert.equal(color(), 'rgb(41, 128, 185)', `${file}: released/focused ${link.textContent}`);
  }
  console.log(file + ': normal/visited blue, active purple, release/focus blue PASS');
  dom.window.close();
}
