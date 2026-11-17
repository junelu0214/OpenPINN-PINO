/* Role/task audit: newcomer -> task recipe -> evidence -> review.
   Actual target validity and full/website parity are checked in manual_pages. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {JSDOM} = require('jsdom');
const root = path.resolve(__dirname, '../../docs/_build/html');
const doms = [];
function page(name) {
  const dom = new JSDOM(fs.readFileSync(path.join(root, `developer_guide${name ? '_'+name : ''}.html`), 'utf8'));
  doms.push(dom);
  return dom.window.document;
}
const start = page('start');
assert(start.getElementById('contribution-plan'));
assert(!start.getElementById('architecture'), 'runtime architecture must not dominate onboarding');
const runtime = page('runtime');
assert.equal(runtime.querySelector('.openpinn-manual > section').id, 'method-recipe');
assert(runtime.getElementById('method-done'));
assert(page('configurator').getElementById('maintenance-recipe'));
assert(page('delivery').getElementById('pr-template'));
assert(page('delivery').getElementById('github-readiness'));
assert(page('reference').getElementById('architecture'));
const landing = page('');
const expected = {
  feedback:'start.html#contribution-plan', algorithm:'runtime.html#method-recipe',
  math:'modeling.html#extension-problem', fix:'configurator.html#maintenance-recipe',
  tests:'delivery.html#testing', pr:'delivery.html#contribution-pr',
};
for (const [node, target] of Object.entries(expected)) {
  assert.equal(landing.querySelector(`[data-node="${node}"]`).getAttribute('href'), 'developer_guide_'+target);
}
assert(landing.querySelector('[role="main"]').textContent.includes('不代表'));
console.log('DEVELOPER_CONTRIBUTION_PATHS_PASS');
for (const dom of doms) dom.window.close();
