const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {JSDOM} = require('jsdom');
const search = require('../../docs/_static/documentation_search.js');
const html = path.resolve(__dirname, '../../docs/_build/html');
function index() {
  const scope = {window:{}};
  vm.runInNewContext(fs.readFileSync(path.join(html, '_static/openpinn_search_index.js'), 'utf8'), scope);
  return scope.window.OpenPINNSearchIndex;
}
function page(query, protocol='file') {
  const url = protocol === 'file' ? 'file:///tmp/OpenPINN%20fresh/search.html' : 'http://localhost/search.html';
  const dom = new JSDOM('<div itemprop="articleBody"><form id="openpinn-search-form"><input name="q"></form><div id="search-results"></div></div>', {url: url+'?q='+encodeURIComponent(query)});
  dom.window.fetch = () => {throw new Error('Search must not fetch local pages');};
  return dom;
}
test('rendered index finds Chinese compounds, manual sections, and English API names',()=>{
  const data = index();
  for(const query of ['圆孔','固结','固结系数','配置生成器','Poisson','make_piecewise_poisson_problem','圆孔 半径']) {
    const found = search.search(data.records,query);
    assert(found.length>0,query);
    assert(found.every(r=>search.terms(query).every(t=>(r.title+' '+r.heading+' '+r.text).toLowerCase().includes(t))));
  }
  assert(search.search(data.records,'圆孔').some(r=>r.href.startsWith('user_guide_geometry.html#')));
  assert(search.search(data.records,'固结系数').some(r=>r.href.startsWith('physics_library.html#')));
  assert.equal(search.search(data.records,'不存在的唯一关键词987654321').length,0);
  assert.equal(search.search(data.records,'   ').length,0);
});
test('offline and HTTP render summaries without fetch; pagination and links preserve query/anchor',()=>{
  for(const protocol of ['file','http']) {
    const dom=page('Poisson',protocol), d=dom.window.document;
    try {
      search.render(d,dom.window,index());
      const results=d.querySelector('#search-results');
      assert.equal(results.dataset.searchState,'complete');
      const total=Number(results.dataset.resultCount);
      assert(total>20);assert.equal(d.querySelectorAll('.search li').length,20);
      d.querySelector('.search-more').click();assert.equal(d.querySelectorAll('.search li').length,Math.min(40,total));
      while(!d.querySelector('.search-more').hidden)d.querySelector('.search-more').click();
      assert.equal(d.querySelectorAll('.search li').length,total);
      assert(d.querySelector('.context').textContent.length>0);
      assert(d.querySelector('mark.openpinn-search-match'));
      const link=new URL(d.querySelector('.search-title').href);
      assert.equal(link.searchParams.get('highlight'),'Poisson');
      assert(link.pathname.endsWith('.html'));
    } finally {dom.window.close();}
  }
});
test('highlight only static HTML text, never diagrams, math, code, or configurator controls',()=>{
  const dom=new JSDOM('<div itemprop="articleBody"><p>圆孔与圆孔</p><svg><title>圆孔</title><desc>圆孔</desc><text>圆孔</text></svg><math><mi>圆孔</mi></math><pre><code>圆孔</code></pre><div data-openpinn-configurator><label>圆孔</label><textarea>圆孔</textarea></div></div>');
  const d=dom.window.document;
  try{
    const svg=d.querySelector('svg').outerHTML, editor=d.querySelector('[data-openpinn-configurator]').outerHTML;
    search.highlightDocument(d,'圆孔');
    assert.equal(d.querySelectorAll('mark').length,2);
    assert.equal(d.querySelector('svg').outerHTML,svg);assert.equal(d.querySelector('[data-openpinn-configurator]').outerHTML,editor);
    search.clearHighlights(d);assert.equal(d.querySelectorAll('mark').length,0);assert.equal(d.querySelector('p').textContent,'圆孔与圆孔');
  }finally{dom.window.close();}
});
test('query and indexed markup remain text; empty, absent-index and oversized queries have clear states',()=>{
  const dom=page('<img'), d=dom.window.document;
  try{
    search.render(d,dom.window,{version:1,records:[{href:'safe.html#section',title:'<img src=x onerror=alert(1)>',heading:'<img',text:'<img src=x onerror=alert(1)>'}]});
    assert.equal(d.querySelectorAll('img,script').length,0);assert(d.querySelector('.context').textContent.includes('<img'));
    search.render(d,dom.window,undefined);assert.equal(d.querySelector('#search-results').dataset.searchState,'error');
    dom.reconfigure({url:'file:///tmp/search.html?q='});search.render(d,dom.window,index());assert.equal(d.querySelector('#search-results').dataset.searchState,'empty');
    dom.reconfigure({url:'file:///tmp/search.html?q='+'x'.repeat(161)});search.render(d,dom.window,index());assert.equal(d.querySelector('#search-results').dataset.searchState,'invalid');
  }finally{dom.window.close();}
});
test('search page uses bundled index and no upstream fetch/highlight runtime',()=>{
  const dom=new JSDOM(fs.readFileSync(path.join(html,'search.html'),'utf8'));
  try{
    const sources=[...dom.window.document.querySelectorAll('script[src]')].map(s=>s.getAttribute('src'));
    assert(sources.some(s=>s.includes('openpinn_search_index.js')));
    assert(sources.some(s=>s.includes('documentation_search.js')));
    for(const name of ['searchtools.js','sphinx_highlight.js','language_data.js'])assert(!sources.some(s=>s.includes(name)),name);
    assert(dom.window.document.querySelector('#openpinn-search-form'));
  }finally{dom.window.close();}
});
test('deferred runtime waits for the later index script while DOM is interactive',()=>{
  const dom=new JSDOM('<form id="openpinn-search-form"><input name="q"></form><div id="search-results"></div>',{url:'file:///tmp/search.html?q=圆孔',runScripts:'outside-only'});
  try{
    Object.defineProperty(dom.window.document,'readyState',{value:'interactive',configurable:true});
    dom.window.eval(fs.readFileSync(path.resolve(__dirname,'../../docs/_static/documentation_search.js'),'utf8'));
    assert.equal(dom.window.document.querySelector('#search-results').children.length,0);
    dom.window.OpenPINNSearchIndex=index();
    dom.window.document.dispatchEvent(new dom.window.Event('DOMContentLoaded'));
    assert.equal(dom.window.document.querySelector('#search-results').dataset.searchState,'complete');
    assert(dom.window.document.querySelector('.context'));
  }finally{dom.window.close();}
});
