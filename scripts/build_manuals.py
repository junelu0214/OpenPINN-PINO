"""Build two offline HTML manuals and auditable source/function appendices.

Run from the repository root: python scripts/build_manuals.py
The prose lives in docs/manuals/_sources; generated files must not be edited.
Only the standard library and the local OpenPINN package are required.
"""
from __future__ import annotations

import ast
import hashlib
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "docs/_ext"))
from manual_sections import MANUAL_PAGES, manual_sections, section_routes
from manual_diagrams import enrich
from manual_frameworks import framework
from openpinn.method_options import METHOD_PARAMETER_SCHEMAS, METHOD_SLOTS
from openpinn.method_registry import list_methods, list_method_variants

DATE = "2026-10-16"
DEVELOPER_REVISION = "2026-10-16"
SOURCES = ROOT / "docs/manuals/_sources"
OUTPUT = ROOT / "docs/manuals/public"

CSS = """
:root{--ink:#203044;--muted:#52657a;--accent:#17688a;--line:#d6e0e8;--paper:#fff;--bg:#f1f5f8}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:1rem}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.85 system-ui,-apple-system,'PingFang SC','Microsoft YaHei',sans-serif}
a{color:var(--accent);text-underline-offset:3px}header{background:#15344b;color:#fff;padding:42px max(5vw,24px)}header h1{font-size:clamp(28px,4vw,44px);line-height:1.3;margin:0 0 14px}header p{max-width:960px;margin:8px 0;color:#d7e7f1}.badge{font-size:13px;letter-spacing:.07em;color:#94cde4}.layout{display:grid;grid-template-columns:270px minmax(0,1fr);max-width:1560px;margin:auto}aside{position:sticky;top:0;height:100vh;overflow:auto;padding:24px 18px;border-right:1px solid var(--line)}nav a{display:block;padding:6px 8px;font-size:14px;text-decoration:none;border-radius:5px}nav a:hover,nav a:focus{background:#dceaf2}main{min-width:0;padding:30px 44px 70px;background:var(--paper)}section{margin:0 0 42px;scroll-margin-top:20px}h2{font-size:25px;border-bottom:2px solid var(--line);padding:10px 0;margin:0 0 18px}h3{font-size:20px;margin:27px 0 10px}h4{font-size:17px;margin:18px 0 8px}p{margin:10px 0}li{margin:5px 0}table{width:100%;border-collapse:collapse;margin:18px 0;display:block;overflow:auto;font-size:14px}td,th{border:1px solid var(--line);padding:9px 12px;vertical-align:top;min-width:95px}th{background:#eaf1f6;text-align:left}tbody tr:nth-child(even){background:#f8fafc}code{font: .9em/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere;background:#eef3f6;padding:1px 4px;border-radius:3px}pre{padding:17px 20px;background:#142b3b;color:#e3eef5;overflow:auto;white-space:pre;font:13px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace;border-radius:7px}pre code{padding:0;color:inherit;background:none;overflow-wrap:normal}.note,.warn,.check{padding:14px 18px;margin:18px 0;border-left:4px solid #2e819e;background:#edf6fa}.warn{border-color:#b77516;background:#fff7e8}.check{border-color:#398464;background:#edf7f1}.small{font-size:13px;color:var(--muted)}details{border:1px solid var(--line);border-radius:6px;margin:12px 0;padding:10px 14px}summary{cursor:pointer;font-weight:650;overflow-wrap:anywhere}.symbol{border-top:1px solid var(--line);padding-top:12px;margin-top:14px}.symbol h4{overflow-wrap:anywhere}.toolbar{display:flex;gap:8px;margin:12px 0;flex-wrap:wrap}input[type=search]{width:100%;padding:10px;border:1px solid #9fb3c2;border-radius:5px;font:inherit}button{padding:8px 10px;border:1px solid #9bb0bf;border-radius:5px;background:white;color:var(--ink);cursor:pointer}button:hover{background:#deedf5}.flow{white-space:pre-wrap;background:#edf4f8;color:#163d53;font-size:14px;border-left:4px solid var(--accent)}.toc-tip{font-size:12px;color:var(--muted)}footer{padding:24px;text-align:center;font-size:13px;color:var(--muted)}[hidden]{display:none!important}.meta{color:var(--muted);font-size:13px}.index-search{max-width:560px}.code-title{font-size:13px;color:var(--muted)}
@media(max-width:950px){.layout{display:block}aside{position:static;height:auto;border-bottom:1px solid var(--line)}nav{columns:2}main{padding:24px 18px}header{padding:28px 20px}}@media(max-width:560px){nav{columns:1}td,th{padding:7px}body{font-size:15px}}
@media print{body{background:white;font-size:10pt}header{background:white;color:black;padding:0 0 18px}header p,.badge{color:#333}.layout{display:block}aside,.toolbar,.index-search,.toc-tip{display:none}main{padding:0}h2,h3,h4,summary{break-after:avoid}pre{background:#f3f3f3;color:#000;white-space:pre-wrap;overflow-wrap:anywhere}table{display:table;font-size:9pt}tr{break-inside:avoid}details{border:0;padding:0}a{color:#000}section{margin-bottom:22px}header{break-after:page}h2{break-before:auto}}
"""
CSS += """
a,a:visited{color:#2980b9}
body a:active{color:#9b59b6!important}
a:focus-visible{outline:2px solid #2980b9;outline-offset:2px}
nav a,nav a:visited{color:#2980b9}
nav a[aria-current="location"],nav a[aria-current="location"]:visited{color:#124d73;background:#dceefb;box-shadow:inset 3px 0 #2980b9;font-weight:700}
nav a:focus-visible{outline:2px solid #2980b9;outline-offset:2px}
"""
CSS += (ROOT / 'docs/_static/manual_diagrams.css').read_text(encoding='utf-8')
JS = """
const nav=document.querySelector('nav');
document.querySelector('#chapter-search').addEventListener('input',e=>{const v=e.target.value.trim().toLowerCase();document.querySelectorAll('nav a').forEach(a=>{const s=document.getElementById(a.hash.slice(1));a.hidden=!!v&&!s.textContent.toLowerCase().includes(v)});nav.querySelectorAll('details').forEach(d=>{d.hidden=[...d.querySelectorAll('a')].every(a=>a.hidden);d.open=!!v&&!d.hidden})});
document.querySelector('#expand').addEventListener('click',()=>document.querySelectorAll('details').forEach(d=>d.open=true));
document.querySelector('#collapse').addEventListener('click',()=>document.querySelectorAll('details').forEach(d=>d.open=false));
document.querySelector('#print').addEventListener('click',()=>{document.querySelectorAll('details').forEach(d=>d.open=true);window.print()});
document.querySelectorAll('[data-filter-target]').forEach(input=>input.addEventListener('input',()=>{const needle=input.value.trim().toLowerCase();const container=document.getElementById(input.dataset.filterTarget);container.querySelectorAll(':scope > details').forEach(d=>{d.hidden=!!needle&&!d.textContent.toLowerCase().includes(needle);if(needle&&!d.hidden)d.open=true})}));
function revealHash(){let id;try{id=decodeURIComponent(location.hash.slice(1))}catch(error){id=''}const el=document.getElementById(id);const section=el&&el.closest('section');nav.querySelectorAll('a').forEach(a=>{if(section&&a.hash==='#'+section.id){a.setAttribute('aria-current','location');a.closest('details').open=true}else a.removeAttribute('aria-current')});if(!el)return;let p=el.parentElement;while(p){if(p.tagName==='DETAILS')p.open=true;p.hidden=false;p=p.parentElement}el.scrollIntoView()}
window.addEventListener('hashchange',revealHash);if(location.hash)revealHash();
"""

def esc(value):
    return html.escape(str(value), quote=True)

def table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'

def code(value):
    return '<code>'+esc(value)+'</code>'

def parameters():
    parts = ['<section id="parameters"><h2>附录 A · 方法选项完整操作表</h2><p>以下名称、默认值、类型和范围直接读取当前版本参数目录。选定方法后在网页填写，或作为 <code>op.method(家族, 变体, 参数名=值)</code> 的关键字传入。默认值不是精度保证；“未提供”表示需用户填写或可省略，见说明。</p>']
    for method in list_methods():
        for variant in list_method_variants(method.name):
            schema = METHOD_PARAMETER_SCHEMAS[(method.name, variant.name)]
            parts.append('<h3>'+esc(method.title+' / '+variant.title)+'</h3><p>'+code(method.name+' / '+variant.name)+'</p>')
            if not schema:
                parts.append('<p>此变体没有专用可编辑超参数。仍可调整通用训练设置。</p>')
                continue
            rows=[]
            for name, item in schema.items():
                bounds=[]
                if item['minimum'] is not None: bounds.append('≥ '+str(item['minimum']))
                if item['maximum'] is not None: bounds.append('≤ '+str(item['maximum']))
                if item['choices']: bounds.append('可选 '+str(item['choices']))
                if item['required']: bounds.append('必填')
                default='未提供' if item['default'] is None else repr(item['default'])
                rows.append([code(name)+'<br>'+esc(item['label']),esc(item['kind'])+'<br>'+code(default),esc('；'.join(bounds) or '无额外数值范围；仍须满足类型与运行校验'),esc(item['description'])+'<br>'+esc(item['effect'])])
            parts.append(table(['参数','类型 / 默认值','限制','用途及调整影响'],rows))
    parts.append('</section>')
    return ''.join(parts)

class DirectCalls(ast.NodeVisitor):
    """Collect lexical calls, excluding separately indexed nested scopes."""
    def __init__(self): self.calls=set()
    def visit_Call(self,node):
        self.calls.add(ast.unparse(node.func))
        self.generic_visit(node)
    def visit_FunctionDef(self,node): pass
    visit_AsyncFunctionDef=visit_FunctionDef
    def visit_ClassDef(self,node): pass
    def visit_Lambda(self,node): pass

def python_inventory(paths):
    modules=[]
    for path in paths:
        source=path.read_text(encoding='utf-8'); tree=ast.parse(source)
        records=[]
        def visit(node, scope=''):
            for child in ast.iter_child_nodes(node):
                if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    name=scope+child.name
                    visitor=DirectCalls()
                    if not isinstance(child,ast.ClassDef):
                        for statement in child.body: visitor.visit(statement)
                    signature=('class '+child.name+'('+', '.join(ast.unparse(b) for b in child.bases)+')') if isinstance(child,ast.ClassDef) else child.name+'('+ast.unparse(child.args)+')'+(' → '+ast.unparse(child.returns) if child.returns else '')
                    records.append(dict(name=name,line=child.lineno,end=child.end_lineno,signature=signature,doc=ast.get_docstring(child) or '',calls=sorted(visitor.calls),kind='class' if isinstance(child,ast.ClassDef) else 'function'))
                    visit(child,name+'.')
                else: visit(child,scope)
        visit(tree)
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)): imports.append(ast.unparse(node))
        modules.append(dict(path=str(path.relative_to(ROOT)),doc=ast.get_docstring(tree) or '',imports=sorted(set(imports)),records=records,sha256=hashlib.sha256(source.encode()).hexdigest()))
    return modules

def function_appendix(modules):
    parts=['<section id="functions"><h2>附录 B · Python 类与函数逐项索引</h2><p>范围：<code>src/openpinn</code>、文档扩展、项目顶层示例、用户模板和维护脚本。逐项列出类、方法、嵌套函数、原始签名、源码说明和函数体内调用表达式。没有 docstring 的函数明确标注，不用名称猜测其行为。</p><div class="warn">这是 AST 静态索引，不是完整运行时调用图。<code>self.xxx</code>、注册工厂、回调与多态对象不作未经验证的目标推断；嵌套函数另列，其调用不计入外层函数。签名默认值的调用、装饰器和匿名 lambda 不属于此处的函数体调用列表。主链路与对象绑定关系请读正文。源码行号只对应本快照。</div><input class="index-search" type="search" placeholder="按文件、函数、参数或调用表达式筛选" aria-label="筛选 Python 函数" data-filter-target="python-index"><div id="python-index">']
    count=0
    for index,module in enumerate(modules):
        parts.append('<details><summary>'+esc(module['path'])+' · '+str(len(module['records']))+' 个定义</summary><p>'+esc(module['doc'])+'</p><details><summary>导入依赖（含条件 / 局部导入）</summary><pre>'+esc('\n'.join(module['imports']))+'</pre></details>')
        for record in module['records']:
            count+=1
            parts.append('<article class="symbol" id="py-'+str(count)+'"><h4>'+esc(record['name'])+'</h4><p class="meta">'+esc(module['path'])+':'+str(record['line'])+'–'+str(record['end'])+' · '+record['kind']+'</p><pre>'+esc(record['signature'])+'</pre>')
            parts.append('<p>'+esc(record['doc'] or '无独立源码 docstring；结合所属模块正文和调用表达式阅读实现。').replace('\n','<br>')+'</p>')
            if record['kind']=='function': parts.append('<p><strong>函数体调用表达式：</strong> '+(' · '.join(code(c) for c in record['calls']) or '无显式调用；可能仅赋值、返回、运算或协议占位。')+'</p>')
            parts.append('</article>')
        parts.append('</details>')
    parts.append('</div><p class="small">合计 '+str(len(modules))+' 个 Python 文件、'+str(count)+' 个类/函数定义；包含嵌套定义，不等于公开 API 数。</p></section>')
    return ''.join(parts),count

def js_appendix(paths):
    parts=['<section id="javascript-index"><h2>附录 C · 前端命名函数定位表</h2><p>按 <code>function 名称(参数)</code> 声明提取的位置索引；不是 JavaScript AST 或完整调用图，不包含匿名事件回调、箭头函数和对象方法。主要事件与生成代码关系已在正文逐步说明。</p>']
    for path in paths:
        source=path.read_text(encoding='utf-8')
        rows=[]
        for match in re.finditer(r'\bfunction\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)',source):
            rows.append([code(match.group(1)+'('+match.group(2)+')'),str(source.count('\n',0,match.start())+1)])
        parts.append('<details><summary>'+esc(str(path.relative_to(ROOT)))+' · '+str(len(rows))+' 个命名声明</summary>'+table(['函数声明','行号'],rows)+'</details>')
    parts.append('</section>')
    return ''.join(parts)

def provenance(paths):
    rows=[]
    for path in paths:
        rows.append([code(str(path.relative_to(ROOT))),code(hashlib.sha256(path.read_bytes()).hexdigest())])
    return '<section id="snapshot"><h2>附录 D · 源码快照与维护记录</h2><p>编写基准日：'+DATE+'。工作目录名称为 v0.6-dev，但 <code>pyproject.toml</code>、包 <code>__version__</code> 与 Sphinx release 仍为 <code>0.2.0</code>。本说明书不把目录名当作已发布版本号。下表用于确认后续代码是否已变化；不是许可证或精度认证。</p><details><summary>展开受核对文件 SHA-256 清单</summary>'+table(['文件','SHA-256'],rows)+'</details></section>'

def render(name,title,subtitle,body):
    manual = 'user' if name == 'user_manual' else 'developer'
    if manual == 'developer':
        subtitle = '面向继续建设 OpenPINN 的贡献者：从反馈与提案，到算法扩展、问题扩展、细节修复、验证和 PR 审阅；架构与函数索引按需查阅。'
        # Contribution tasks first; preserve the technical reference verbatim.
        source_sections = manual_sections('<main>'+body+'</main>')
        body = ''.join(source_sections[anchor][1] for _, _, _, anchors in MANUAL_PAGES[manual] for anchor in anchors)
    body = enrich(manual, body)
    sections = manual_sections('<main>'+body+'</main>')
    section_routes(manual, sections)
    navigation = ''.join('<details><summary>'+esc(group_title)+'</summary>'+''.join('<a href="#'+anchor+'">'+sections[anchor][0]+'</a>' for anchor in anchors)+'</details>' for _,group_title,_,anchors in MANUAL_PAGES[manual])
    other='developer_manual.html' if name=='user_manual' else 'user_manual.html'
    other_title='开发者说明书' if name=='user_manual' else '使用者说明书'
    return '<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="'+esc(subtitle)+'"><title>OpenPINN · '+title+'</title><style>'+CSS+'</style></head><body><header><div class="badge">OPENPINN / '+DATE+' / 源码核对版</div><h1>'+title+'</h1><p>'+subtitle+'</p><p>适用：当前本地 v0.6-dev 工作目录（包版本 0.2.0） · 离线 HTML · 可检索与打印</p></header><div class="layout"><aside><a href="'+other+'">切换至'+other_title+'</a><div class="toolbar"><button id="expand">展开附录</button><button id="collapse">折叠附录</button><button id="print">打印 / 保存 PDF</button></div><label for="chapter-search">检索章节</label><input id="chapter-search" type="search" placeholder="输入关键词"><p class="toc-tip">检索只筛选目录，不隐藏正文。查找正文可先展开附录，再用浏览器查找。无脚本时仍可完整阅读。</p><nav aria-label="章节目录">'+navigation+'</nav><noscript><p>浏览器脚本已禁用，请使用浏览器查找或滚动阅读。</p></noscript></aside><main>'+body+'</main></div><footer>OpenPINN 双说明书 · 以源码与实际接口为准 · 完整单文件归档版 · 网站按六个子页面阅读 · 不依赖 CDN</footer><script>'+JS+'</script></body></html>'

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    py_paths=sorted((ROOT/'src/openpinn').rglob('*.py'))+sorted((ROOT/'docs/_ext').glob('*.py'))+sorted((ROOT/'examples').glob('*.py'))+sorted((ROOT/'user_projects').rglob('*.py'))+sorted((ROOT/'scripts').glob('*.py'))
    # This generator is described in the maintenance section, not recursively indexed.
    py_paths=[p for p in py_paths if p.resolve()!=Path(__file__).resolve()]
    js_paths=sorted((ROOT/'docs/_static').glob('*.js'))
    modules=python_inventory(py_paths)
    api,count=function_appendix(modules)
    handoff_paths=[ROOT/name for name in (
        'package.json', 'package-lock.json', 'scripts/build_docs.sh',
        'scripts/build_docs.ps1', 'scripts/check.sh', 'scripts/check.ps1',
        'examples/geometries/analytic_two_holes.dxf',
        'docs/_static/manual_diagrams.css',
        'src/openpinn/physics_catalog.json', 'docs/_static/physics_editor.css',
        'docs/physics_library.rst', 'docs/method_configurator.rst',
        'docs/_static/method_configurator.css', 'catalog/README.md', 'catalog/ACCEPTANCE.md',
    )]
    tracked=sorted(set(py_paths+js_paths+[ROOT/'pyproject.toml',ROOT/'docs/conf.py',Path(__file__).resolve()]+list(SOURCES.glob('*.html'))+handoff_paths+list((ROOT/'docs').glob('*guide*.rst'))))
    user=(SOURCES/'user_content.html').read_text(encoding='utf-8')+parameters()
    developer=(SOURCES/'developer_content.html').read_text(encoding='utf-8')+(SOURCES/'developer_contributing.html').read_text(encoding='utf-8')+api+js_appendix(js_paths)+provenance(tracked)
    for name,title,subtitle,body in [('user_manual','使用者说明书','只说明软件怎样安装、配置、运行、查看结果与处理常见问题；不展开内部代码设计。',user),('developer_manual','开发者说明书','面向多人协作维护：代码结构、数据契约、模块关系、关键函数调用链、扩展步骤与验收方式。',developer)]:
        document = render(name,title,subtitle,body)
        if name == 'developer_manual':
            document = document.replace('OPENPINN / '+DATE+' / 源码核对版', 'OPENPINN / '+DEVELOPER_REVISION+' / 贡献者指南', 1)
        document = document.replace('</header>', '</header><div class="manual-framework-standalone">' + framework('user' if name == 'user_manual' else 'developer', offline=True) + '</div>', 1)
        path=OUTPUT/(name+'.html');path.write_text(document,encoding='utf-8')
        print(f'{path}: {path.stat().st_size:,} bytes')
    report={'date':max(DATE, DEVELOPER_REVISION),'user_basis_date':DATE,'developer_revision':DEVELOPER_REVISION,'python_files':len(modules),'python_definitions':count,'js_files':len(js_paths),'method_variants':len(METHOD_PARAMETER_SCHEMAS),'tracked_files':len(tracked)}
    (OUTPUT/'manual_build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__': main()
