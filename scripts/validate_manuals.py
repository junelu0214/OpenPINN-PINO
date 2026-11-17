"""Validate offline manual structure and execute explicitly marked tiny examples."""
from __future__ import annotations

import ast
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "docs/manuals/public"

class ManualParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids=[]; self.links=[]; self.samples=[]; self.sections=0
        self.pre=None; self.text=[]; self.scripts=[]; self.script=False; self.script_text=[]
        self.remote=[]

    def handle_starttag(self,tag,attrs):
        data=dict(attrs)
        if 'id' in data: self.ids.append(data['id'])
        if tag=='a' and 'href' in data: self.links.append(data['href'])
        if tag=='section': self.sections+=1
        if tag=='pre': self.pre=data; self.text=[]
        if tag=='script': self.script=True; self.script_text=[]
        for key in ('src','href'):
            if tag in ('script','link','img','iframe') and data.get(key,'').startswith(('http:','https:','//')): self.remote.append(data[key])

    def handle_data(self,data):
        if self.pre is not None: self.text.append(data)
        if self.script: self.script_text.append(data)

    def handle_endtag(self,tag):
        if tag=='pre' and self.pre is not None:
            if self.pre.get('data-language')=='python': self.samples.append((self.pre.get('data-smoke'),''.join(self.text)))
            self.pre=None
        if tag=='script': self.scripts.append(''.join(self.script_text)); self.script=False

def main():
    started=time.perf_counter()
    workspace=Path(tempfile.mkdtemp(prefix='openpinn-manual-validation-'))
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    report={'basis_date':'2026-10-12','status':'passed','workspace':str(workspace),'html':[],'smoke_examples':[],
            'not_checked':['真实浏览器视觉排版与缩放/拖拽','正式收敛精度或多种子优越性','手册中所有长预算命令']}
    for name in ('user_manual','developer_manual'):
        path=PUBLIC/(name+'.html'); document=path.read_text(encoding='utf-8')
        parser=ManualParser();parser.feed(document)
        assert document.startswith('<!doctype html>'),name
        assert len(parser.ids)==len(set(parser.ids)),f'{name}: duplicate ids'
        assert not parser.remote,f'{name}: remote runtime assets'
        for target in parser.links:
            if target.startswith('#'): assert target[1:] in parser.ids,(name,target)
            elif not target.startswith(('https:','http:','mailto:')): assert (path.parent/target).is_file(),(name,target)
        for script in parser.scripts:
            checked=subprocess.run(['node','--check'],input=script,text=True,capture_output=True)
            assert checked.returncode==0,checked.stderr
        for sample_id,source in parser.samples:
            ast.parse(source)
            if sample_id:
                case_started=time.perf_counter()
                completed=subprocess.run([sys.executable,'-c',source],cwd=workspace,env=env,text=True,capture_output=True,timeout=120)
                outcome={'name':sample_id,'returncode':completed.returncode,'seconds':round(time.perf_counter()-case_started,3),'stdout_tail':completed.stdout[-1500:],'stderr':completed.stderr[-3000:]}
                report['smoke_examples'].append(outcome)
                if completed.returncode: report['status']='failed'
                print(f"{sample_id}: {'PASS' if completed.returncode==0 else 'FAIL'}",flush=True)
        report['html'].append({'file':path.name,'bytes':path.stat().st_size,'sections':parser.sections,'ids':len(parser.ids),'python_examples_syntax_checked':len(parser.samples),'no_remote_runtime_assets':True})
    report['elapsed_seconds']=round(time.perf_counter()-started,3)
    (PUBLIC/'manual_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['status']=='passed' else 1

if __name__=='__main__': raise SystemExit(main())
