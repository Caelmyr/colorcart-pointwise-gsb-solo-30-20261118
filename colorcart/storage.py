import io,json,os,threading,uuid,hashlib,zipfile,copy
from pathlib import Path
import numpy as np
from image_lab.validation import Invalid,Conflict
from .core import png,validate as validate_pattern,summary,render,swatch_sheet

def valid_image(a):
    if a.ndim!=3 or a.shape[2]!=4 or max(a.shape[:2])>128:raise Invalid("图像数据无效")
    return a

def preview(p,overlay=False):return png(render(p))

class Storage:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock();self.document=self.root/'workspace.json'
        if not self.document.exists():self.write(dict(revision=0,images=[],patterns=[],notes=''))
    def read(self):return json.loads(self.document.read_text())
    def write(self,d):
        tmp=self.document.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));os.replace(tmp,self.document)
    def path(self,key):
        if not isinstance(key,str) or len(key)!=32 or any(x not in '0123456789abcdef' for x in key):raise Invalid('无效图像ID')
        return self.root/(key+'.npy')
    def image(self,key):
        with self.lock:
            data=self.read();row=next((x for x in data['images'] if x['id']==key),None)
            if row is None:raise Invalid('没有此照片')
            raw=self.path(key).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise Invalid('照片校验失败')
            return row,valid_image(np.load(io.BytesIO(raw),allow_pickle=False))
    def save_image(self,name,a):
        a=valid_image(a);out=io.BytesIO();np.save(out,a,allow_pickle=False);raw=out.getvalue();key=uuid.uuid4().hex
        row=dict(id=key,name=str(name)[:100],width=a.shape[1],height=a.shape[0],sha256=hashlib.sha256(raw).hexdigest())
        with self.lock:
            d=self.read();self.path(key).write_bytes(raw);d['images'].append(row);d['revision']+=1;self.write(d)
        return row
    def pattern(self,key):
        with self.lock:
            row=next((p for p in self.read()['patterns'] if p['id']==key),None)
            if row is None:raise Invalid('没有此图纸')
            return copy.deepcopy(validate_pattern(row))
    def save_pattern(self,p,source):
        q=copy.deepcopy(validate_pattern(p));q.update(id=uuid.uuid4().hex,revision=0,source=source)
        with self.lock:
            d=self.read();d['patterns'].append(q);d['revision']+=1;self.write(d)
        return q
    def change(self,key,revision,callback):
        with self.lock:
            d=self.read();p=next((p for p in d['patterns'] if p['id']==key),None)
            if p is None:raise Invalid('没有此图纸')
            if revision!=p['revision']:raise Conflict('图纸已改变，请刷新后重新编辑')
            callback(p);validate_pattern(p);p['revision']+=1;d['revision']+=1;self.write(d);return copy.deepcopy(p)
    def state(self):
        with self.lock:
            d=self.read();d['patterns']=[{k:v for k,v in p.items() if k not in ('original','labels','undo','redo')}|dict(summary=summary(p)) for p in d['patterns']];return d
    def notes(self,text,revision):
        with self.lock:
            d=self.read()
            if revision!=d['revision']:raise Conflict('工作区已改变，请刷新后重试')
            d['notes']=str(text)[:10000];d['revision']+=1;self.write(d);return d
    def export(self,key):
        p=self.pattern(key);out=io.BytesIO()
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('pattern.png',preview(p,False));z.writestr('pattern.json',json.dumps(p,ensure_ascii=False,indent=2));z.writestr('regions.json',json.dumps(summary(p),ensure_ascii=False,indent=2));z.writestr('palette.png',swatch_sheet(p));z.writestr('original.png',png(render(p,'original')))
            z.writestr('README.txt','ColorCart平涂插画换色包。pattern.png为当前换色图，original.png为分色前图像；pattern.json保存区域、色板、锁色和编辑历史；regions.json为面积和四邻接摘要。合成演示不是实拍。')
        return out.getvalue()
