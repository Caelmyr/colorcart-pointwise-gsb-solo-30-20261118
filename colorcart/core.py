import io,copy,hashlib
import numpy as np
import cv2
from PIL import Image,ImageDraw
from image_lab.validation import Invalid

PALETTE=['#ef8354','#4f5d75','#bfc0c0','#82b29a','#f2cc8f','#9a8c98']

def integer(v,lo,hi,name):
    if isinstance(v,bool) or not isinstance(v,(int,np.integer)) or not lo<=int(v)<=hi:raise Invalid(name+'超出范围')
    return int(v)

def rgb(s):
    if not isinstance(s,str) or len(s)!=7 or s[0]!='#':raise Invalid('颜色需为#RRGGBB')
    try:return tuple(int(s[i:i+2],16) for i in (1,3,5))
    except ValueError:raise Invalid('颜色需为#RRGGBB')

def png(a):
    out=io.BytesIO();Image.fromarray(np.asarray(a,dtype=np.uint8)).save(out,format='PNG');return out.getvalue()

def decode(raw):
    try:
        im=Image.open(io.BytesIO(raw));im.load()
    except Exception:raise Invalid('无法读取图片，请上传PNG/JPEG')
    if max(im.size)>2048 or min(im.size)<4:raise Invalid('图片边长需在4–2048像素')
    im=im.convert('RGBA');im.thumbnail((128,128),Image.Resampling.NEAREST);return np.asarray(im).copy()

def demo():
    a=np.zeros((96,120,4),np.uint8);colors=[(225,139,105),(224,161,120),(175,181,147),(126,163,160),(173,151,163),(205,179,139)]
    for y in range(3):
        for x in range(4):a[y*32:(y+1)*32,x*30:(x+1)*30]=(*colors[(x+2*y)%6],255)
    a[16:48,15:45]=(*colors[5],255)
    return a

def validate(p):
    labels=np.asarray(p['labels'],dtype=np.int32);original=np.asarray(p['original'],dtype=np.uint8)
    if labels.ndim!=2 or max(labels.shape)>128 or min(labels.shape)<4 or original.shape!=(*labels.shape,4):raise Invalid('画布数据无效')
    ids={int(x) for x in np.unique(labels) if x}
    if not ids or len(ids)>64 or set(p['regions'])!={str(x) for x in ids}:raise Invalid('区域数据无效')
    if not isinstance(p['palette'],list) or not 2<=len(p['palette'])<=8:raise Invalid('色板需2–8色')
    for c in p['palette']:rgb(c)
    for key,r in p['regions'].items():
        rgb(r['color'])
        if not isinstance(r['locked'],bool):raise Invalid('锁色值无效')
        if len(str(r.get('name','')))>80:raise Invalid('区域名称过长')
    return p

def create(a,b):
    count=integer(b.get('colors',6),2,8,'分色数量');min_area=integer(b.get('min_area',3),1,64,'最小面积')
    im=Image.fromarray(a[:,:,:3]).quantize(colors=count,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE).convert('RGB');flat=np.asarray(im)
    labels=np.zeros(a.shape[:2],np.int32);regions={};ident=1
    for c in np.unique(flat.reshape(-1,3),axis=0):
        mask=(np.all(flat==c,axis=2)&(a[:,:,3]>=128)).astype(np.uint8)
        n,comp,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=4)
        for k in range(1,n):
            area=int(stats[k,cv2.CC_STAT_AREA])
            if area<min_area:continue
            if ident>64:raise Invalid('分色产生超过64区，请用准备页简化图像或提高最小面积')
            labels[comp==k]=ident
            source=np.median(a[:,:,:3][comp==k],axis=0).astype(int).tolist();color='#'+''.join(f'{v:02x}' for v in source)
            regions[str(ident)]=dict(color=color,source_color=color,locked=False,name=f'区域{ident}')
            ident+=1
    if not regions:raise Invalid('没有可用不透明区域')
    return validate(dict(name=str(b.get('name','未命名插画'))[:80],labels=labels.tolist(),original=a.tolist(),regions=regions,palette=PALETTE.copy(),undo=[],redo=[],notes='',source=dict(colors=count,min_area=min_area)))

def edges(p):
    a=np.asarray(p['labels']);result=set()
    for x,y in ((a[:,:-1],a[:,1:]),(a[:-1,:],a[1:,:])):
        mask=(x!=y)&(x>0)&(y>0)
        for u,v in zip(x[mask],y[mask]):result.add(tuple(sorted((int(u),int(v)))))
    return sorted(result)

def summary(p):
    a=np.asarray(p['labels']);pairs=edges(p)
    return dict(width=a.shape[1],height=a.shape[0],region_count=len(p['regions']),adjacency=[list(x) for x in pairs],equal_neighbors=[list(x) for x in pairs if p['regions'][str(x[0])]['color']==p['regions'][str(x[1])]['color']],regions=[dict(id=int(k),area=int(np.sum(a==int(k))),**v) for k,v in p['regions'].items()])

def snapshot(p):return {k:copy.deepcopy(p[k]) for k in ('name','labels','regions','palette','notes')}

def checkpoint(p):
    p['undo'].append(snapshot(p));p['undo']=p['undo'][-20:];p['redo']=[]

def edit(p,b):
    action=b.get('action');ids=b.get('regions',[])
    if not isinstance(ids,list) or len(ids)>64 or any(str(x) not in p['regions'] for x in ids):raise Invalid('请选择存在的区域')
    if action in ('paint','lock','rename','merge') and not ids:raise Invalid('请先选择区域')
    checkpoint(p)
    if action=='paint':
        color=b.get('color');rgb(color)
        for key in ids:
            if not p['regions'][str(key)]['locked']:p['regions'][str(key)]['color']=color.lower()
    elif action=='lock':
        if not isinstance(b.get('locked'),bool):raise Invalid('请选择锁定状态')
        for key in ids:p['regions'][str(key)]['locked']=b['locked']
    elif action=='rename':
        for key in ids:p['regions'][str(key)]['name']=str(b.get('name',''))[:80]
    elif action=='palette':
        colors=b.get('palette');p['palette']=copy.deepcopy(colors)
    elif action=='notes':p['notes']=str(b.get('notes',''))[:10000]
    elif action=='merge':
        ids=set(map(int,ids));adj=edges(p);seen={min(ids)}
        while True:
            nxt=seen|{u for u,v in adj if v in seen and u in ids}|{v for u,v in adj if u in seen and v in ids}
            if nxt==seen:break
            seen=nxt
        if seen!=ids:raise Invalid('只能合并互相连通的相邻区域')
        if any(p['regions'][str(x)]['locked'] for x in ids):raise Invalid('请先解锁要合并的区域')
        dest=min(ids);a=np.asarray(p['labels']);a[np.isin(a,list(ids))]=dest;p['labels']=a.tolist()
        for key in ids-{dest}:del p['regions'][str(key)]
    elif action=='reset':
        for r in p['regions'].values():
            if not r['locked']:r['color']=r['source_color']
    else:raise Invalid('未知编辑操作')
    return validate(p)

def history(p,action):
    if action not in ('undo','redo'):raise Invalid('未知历史操作')
    src=p[action];dest=p['redo' if action=='undo' else 'undo']
    if not src:raise Invalid('没有可用历史')
    dest.append(snapshot(p));p.update(src.pop());return validate(p)

def render(p,mode='color',selected=None):
    a=np.asarray(p['labels']);out=np.zeros((*a.shape,4),np.uint8)
    if mode=='original':out=np.asarray(p['original'],np.uint8).copy()
    elif mode in ('color','outline'):
        for key,r in p['regions'].items():out[a==int(key)]=(*rgb(r['color']),255)
        if mode=='outline':
            border=np.zeros(a.shape,bool);border[:,:-1]|=a[:,:-1]!=a[:,1:];border[:-1,:]|=a[:-1,:]!=a[1:,:];out[border]=(35,38,40,255)
    else:raise Invalid('未知视图')
    if selected:
        mask=np.isin(a,selected);out[mask,:3]=(out[mask,:3].astype(float)*.5+np.array([255,240,80])*.5).astype(np.uint8)
    return out

def swatch_sheet(p):
    im=Image.new('RGB',(320,40*len(p['palette'])),(255,255,255));draw=ImageDraw.Draw(im)
    for i,c in enumerate(p['palette']):draw.rectangle((0,i*40,100,i*40+39),fill=rgb(c));draw.text((115,i*40+12),f'{i+1}  {c}',fill=(30,30,30))
    return png(im)
