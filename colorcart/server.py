import io,copy
from flask import request,jsonify,render_template,send_file
from image_lab.validation import Invalid
from .storage import Storage
from .core import decode,demo,create,edit,history,render,summary,png,swatch_sheet,validate

def install(app,root):
    store=Storage(root);app.extensions['colorcart']=store
    def body():
        b=request.get_json(silent=True)
        if not isinstance(b,dict):raise Invalid('请求需为JSON对象')
        return b
    def response(p):return jsonify(pattern=p,summary=summary(p))
    @app.get('/')
    @app.get('/colorcart')
    def colorcart_page():return render_template('colorcart.html')
    @app.get('/colorcart/state')
    def colorcart_state():return jsonify(store.state())
    @app.post('/colorcart/demo')
    def colorcart_demo():return jsonify(image=store.save_image('合成平涂小镇',demo()))
    @app.post('/colorcart/upload')
    def colorcart_upload():
        f=request.files.get('file')
        if not f:raise Invalid('请选择图片')
        return jsonify(image=store.save_image(f.filename,decode(f.read())))
    @app.get('/colorcart/images/<key>.png')
    def colorcart_image(key):return send_file(io.BytesIO(png(store.image(key)[1])),mimetype='image/png')
    @app.post('/colorcart/patterns')
    def colorcart_create():
        b=body();row,a=store.image(b.get('image_id'));return response(store.save_pattern(create(a,b),dict(image_id=row['id'],image_sha256=row['sha256'])))
    @app.get('/colorcart/patterns/<key>')
    def colorcart_pattern(key):return response(store.pattern(key))
    @app.post('/colorcart/patterns/<key>/edit')
    def colorcart_edit(key):
        b=body();return response(store.change(key,b.get('revision'),lambda p:edit(p,b)))
    @app.post('/colorcart/patterns/<key>/history')
    def colorcart_history(key):
        b=body();return response(store.change(key,b.get('revision'),lambda p:history(p,b.get('action'))))
    @app.post('/colorcart/patterns/<key>/duplicate')
    def colorcart_duplicate(key):
        p=store.pattern(key);p['name']+=' 副本';p['undo']=[];p['redo']=[];return response(store.save_pattern(p,dict(pattern_id=key,revision=p['revision'])))
    @app.post('/colorcart/import')
    def colorcart_import():
        p=copy.deepcopy(body().get('pattern'));validate(p);return response(store.save_pattern(p,dict(imported=True)))
    @app.get('/colorcart/patterns/<key>.png')
    def colorcart_png(key):return send_file(io.BytesIO(png(render(store.pattern(key),request.args.get('mode','color')))),mimetype='image/png',download_name='colorcart.png')
    @app.get('/colorcart/patterns/<key>/palette.png')
    def colorcart_palette(key):return send_file(io.BytesIO(swatch_sheet(store.pattern(key))),mimetype='image/png',download_name='palette.png',as_attachment=True)
    @app.get('/colorcart/patterns/<key>.zip')
    def colorcart_zip(key):return send_file(io.BytesIO(store.export(key)),mimetype='application/zip',download_name='colorcart.zip',as_attachment=True)
