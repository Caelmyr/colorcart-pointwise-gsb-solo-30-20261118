import os
from image_lab import create_app
from colorcart.server import install
from waitress import serve
root=os.environ.get('COLORCART_DATA_DIR','data')
app=create_app(os.path.join(root,'preparation'))
install(app,os.path.join(root,'illustrations'))
if __name__=='__main__':serve(app,host='0.0.0.0',port=int(os.environ.get('PORT','18933')),threads=4)
