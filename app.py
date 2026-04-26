from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model('bengali_handwritten_cnn.h5')
IMG_SIZE=(64,64)

@app.route('/', methods=['GET','POST'])
def index():
    prediction=None
    confidence=None
    error=None
    if request.method=='POST':
        f=request.files.get('image')
        if f:
            try:
                img=Image.open(f).convert('RGB').resize(IMG_SIZE)
                arr=np.array(img)/255.0
                arr=np.expand_dims(arr,0)
                pred=model.predict(arr, verbose=0)[0]
                cls=int(np.argmax(pred))
                conf=float(np.max(pred))*100
                prediction=f'Predicted Class: {cls}'
                confidence=f'{conf:.2f}%'
            except Exception as e:
                error=str(e)
    return render_template('index.html', prediction=prediction, confidence=confidence, error=error)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)