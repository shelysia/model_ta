from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')  # Ganti dengan path model kamu

HTML_TEMPLATE = '''
<!doctype html>
<title>Upload Image for Prediction</title>
<h2>🧠 CNN Image Prediction</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=image>
  <input type=submit value=Predict>
</form>
{% if prediction %}
  <h3>Predicted Class: {{ prediction.predicted_class }}</h3>
  <h3>Confidence: {{ prediction.confidence }}</h3>
{% elif error %}
  <h3 style="color:red;">Error: {{ error }}</h3>
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        if 'image' not in request.files:
            error = 'No image provided'
        else:
            image_file = request.files['image']
            try:
                image = Image.open(image_file).convert('RGB').resize((128, 128))
                image_array = np.array(image).astype('float32') / 255.0
                image_array = np.expand_dims(image_array, axis=0)
                
                pred = model.predict(image_array)[0]
                predicted_class = int(np.argmax(pred))
                confidence = float(np.max(pred))

                pred = model.predict(image_array)[0][0]  # ambil nilai float-nya

                if pred >= 0.5:
                    predicted_class = 1  # Sehat
                else:
                    predicted_class = 0  # Mastitis

                confidence = pred if predicted_class == 1 else 1 - pred

                if predicted_class == 0:
                    predict = "Mastitis"
                else:
                    predict = "Sehat"

                prediction = {
                    'predicted_class': predict,
                    'confidence': round(confidence * 100, 2)
                }

            except Exception as e:
                error = str(e)
    
    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error)

@app.route('/predict', methods=['POST'])
def predict_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    
    try:
        image = Image.open(image_file).convert('RGB').resize((128, 128))
        image_array = np.array(image).astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        prediction = model.predict(image_array)[0]
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        if predicted_class == 0 :
            predict = "Mastitis"
        elif predicted_class == 1 :
            predict = "Sehat"

        return jsonify({
            'predicted_class': predict,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
