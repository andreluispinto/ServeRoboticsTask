import tensorflow as tf
from flask import Flask, request, jsonify

# Load the trained model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input image data from the request
    image_data = request.get_json()['image']

    # Preprocess the image
    image = tf.convert_to_tensor(image_data, dtype=tf.float32)
    image = tf.reshape(image, (1, 28, 28, 1))
    image = image / 255.0

    # Make the prediction
    predictions = model.predict(image)
    predicted_digit = int(tf.argmax(predictions[0]))
    predicted_probability = float(tf.reduce_max(predictions[0]))

    # Return the prediction as JSON
    return jsonify({'predicted_digit': predicted_digit, 'predicted_probability': predicted_probability})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)