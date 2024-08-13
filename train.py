import logging
from flask import Flask, request, jsonify

'''This code trains the MNIST model and deploys it using a Flask-based web server.
The trained model is saved to a file for later use.'''

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input image data from the request
    image_data = request.get_json()['image']

    # Preprocess the image
    image = tf.convert_to_tensor(image_data, dtype=tf.float32)
    image = tf.reshape(image, (1, 28, 28, 1))
    image = image / 255.0

    try:
        # Make the prediction
        predictions = model.predict(image)
        predicted_digit = int(tf.argmax(predictions[0]))
        predicted_probability = float(tf.reduce_max(predictions[0]))

        # Log the prediction
        logging.info(f'Predicted digit: {predicted_digit}, Probability: {predicted_probability}')

        # Return the prediction as JSON
        return jsonify({'predicted_digit': predicted_digit, 'predicted_probability': predicted_probability})
    except Exception as e:
        # Log the error
        logging.error(f'Error in prediction: {str(e)}')
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
