import tensorflow as tf
from flask import Flask, request, jsonify

'''This code trains the MNIST model and deploys it using a Flask-based web server. 
The trained model is saved to a file for later use.'''


# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, tf.keras.utils.to_categorical(y_train), epochs=10, batch_size=32, validation_data=(x_test, tf.keras.utils.to_categorical(y_test)))

# Save the trained model
model.save('mnist_model.h5')

# Create the Flask app for deployment
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
