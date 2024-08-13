# ServeRoboticsTask

This pipeline includes the following steps:

Load the MNIST dataset using the TensorFlow Keras API.
Preprocess the data by normalizing the pixel values and reshaping the images to fit the model input.
Define the model architecture using the TensorFlow Keras Sequential API. The model includes three convolutional layers, two max-pooling layers, two fully connected layers, and an output layer.
Compile the model with the Adam optimizer and categorical cross-entropy loss.
Train the model for 10 epochs with a batch size of 32, using the validation data to monitor performance.
Evaluate the model on the test set and print the test loss and accuracy.
Save the trained model to a file for later use.
For the inference server, you can use the saved model to make predictions on new images.

# Implementation
This implementation uses the Flask web framework to create a REST API endpoint that accepts raw image data as input and returns the predicted digit and its probability as output. The server loads the pre-trained model and uses it to make predictions on the input image.

To use the inference server, you can send a POST request to the /predict endpoint with the image data in the request body, like the json file.

The server will respond with the predicted digit and its probability, please look at predicted.py file.

# Requirements:
1. Environment Setup:
Set up a reproducible environment using Docker or other containerization technologies.
Provide a Dockerfile that creates an environment with all dependencies necessary to train and serve a model.

3. Data Handling:
Develop a pipeline for loading and preprocessing the MNIST dataset efficiently.
Utilize a data versioning tool/platform (e.g., DVC, MLflow) to ensure reproducibility.

4. Model Training and Deployment:
Train a simple model (e.g., a neural network) to classify MNIST images.
Deploy the trained model using a serving platform such as TensorFlow Serving, FastAPI, or any other suitable technology.
Ensure that the deployment can handle live inference requests.

Bonus points:
1. Monitoring and Logging:
Implement logs for both training and inference processes using a logging framework.
Set up basic monitoring for the deployed model, tracking metrics such as request rate, latency, and error rates.

2. Infrastructure As Code:
Utilize Infrastructure as Code (IaC) tools (e.g., Terraform, Ansible) to automate the setup of required cloud infrastructure.
Document your cloud architecture and ensure it is scalable and cost-effective.
