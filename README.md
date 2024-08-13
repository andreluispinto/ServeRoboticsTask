# ServeRoboticsTask

Requirements:
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
