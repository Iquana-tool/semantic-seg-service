# coral-back-ai-training
This repository contains the code for a microservice controlling semantic segmentation models.

---

# Features
- **Model Training**: Train segmentation models using custom datasets.
- **Model Evaluation**: Evaluate trained models on validation datasets.
- **Model Deployment**: Deploy trained models for inference.
- **Model Management**: Manage multiple versions of models.
- **API Endpoints**: Expose endpoints for training, evaluation, and deployment.

---
# Setup Instructions
## Docker compose (Recommended)
1. Install docker and docker-compose
2. Run `docker compose up --build`. Alternatively if you want CUDA enabled, specify the `docker-compose-cuda.yml`.
## Manual Setup
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up the environment variables in a `.env` file:
   ```plaintext
   MODEL_PATH=/path/to/model/dir
   LOG_PATH=/path/to/log/dir
   DATA_PATH=/path/to/dataset/dir
   JOBS_PATH=accuracy,precision,recall
   ```
3. Run the microservice:
   ```bash
   fastapi run main.py --port [your_port_number]
   ```
## Docker Setup
1. Build the Docker image (Alternatively build the CUDA version by specifying the image):
   ```bash
   docker build -t coral-back-ai-training .
   ```
2. Run the Docker container:
   ```bash
    docker run -d -p [your_port_number]:8000 --env-file .env coral-back-ai-training
    ```
---
# API Documentation
For detailed API documentation, please refer to the [OpenAPI documentation](http://localhost:[your_port_number]/docs) after running the service.

---
# Model Zoo
These models are included in the repository by default:
- **UNet**: A popular architecture for image segmentation tasks.
- **UNet++**: An improved version of UNet with better performance.
- **DeepLabV3**: A state-of-the-art model for semantic segmentation.
- **DeepLabV3+**: Advance version of DeepLabV3.
## Adding Custom Models
You can add any model you want as long as it is a pytorch implementation. You can follow these steps to add your own 
model:
1. Place the model file in the `models` directory. The model should implement a class that inherits from `torch.nn.Module` and has a `forward` method.
2. Add your model to MODEL_REGISTRY in `models/__init__.py`. You can orient at the existing models for the correct format.
3. Done! Your model should appear as a base model now.
