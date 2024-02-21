# Project Name

This project is a solution for training and making predictions using neural network models built with PyTorch. It provides scripts for training, predicting, and modular operations for model handling. With support for GPU acceleration, automatic checkpoint saving, and compatibility with popular architectures like VGG and ResNet, the project allows users to  create, train, and utilize deep learning models for various tasks.

## Key Features

- **GPU Support:** The project is configured to leverage GPU acceleration if available, enhancing the training speed.
- **Checkpoint Saving:** During the training process, the script automatically saves checkpoints, allowing the user to resume training from the last epoch in case of interruption.
- **Supported Architectures:** The project supports popular architectures like VGG and ResNet, providing flexibility in choosing the architecture that best suits the task.

## Files

- **train.py:** Script for training a neural network model.
- **predict.py:** Script for making predictions using a trained model.
- **model_operations.py:** Module containing classes and functions related to the neural network model.
- **utils.py:** Utility functions for data loading and processing.

## Requirements

- Python (>=3.x)
- PyTorch (2.2.0)
- torchvision (0.17.0)
- matplotlib (3.8.1)
- numpy (1.22.3)

## Getting Started

1. Clone the repository.
   ```
   git clone https://github.com/your-username/your-repo.git
   ```
2. Install dependencies.
   ```
   pip install -r requirements.txt
   ``` 
4. Run the training script.
   ```
   python3 train.py <data_dir> --save_dir <save_dir> --arch <arch> --learning_rate <learning_rate> --hidden_units <hidden_units> --epochs <epochs> --batch_size <batch_size> --gpu --resume_training
   ```
   Example:
   ```
   python3 train.py flowers --arch vgg19 --learning_rate 0.0001 --hidden_units 512 --epochs 10 --batch_size 32 --gpu
   ```
6. Make predictions using the trained model.
   ```
   python3 predict.py <data_dir> <image_path> --checkpoint <checkpoint_path> --top_k <top_k> --gpu
   ```
   Example:
   ```
   python3 predict.py flowers image.jpg --top_k 3 --gpu
   ```
