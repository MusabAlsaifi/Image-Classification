# Imports
import os
import torch
from typing import Tuple, List
from torch import nn, optim
from torchvision import models
from utils import trainloader, validateloader, testloader, process_image


# Dictionary containing supported architectures
ARCHITECTURE_DICT = {
    'vgg11': 'VGG11_Weights.DEFAULT',
    'vgg13': 'VGG13_Weights.DEFAULT',
    'vgg16': 'VGG16_Weights.DEFAULT',
    'vgg19': 'VGG19_Weights.DEFAULT',
    'resnet18': 'ResNet18_Weights.DEFAULT',
    'resnet34': 'ResNet34_Weights.DEFAULT',
    'resnet50': 'ResNet50_Weights.DEFAULT',
    'resnet101': 'ResNet101_Weights.DEFAULT',
    'resnet152': 'ResNet152_Weights.DEFAULT'
}


class Architecture:
    @staticmethod
    def create_archetype(arch_name: str) -> nn.Module:
        """
        Create an instance of a specified architecture.

        Parameters:
        - arch_name (str): Name of the architecture to create.

        Returns:
        - model: An instance of the specified architecture.
        """
        if arch_name in ARCHITECTURE_DICT:
            # Create an instance of the specified architecture with pretrained weights 
            return getattr(models, arch_name)(weights=ARCHITECTURE_DICT[arch_name])
        else:
            print("Available Architecture are:")
            for arch in ARCHITECTURE_DICT.keys():
                print(arch)
            raise ValueError(f"Unsupported Architecture: {arch_name}")
        

class CustomClassifier:
    @staticmethod
    def create_classifier(in_features: int, hidden_units: int, out_features: int) -> nn.Module:
        """
        Create a custom classifier module.

        Parameters:
        - in_features (int): Number of input features.
        - hidden_units (int): Number of hidden units in the classifier.
        - out_features (int): Number of output features.

        Returns:
        - classifier (nn.Module): Custom classifier module.
        """
        classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, out_features),
            nn.LogSoftmax(dim=1)
        )

        return classifier
    

class Train:
    def __init__(self, train_data: str, validate_data: str, arch: str, learning_rate: float,
                  epochs: int, hidden_units: int, device: torch.device, batch_size: int):
        """
        Initialize the training process for a neural network model.

        Parameters:
        - train_data (str): Path to the training dataset.
        - validate_data (str): Path to the validation dataset.
        - arch (str): Name of the architecture to use.
        - learning_rate (float): Learning rate for the optimizer.
        - epochs (int): Number of training epochs.
        - hidden_units (int): Number of hidden units in the classifier.
        - device (torch.device): Device to use for training (e.g., "cuda" or "cpu").
        - batch_size (int): Size of each mini-batch during training.
        """
        self.arch = arch
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_units = hidden_units
        self.device = device
        self.batch_size = batch_size
        self.train_data = trainloader(self.device, train_data, self.batch_size)
        self.validate_data = validateloader(self.device, validate_data, self.batch_size)
        self.output = len(self.train_data.dataset.classes)
        self.input_size = None
        self.model, self.criterion, self.optimizer = self.initialize_model()
        self.model.class_to_idx = self.train_data.dataset.class_to_idx
      

    def initialize_model(self) -> Tuple[nn.Module, nn.NLLLoss, optim.Optimizer]:
        """
        Initialize the neural network model for training.

        This method creates an instance of the specified architecture, freezes its
        pre-trained parameters, and replaces the classifier with a new one suitable
        for the specific task.

        Parameters:
        - arch (str): Name of the architecture to use (default: "vgg16").
        - learning_rate (float): Learning rate for the optimizer (default: 0.001).
        - hidden_units (int): Number of hidden units in the classifier (default: 512).

        Returns:
        - model: An initialized neural network model.
        """
        print("Initializing model...")

        # Create the model 
        model = Architecture.create_archetype(self.arch) 
    
        # Freeze the parameters of the model
        for params in model.parameters():
            params.requires_grad = False
        
        # For VGG architectures
        if 'vgg' in self.arch:

            # get in_features and hidden units 
            self.input_size = model.classifier[0].in_features

            # If hidden units not specified by the user, then extract the defualt of the architechture
            if self.hidden_units == None:
                self.hidden_units = model.classifier[0].out_features

            # Create classifier
            model.classifier = CustomClassifier.create_classifier(
                in_features=self.input_size,
                hidden_units=self.hidden_units,
                out_features=self.output
            )

            optimizer = optim.Adam(model.classifier.parameters(), lr=self.learning_rate)
            print(model.classifier)

        # For ResNet architectures
        elif 'resnet' in self.arch:

            # get in_features and hidden units
            self.input_size = model.fc.in_features

            # If hidden units not specified by the user, then extract the defualt of the architechture
            if self.hidden_units == None:
                self.hidden_units = model.fc.out_features

            # Create classifier
            model.fc = CustomClassifier.create_classifier(
                in_features=self.input_size,
                hidden_units=self.hidden_units,
                out_features=self.output
            )

            optimizer = optim.Adam(model.fc.parameters(), lr=self.learning_rate)
            print(model.fc)

        criterion = nn.NLLLoss()
        model.to(self.device)

        return model, criterion, optimizer
    

    def train(self, checkpoint_path: str, resume_training: bool, checkpoint_interval: int = 1):
        """
        Train the model on the specified training dataset and validate it on the validation dataset.

        Parameters:
        - checkpoint_path (str): Path to save the model checkpoints (default: "checkpoint.pth").
        - checkpoint_interval (int): Number of epochs between each model checkpoint (default: 1).
        """
        print("Training...")
        steps = 0
        running_loss = 0
        print_every = 50

        # Check if a there is a checkpoint and --resume_traning flag is passed to args
        if os.path.isfile(checkpoint_path) and resume_training:

            # Load checkpoint and resume training
            checkpoint = torch.load(checkpoint_path)

            # Start from the next epoch
            start_epoch = checkpoint['epoch'] + 1  
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            running_loss = checkpoint['running_loss']
            print(f"Resuming training from epoch {start_epoch + 1}")
        else:
            # Start training from the beginning
            start_epoch = 0

        for epoch in range(start_epoch, self.epochs):
            print(f'\nepoch No. {epoch + 1}')

            for images, labels in self.train_data:
                steps += 1

                # Move data to the device
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                # forward pass
                logps = self.model(images)

                # Calculate the loss
                loss = self.criterion(logps, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    self.model.eval()
                    test_loss = 0
                    accuracy = 0

                    # To accelerates execution and reduces the amount of required memory
                    with torch.no_grad():
                        # Evaluate the model on the validation set
                        for images, labels in self.validate_data:
                            images, labels = images.to(self.device), labels.to(self.device)

                            logps = self.model(images)
                            loss = self.criterion(logps, labels)
                            test_loss += loss.item()

                            # Calculate the accuracy
                            ps = torch.exp(logps)
                            top_ps, top_class = ps.topk(1, dim=1)
                            equality = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{self.epochs} .. "
                        f"Train loss: {running_loss/print_every:.3f} .. "
                        f"Test loss: {test_loss/len(self.validate_data):.3f} .. "
                        f"Test accuracy: {accuracy/len(self.validate_data):.3f}")


                    running_loss = 0
                    self.model.train()

            # Save the model every epoch
            if epoch % checkpoint_interval == 0:
                self.save_checkpoint(epoch, checkpoint_path, running_loss)

        print("\nTraining's finished.")


    def save_checkpoint(self, epoch: int, checkpoint_path: str, running_loss: float):
        """
        Save a checkpoint of the current model state.

        Parameters:
        - epoch (int): Current epoch number.
        - checkpoint_path (str): Path to save the checkpoint.
        """
        print("Saving checkpoint...")
        
        checkpoint = {
            "epoch": epoch,
            "running_loss": running_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hidden_units": self.hidden_units,
            "output": self.output,
            "input_size": self.input_size,
            "arch_name": self.arch,
            "learning_rate": self.learning_rate,
            "class_to_idx": self.model.class_to_idx
        }

        # Save the checkpoint to the specified checkpoint path
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}: {checkpoint_path}")


class Predict:
    def __init__(self, checkpoint_path: str, image_path: str, top_k: int, device: torch.device, test_data: str):
        """
        Initialize the Predict class.

        Parameters:
        - checkpoint_path (str): Path to the checkpoint file.
        - image_path (str): Path to the input image_path for prediction.
        - top_k (int): Number of top classes to predict.
        - device (torch.device): Device to use for prediction (e.g., "cuda" or "cpu").
        """
        self.checkpoint_path = checkpoint_path
        self.image_path = image_path
        self.top_k = top_k
        self.device = device
        self.test_data = testloader(self.device, test_data, 1)
        self.output = len(self.test_data.dataset.classes)
        self.model = self.load_model()

    def load_model(self) -> nn.Module:
        """
        Load a trained model from a checkpoint file.

        Returns:
        - model: Loaded neural network model.
        """
        try:
            print("Loading model...")
            checkpoint = torch.load(self.checkpoint_path, map_location=torch.device(self.device))

            # Retrieve the name of the Architecture 
            arch_name = checkpoint["arch_name"]
            print(f"Architecture name in checkpoint: {arch_name}")

            # Create and load the model saved in the checkpoint
            model = models.__dict__[arch_name]()

            # Retrieve the number of hidden units from the checkpoint
            input_size = checkpoint["input_size"]
            hidden_units = checkpoint["hidden_units"]

            # check the architecture name
            # if it's VGG architecture
            if 'vgg' in arch_name:
                # Creating the classifier
                model.classifier = CustomClassifier.create_classifier(
                    in_features=input_size,
                    hidden_units=hidden_units,
                    out_features=self.output
                )

            # if RESNET architecture
            elif 'resnet' in arch_name:
                # Creating the classifier
                model.fc = CustomClassifier.create_classifier(
                in_features=input_size,
                hidden_units=hidden_units,
                out_features=self.output
            )

            # Load the model state dictionary from the checkpoint
            model.load_state_dict(checkpoint["model_state_dict"])

            # Load class-to-index mapping from the checkpoint
            model.class_to_idx = checkpoint['class_to_idx']

            # Move the model to the device and set it to evaluation mode
            model.to(self.device)
            model.eval()
            
            return model
        
        except FileNotFoundError:
            # If the checkpoint file not found 
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")


    def predict(self) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Process an input image_path for prediction.

        Parameters:
        - image_path (str): Path to the input image_path.
        - top_k (int): Top K predictions
        - device (str): Device to use for prediction (e.g., "cuda" or "cpu").

        Returns:
        - Tuple[torch.Tensor, List[str], torch.Tensor]: Tuple containing topk probabilities, 
          corresponding class labels, and the image_path tensor after processing.
        """
        model = self.model
        # Implement the code to predict the class from an image_path file
        img_tensor = process_image(self.image_path)
        
        # Move the image_path tensor to the appropriate device
        img_tensor = img_tensor.to(self.device)

        # Set the model to evaluation mode
        model.eval()

        # Perform inference
        with torch.no_grad():
            output = model(img_tensor)

        probabilities = torch.exp(output[0])

        # Get the topk predicted classes and corresponding probabilities
        topk_probabilities, topk_indices = torch.topk(probabilities, self.top_k)
        
        # Convert indices to class labels using the model's class_to_idx attribute
        idx_to_class = {idx: label for label, idx in model.class_to_idx.items()}
        topk_classes = [idx_to_class[idx.item()] for idx in topk_indices]

        return topk_probabilities, topk_classes, img_tensor