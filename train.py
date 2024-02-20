import os
import torch
import argparse
from model_operations import Train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training a neural network model")

    # Required argument
    parser.add_argument("data_dir", type=str, help="Path to the dataset directory")

    # Optional argument
    parser.add_argument("--save_dir", type=str, default="checkpoint.pth", help="Path to save checkpoint")
    parser.add_argument("--arch", type=str, default="vgg16", help="Name of the architecture to use")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--hidden_units", type=int, default=None, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=5, help="Number of traning epoch")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of training batches")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if specified")
    parser.add_argument("--resume_training", action="store_true", help="Resume training from saved checkpoint")

    args = parser.parse_args()
    
    # Create full path for training and validation datasets 
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    
    # Set the device based on the --gpu flag
    if args.gpu and torch.cuda.is_available():
        device = "cuda"
    # using GPU for MacOS
    elif args.gpu and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using {device} device")

    trainer = Train(
        train_data=train_dir,
        validate_data=valid_dir,
        arch=args.arch,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device
    )

    # Start or continue training the model and save checkpoints
    trainer.train(
        checkpoint_path=args.save_dir,
        resume_training=(True if args.resume_training else False),
    )