# main.py

import argparse
import os
from dataset import dataset_setup
from model import NeuralNetwork
from train import train_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network on image data.')
    parser.add_argument('--dataset_dir', type=str, help='Directory where the dataset is located.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    # Add other relevant arguments here

    args = parser.parse_args()
    return args

def main():
    # Parse command-line arguments
    args = parse_args()

    # Setup the dataset
    train_data, validation_data = dataset_setup(dataset_type='train', data_aug=True)

    # Create the neural network model
    _, _, _, model = NeuralNetwork(
        input_shape=(256, 256),
        output_activation='sigmoid',
        minibatch_size=args.batch_size,
        is_training=True
    )

    # Train the model
    train_model(
        train_data=train_data,
        val_data=validation_data,
        train_steps=100,  # Define appropriately or calculate based on your dataset
        val_steps=50,     # Define appropriately or calculate based on your dataset
        out_dir='path/to/save/models',  # Replace with the path where you want to save the model
        args=args
    )

if __name__ == '__main__':
    main()
