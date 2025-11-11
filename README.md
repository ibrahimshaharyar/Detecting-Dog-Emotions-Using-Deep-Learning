# Detecting-Dog-Emotions-Using-Deep-Learning
🐶 Dog Emotion Classification using PyTorch

This project implements a Convolutional Neural Network (CNN) to classify dog emotions from images.
It’s based on the Dataquest Dog Emotions Project
 and expands it with improved model structure, data augmentation, and normalization.

🚀 Project Overview

The goal is to train a deep learning model that can recognize a dog’s emotional expression (e.g., happy, sad, angry) from its image.

Key Features

Custom Dataset Class (DogDataset) for:

Image loading via torchvision.io.read_image

On-the-fly augmentations (horizontal flip, rotation, auto-augment)

Normalization with ImageNet mean/std

CNN Architecture (NeuralNetwork):

Multiple Conv2d + BatchNorm2d + ReLU layers

Max pooling for spatial downsampling

Fully-connected classifier layer for emotion prediction

Training Loop:

Mini-batch training with optimizer, loss computation, and GPU acceleration

Split into 80% train / 20% test sets using torch.utils.data.random_split

🧠 Model Architecture
Conv2d(3 → 64, kernel=4, stride=2)
BatchNorm2d(64)
ReLU
MaxPool2d(2x2)
Conv2d(64 → 64, kernel=2)
BatchNorm2d(64)
Conv2d(64 → 64, kernel=2)
BatchNorm2d(64)
ReLU
MaxPool2d(2x2)
Flatten
Linear(64*46*46 → 64)
Linear(64 → num_classes)

📦 Installation
git clone https://github.com/ibrahimshaharyar/Detecting-Dog-Emotions-Using-Deep-Learning.git
cd Detecting-Dog-Emotions-Using-Deep-Learning
pip install torch torchvision pandas numpy matplotlib

🏋️‍♂️ Training
from torch.utils.data import DataLoader
from model import NeuralNetwork, DogDataset

# create dataset and dataloader
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

model = NeuralNetwork(classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    for batch, (images, labels, paths) in enumerate(train_loader):
        ...

📈 Future Work

Add validation accuracy tracking

Implement learning-rate scheduler

Visualize Grad-CAM heatmaps

Deploy model as a simple web app

📜 License

This project is open-source and available under the MIT License.
