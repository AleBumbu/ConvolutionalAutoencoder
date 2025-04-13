import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
import os

#---------------------------------------------------------
# Convolutional autoencoder with a 12:1 compression ratio
#---------------------------------------------------------

# Data formatting
def resizeImages(dataset, size):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
    ])
    dataset = torch.from_numpy(dataset).reshape(-1, 150, 225, 3).permute(0, 3, 1, 2)
    dataset = torch.stack([transform(img) for img in dataset]) / 255.0
    return dataset

def loadData():
    inputs1 = np.load("subset_1.npy")  # rainy
    inputs2 = np.load("subset_2.npy")  # sunny
    inputs3 = np.load("subset_3.npy")  # foggy + sunset

    fullDataset = np.concatenate((inputs1, inputs2, inputs3), axis=0)
    np.random.shuffle(fullDataset)

    fullDataset = resizeImages(fullDataset, 256)
    trainingData, testingData = train_test_split(fullDataset, test_size=0.2)
    return trainingData, testingData


# Model
class ConvAutoencoder(nn.Module):    #convolutional autoencoder
    def __init__(self, dropoutRate = 0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),    #input = 3*256*256
            nn.ReLU(),
            nn.Dropout(dropoutRate),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Dropout(dropoutRate),
            
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),    #latent space = 16*32*32
            nn.ReLU(),

        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    #input = 16*32*32
            nn.ReLU(),
            nn.Dropout(dropoutRate),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.Dropout(dropoutRate),
            
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    #output = 3*256*256
            nn.Sigmoid()  # Ensures values are between 0-1
        )
        
    def forward(self, a):
        encode = self.encoder(a)
        decode = self.decoder(encode)
        return decode
    

# Hyperparameter tuning
def hyperparameterTuning(trainData, testData, parameterGrid):
    bestLoss = float('inf')
    bestParams = None

    for params in ParameterGrid(parameterGrid):
        print(f"Testing {params}")
        model = ConvAutoencoder(dropoutRate=params['dropoutRate'])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learningRate'])

        # Training
        model.train()
        for epoch in range(5):  # Small number of epochs for tuning
            epochLoss = 0
            for images in trainData:
                optimizer.zero_grad()
                output = model(images.unsqueeze(0))
                loss = criterion(output, images.unsqueeze(0))
                loss.backward()
                optimizer.step()
                epochLoss += loss.item()

        # Validation
        model.eval()
        valLoss = 0
        with torch.no_grad():
            for images in testData:
                output = model(images.unsqueeze(0))
                loss = criterion(output, images.unsqueeze(0))
                valLoss += loss.item()

        avgValLoss = valLoss / len(testData)
        print(f"Validation Loss: {avgValLoss}")

        if avgValLoss < bestLoss:
            bestLoss = avgValLoss
            bestParams = params

    print(f"Best Parameters: {bestParams}, Best Loss: {bestLoss}")
    return bestParams

#Training
def trainModel(model, trainData, testData, epochs=100, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainLosses = []
    valLosses = []
    printEvery = 5
    counter = 0

    for epoch in range(epochs):
        model.train()
        epochLoss = 0
        for images in trainData:
            optimizer.zero_grad()
            output = model(images.unsqueeze(0))
            loss = criterion(output, images.unsqueeze(0))
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
        
        trainLosses.append(epochLoss / len(trainData))

        # Validation
        model.eval()
        valLosses = 0
        with torch.no_grad():
            for images in testData:
                output = model(images.unsqueeze(0))
                loss = criterion(output, images.unsqueeze(0))
                valLosses += loss.item()
        
        val_losses.append(valLosses / len(testData))

        counter += 1
        if counter == printEvery:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {trainLosses[-1]:.6f}, Val Loss: {valLosses[-1]:.6f}")
            counter = 0

    return trainLosses, valLosses


def saveModel(model, path="autoencoder.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")

def loadModel(dropoutRate=0.2, path="autoencoder.pth"):
    model = ConvAutoencoder(dropoutRate=dropoutRate)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

#Graph creation
def visualizeReconstruction(model, testData, startIndex=0):
    model.eval()
    fig, axes = plt.subplots(2, 7, figsize=(30, 10))

    for i in range(7):
        idx = startIndex + i
        image = testData[idx]
        reconstructed = model(image.unsqueeze(0))
        reconstructed = reconstructed.squeeze(0).detach().numpy()

        # Original
        axes[0, i].imshow(image.permute(1, 2, 0).numpy())
        axes[0, i].axis('off')

        # Reconstructed
        axes[1, i].imshow(reconstructed.transpose(1, 2, 0))
        axes[1, i].axis('off')

    axes[0, 0].set_title('Original')
    axes[1, 0].set_title('Reconstructed')
    plt.show()

if __name__ == "__main__":
    modelFile = "autoencoder.pth"
    
    if not os.path.exists(modelFile):
        print("Training model...")
        trainingData, testingData = loadData()

        paramGrid = {
            "dropoutRate": [0.2, 0.3, 0.4],
            "learningRate": [1e-3, 1e-4, 1e-5]
        }
        besParams = hyperparameterTuning(trainingData, testingData, paramGrid)

        model = ConvAutoencoder(dropoutRate=besParams['dropoutRate'])
        train_losses, val_losses = trainModel(model, trainingData, testingData, epochs=100, lr=besParams['learningRate'])

        saveModel(model, modelFile)

        # Plot losses
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.show()

    else:
        print("Loading existing model...")
        model = loadModel(dropoutRate=0.2, path=modelFile)

    _, testingData = loadData()

    visualizeReconstruction(model, testingData, startIndex=7)