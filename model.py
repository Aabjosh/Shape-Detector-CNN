# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# initial work
BATCH_SIZE = 2
EPOCHS = 30
SAMPLE_DIMENSION = 64
DIRECTORY = "basicshapes/shapes"

# preprocessing of images to grayscale, same size, tensor format
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),  # this is critical
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_set = datasets.ImageFolder(root=DIRECTORY, transform=preprocess)
print(train_set.class_to_idx)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

# create CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(

            # parsing steps and their arguments below

            # Conv2d:                       number of input channels (grayscale = 1), 
            #                               output channels = number of 'feature maps' or details taken away, 
            #                               kernel size = size of window that is looking at pixels (side dimension) (larger = less detailed but contextual for example),
            #                               padding = adds pixels to meet the scale size (padding of 1 gives same as input size, 0 means no pixels added)
            #                               OPTIONAL: stride = number of pixels the kernel moves, used in MaxPool2d

            # ReLU (Rectified Linear Unit): ReLU(x)=max(0,x)
            #                               Takes any negative numbers and makes them '0'
            #                               makes the data non-linear so that the network isn't a linear function
            #                               Efficient, allows for 'complex pattern' training, avoids issues like the vanishing gradient problem

            # MaxPool2d:                    MaxPool2d performs downsampling by sliding a kernel window over the input feature map,
            #                               taking the maximum value within each window. This reduces the spatial dimensions
            #                               (height and width), while preserving the most important features.
            #                               The stride controls how much the kernel moves each step, effectively skipping pixels,
            #                               which results in a smaller feature map with concentrated, important info.
            #                               As the network progresses, the number of output channels (feature maps) often increases,
            #                               allowing the model to learn more complex features at lower resolutions.



            nn.Conv2d(1, 16, 3, padding=1),  # (B,16,64,64)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # (B,16,32,32) (moves across layers: 2x2 view, jumps every 2 pixels)

            nn.Conv2d(16, 32, 3, padding=1), # (B,32,32,32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # (B,32,16,16) (moves across layers: 2x2 view, jumps every 2 pixels)

            nn.Conv2d(32, 64, 3, padding=1), # (B,64,16,16)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # (B,64,8,8) (moves across layers: 2x2 view, jumps every 2 pixels)

            # flatten the network to be used as a one dimensional vector, nonlinearize with ReLU as before. 'Linear' is complex and involves the use of dot product (scalar projection of vectors, gives scalar that tells how much the vectors point in the same direction)
            # using: output = tensor * weights^T + bias
            nn.Flatten(),                    # (B, 64*8*8)
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 classes remain 
        )

    # call model when moving forward through the module subclass (CNN)
    def forward(self, x):
        return self.model(x)

# only run when in this file directly
if __name__ == "__main__":
        
    # loop for training the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("USING CUDA")
        
    model = CNN().to(device)

    # error loss function
    criterion = nn.CrossEntropyLoss()

    # optimizing for error
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    print("Training...")

    # used so we don't accidentally save overfitted and collapsed training data
    best_accuracy = 0

    # so we can visualize the training model (appending data)
    losses = []
    accuracies = []

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # reset the trained gradients from the previous batch (if existent), pass images through the module CNN
            optimizer.zero_grad()
            outputs = model(images)

            # calc the error, figure out how the weights contributed to loss, update with Adam
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 'item' converts from tensor to a plain number
            total_loss += loss.item()

            # don't take highest scores, take class indexes of highest score / guess
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total * 100
        
        # add losses and accuracy scores to be used in matplotlib after every epoch
        losses.append(total_loss)
        accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{EPOCHS} \nLoss: {total_loss:.4f}\nAccuracy: {accuracy:.2f}%")

        # ONLY SAVE THE MODEL IF ITS MORE ACCURATE THAN PREVIOUSLY
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), "shapeGuesser_cnn.pth")
            print("Model saved as: " + "shapeGuesser_cnn.pth")
            best_accuracy = accuracy
    

    print(model)

    # Visualize the training

    # metrics/datasets
    plt.plot(losses, label="LOSS")
    plt.plot(accuracies, label="ACCURACY (%)")

    # graph labels
    plt.xlabel(f"Epoch: Total Number = {EPOCHS}")
    plt.ylabel("Value")

    # display legend, graph
    plt.legend()
    plt.show()