# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
The problem is to design and develop a Convolutional Deep Neural Network (CNN) that can automatically classify grayscale images into predefined categories. The model must learn important spatial features such as edges, textures, and shapes from image data and accurately predict the correct class label.

##  NEURAL NETWORK DIAGRAM
<img width="933" height="646" alt="image" src="https://github.com/user-attachments/assets/305aa599-e7f5-4ce5-bd1c-986bb93b230f" />



## DESIGN STEPS
### STEP 1: 

Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.
### STEP 2: 
Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

### STEP 3: 
Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

### STEP 4: 
Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.

### STEP 5: 
Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

### STEP 6: 
Evaluate the trained model on test images and verify the classification accuracy for new unseen images



## PROGRAM

### Name:

### Register Number:

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32, kernel_size=3, padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, padding=1)
        self.pool=nn.MaxPool2d (kernel_size=2, stride=2)
        self.fc1=nn.Linear (128*3*3,128)
        self.fc2=nn.Linear (128,64)
        self.fc3=nn.Linear (64,10)

    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)


# Train the Model
def train_model(model, train_loader, num_epochs=3):
   for epoch in range (num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

      print('Name:Sangeetha S')
      print('Register Number:212224040287')
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch

<img width="842" height="309" alt="image" src="https://github.com/user-attachments/assets/f19213b0-13a0-4080-993a-d4a1df7e1f2a" />

## Confusion Matrix

<img width="797" height="589" alt="image" src="https://github.com/user-attachments/assets/360542c6-6397-4f70-8668-0503d6259e05" />

## Classification Report
<img width="576" height="308" alt="image" src="https://github.com/user-attachments/assets/27b5a9f8-6d5b-4c25-be06-d0f0198f9e17" />

### New Sample Data Prediction
<img width="637" height="444" alt="image" src="https://github.com/user-attachments/assets/2a672c2c-9dd6-44ba-9cee-3209903738d7" />

## RESULT
The Convolutional Neural Network (CNN) model was successfully trained and achieved good classification performance on the given image dataset.

