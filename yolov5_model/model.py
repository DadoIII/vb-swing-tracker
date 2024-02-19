import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
#from custom_dataset import CustomDataset  # Define your custom dataset class

print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_properties(0).total_memory)

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # load on CPU
model.cuda()  # GPU

# Find the last convolutional layer
last_conv_layer = None
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        last_conv_layer = module

# Check if the last convolutional layer is found
if last_conv_layer is not None:
    input_channels = last_conv_layer.in_channels
    print("Number of input channels for the last convolutional layer:", input_channels)
else:
    print("Last convolutional layer not found in the model.")



# Modify Model Architecture
# Replace the last layer(s) of the model with your custom output layer
# Example:
model.model[-1] = nn.Conv2d(in_channels=512, out_channels=6, kernel_size=1)

# Image
#im = '../images/unlabeled_images/car.webp'

# Inference
#results = model(im)

# results.pandas().xyxy[0]
# results.print()
# results.show()
# results.save()


# =========================================

# Step 3: Fine-tune the model
# Prepare dataset and dataloaders
# train_dataset = CustomDataset(train=True, transform=transforms.Compose([...]))  # Define your custom dataset class
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()  # Example loss function
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_dataloader:
#         images, labels = images.cuda(), labels.cuda()  # Move data to GPU
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader)}')

# Step 4: Evaluation
# Prepare validation dataset and dataloader
# val_dataset = CustomDataset(train=False, transform=transforms.Compose([...]))  # Define your custom dataset class
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Evaluate model on validation set
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in val_dataloader:
#         images, labels = images.cuda(), labels.cuda()  # Move data to GPU
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     accuracy = 100 * correct / total
#     print(f'Validation Accuracy: {accuracy}%')

# Step 5: Deployment
# Once satisfied with performance, deploy the model for inference