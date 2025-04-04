from models.model import SkinToneCNNModel
from models.dataset import SkinToneDataset
from mlaicore.utils.utils import Utils
from mlaicore.utils.train_utils import TrainUtils
from torchvision import transforms
import torch as T
from models.loss_func import OrdinalRegressionLoss

import time  # Import the time module for timing
from dotenv import load_dotenv
import os
load_dotenv()

device = Utils.set_accelerator()

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

train_dataset = SkinToneDataset(csv_file=os.getenv("TRAIN_DATASET_PATH"), transform=transform)


network = SkinToneCNNModel(num_classes=10,image_size=512,channels=3).to(device)
epochs = 50
learning_rate=0.0001
batch_size=32
epoch_log_interval=10
checkpoint_path=os.getenv("CHECKPOINT_PATH")

loss_func = T.nn.CrossEntropyLoss()  # does log-softmax()
optimizer = T.optim.SGD(network.parameters(), lr=learning_rate, weight_decay=1e-4)

training_loader=T.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

start_epoch = TrainUtils.load_checkpoint(checkpoint_path, network, optimizer)

print("\nbat_size = %3d " % batch_size)
print("loss = " + str(loss_func))
print("optimizer = SGD")
print("max_epochs = %3d " % epochs)
print("lrn_rate = %0.4f " % learning_rate)
print("\nStarting training")

network.train()

for epoch in range(0, epochs):
    epoch_loss = 0
    epoch_start_time = time.time()  # Start timing the epoch
    for (batch_id, batch) in enumerate(training_loader):
        batch_start_time = time.time()  # Start timing the batch
        (images, label) = batch
        optimizer.zero_grad()
        label = label.to(device)
        images = images.permute(0, 1, 2, 3).float().to(device)  # Convert [batch_size, height, width, channels] to [batch_size, channels, height, width]
        
        output = network(images)
        loss = loss_func(output, label)
        epoch_loss += loss.item()
        
        loss.backward()  # Gradient descent
        optimizer.step()  # Update weights
        
        batch_end_time = time.time()  # End timing the batch
        print(f"Batch {batch_id}, loss {loss:0.4f} completed in {batch_end_time - batch_start_time:.2f} seconds")
    
    epoch_end_time = time.time()  # End timing the epoch
    if epoch % epoch_log_interval == 0:
        print("epoch = %4d loss = %0.4f completed in %0.2f seconds" % (epoch, epoch_loss, epoch_end_time - epoch_start_time))
 
    TrainUtils.save_checkpoint(epoch, network, optimizer, epoch_loss, checkpoint_path)

# Save the trained model for inference
model_save_path = os.getenv("MODEL_SAVE_PATH", "skin_tone_model.pth")  # Default path if not set in environment
T.save(network.state_dict(), model_save_path)
print(f"Model saved for inference at {model_save_path}")

print("Done")
 
# Add test dataset and loader
test_dataset = SkinToneDataset(csv_file=os.getenv("TEST_DATASET_PATH"), transform=transform)
test_loader = T.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
       
# Test loop
network.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with T.no_grad():  # Disable gradient computation for testing
    for (images, label) in test_loader:
        label = label.to(device)
        images = images.permute(0, 1, 2, 3).float().to(device)

        output = network(images)
        _, predicted = T.max(output, 1)  # Get the class with the highest score
        total += label.size(0)
        correct += (predicted == label).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")



