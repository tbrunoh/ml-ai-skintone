from models.model import SkinToneCNNModel
from models.dataset import SkinToneDataset
from mlaicore.utils.utils import Utils
from torchvision import transforms
import torch as T
from models.loss_func import OrdinalRegressionLoss
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

loss_func = T.nn.CrossEntropyLoss()  # does log-softmax()
optimizer = T.optim.SGD(network.parameters(), lr=learning_rate, weight_decay=1e-4)

training_loader=T.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

print("\nbat_size = %3d " % batch_size)
print("loss = " + str(loss_func))
print("optimizer = SGD")
print("max_epochs = %3d " % epochs)
print("lrn_rate = %0.4f " % learning_rate)
print("\nStarting training")

network.train()
for epoch in range(0,epochs):
    epoch_loss=0
    for(batch_id,batch) in enumerate(training_loader):
        (images,label)=batch
        optimizer.zero_grad()
        label=label.to(device)
        images = images.permute(0, 1, 2, 3).float().to(device)  # Convert [batch_size, height, width, channels] to [batch_size, channels, height, width]
        # print(images.shape)
        
        output = network(images)
        loss = loss_func(output, label)
        epoch_loss+=loss.item()
        
        loss.backward() #gradient descent
        optimizer.step() #update weights
    if epoch % epoch_log_interval == 0:
        print("epoch = %4d loss = %0.4f" % (epoch, epoch_loss))
    print("Done ") 
        
        




