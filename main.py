import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 1
batch_size = 100
learning_rate = 0.001

# MNIST dataset
'''
train_dataset = torchvision.datasets.MNIST(root='../../data/',
										   train=True, 
										   transform=transforms.ToTensor(),
										   download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
										  train=False, 
										  transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size, 
										   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										  batch_size=batch_size, 
										  shuffle=False)
'''                               

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
	def __init__(self, num_classes=1):
		super(ConvNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=4, stride=2,padding=1))
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
			nn.MaxPool2d(kernel_size=4,stride=2,padding=1))
		self.layer3 = nn.Sequential(
			nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=4,stride=2,padding=1)
			)
		self.layer4 = nn.Sequential(
			nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
			nn.ReLU(),
			nn.Conv2d(512,2,kernel_size=1,stride=1,padding=0),
			nn.Upsample(size=(256,256),mode='bilinear'))
		
	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		return out

model = ConvNet(num_classes).to(device)
print(model)
img = cv2.imread('31.jpg')
img = img[:,:,:3]
img = cv2.resize(img,(256,256))
img = np.moveaxis(img, -1, 0)
img = np.expand_dims(img,axis=0)
img = torch.from_numpy(img)
img = img.type('torch.FloatTensor')
output = model(img)
print(output.shape)
print(type(output))
'''

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)
		
		# Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)
		
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if (i+1) % 100 == 0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
				   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
'''