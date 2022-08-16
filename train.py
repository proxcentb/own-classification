import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Net
from torchsummary import summary
from tqdm import tqdm
from torchvision.transforms import Compose, Compose, ToTensor, RandomAffine, Grayscale
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from variables import batch_size, device, epochs, learning_rate, momentum_value, trainPath, testPath

trainTransforms = Compose([
	Grayscale(),
    RandomAffine(degrees=(-10, 10), translate=(0.2, 0.2), scale=(0.5, 1.1)), 
    ToTensor()
])
testTransforms = Compose([
	Grayscale(),
    RandomAffine(degrees=(-10, 10), translate=(0.2, 0.2), scale=(0.5, 1.1)), 
    ToTensor()
])

trainDataset = ConcatDataset([ImageFolder(root=trainPath, transform=trainTransforms) for i in range(0, 400)])
testDataset = ConcatDataset([ImageFolder(root=testPath, transform=testTransforms) for i in range(0, 400)])

train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testDataset, batch_size=batch_size)

labels = trainDataset.datasets[0].class_to_idx

model = Net(len(labels)).to(device)
summary(model, input_size=(1, 80, 80))

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'epoch: {epoch} loss={loss.item()} batch_id={batch_idx}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():   
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, train=False)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy Test: {}/{} ({:.1f}%)'.format(
        test_loss, 
        correct,
        len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    ))

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_value)

for epoch in range(0, epochs):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

torch.save(model, "model.pth")