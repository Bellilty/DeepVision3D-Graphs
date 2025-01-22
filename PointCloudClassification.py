import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale, Compose
from torch_geometric.loader import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import random


seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


path = 'ModelNet10'
transform = Compose([SamplePoints(1024), NormalizeScale()])
train_dataset = ModelNet(path, name='10', train=True, transform=transform)
test_dataset = ModelNet(path, name='10', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


class Tnet(nn.Module):
    def __init__(self, k=3):
        super(Tnet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3.weight.data.fill_(0)
        self.fc3.bias.data.copy_(torch.eye(k).view(-1))

    def forward(self, x):
        bs = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            init = init.cuda()
        x = x.view(-1, self.k, self.k) + init
        return x


class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNet, self).__init__()
        self.tnet1 = Tnet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.4)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        trans = self.tnet1(x)
        x = torch.bmm(torch.transpose(x, 1, 2), trans).transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return self.log_softmax(x)


def pointnet_loss(outputs, labels):
    criterion = nn.NLLLoss()
    return criterion(outputs, labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointNet(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


#os.makedirs('results', exist_ok=True)

def train(model, train_loader, optimizer, epochs=19):
    model.train()
    train_loss_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            if hasattr(data, 'pos') and hasattr(data, 'y'):
                inputs = data.pos.view(-1, 1024, 3).transpose(1, 2).to(device)
                labels = data.y.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = pointnet_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i % 25 == 1:
                    print(f'Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}, Loss: {running_loss / 10:.3f}')
                    running_loss = 0.0
        
        train_loss_history.append(running_loss / len(train_loader))

        #test(model, test_loader)

    #plt.figure(figsize=(10, 5))
    #plt.plot(range(1, epochs + 1), train_loss_history, label='Training Loss')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.title('Training Loss Over Epochs')
    #plt.legend()
    #plt.savefig('results/training_loss_graph.png')
    #plt.close()
    



def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    successful_samples = {i: [] for i in range(10)}
    failed_samples = {i: [] for i in range(10)}
    with torch.no_grad():
        for data in test_loader:
            if hasattr(data, 'pos') and hasattr(data, 'y'):
                inputs = data.pos.view(-1, 1024, 3).transpose(1, 2).to(device)
                labels = data.y.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
                
def test_and_save_samples(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    successful_samples = {i: [] for i in range(10)}
    failed_samples = {i: [] for i in range(10)}
    predictions = {i: [] for i in range(10)}  # Store predicted labels for failed samples

    with torch.no_grad():
        for data in test_loader:
            if hasattr(data, 'pos') and hasattr(data, 'y'):
                inputs = data.pos.view(-1, 1024, 3).transpose(1, 2).to(device)
                labels = data.y.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for i in range(inputs.size(0)):
                    label = labels[i].item()
                    if predicted[i] == labels[i]:
                        if len(successful_samples[label]) < 3:
                            successful_samples[label].append(inputs[i].cpu().numpy())
                    else:
                        if len(failed_samples[label]) < 3:
                            failed_samples[label].append(inputs[i].cpu().numpy())
                            predictions[label].append(predicted[i].item())  # Store predicted label

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    save_pointcloud_images(successful_samples, 'successful')
    save_pointcloud_images(failed_samples, 'failed', predictions)  # Pass predictions

    return accuracy

def save_pointcloud_images(pointcloud_dict, label, predictions=None):
    """Saves point cloud images for the report, with predicted labels for misclassified samples."""
    for class_label, clouds in pointcloud_dict.items():
        for idx, cloud in enumerate(clouds):
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            cloud = cloud.transpose(1, 0)  # Convert back to (num_points, num_features)
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=cloud[:, 2], cmap='viridis')
            
            if label == 'failed' and predictions and len(predictions[class_label]) > idx:
                pred_label = predictions[class_label][idx]
                ax.set_title(f'Failed Example {idx + 1} - True: {class_label}, Pred: {pred_label}')
            else:
                ax.set_title(f'{label.capitalize()} Example {idx + 1} - Label {class_label}')
                
            plt.savefig(f'results/{label}_example_label_{class_label}_{idx + 1}.png')
            plt.close()


train(model, train_loader, optimizer)
test(model, test_loader)
#test_accuracy = test_and_save_samples(model, test_loader)
