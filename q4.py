import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchviz import make_dot
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)) 
])

class FNN(nn.Module):
    
    def __init__(self, input_size=784, hidden_layer=128, output=10):
        super(FNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, output)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        x = self.fc3(x)
        
        return x
    
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)




model = FNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

train_loss_history = []
test_loss_history = []
train_accuracy_history = []
test_accuracy_history = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = torch.max(outputs, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()
    
    train_loss = running_loss / len(train_dataset)
    train_loss_history.append(train_loss)
    train_accuracy = correct_train / total_train
    train_accuracy_history.append(train_accuracy)
    
    # Evaluate on test set
    model.eval()
    correct_test = 0
    total_test = 0
    test_running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total_test += targets.size(0)
            correct_test += (predicted == targets).sum().item()
    
    test_loss = test_running_loss / len(test_dataset)
    test_loss_history.append(test_loss)
    test_accuracy = correct_test / total_test
    test_accuracy_history.append(test_accuracy)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

dummy_input = torch.randn(1, 784)
output = model(dummy_input)
graph = make_dot(output, params=dict(model.named_parameters()))
graph.render("feedforward_graph")
graph.format = 'png'
graph.view()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_accuracy_history, label='Training Accuracy')
plt.plot(range(1, epochs+1), test_accuracy_history, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_loss_history, label='Training Loss')
plt.plot(range(1, epochs+1), test_loss_history, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

plt.tight_layout()
plt.show()