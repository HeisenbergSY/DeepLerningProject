import torch
from torch import nn, optim
import torch.cuda.amp as amp
import copy

class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience  # Number of epochs to wait after the last improvement
        self.delta = delta  # Minimum change to qualify as an improvement
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.val_loss_min = val_loss
        self.best_model = copy.deepcopy(model.state_dict())

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, weight_decay=1e-4, device='cuda'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = amp.GradScaler()
    early_stopping = EarlyStopping(patience=3, delta=0.001)

    final_epoch = 0
    best_val_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.unsqueeze(1).float().to(device)
            optimizer.zero_grad()
            
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        val_loss = validate_model(model, val_loader, device, criterion)
        val_accuracy = validate_model_accuracy(model, val_loader, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%')

        final_epoch = epoch + 1
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(early_stopping.best_model)
    return final_epoch, best_val_accuracy  # Return the last epoch and best validation accuracy

def validate_model(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.unsqueeze(1).float().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def validate_model_accuracy(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.unsqueeze(1).float().to(device)
            outputs = model(images)
            predicted = (outputs > 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
