import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def test_model(model, test_loader, device='cuda'):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.unsqueeze(1).float().to(device)
            outputs = model(images)
            predicted = (outputs > 0).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    all_labels = [int(x) for x in all_labels]
    all_predictions = [int(x) for x in all_predictions]

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    auc_roc = roc_auc_score(all_labels, all_predictions)

    return accuracy, precision, recall, f1, auc_roc
