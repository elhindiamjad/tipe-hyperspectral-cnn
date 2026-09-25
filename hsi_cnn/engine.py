"""Training, evaluation and prediction loops."""
import numpy as np
import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Return (mean loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a labeled loader. Return (mean loss, accuracy, predictions, targets)."""
    model.eval()
    total_loss, preds, trues = 0.0, [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item() * inputs.size(0)
        preds.append(outputs.argmax(1).cpu().numpy())
        trues.append(targets.cpu().numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    return total_loss / len(trues), float((preds == trues).mean()), preds, trues


@torch.no_grad()
def predict(model, loader, device):
    """Predict classes for an unlabeled loader (batched, much faster than pixel by pixel)."""
    model.eval()
    return np.concatenate([model(x.to(device)).argmax(1).cpu().numpy() for x in loader])
