import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import balanced_accuracy_score
import os

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 30
LR         = 3e-5          # lower than before
SAVE_PATH  = '/content/skin-classifier/models/best_model.pth'
PATIENCE   = 7             # early stopping patience

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), balanced_accuracy_score(all_labels, all_preds)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), balanced_accuracy_score(all_labels, all_preds)

def train(model, train_loader, val_loader, class_weights):
    model     = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc    = 0.0
    epochs_no_improve = 0
    history         = []

    print(f"Training on: {DEVICE}")
    print("-" * 55)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc':<14}{'Val Loss':<14}{'Val Acc'}")
    print("-" * 55)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc     = validate(model, val_loader, criterion)
        scheduler.step()

        history.append({
            'epoch'     : epoch + 1,
            'train_loss': train_loss,
            'train_acc' : train_acc,
            'val_loss'  : val_loss,
            'val_acc'   : val_acc
        })

        print(f"{epoch+1:<8}{train_loss:<14.4f}{train_acc:<14.4f}{val_loss:<14.4f}{val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"         ✓ Saved best model (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"         No improvement ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("-" * 55)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return history