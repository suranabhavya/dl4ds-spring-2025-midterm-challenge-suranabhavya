import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import wandb
import json
from torch.optim.lr_scheduler import CosineAnnealingLR

class PretrainedModel(nn.Module):
    def __init__(self, num_classes=100):
        super(PretrainedModel, self).__init__()
        # Use ResNet50 as the base model with ImageNet1K_v2 weights
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Replace the classifier with a custom one for CIFAR-100
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    CONFIG = {
        "model": "ResNet50-Transfer",
        "batch_size": 256,  # Higher batch size for faster training
        "learning_rate": 0.001,  # Lower learning rate for fine-tuning
        "epochs": 50,  # More epochs for better convergence
        "num_workers": 16,
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
        "mixup_alpha": 0.2,  # Mixup augmentation strength
        "weight_decay": 1e-4,  # L2 regularization
        "label_smoothing": 0.1,  # Label smoothing factor
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    # Set seed for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    if CONFIG["device"] == "cuda":
        torch.cuda.manual_seed(CONFIG["seed"])
    
    # Advanced data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),  # Auto-augmentation strategy
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR-100 specific
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),  # Random erasing
    ])

    # No augmentation for validation/test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Data Loading
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                           download=True, transform=transform_train)
    
    # Split train into train and validation (90/10 split for more training data)
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    # Override transform for validation set
    valset.dataset.transform = transform_test
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=CONFIG["batch_size"], shuffle=True, 
        num_workers=CONFIG["num_workers"], pin_memory=True)
    
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=True)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=True)
    
    # Initialize model
    model = PretrainedModel(num_classes=100)
    model = model.to(CONFIG["device"])
    
    print("\nModel summary:")
    print(f"{model}\n")
    
    # Loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    
    # Two parameter groups: one for the base model (smaller lr) and one for the classifier (larger lr)
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': CONFIG["learning_rate"] * 0.1},
        {'params': model.base_model.fc.parameters()}
    ], lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    
    # Cosine annealing scheduler for better convergence
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=CONFIG["learning_rate"] / 100)
    
    # Initialize wandb
    wandb.login(key="acb1911b95fa503cdf084d582cc0a800c733b1cc")
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)
    
    # Training Loop
    best_val_acc = 0.0
    
    for epoch in range(CONFIG["epochs"]):
        # Progressively unfreeze layers as training progresses
        if epoch == 20:  # Unfreeze more layers after 20 epochs
            for param in model.base_model.layer4.parameters():
                param.requires_grad = True
        
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
            
    wandb.finish()
    
    # Evaluation
    import eval_cifar100
    import eval_ood
    
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood_part3.csv", index=False)
    print("submission_ood_part3.csv created successfully.")

if __name__ == '__main__':
    main()