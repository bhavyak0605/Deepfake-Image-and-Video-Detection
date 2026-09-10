import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import classification_report

if __name__ == '__main__':

    # =====================================================
    # DEVICE
    # =====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    # =====================================================
    # DATASET PATH
    # =====================================================
    data_dir = r"C:\Users\Saqib\Downloads\Fake\Fake\dataset"
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")

    # =====================================================
    # DATA AUGMENTATION
    # =====================================================
    train_transform = transforms.Compose([

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(10),

        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),

        transforms.RandomGrayscale(p=0.1),

        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

        transforms.RandomPerspective(
            distortion_scale=0.2,
            p=0.3
        ),

        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)

    print("Train Images:", len(train_dataset))
    print("Val Images:", len(val_dataset))

    # =====================================================
    # DATALOADERS
    # =====================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    # =====================================================
    # MODEL
    # =====================================================
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    # Freeze early layers, train last blocks
    for param in model.features[:-3].parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(1280, 2)
    model = model.to(device)

    # =====================================================
    # LOSS + OPTIMIZER
    # =====================================================
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=2,
        factor=0.5
    )

    scaler = GradScaler()

    # =====================================================
    # CHECKPOINT SYSTEM
    # =====================================================
    start_epoch = 0
    best_val_acc = 0

    if os.path.exists("checkpoint.pth"):

        print("Loading checkpoint...")

        checkpoint = torch.load("checkpoint.pth", map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])

        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']

        print(f"Resuming from epoch {start_epoch}")

    elif os.path.exists("best_model.pth"):

        print("Loading best model weights...")

        model.load_state_dict(torch.load("best_model.pth", map_location=device))

        start_epoch = 1

        print("Continuing training from epoch 2")

    # =====================================================
    # TRAINING SETTINGS
    # =====================================================
    epochs = 3
    patience = 4
    counter = 0

    print(f"Total epochs already trained: {start_epoch}")

    # =====================================================
    # TRAINING LOOP
    # =====================================================
    for epoch in range(start_epoch, start_epoch + epochs):

        model.train()

        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type):

                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = outputs.max(1)

            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # =====================================================
        # VALIDATION
        # =====================================================
        model.eval()

        val_correct = 0
        val_total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for images, labels in tqdm(val_loader, desc="Validating"):

                images = images.to(device)
                labels = labels.to(device)

                with autocast(device_type=device.type):

                    outputs = model(images)

                _, preds = outputs.max(1)

                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100 * val_correct / val_total

        print(f"\nEpoch [{epoch+1}]")
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        scheduler.step(val_acc)

        # =====================================================
        # SAVE CHECKPOINT
        # =====================================================
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'best_val_acc': best_val_acc
        }, "checkpoint.pth")

        # =====================================================
        # SAVE BEST MODEL
        # =====================================================
        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(model.state_dict(), "best_model.pth")

            print("Best model updated.")

            counter = 0

        else:

            counter += 1

            if counter >= patience:

                print("Early stopping triggered.")
                break

    # =====================================================
    # FINAL REPORT
    # =====================================================
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds))

    print("\nTraining Complete.")