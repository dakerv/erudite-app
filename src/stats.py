import torch

for epoch in range(1, 16):
    checkpoint = torch.load(
        f"models (experiment two)/efficientnet_b0_epoch_{epoch}.pth",
        map_location="cpu"
    )

    print(f"\nEpoch {epoch}")
    print(f"Training Loss: {checkpoint['train_loss']:.4f}")
    print(f"Training Accuracy: {checkpoint['train_accuracy']:.2f}%")
    print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"Validation Accuracy: {checkpoint['val_accuracy']:.2f}%")