import torch

for epoch in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    checkpoint = torch.load(
        f"models (experiment two)/efficientnet_b0_epoch_{epoch}.pth",
        map_location="cpu"
    )

    print(f"\nEpoch {epoch}")
    print(f"\nTraining Loss: {checkpoint['train_loss']: .4f}")
    print(f"\nTraining Accuracy: {checkpoint['train_accuracy']:.2f}%")
    print(f"\nValidation Loss: {checkpoint['val_loss']: .4f}")
    print(f"\nValidation Accuracy: {checkpoint['val_accuracy']:.2f}%")