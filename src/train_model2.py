
"""
EfficientNet-B0 Training Pipeline
---------------------------------

Trains the EfficientNet-B0 convolutional neural network using
transfer learning for three-class deepfake image classification.

Classes:
--------
- Real
- Swapped
- Synthetic

Pipeline:
---------
1. Load preprocessed dataset
2. Create EfficientNet-B0 model
3. Replace ImageNet classifier with three-class classifier
4. Train model
5. Validate model performance
6. Save best performing model
7. Save epoch checkpoints
8. Save recovery checkpoints

Author: Vanessa Daker
"""

import time
from pathlib import Path

import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from dataset_loader import create_dataloaders


# =================
# Configuration
# =================

DATASET_DIR = "dataset"

EXPERIMENT_DIR = Path("models (experiment two)")

MODEL_SAVE_PATH = EXPERIMENT_DIR / "efficientnet_b0.pth"

RESUME_CHECKPOINT_PATH = (
    EXPERIMENT_DIR / "training_checkpoint.pth"
)

CHECKPOINT_FREQUENCY = 50

# Change to True ONLY when recovering an interrupted run.
RESUME_TRAINING = False

BEST_VAL_ACCURACY = 0.0

BATCH_SIZE = 8

EPOCHS = 15

LEARNING_RATE = 0.0001

NUM_CLASSES = 3

DEVICE = "cpu"


# ==========================
# Create Model Directory
# ==========================

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =================
# Load the Dataset
# =================

train_loader, val_loader, test_loader = create_dataloaders(
    DATASET_DIR,
    BATCH_SIZE
)

print("\nDataset loaded successfully!")

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Testing batches: {len(test_loader)}")


# =================
# Model Creation
# =================

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(
    weights=weights
)

# EfficientNet-B0 normally predicts 1000 ImageNet classes.
# Replace the classifier with a three-class classifier.

model.classifier[1] = nn.Linear(
    1280,
    NUM_CLASSES
)

model.to(DEVICE)

print("\nModel created successfully!")


# ===========================
# Loss Function and Optimizer
# ===========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nLoss function and optimizer created successfully!")

print(f"Loss function: {criterion}")
print("Optimizer: Adam")
print(f"Learning rate: {LEARNING_RATE}")


# =================
# Training History
# =================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}


# ========================
# Resume Training Settings
# ========================

start_epoch = 0
start_batch = 0

resume_running_train_loss = 0.0
resume_correct_train = 0
resume_total_train = 0


# ==========================
# Continue From Previous Epoch
# ==========================

PREVIOUS_EPOCH_CHECKPOINT = (
    EXPERIMENT_DIR / "efficientnet_b0_epoch_12.pth"
)

if PREVIOUS_EPOCH_CHECKPOINT.exists():

    checkpoint = torch.load(
        PREVIOUS_EPOCH_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    BEST_VAL_ACCURACY = checkpoint["val_accuracy"]

    print(
        "\nPrevious Experiment 2 epoch checkpoint loaded."
    )

    print(
        f"Continuing from epoch {start_epoch + 1}"
    )

    print(
        f"Previous validation accuracy: "
        f"{BEST_VAL_ACCURACY:.2f}%"
    )

    # ==========================
    # Restore True Best Accuracy
    # ==========================

    if MODEL_SAVE_PATH.exists():

        best_checkpoint = torch.load(
            MODEL_SAVE_PATH,
            map_location=DEVICE
        )

        BEST_VAL_ACCURACY = (
            best_checkpoint["val_accuracy"]
        )

        print(
            "\nBest Experiment 2 model found."
        )

        print(
            f"Best validation accuracy restored: "
            f"{BEST_VAL_ACCURACY:.2f}%"
        )



# ==========================
# Load Recovery Checkpoint
# ==========================

if RESUME_TRAINING:

    checkpoint_path = Path(
        RESUME_CHECKPOINT_PATH
    )

    if checkpoint_path.exists():

        print("\nRecovery checkpoint found.")
        print("Loading checkpoint...")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"]

        start_batch = checkpoint["batch"]

        resume_running_train_loss = (
            checkpoint["running_train_loss"]
        )

        resume_correct_train = (
            checkpoint["correct_train"]
        )

        resume_total_train = (
            checkpoint["total_train"]
        )

        BEST_VAL_ACCURACY = (
            checkpoint["best_val_accuracy"]
        )

        history = checkpoint["history"]

        print("\nCheckpoint loaded successfully!")

        print(
            f"Resuming from epoch "
            f"{start_epoch + 1}"
        )

        print(
            f"Resuming from batch "
            f"{start_batch + 1}"
        )

    else:

        print(
            "\nNo recovery checkpoint found."
        )

        print(
            "Starting training from the beginning."
        )

else:

    print(
        "\nStarting Experiment 2 from the "
        "ImageNet-pretrained EfficientNet-B0."
    )


# =========================
# Training and Validation
# =========================

print("\nStarting training...")

for epoch in range(start_epoch, EPOCHS):

    print(
        f"\nStarting epoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    epoch_start_time = time.time()

    model.train()

    # =====================
    # Training Variables
    # =====================

    if (
        epoch == start_epoch
        and start_batch > 0
    ):

        running_train_loss = (
            resume_running_train_loss
        )

        correct_train = (
            resume_correct_train
        )

        total_train = (
            resume_total_train
        )

        print(
            f"Resuming from batch "
            f"{start_batch + 1}"
        )

    else:

        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

    start_time = time.time()


    # =====================
    # Training Loop
    # =====================

    for batch_index, (images, labels) in enumerate(
        train_loader
    ):

        # Skip batches already completed
        # before an interruption.

        if batch_index < start_batch:
            continue

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


        # =====================
        # Training Statistics
        # =====================

        running_train_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total_train += labels.size(0)

        correct_train += (
            predicted == labels
        ).sum().item()


        # =====================
        # Progress Display
        # =====================

        if (batch_index + 1) % 10 == 0:

            elapsed = (
                time.time() - start_time
            )

            print(
                f"Batch {batch_index + 1}/"
                f"{len(train_loader)} "
                f"| Loss: {loss.item():.4f} "
                f"| Time: {elapsed:.2f}s"
            )


        # =========================
        # Recovery Checkpoint
        # =========================

        if (
            (batch_index + 1)
            % CHECKPOINT_FREQUENCY
            == 0
        ):

            torch.save(
                {
                    "epoch": epoch,
                    "batch": batch_index + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "running_train_loss":
                        running_train_loss,

                    "correct_train":
                        correct_train,

                    "total_train":
                        total_train,

                    "best_val_accuracy":
                        BEST_VAL_ACCURACY,

                    "history":
                        history
                },
                RESUME_CHECKPOINT_PATH
            )

            print(
                f"\nRecovery checkpoint saved "
                f"at batch {batch_index + 1}"
            )


    # =====================
    # Training Results
    # =====================

    train_loss = (
        running_train_loss
        / len(train_loader)
    )

    train_accuracy = (
        100
        * correct_train
        / total_train
    )


    # =====================
    # Validation Phase
    # =====================

    model.eval()

    running_val_loss = 0.0

    correct_val = 0

    total_val = 0

    print("\nStarting validation...")


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_val_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total_val += labels.size(0)

            correct_val += (
                predicted == labels
            ).sum().item()


    # =====================
    # Validation Results
    # =====================

    val_loss = (
        running_val_loss
        / len(val_loader)
    )

    val_accuracy = (
        100
        * correct_val
        / total_val
    )


    # =====================
    # Store Epoch Results
    # =====================

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )


    # =====================
    # Save Every Epoch
    # =====================

    epoch_checkpoint_path = (
        EXPERIMENT_DIR
        / f"efficientnet_b0_epoch_{epoch + 1}.pth"
    )

    torch.save(
        {
            "epoch": epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "val_accuracy":
                val_accuracy,

            "val_loss":
                val_loss
        },
        epoch_checkpoint_path
    )

    print(
        f"\nEpoch checkpoint saved to:"
        f"\n{epoch_checkpoint_path}"
    )


    # =================
    # Save Best Model
    # =================

    if val_accuracy > BEST_VAL_ACCURACY:

        BEST_VAL_ACCURACY = val_accuracy

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_accuracy":
                    val_accuracy,

                "val_loss":
                    val_loss
            },
            MODEL_SAVE_PATH
        )

        print("\nNew best model saved!")

        print(
            f"Best validation accuracy: "
            f"{BEST_VAL_ACCURACY:.2f}%"
        )


    # =================
    # Epoch Summary
    # =================

    epoch_time = (
        time.time()
        - epoch_start_time
    )

    print(
        "\n" + "=" * 60
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] completed"
    )

    print(
        f"Training Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Training Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Epoch Time: "
        f"{epoch_time:.2f} seconds"
    )

    print(
        "=" * 60
    )


    # =========================
    # Reset Resume Position
    # =========================

    start_batch = 0

    resume_running_train_loss = 0.0

    resume_correct_train = 0

    resume_total_train = 0
