
import os
import random
import json
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =========================================================
# 1. CAU HINH
# =========================================================

SEED = 42
EPOCHS = 50
BATCH_SIZE = 16          # GTX 1060 3GB: bat dau voi batch size nho
LEARNING_RATE = 0.001
IMAGE_SIZE = 96
NUM_CLASSES = 10
NUM_WORKERS = 0          # Phu hop voi Windows
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("ALEXNET - FASHION-MNIST")
print("=" * 60)
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
else:
    print("CUDA khong kha dung, dang chay tren CPU.")

print(f"So epoch: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Kich thuoc anh: {IMAGE_SIZE} x {IMAGE_SIZE}")
print("=" * 60)


# =========================================================
# 2. DATA AUGMENTATION VA TIEN XU LY DU LIEU
# =========================================================

# Fashion-MNIST co anh xam 28x28.
# AlexNet can kich thuoc lon hon, nen resize len 96x96.
# Chuyen anh xam thanh 3 kenh de phu hop voi kien truc AlexNet.

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])


# =========================================================
# 3. TAI DATASET FASHION-MNIST
# =========================================================

print("\nDang tai/kiem tra Fashion-MNIST...")

train_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=test_transform
)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print(f"So anh train: {len(train_dataset)}")
print(f"So anh test: {len(test_dataset)}")
print(f"So lop: {NUM_CLASSES}")


# =========================================================
# 4. XAY DUNG MO HINH ALEXNET
# =========================================================

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Input: 3 x 96 x 96
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=11,
                stride=4,
                padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(
                in_channels=64,
                out_channels=192,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(
                in_channels=192,
                out_channels=384,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Adaptive pooling dam bao dau vao classifier co kich thuoc co dinh.
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


model = AlexNet(num_classes=NUM_CLASSES).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTong so tham so mo hinh: {total_params:,}")


# =========================================================
# 5. HAM LOSS VA OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=15,
    gamma=0.5
)


# =========================================================
# 6. HAM HUAN LUYEN MOT EPOCH
# =========================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True
        )
        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# =========================================================
# 7. HAM DANH GIA
# =========================================================

def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(
                device,
                non_blocking=True
            )
            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += batch_size

            for i in range(NUM_CLASSES):
                mask = labels == i
                class_total[i] += mask.sum().item()
                class_correct[i] += (
                    (predictions == labels) & mask
                ).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    class_accuracy = {}
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            class_accuracy[name] = (
                100.0 * class_correct[i] / class_total[i]
            )
        else:
            class_accuracy[name] = 0.0

    return epoch_loss, epoch_accuracy, class_accuracy


# =========================================================
# 8. VONG LAP HUAN LUYEN 50 EPOCH
# =========================================================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
    "learning_rate": []
}

best_accuracy = 0.0
best_epoch = 0
start_time = time.time()

print("\nBat dau huan luyen...\n")

try:
    for epoch in range(EPOCHS):
        epoch_start = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        test_loss, test_accuracy, _ = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        history["learning_rate"].append(current_lr)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.2f}% | "
            f"Time: {epoch_time / 60:.1f} phut"
        )

        # Luu mo hinh tot nhat theo test accuracy.
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_accuracy": best_accuracy,
                    "class_names": class_names
                },
                os.path.join(OUTPUT_DIR, "best_model.pth")
            )

            print(
                f"  -> Da luu best_model.pth "
                f"(Test Acc: {best_accuracy:.2f}%)"
            )

        # Luu tien do sau moi epoch de tranh mat lich su.
        with open(
            os.path.join(OUTPUT_DIR, "history.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(history, f, indent=4)

except KeyboardInterrupt:
    print("\nDa dung huan luyen theo yeu cau.")
    print("Lich su cac epoch da hoan thanh duoc luu trong outputs/history.json.")


# =========================================================
# 9. LUU MO HINH CUOI CUNG
# =========================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "epochs_completed": len(history["train_loss"])
    },
    os.path.join(OUTPUT_DIR, "last_model.pth")
)


# =========================================================
# 10. VE BIEU DO
# =========================================================

completed_epochs = len(history["train_loss"])

if completed_epochs > 0:
    epochs_range = range(1, completed_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        epochs_range,
        history["train_loss"],
        label="Train Loss"
    )
    plt.plot(
        epochs_range,
        history["test_loss"],
        label="Test Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        epochs_range,
        history["train_accuracy"],
        label="Train Accuracy"
    )
    plt.plot(
        epochs_range,
        history["test_accuracy"],
        label="Test Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "training_curves.png"),
        dpi=150
    )
    plt.close()

    # Danh gia theo tung lop bang mo hinh hien tai.
    _, final_accuracy, per_class_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    results = {
        "epochs_completed": completed_epochs,
        "best_epoch": best_epoch,
        "best_test_accuracy": best_accuracy,
        "final_test_accuracy": final_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "device": str(device)
    }

    with open(
        os.path.join(OUTPUT_DIR, "results.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("KET THUC")
    print("=" * 60)
    print(f"So epoch da hoan thanh: {completed_epochs}/{EPOCHS}")
    print(f"Epoch co test accuracy tot nhat: {best_epoch}")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print(f"Final test accuracy: {final_accuracy:.2f}%")
    print(f"Tong thoi gian: {(time.time() - start_time) / 60:.1f} phut")

    print("\nDo chinh xac theo tung lop:")
    for name, accuracy in per_class_accuracy.items():
        print(f"  {name}: {accuracy:.2f}%")

    print(f"\nKet qua duoc luu tai: {os.path.abspath(OUTPUT_DIR)}")
    print(" - best_model.pth")
    print(" - last_model.pth")
    print(" - history.json")
    print(" - results.json")
    print(" - training_curves.png")

else:
    print("\nChua hoan thanh epoch nao.")
    print("Hay chay lai train.py de bat dau huan luyen.")
