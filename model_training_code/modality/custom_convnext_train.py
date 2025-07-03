# ================== IMPORTS ===================
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import pydicom

warnings.filterwarnings("ignore")


class DicomDataset(Dataset):
    """
    DICOM formatındaki görüntüleri okuyup, sınıf etiketleriyle birlikte veri seti oluşturan Dataset.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(".dcm"):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)

        img -= img.min()
        img /= (img.max() + 1e-5)
        img = np.stack([img] * 3, axis=-1)

        img = Image.fromarray((img * 255).astype(np.uint8))

        if self.transform:
            img = self.transform(img)

        return img, label


# ================== CONFIG ===================
run_name = "experiment"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri seti dizinlerini kendi yapına göre ayarla
train_dir = "path/to/train_directory"
test_dir = "path/to/test_directory"

# Veri ön işleme dönüşümleri
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = DicomDataset(train_dir, transform=transform)
test_dataset = DicomDataset(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)


class CustomConvNeXt(nn.Module):
    """
    ConvNeXt base backbone + özel classifier katmanı.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model("convnext_base", pretrained=True, num_classes=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.global_avg_pool(x)             
        x = x.view(x.size(0), -1)               
        return self.classifier(x)


class FocalLoss(nn.Module):
    """
    Focal Loss fonksiyonu, dengesiz veri için sınıf ağırlıkları ile birlikte kullanılabilir.
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean', device='cpu'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.device = device

        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(self.alpha, dtype=torch.float).to(device)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # Doğru sınıfa ait tahmin olasılığı
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ================== TRAIN SETUP ===================
# Sınıf ağırlıklarını dengele
all_labels = [label for _, label in train_dataset.samples]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

model = CustomConvNeXt(num_classes=num_classes).to(device)
criterion = FocalLoss(alpha=class_weights, gamma=2, device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# ================== TRAIN LOOP ===================
best_f1 = 0
train_losses, val_losses = [], []
train_f1s, val_f1s = [], []
train_accs, val_accs = [], []

for epoch in range(1, 51):
    print(f"\nEpoch {epoch}")
    model.train()
    running_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    train_f1s.append(train_f1)

    model.eval()
    val_loss, correct, total = 0, 0, 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(test_loader)
    val_acc = correct / total
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_f1s.append(val_f1)

    print(f"Train Loss: {train_loss:.6f}, Acc: {train_acc:.6f}, F1: {train_f1:.6f}")
    print(f"Val   Loss: {val_loss:.6f}, Acc: {val_acc:.6f}, F1: {val_f1:.6f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), f"{run_name}_best_model.pth")
        print("Best model saved.")

# Son epoch modeli kaydet
torch.save(model.state_dict(), f"{run_name}_last_epoch_model.pth")


# ================== EVALUATION ===================
def plot_confusion_and_roc(model_path, loader, dataset, split_name):
    """
    Modelin tahmin sonuçları üzerinden karışıklık matrisi ve ROC eğrilerini çizip kaydeder.
    """
    model = CustomConvNeXt(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    classes = dataset.classes
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.title(f"{split_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{run_name}_{split_name}_confusion_{os.path.basename(model_path).split('.')[0]}.png")
    plt.close()

    # ROC eğrisi hesaplama
    all_labels_np = np.array(all_labels)
    all_probs_np = np.array(all_probs)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve((all_labels_np == i).astype(int), all_probs_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{split_name} ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"{run_name}_{split_name}_roc_{os.path.basename(model_path).split('.')[0]}.png")
    plt.close()

    print(f"\n=== {split_name.upper()} - {os.path.basename(model_path)} ===")
    print(classification_report(all_labels, all_preds, target_names=classes))


# --- En iyi modeli değerlendir ---
plot_confusion_and_roc(f"{run_name}_best_model.pth", train_loader, train_dataset, "train")
plot_confusion_and_roc(f"{run_name}_best_model.pth", test_loader, test_dataset, "test")

# --- Son epoch modelini değerlendir ---
plot_confusion_and_roc(f"{run_name}_last_epoch_model.pth", train_loader, train_dataset, "train")
plot_confusion_and_roc(f"{run_name}_last_epoch_model.pth", test_loader, test_dataset, "test")

print("\n✅ Training and Evaluation Completed.")