<<<<<<< HEAD
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import timm
import pydicom
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomConvNeXt(nn.Module):
    """
    ConvNeXt Base model, feature extractor olarak kullanılır.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("convnext_base", pretrained=False, num_classes=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)

class MLP(nn.Module):
    """
    Özellik vektörleri üzerinde sınıflandırma için MLP.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.layers(x)

class VakaDataset(Dataset):
    """
    Vaka bazlı, modaliteye göre DICOM görüntüleri yükleyen dataset.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for vaka in os.listdir(cls_path):
                vaka_path = os.path.join(cls_path, vaka)
                if not os.path.isdir(vaka_path): 
                    continue
                modalite_dict = {}
                for mod in ['adc','dwi','ct','t2a']:
                    mod_path = os.path.join(vaka_path, mod)
                    if os.path.isdir(mod_path):
                        dcm_files = [os.path.join(mod_path, f) for f in os.listdir(mod_path) if f.lower().endswith('.dcm')]
                        if dcm_files:
                            modalite_dict[mod] = dcm_files
                if modalite_dict:
                    self.samples.append((modalite_dict, self.class_to_idx[cls]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        modalite_dict, label = self.samples[idx]
        images_per_modality = {}
        for mod, paths in modalite_dict.items():
            imgs = []
            for p in paths:
                ds = pydicom.dcmread(p)
                img = ds.pixel_array.astype(np.float32)
                img -= img.min()
                img /= (img.max() + 1e-5)
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                img = Image.fromarray((img*255).astype(np.uint8))
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            images_per_modality[mod] = torch.stack(imgs)
        return images_per_modality, label

def collate_fn(batch):
    modalite_dicts = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return modalite_dicts, labels

def extract_features(dataloader, models):
    features, labels = [], []
    for modalite_dicts, label_batch in tqdm(dataloader, desc="Extracting features"):
        for modalite_dict, label in zip(modalite_dicts, label_batch):
            vaka_feats = []
            for mod in ['adc','dwi','ct','t2a']:
                if mod in modalite_dict:
                    imgs = modalite_dict[mod].to(device)
                    with torch.no_grad():
                        feat = models[mod](imgs)
                        avg_feat = feat.mean(dim=0)
                    vaka_feats.append(avg_feat.cpu())
                else:
                    feat_dim = list(models.values())[0].backbone.num_features
                    vaka_feats.append(torch.zeros(feat_dim))
            features.append(torch.cat(vaka_feats))
            labels.append(label)
    return torch.stack(features), torch.tensor(labels)

def load_feature_extractors(model_dir):
    """
    Belirtilen dizinden modalite modellerini yükler.
    """
    modalities = ['adc','dwi','ct','t2a']
    feature_extractors = {}
    for mod in modalities:
        model_path = os.path.join(model_dir, f"{mod}_gap_focal_best_model.pth")
        model = CustomConvNeXt().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.eval()
        feature_extractors[mod] = model
    return feature_extractors

if __name__ == "__main__":
    # Ayarlanabilir parametreler:
    train_dir = "path/to/train"
    test_dir = "path/to/test"
    model_dir = "path/to/models"

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = VakaDataset(train_dir, transform=transform)
    test_dataset = VakaDataset(test_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    feature_extractors = load_feature_extractors(model_dir)

    print("Extracting train features...")
    train_features, train_labels = extract_features(train_loader, feature_extractors)

    mlp_input_dim = train_features.shape[1]
    num_classes = len(train_dataset.classes)
    mlp_model = MLP(mlp_input_dim, num_classes).to(device)

    all_labels = [label for _, label in train_dataset.samples]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-4)

    print("Extracting test features...")
    test_features, test_labels = extract_features(test_loader, feature_extractors)

    epochs = 5
    batch_size = 16
    train_data = TensorDataset(train_features, train_labels)
    train_loader_mlp = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        mlp_model.train()
        for batch_feats, batch_labels in train_loader_mlp:
            batch_feats = batch_feats.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = mlp_model(batch_feats)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mlp_model.eval()
        with torch.no_grad():
            val_outputs = mlp_model(test_features.to(device))
            val_preds = torch.argmax(val_outputs, dim=1).cpu()
            acc = accuracy_score(test_labels, val_preds)
            f1 = f1_score(test_labels, val_preds, average="macro")

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader_mlp):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    torch.save(mlp_model.state_dict(), "mlp_model_adc_dwi_ct_t2a.pth")
    print("Model saved as mlp_model_adc_dwi_ct_t2a.pth")
=======
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import timm
import pydicom
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomConvNeXt(nn.Module):
    """
    ConvNeXt Base model, feature extractor olarak kullanılır.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("convnext_base", pretrained=False, num_classes=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)

class MLP(nn.Module):
    """
    Özellik vektörleri üzerinde sınıflandırma için MLP.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.layers(x)

class VakaDataset(Dataset):
    """
    Vaka bazlı, modaliteye göre DICOM görüntüleri yükleyen dataset.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for vaka in os.listdir(cls_path):
                vaka_path = os.path.join(cls_path, vaka)
                if not os.path.isdir(vaka_path): 
                    continue
                modalite_dict = {}
                for mod in ['adc','dwi','ct','t2a']:
                    mod_path = os.path.join(vaka_path, mod)
                    if os.path.isdir(mod_path):
                        dcm_files = [os.path.join(mod_path, f) for f in os.listdir(mod_path) if f.lower().endswith('.dcm')]
                        if dcm_files:
                            modalite_dict[mod] = dcm_files
                if modalite_dict:
                    self.samples.append((modalite_dict, self.class_to_idx[cls]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        modalite_dict, label = self.samples[idx]
        images_per_modality = {}
        for mod, paths in modalite_dict.items():
            imgs = []
            for p in paths:
                ds = pydicom.dcmread(p)
                img = ds.pixel_array.astype(np.float32)
                img -= img.min()
                img /= (img.max() + 1e-5)
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                img = Image.fromarray((img*255).astype(np.uint8))
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            images_per_modality[mod] = torch.stack(imgs)
        return images_per_modality, label

def collate_fn(batch):
    modalite_dicts = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return modalite_dicts, labels

def extract_features(dataloader, models):
    features, labels = [], []
    for modalite_dicts, label_batch in tqdm(dataloader, desc="Extracting features"):
        for modalite_dict, label in zip(modalite_dicts, label_batch):
            vaka_feats = []
            for mod in ['adc','dwi','ct','t2a']:
                if mod in modalite_dict:
                    imgs = modalite_dict[mod].to(device)
                    with torch.no_grad():
                        feat = models[mod](imgs)
                        avg_feat = feat.mean(dim=0)
                    vaka_feats.append(avg_feat.cpu())
                else:
                    feat_dim = list(models.values())[0].backbone.num_features
                    vaka_feats.append(torch.zeros(feat_dim))
            features.append(torch.cat(vaka_feats))
            labels.append(label)
    return torch.stack(features), torch.tensor(labels)

def load_feature_extractors(model_dir):
    """
    Belirtilen dizinden modalite modellerini yükler.
    """
    modalities = ['adc','dwi','ct','t2a']
    feature_extractors = {}
    for mod in modalities:
        model_path = os.path.join(model_dir, f"{mod}_gap_focal_best_model.pth")
        model = CustomConvNeXt().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.eval()
        feature_extractors[mod] = model
    return feature_extractors

if __name__ == "__main__":
    # Ayarlanabilir parametreler:
    train_dir = "path/to/train"
    test_dir = "path/to/test"
    model_dir = "path/to/models"

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = VakaDataset(train_dir, transform=transform)
    test_dataset = VakaDataset(test_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    feature_extractors = load_feature_extractors(model_dir)

    print("Extracting train features...")
    train_features, train_labels = extract_features(train_loader, feature_extractors)

    mlp_input_dim = train_features.shape[1]
    num_classes = len(train_dataset.classes)
    mlp_model = MLP(mlp_input_dim, num_classes).to(device)

    all_labels = [label for _, label in train_dataset.samples]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-4)

    print("Extracting test features...")
    test_features, test_labels = extract_features(test_loader, feature_extractors)

    epochs = 5
    batch_size = 16
    train_data = TensorDataset(train_features, train_labels)
    train_loader_mlp = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        mlp_model.train()
        for batch_feats, batch_labels in train_loader_mlp:
            batch_feats = batch_feats.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = mlp_model(batch_feats)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mlp_model.eval()
        with torch.no_grad():
            val_outputs = mlp_model(test_features.to(device))
            val_preds = torch.argmax(val_outputs, dim=1).cpu()
            acc = accuracy_score(test_labels, val_preds)
            f1 = f1_score(test_labels, val_preds, average="macro")

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader_mlp):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    torch.save(mlp_model.state_dict(), "mlp_model_adc_dwi_ct_t2a.pth")
    print("Model saved as mlp_model_adc_dwi_ct_t2a.pth")
>>>>>>> bbb7611177ae268e5489c880ecb9f1d964437d2c
