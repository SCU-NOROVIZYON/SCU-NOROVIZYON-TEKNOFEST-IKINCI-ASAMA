import os
import torch
import torch.nn as nn
import timm
import pydicom
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import itertools
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomConvNeXt(nn.Module):
    """
    ConvNeXt backbone, özellik çıkarımı için.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("convnext_base", pretrained=False, num_classes=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)


class MLP(nn.Module):
    """
    Özelliklerden sınıf tahmini yapan Çok Katmanlı Algılayıcı (MLP).
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
    Vaka klasörlerinden, modaliteler bazında DICOM görüntüleri yükleyen Dataset.
    Modaliteler: 'adc', 'dwi', 't2a', 'ct'
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for vaka in os.listdir(cls_path):
                vaka_path = os.path.join(cls_path, vaka)
                if not os.path.isdir(vaka_path):
                    continue
                modalite_dict = {}
                for mod in ['adc', 'dwi', 't2a', 'ct']:
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
                    img = np.stack([img] * 3, axis=-1)

                img = Image.fromarray((img * 255).astype(np.uint8))
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            images_per_modality[mod] = torch.stack(imgs)

        return images_per_modality, label


def collate_fn(batch):
    """
    Batch için özel collate function.
    """
    modalite_dicts = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return modalite_dicts, labels


def load_feature_extractors():
    """
    Önceden eğitilmiş ConvNeXt modellerini modalitelere göre yükler.
    """
    modalities = ['adc', 'dwi', 't2a', 'ct']
    feature_extractors = {}
    for mod in modalities:
        model = CustomConvNeXt().to(device)
        # Kendi modellerinizin yolunu buraya yazın
        model_path = f"path/to/models/{mod}_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.eval()
        feature_extractors[mod] = model
    return feature_extractors


def extract_features_for_combination(dataloader, models, combination):
    """
    Verilen modalite kombinasyonu için tüm veriden özellik çıkarımı yapar.
    """
    features, labels = [], []
    feat_dim = list(models.values())[0].backbone.num_features

    for modalite_dicts, label_batch in tqdm(dataloader, desc=f"Extracting features for {combination}"):
        for modalite_dict, label in zip(modalite_dicts, label_batch):
            vaka_feats = []
            for mod in combination:
                if mod in modalite_dict:
                    imgs = modalite_dict[mod].to(device)
                    with torch.no_grad():
                        feat = models[mod](imgs)
                        avg_feat = feat.mean(dim=0)  
                    vaka_feats.append(avg_feat.cpu())
                else:
                    vaka_feats.append(torch.zeros(feat_dim))
            features.append(torch.cat(vaka_feats))
            labels.append(label)
    return torch.stack(features), torch.tensor(labels)


def train_mlp(train_loader, feature_extractors, combination, num_classes, epochs=10, lr=1e-3):
    """
    Verilen modalite kombinasyonu için MLP modeli eğitir.
    """
    train_features, train_labels = extract_features_for_combination(train_loader, feature_extractors, combination)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)

    mlp_input_dim = train_features.shape[1]
    mlp_model = MLP(mlp_input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr)

    mlp_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = mlp_model(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        if epoch in [0, 4, 9]:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    save_path = f"mlp_model_{'_'.join(combination)}.pth"
    torch.save(mlp_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return mlp_model


def test_mlp(mlp_model, test_loader, feature_extractors, combination, test_dataset_classes):
    """
    Test verisi üzerinde MLP modelini değerlendirir ve sonuçları yazdırır.
    """
    test_features, test_labels = extract_features_for_combination(test_loader, feature_extractors, combination)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    mlp_model.eval()
    with torch.no_grad():
        outputs = mlp_model(test_features)
        preds = torch.argmax(outputs, dim=1).cpu()

    print(f"\n=== Test Results for combination: {combination} ===")
    print("Classification Report:")
    print(classification_report(test_labels.cpu(), preds, target_names=test_dataset_classes, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels.cpu(), preds))


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Veri setlerinin dizinlerini kendi bilgisayarına göre ayarla
    train_dataset = VakaDataset("path/to/train", transform=transform)
    test_dataset = VakaDataset("path/to/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    feature_extractors = load_feature_extractors()

    modalities = ['adc', 'dwi', 't2a', 'ct']
    num_classes = len(train_dataset.classes)

    # Modalite kombinasyonları için MLP eğitim ve test döngüsü
    for r in range(2, len(modalities) + 1):
        for combination in itertools.combinations(modalities, r):
            print(f"\nTraining MLP for combination: {combination}")
            mlp_model = train_mlp(train_loader, feature_extractors, combination, num_classes, epochs=10, lr=1e-3)
            test_mlp(mlp_model, test_loader, feature_extractors, combination, test_dataset.classes)