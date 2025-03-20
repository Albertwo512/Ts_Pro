import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# =======================
# Dataset para Mel-Spectrograma (solo con clase 0)
# =======================
class MelSpectrogramDataset(Dataset):
    def __init__(self, mel_dir, mel_dim=128, max_length=400):
        self.mel_files = [os.path.join(mel_dir, f) for f in os.listdir(mel_dir) if f.endswith('.npy') and '0' in f]
        self.labels = [0 for _ in range(len(self.mel_files))]  # Etiqueta 0 para todo el dataset

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        mel_spec = np.load(self.mel_files[idx])
        label = self.labels[idx]
        return torch.tensor(mel_spec, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# =======================
# Modelo ResNet18
# =======================
def create_resnet18_model(input_channels=1, num_classes=3):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Ajuste para 3 clases
    return model

# =======================
# Función de entrenamiento
# =======================
def train_model(model, train_loader, num_epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for mel_spec, labels in train_loader:
            mel_spec, labels = mel_spec.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(mel_spec.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# =======================
# Preparar los datos
# =======================
mel_dir_train = "Numero frances dataset/mel_0"  # Ruta a la carpeta de la clase 0
dataset = MelSpectrogramDataset(mel_dir_train)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# =======================
# Crear y entrenar el modelo
# =======================
model = create_resnet18_model(input_channels=1, num_classes=3)  # Cambiar número de clases si es necesario
train_model(model, train_loader, num_epochs=30)

# =======================
# Guardar el modelo
# =======================
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Modelo guardado en {filepath}")

save_model(model, 'R-M-v0.pth')  # Guardar el modelo entrenado
