import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Dataset para Mel-Spectrograma (con tres clases)
class MelSpectrogramDataset(Dataset):
    def __init__(self, mel_dir):
        # Obtener todos los archivos .npy en la carpeta
        self.mel_files = [os.path.join(mel_dir, f) for f in os.listdir(mel_dir) if f.endswith('.npy')]
        # Extraer etiquetas de los nombres de los archivos
        self.labels = [int(f.split("_")[0]) for f in os.listdir(mel_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        mel_spec = np.load(self.mel_files[idx])  # Cargar el espectrograma
        label = self.labels[idx]  # Obtener la etiqueta
        return torch.tensor(mel_spec, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Modelo ResNet18
def create_resnet18_model(input_channels=1, num_classes=3):  # Usando 3 clases
    model = models.resnet18(weights=None)  # Sin pesos preentrenados
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Cambiar la salida a 3 clases
    return model

# Función de entrenamiento
def train_model(model, train_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()  # Usamos CrossEntropyLoss para clasificación
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for mel_spec, labels in train_loader:
            mel_spec, labels = mel_spec.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(mel_spec.unsqueeze(1))  # Añadimos la dimensión del canal (1, C, H, W)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Cargar el modelo previamente entrenado
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"Modelo cargado correctamente desde {model_path}")
    return model

# Cargar y preparar los datos
mel_dir_train = "Numero frances dataset"  # Carpeta donde están los archivos .npy (clases 0, 1, 2)
dataset_train = MelSpectrogramDataset(mel_dir_train)
train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)

# Crear el modelo
model = create_resnet18_model(input_channels=1, num_classes=3)

# Cargar el modelo previamente entrenado con las clases 0 y 1
model_path = "Ts_Pro/R-M-v0.pth"  # Modelo previamente entrenado con clases 0 y 1
model = load_model(model, model_path)

# Reentrenar el modelo solo con la clase 2 (y asegurar que no pierda conocimiento de las clases anteriores)
train_model(model, train_loader, num_epochs=50)

# Guardar el modelo después del reentrenamiento
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Modelo guardado en {filepath}")

save_model(model, "R-M-FINAL-v2.pth")
