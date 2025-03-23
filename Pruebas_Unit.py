import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ==========================
# CONFIGURACIÓN
# ==========================
mel_dir_test = "Numero frances dataset\mel_Test_1"  # Carpeta de prueba con los archivos .npy
model_path = "R-M-FINAL-v1.pth"  # Modelo entrenado con 0, 1 y 2
batch_size = 1  # Solo vamos a procesar un solo dato
num_classes = 3  # Ajusta según las clases usadas en el entrenamiento

# ==========================
# Dataset para Mel-Spectrograma
# ==========================
class MelSpectrogramDataset(Dataset):
    def __init__(self, mel_dir):
        self.mel_files = [os.path.join(mel_dir, f) for f in os.listdir(mel_dir) if f.endswith('.npy')]
        self.labels = [int(f.split("_")[0]) for f in os.listdir(mel_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        mel_spec = np.load(self.mel_files[idx])
        label = self.labels[idx]
        return torch.tensor(mel_spec, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ==========================
# Modelo ResNet18
# ==========================
def create_resnet18_model(input_channels=1, num_classes=3):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ==========================
# Realizar la prueba con un solo dato aleatorio
# ==========================
def test_random_sample(model, mel_dir_test):
    # Seleccionamos un archivo aleatorio de la carpeta
    random_file = random.choice([f for f in os.listdir(mel_dir_test) if f.endswith('.npy')])
    mel_sample_path = os.path.join(mel_dir_test, random_file)

    print(f"Archivo seleccionado para prueba: {mel_sample_path}")

    # Cargar el archivo Mel-Spectrograma
    mel_spec = np.load(mel_sample_path)
    mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
    
    # Asegúrate de agregar la dimensión del canal (para ResNet)
    mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)  # Forma: (1, 1, H, W)
    
    # Mover al dispositivo adecuado (GPU o CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    mel_spec = mel_spec.to(device)

    # Obtener la etiqueta real
    label = int(random_file.split("_")[0])  # La etiqueta se extrae del nombre del archivo

    # Hacer la predicción
    model.eval()
    with torch.no_grad():
        output = model(mel_spec)
        _, predicted = torch.max(output, 1)

    print(f"Etiqueta real: {label}")
    print(f"Predicción: {predicted.item()}")

    # Calcular y mostrar las métricas
    precision = precision_score([label], [predicted.item()], average='micro', zero_division=0)
    recall = recall_score([label], [predicted.item()], average='micro', zero_division=0)
    f1 = f1_score([label], [predicted.item()], average='micro', zero_division=0)
    accuracy = accuracy_score([label], [predicted.item()])

    # Mostrar las métricas en porcentaje
    print(f"\n🎯 Métricas de la predicción con el archivo seleccionado:")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return predicted.item()

# ==========================
# Cargar y evaluar el modelo
# ==========================
if __name__ == "__main__":
    # Cargar el modelo entrenado
    print("\n📥 Cargando modelo entrenado...")
    model = create_resnet18_model(input_channels=1, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    print("✅ Modelo cargado correctamente.")

    # Realizar la prueba con un archivo aleatorio
    print("\n🔬 Realizando prueba con un solo dato aleatorio...")
    predicted_label = test_random_sample(model, mel_dir_test)

    print(f"El modelo predijo la etiqueta: {predicted_label}")
