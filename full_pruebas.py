import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import models

# ==========================
# CONFIGURACIÓN
# ==========================
mel_dir_test = "Numero frances dataset/mel_Test_2"  # Carpeta de prueba
model_path = "R-M-FINAL-v1.pth"  # Modelo entrenado con 0, 1 y 2
batch_size = 16
num_classes = 3  # Ajusta según las clases usadas en el entrenamiento

# ==========================
# Dataset para Mel-Spectrograma
# ==========================
class MelSpectrogramDataset(Dataset):
    def __init__(self, mel_dir):
        self.mel_files = [os.path.join(mel_dir, f) for f in os.listdir(mel_dir) if f.endswith('.npy')]
        
        # Aquí agregamos un manejo de errores en la conversión de etiquetas
        self.labels = []
        for f in os.listdir(mel_dir):
            if f.endswith('.npy'):
                # Intentar extraer la etiqueta numérica
                try:
                    label = int(f.split("_")[0])  # Intenta convertir el prefijo del archivo en un número
                    self.labels.append(label)
                except ValueError:
                    print(f"⚠️ Advertencia: No se pudo convertir '{f.split('_')[0]}' a un número.")
                    self.labels.append(-1)  # Puedes asignar un valor predeterminado en caso de error

        # Verificación de etiquetas
        print(f"Primeras etiquetas: {self.labels[:10]}")  # Imprimir las primeras etiquetas para ver si están correctas

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
# Evaluar el modelo y generar métricas
# ==========================
def evaluar_modelo(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for mel_spec, labels in test_loader:
            mel_spec, labels = mel_spec.to(device), labels.to(device)
            outputs = model(mel_spec.unsqueeze(1))  # Asegúrate de agregar la dimensión del canal
            _, predicted = torch.max(outputs, 1)            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Precisión
    accuracy = 100 * sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)
    print(f"\n🎯 Precisión en el conjunto de prueba: {accuracy:.2f}%")

    # Reporte detallado
    print("\n📊 Reporte de clasificación:\n", classification_report(all_labels, all_preds, digits=4))

    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel("Predicciones")
    plt.ylabel("Etiquetas Reales")
    plt.title("Matriz de Confusión")
    plt.show()

# ==========================
# Cargar y evaluar el modelo
# ==========================
if __name__ == "__main__":
    # Cargar dataset de prueba
    test_dataset = MelSpectrogramDataset(mel_dir_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Cargar modelo entrenado
    print("\n📥 Cargando modelo entrenado...")
    model = create_resnet18_model(input_channels=1, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    print("✅ Modelo cargado correctamente.")

    # Evaluar el modelo
    print("\n🔬 Evaluando el modelo con datos de prueba...")
    evaluar_modelo(model, test_loader)
