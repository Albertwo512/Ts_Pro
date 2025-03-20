import os

# =======================
# Función para cambiar las etiquetas en los nombres de los archivos en orden secuencial
# =======================
def cambiar_etiquetas_secuenciales(mel_dir, nueva_etiqueta):
    """
    Cambia los nombres de los archivos .npy en una carpeta asignándoles un nuevo número de etiqueta
    y un número secuencial.

    :param mel_dir: Ruta de la carpeta con los archivos .npy
    :param nueva_etiqueta: Nuevo número de etiqueta (int) que se usará en los nuevos nombres
    """
    archivos = [f for f in os.listdir(mel_dir) if f.endswith('.npy')]  # Filtrar archivos .npy
    archivos.sort()  # Ordenar para mantener una numeración lógica
    
    for idx, filename in enumerate(archivos, start=1):
        nuevo_nombre = f"{nueva_etiqueta}_{idx}.npy"  # Crear nuevo nombre con etiqueta y número secuencial
        
        old_path = os.path.join(mel_dir, filename)
        new_path = os.path.join(mel_dir, nuevo_nombre)

        os.rename(old_path, new_path)  # Renombrar el archivo
        print(f"✅ {filename} → {nuevo_nombre}")

# =======================
# Parámetros
# =======================
mel_dir = "Numero frances dataset/mel_2"  # Ruta de la carpeta con los archivos
nueva_etiqueta = 2  # Número que deseas asignar como etiqueta

# =======================
# Ejecutar la función
# =======================
cambiar_etiquetas_secuenciales(mel_dir, nueva_etiqueta)
