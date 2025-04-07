import os
import random
import shutil

# Función para separar el 20% de los archivos aleatoriamente
def separate_random_data(input_folder, output_folder, percentage=0.2):
    # Asegurarse de que la carpeta de salida exista
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Obtener todos los archivos en la carpeta de entrada
    all_files = [file for file in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, file))]
    
    # Calcular cuántos archivos se deben seleccionar (20% de todos los archivos)
    num_files_to_move = int(len(all_files) * percentage)
    
    # Seleccionar aleatoriamente el 20% de los archivos
    files_to_move = random.sample(all_files, num_files_to_move)
    
    # Mover los archivos seleccionados a la carpeta de salida
    for file_name in files_to_move:
        source_path = os.path.join(input_folder, file_name)
        destination_path = os.path.join(output_folder, file_name)
        shutil.move(source_path, destination_path)
        print(f"Archivo movido: {file_name}")
    
    print(f"Se movieron {num_files_to_move} archivos aleatorios.")

# Ruta de la carpeta de entrada y la carpeta de salida
input_folder = 'mels'  # Carpeta que contiene los archivos
output_folder = 'pruebas'  # Carpeta donde se moverán los archivos

# Separar el 20% de los archivos aleatoriamente
separate_random_data(input_folder, output_folder, percentage=0.2)
