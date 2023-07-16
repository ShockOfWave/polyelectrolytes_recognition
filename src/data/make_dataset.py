import os
import numpy as np
import pandas as pd
from rich.progress import track
from rich.markdown import Markdown
from rich.console import Console
import cv2
from PIL import Image

from src.paths.paths import get_project_path
from src.barcode.logic import list_txt_files, list_csv_files
from src.barcode.convert import matrix_convert_save
from src.barcode.AFM_analize_folder import minmax_db, autocorr_function, persistance_db

from src.bottleneck.bottleneck_final import calc_bottlenecks

def create_afm_image_from_csv(list_of_image_paths: list):
    for image_path in track(list_of_image_paths, description='Creating images from AFM csv...'):
        df = pd.read_csv(image_path)
        df = df.drop(df.columns[0], axis=1)
        image = Image.fromarray(df.values.astype(np.uint8))
        
        if image.mode != 'RGB':
            image.convert('RGB')
                
        image.save(os.path.join(os.sep.join(image_path.split(os.sep)[:-1]), 'image.png'), 'PNG')
        

def create_barcodes(path_to_data: str = os.path.join(get_project_path(), 'data')):
    console = Console(record=False)
    console.print(Markdown('# Start searching for files...'))
    width_line = 0.0196
    list_files_to_convert = list_txt_files(path_to_data)
    console.print(Markdown('# Start converting files...'))
    matrix_convert_save(list_files_to_convert)
    list_of_csv_files = list_csv_files(path_to_data)
    create_afm_image_from_csv(list_of_csv_files)
    console.print(Markdown('# Start calculating...'))
    minmax_db(list_of_csv_files)
    autocorr_function(list_of_csv_files, width_line)
    persistance_db(list_of_csv_files)
    console.print(Markdown('# Barcode successfully created!'))
    
def calc_bottleneck_dist(path_to_data: str = os.path.join(get_project_path(), 'data')):
    console = Console(record=False)
    console.print(Markdown('# Start calculating bottleneck distances...'))
    calc_bottlenecks(path_to_data)
    console.print(Markdown('# Bottleneck distances successfully calculated!'))

def create_npy_files(list_of_paths):
    final_list = list()
    name = None
    if list_of_paths[0].endswith('barcode.png'):
        name = 'barcode'
    if list_of_paths[0].endswith('diagram.png'):
        name = 'diagram'
    if list_of_paths[0].endswith('image.png'):
        name = 'AFM'

    for path in track(list_of_paths, description=f"[red]Reading {name} images..."):
        image = cv2.imread(path)
        image = cv2.resize(image, (256, 256))
        image = image/255.0
        image = np.array(image, dtype=np.float32)
        final_list.append(image)
    
    return np.array(final_list, dtype=np.float32)

def prepare_tables(tables, coef):
    readed_table = list()
    name = None
    if tables[0].endswith('auto.csv'):
        name = 'auto'
    if tables[0].endswith('min_max_ix(3x3).csv'):
        name = 'min max'
    if tables[0].endswith('flattened_submat(3x3).csv'):
        name = 'flattened submat'
    
    current_index = 0
    for file in track(tables, description=f'[green]Reading {name} tables'):
        table = pd.read_csv(file)
        table.drop(table.columns[0], axis=1, inplace=True)
        features_list, features = list(), list()
        for i in table.columns[:coef]:
            for j in range(len(table)):
                features_list.append(f'{i}_{j}')
                features.append(table[i].values[j])
                current_index+=1
        final_table = pd.DataFrame(data=[features], columns=features_list)
        readed_table.append(final_table)
    return np.array(pd.concat(readed_table, ignore_index=True).values, dtype=np.float32)

def create_dataset(path_to_data: str = os.path.join(get_project_path(), 'data')):
    
    console = Console(record=False)
    
    list_folders = []

    for root, dirs, files in os.walk(path_to_data):
        for dir in dirs:
            if not dir in ['no_layers', '1_layer', '2_layers', '3_layers', '4_layers']:
                list_folders.append(os.path.join(root, dir))
                
    auto_tables, min_max_tables, flattened_tables, barcode_images, diagram_images, afm_images, y_data = list(), list(), list(), list(), list(), list(), list()

    for path in list_folders:
        tmp_files_list = os.listdir(path)
        tmp_files_path_list = list()
        for file in tmp_files_list:
            tmp_files_path_list.append(os.path.join(path, file))
        
        for file in tmp_files_path_list:
            if file.endswith('auto.csv'):
                auto_tables.append(file)
                y_data.append(file.split(os.sep)[-3])
            elif file.endswith('min_max_ix(3x3).csv'):
                min_max_tables.append(file)
            elif file.endswith('flattened_submat(3x3).csv'):
                flattened_tables.append(file)
            elif file.endswith('barcode.png'):
                barcode_images.append(file)
            elif file.endswith('persistence_diagram.png'):
                diagram_images.append(file)
            elif file.endswith('image.png'):
                afm_images.append(file)
                
    auto_tables, min_max_tables, flattened_tables, barcode_images, diagram_images, afm_images, y_data = np.array(auto_tables), np.array(min_max_tables), np.array(flattened_tables), np.array(barcode_images), np.array(diagram_images), np.array(afm_images), np.array(y_data)
    
    if not os.path.exists(os.path.join(get_project_path(), 'prepared_data')):
        os.makedirs(os.path.join(get_project_path(), 'prepared_data'))
    
    console.print(Markdown('# Start barcode images dataset creating...'))
    barcode_files = create_npy_files(barcode_images)
    np.save(os.path.join(get_project_path(), 'prepared_data', 'barcode_images.npy'), barcode_files, allow_pickle=True)
    console.print(Markdown(f'# Barcode images dataset saved in {os.path.join(get_project_path(), "prepared_data", "barcode_images.npy")}'))
    console.print(Markdown('# Start diagram images dataset creating...'))
    diagram_files = create_npy_files(diagram_images)
    np.save(os.path.join(get_project_path(), 'prepared_data', 'diagram_images.npy'), diagram_files, allow_pickle=True)
    console.print(Markdown(f'# Diagram images dataset saved in {os.path.join(get_project_path(), "prepared_data", "diagram_images.npy")}'))
    console.print(Markdown('# Start AFM images dataset creating...'))
    afm_files = create_npy_files(afm_images)
    np.save(os.path.join(get_project_path(), 'prepared_data', 'afm_images.npy'), afm_files, allow_pickle=True)
    console.print(Markdown(f'# AFM images dataset saved in {os.path.join(get_project_path(), "prepared_data", "afm_images.npy")}'))
    console.print(Markdown('# Start auto table dataset creating...'))
    auto_files = prepare_tables(auto_tables, coef=3)
    np.save(os.path.join(get_project_path(), 'prepared_data', 'auto_tables.npy'), auto_files, allow_pickle=True)
    console.print(Markdown(f'# Auto table saved in {os.path.join(get_project_path(), "prepared_data", "auto_tables.npy")}'))
    console.print(Markdown('# Start min/max table dataset creating...'))
    min_max_files = prepare_tables(min_max_tables, coef=2)
    np.save(os.path.join(get_project_path(), 'prepared_data', 'min_max_tables.npy'), min_max_files, allow_pickle=True)
    console.print(Markdown(f'# Min/max table saved in {os.path.join(get_project_path(), "prepared_data", "min_max_tables.npy")}'))
    console.print(Markdown('# Start flattened table dataset creating...'))
    flattened_files = prepare_tables(flattened_tables, coef=-1)
    np.save(os.path.join(get_project_path(), 'prepared_data', 'flattened_tables.npy'), flattened_files, allow_pickle=True)
    console.print(Markdown(f'# Flattened table saved in {os.path.join(get_project_path(), "prepared_data", "flattened_tables.npy")}'))
    np.save(os.path.join(get_project_path(), 'prepared_data', 'labels.npy'), y_data, allow_pickle=True)
    console.print(Markdown(f'# Labels saved in {os.path.join(get_project_path(), "prepared_data", "labels.npy")}'))
    
if __name__ == "__main__":
    # create_barcodes()
    # calc_bottleneck_dist()
    create_dataset()