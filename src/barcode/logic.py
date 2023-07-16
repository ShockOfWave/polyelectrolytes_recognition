import os
from pathlib import Path

def get_project_path():
    path = Path(__file__).parent.parent
    return path

def list_txt_files(folder):
    list_txt_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.txt'):
                list_txt_files.append(os.path.join(root, file))

    return list_txt_files

def list_csv_files(folder):
    list_csv_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('csv'):
                list_csv_files.append(os.path.join(root, file))
    return list_csv_files
