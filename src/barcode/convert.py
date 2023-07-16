import numpy as np
import os
import cv2
import shutil
from rich.progress import track

def matrix_convert_save(dataset):
    for file in track(dataset, description='[green]Converting txt to csv...'):
        filik = open(file, "r")
        listik = []
        save_path = os.path.join(os.sep.join(file.split(os.sep)[:-1]), file.split(os.sep)[-1][:-4])
        for line in filik:
            stripped_line = line.strip()
            line_list = stripped_line.split()
            listik.append(line_list)
        dfs = listik[4:]
        for x in range(4):
            dfs = np.array(dfs, dtype=np.float64)
            dfs = cv2.rotate(dfs, cv2.ROTATE_90_CLOCKWISE)
            dfs = dfs.tolist()
            dflength = len(dfs)
            first_line = ['DataLine']
            for i in range(dflength):
                first_line.append('Pos = '+str(i))
            dfs_new = [[float(item)*10**9 for item in list] for list in dfs]
            j = 0
            for line in dfs_new:
                line.insert(0, j)
                j = j+1
            dfs_new.insert(0, first_line)
            file_name = f'{(x+1)*90}_'+file.split(os.sep)[-1][:-4]
            
            save_path = os.path.join(os.sep.join(file.split(os.sep)[:-1]), file_name)
            
            if os.path.exists(save_path):
                pass
            else:
                os.mkdir(save_path)
            
            np.savetxt(os.path.join(save_path, (file_name + '.csv')), dfs_new, fmt='%s', delimiter=',')



