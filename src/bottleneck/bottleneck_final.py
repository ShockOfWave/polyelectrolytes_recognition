import os
from gudhi.hera import bottleneck_distance, wasserstein_distance
import pandas as pd
from copy import deepcopy
from src.bottleneck.bottleneck_finder import calc_only_persistence
from rich.markdown import Markdown
from rich.console import Console
from rich.progress import track
from src.paths.paths import get_project_path

def calc_bottlenecks(path_to_data):
    
    files_list: list = []
    
    for root, dirs, files in os.walk(path_to_data):
        for file in files:
            if file.endswith('.csv') and not file.endswith('(3x3).csv') and not file.endswith('_auto.csv') and not file.endswith('output.csv'):
                files_list.append(os.path.join(root, file))
                
    main_df_bottleneck: list = []
    main_df_wasserstein: list = []
    
    diags: list = []
    names: list = []
    
    for file in track(files_list, description='[green]Preparing files for bottleneck...'):
        diags.append(calc_only_persistence(file))
        names.append(file.split(os.sep)[:-4])
        
    for i, diag in track(enumerate(diag), description='Calculating bottleneck distance...'):
        bottleneck_dist: list = []
        wasserstein_dist: list = []
        columns_list: list = []
        
        tmp_diags = deepcopy(diags)
        
        for j, tmp_diag in enumerate(tmp_diags):
            bottleneck_dist.append(bottleneck_distance(diag, tmp_diag))
            wasserstein_dist.append(wasserstein_distance(diag, tmp_diag))
            columns_list.append(names[j])
        
        tmp_df_bottleneck = pd.DataFrame(data=[bottleneck_dist], columns=columns_list, index=[names[i]])
        tmp_df_wasserstein = pd.DataFrame(data=[wasserstein_dist], columns=columns_list, index=[names[i]])
        main_df_bottleneck.append(tmp_df_bottleneck)
        main_df_wasserstein.append(tmp_df_wasserstein)
        
    save_path = os.path.join(get_project_path(), 'bottleneck_wasserstein_dist')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    pd.concat(main_df_bottleneck).to_csv(os.path.join(save_path, 'results_bottleneck.csv'))
    pd.concat(main_df_wasserstein).to_csv(os.path.join(save_path, 'results_wasserstein.csv'))

# files_list: list = []

# console = Console(record=True)
# console.print(Markdown('# Start searching for files...'))

# for root, dirs, files in os.walk('output'):
#     for file in files:
#         if file == str(os.path.join(root, file).split(os.sep)[-2]+'.csv'):
#             files_list.append(os.path.join(root, file))
            
# console.print(Markdown('# Searching for files finished!'))

# main_df_bottleneck: list = []
# main_df_wasserstein: list = []

# diags: list = []
# names: list = []

# console.print(Markdown('# Start to process diagrams...'))

# for file in track(files_list, description='[green]Processing...'):
#     diags.append(calc_only_persistence(file))
#     names.append(file.split(os.sep)[-2])

# console.print(Markdown('# Diagrams processing finished!'))
# console.print(Markdown('# Start to calculating bottleneck and wasserstein distanses...'))

# for i, diag in track(enumerate(diags), description='Processing...'):
#     bottleneck_dist: list = []
#     wasserstein_dist: list = []
#     columns_list: list = []
    
#     tmp_diags = deepcopy(diags)
    
#     for j, tmp_diag in enumerate(tmp_diags):
#         bottleneck_dist.append(bottleneck_distance(diag, tmp_diag))
#         wasserstein_dist.append(wasserstein_distance(diag, tmp_diag))
#         columns_list.append(names[j])
        
#     tmp_df_bottleneck = pd.DataFrame(data=[bottleneck_dist], columns=columns_list, index=[names[i]])
#     tmp_df_wasserstein = pd.DataFrame(data=[wasserstein_dist], columns=columns_list, index=[names[i]])
#     main_df_bottleneck.append(tmp_df_bottleneck)
#     main_df_wasserstein.append(tmp_df_wasserstein)
    
# pd.concat(main_df_bottleneck).to_csv('results_bottleneck.csv')
# pd.concat(main_df_wasserstein).to_csv('results_wasserstein.csv')
# console.save_html('log.html')
# MARKDOWN = """
# # Finished!

# Bottleneck distanses saved in results_bottleneck.csv file
# Wasserstein distanses saved in results_wasserstein.csv file

# Console output saved in log.html file
# """
# console.print(Markdown(MARKDOWN))
