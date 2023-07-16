import pandas as pd
import numpy as np
import gudhi


def calc_bottleneck_for_two_files(dataframe_1: str, dataframe_2: str):
    first_source = pd.read_csv(dataframe_1)
    second_source = pd.read_csv(dataframe_2)

    first_source = first_source.to_numpy()
    second_source = second_source.to_numpy()

    gudhi.persistence_graphical_tools._gudhi_matplotlib_use_tex = False

    first_rips_complex = gudhi.RipsComplex(distance_matrix=first_source,
                                           max_edge_length=100.0)  # max_edge_length=100, 250

    first_simplex_tree = first_rips_complex.create_simplex_tree(max_dimension=3)
    first_diag_pers = first_simplex_tree.persistence(min_persistence=0)
    first_diag = first_simplex_tree.persistence_intervals_in_dimension(1)
    
    second_rips_complex = gudhi.RipsComplex(distance_matrix=second_source,
                                            max_edge_length=100.0)  # max_edge_length=100, 250
    second_simplex_tree = second_rips_complex.create_simplex_tree(max_dimension=3)
    second_diag_pers = second_simplex_tree.persistence(min_persistence=0)
    second_diag = second_simplex_tree.persistence_intervals_in_dimension(1)
    

    results = gudhi.bottleneck_distance(diagram_1=first_diag, diagram_2=second_diag)
    return results

def calc_only_persistence(dataframe: str):
    df = pd.read_csv(dataframe)
    df = df.to_numpy()
    
    gudhi.persistence_graphical_tools._gudhi_matplotlib_use_tex = False
    
    rips = gudhi.RipsComplex(distance_matrix=df, max_edge_length=100.0)
    simplex_tree = rips.create_simplex_tree(max_dimension=3)
    diag_pers = simplex_tree.persistence(min_persistence=0)
    diag = simplex_tree.persistence_intervals_in_dimension(1)
    
    return diag
