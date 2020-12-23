# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:42:31 2020

@author: lucas
"""
import os
import networkx 
from networkx.algorithms.components.connected import connected_components
import numpy as np


def get_batch_names(path, layer="Image"):
    fn_lst = [x for x in os.listdir(path) if layer in x]
    fn_lst = [x.split("-")[0] for x in fn_lst]
    return(fn_lst)

def to_graph(l):
    """Turn list of overlapping plant masks into graph"""
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current  

def combine_plant_parts(overlap_lst):
    G = to_graph(overlap_lst)
    return(list(connected_components(G)))
    

def remove_outliers(array, max_deviations = 3):
    # Remove image outliers
    mean = np.mean(array)
    standard_deviation = np.std(array)
    distance_from_mean = abs(array - mean)
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    return np.where(not_outlier, array, 0)
