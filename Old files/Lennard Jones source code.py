"""
Computer modelling report: Lennard - Jones System

PROGRAMME DESCRIPTION GOES HERE

author: Benedict Tan s1934251 

"""

import sys
import math
import numpy as np
import matplotlib.pyplot as pyplot
from particle3D import Particle3D
from scipy.signal import find_peaks


def pbc(x, L): 
    
    """
    Image of x
    
    :param x: position vector as a numpy array [x_1,x_2,x_3]
    :param L: side length of cube (must be a positive number)
    :return: image of x inside cube 0 ≤ x_i < l
    """
    
    L_vector=L*np.array([1,1,1])
    image_of_x=np.mod(x, L_vector)
    return image_of_x

def mic(r_ij, L):
    """
    Returns nearest neighbour location for particle i inside the cube of 
    side length L.
    
    :param r_ij: r_ij=r_1-r_2
    :param L: side length of cube (must be a positive number)
    :return: distance between particle i and the closest image of particle j
    to particle i
    """
    
    L_vector=L*np.array([1,1,1])
    MIC=((r_ij+0.5*L_vector)%L_vector)-0.5*L_vector
    return MIC

def pair_sep(particle_list, i):
    """
    Finds the pairwise separation between particle i and the remaining
    particles within a distance of 3.5*sigma (mic applied)
    
    :param particle_list: list of N Particle3D instances
    :param i: index of the chosen particle
    :return: list containing N-1 separations between the particles (mic 
    applied)
    """
    
    L=3
    
    for n in range(0, i-1):
        sep_list_1=[]
        r_ij=particle_list[i].pos-particle_list[n]
        mic_r_ij=mic(r_ij, L)
        sep_list_1.append(mic_r_ij)
        
    for n in range(i+1, len(particle_list)):
        sep_list_2=[]
        r_ij=particle_list[i].pos-particle_list[n]
        mic_r_ij=mic(r_ij, L)
        sep_list_2.append(mic_r_ij)
        



print(mic(np.array([3,3,3]), 5))
    



    
    

