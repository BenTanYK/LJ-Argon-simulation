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

""" Define box length L and integration timestep dt"""

L=3
dt=0.01
numstep=1000

p1=Particle3D('p1', 1, np.array([9+2**(1/6)+0.1,0,0]), np.array([0.01, 0, 0]))
p2=Particle3D('p2', 1, np.array([9,0,0]), np.array([-0.01, 0, 0]))

    
particles=np.array([p1, p2])


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

def mic(p1, p2, L):
    """
    Returns vector r_ij', where r_ij'=r_1-r_2', where r_2' is the nearest 
    neighbour to particle 1 in the cube of side length L.
    
    :param r_ij: r_ij=r_1-r_2
    :param L: side length of cube (must be a positive number)
    :return: distance between particle i and the closest image of particle j
    to particle i
    """
    
    r_ij=p1.pos-p2.pos
    L_vector=L*np.array([1,1,1])

    MIC=(np.mod((r_ij+0.5*L_vector), L_vector))-0.5*L_vector
    return MIC



def pair_seps(particles):
    """
    Computes NxNx3 numpy array containing pair separation vectors between
    all N particles.
    
    :param particles: numpy array of N Particle3D instances
    :return: numpy array containing separations between the particles 
    """

    #create empty array to hold pair separation vectors
    pair_seps = np.empty(shape=(len(particles), len(particles), 3), dtype=float) 
    
    #loop through all N particles, adding separation vectors to each element
    for x in range(len(particles)):
        
    #Given pair_seps is an antisymmetric matrix, only compute upper right triangle
        
        for y in range(x, len(particles)):
            
            pair_seps[x,y]=mic(particles[x], particles[y], L)
            
    #Take lower triangle elements as negative values of transpose element
        
        for y in range(0,x):
            
            pair_seps[x,y]=-1*pair_seps[y,x]
    
    return pair_seps

def scalar_seps(pair_seps):
    
    """
    Computes NxN numpy array containing scalar separations between all N
    particles 
    
    :param pair_seps: NxNx3 numpy array of particle separations
    :return: numpy array containing scalar separations between all particles 
    """

    #create empty array to hold scalar separation vectors
    scalar_seps = np.empty(shape=(len(particles), len(particles)), dtype=float) 
    
    #loop through all N particles, adding separation vectors to each element
    for x in range(len(particles)):
        
    #Given scalar_seps is a symmetric matrix, only compute upper right triangle
        
        for y in range(x, len(particles)):
            
            scalar_seps[x,y]=np.linalg.norm(pair_seps[x,y])
            
    #Take lower left triangle elements as equal to transpose elements
        
        for y in range(0,x):
            
            scalar_seps[x,y]=scalar_seps[y,x]
    
    return scalar_seps    
    

def update_all_forces(pair_seps, scalar_seps):
    """
    Computes NxNx3 numpy array containing force vectors acting on all N 
    particles.
    
    :param pair_seps: NxNx3 numpy array of particle separations
    :return: numpy array containing forces acting on all particles 
    """
    
    #create empty array to force vectors
    forces = np.empty(shape=(len(particles), len(particles), 3), dtype=float) 
    
    #loop through all N particles, adding separation vectors to each element
    for x in range(len(particles)):
        
    #Given forces is an antisymmetric matrix, only compute upper right triangle
        
        for y in range(x, len(particles)):
            r=scalar_seps[x,y]
            forces[x,y]=48*(1/r**14-1/(2*r**8))*pair_seps[x,y]
                        
            
    #Take lower left triangle elements as negative values of transpose element
        
        for y in range(0,x):            
            forces[x,y]=-1*forces[y,x]
    
    return forces 

            
print(pair_seps(particles))
print(scalar_seps(pair_seps(particles)))
print(update_all_forces(pair_seps(particles), scalar_seps(pair_seps(particles))))












