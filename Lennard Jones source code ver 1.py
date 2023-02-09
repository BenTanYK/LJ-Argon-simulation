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

""" Define box length L and integration timestep dt"""
L=3
dt=0.01
numstep=2000


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
    Returns vector r_ij', where r_ij'=r_1-r_2', where r_2' is the nearest 
    neighbour to particle 1 in the cube of side length L.
    
    :param r_ij: r_ij=r_1-r_2
    :param L: side length of cube (must be a positive number)
    :return: distance between particle i and the closest image of particle j
    to particle i
    """
    
    L_vector=L*np.array([1,1,1])

    MIC=(np.mod((r_ij+0.5*L_vector), L_vector))-0.5*L_vector
    return MIC

def pair_sep(particles, i):
    """
    Finds the pairwise separation between particle i and the remaining
    particles within a distance of 3.5*sigma (mic applied)
    
    :param particles: numpy array of N Particle3D instances
    :param i: index of the chosen particle
    :return: numpy array containing N separations between the particles 
    (mic applied, 0 separation for particle i itself)
    """

    sep=np.array([])
    scalar_sep=np.array([])
    
    for n in range(len(particles)):      
        r_ij=pbc(particles[i].pos, L)-pbc(particles[n].pos, L)
        mic_r_ij=mic(r_ij, L)
        sep=np.append(sep, mic_r_ij)
            
    for m in range(len(particles)):
        distance=np.array([sep[3*m], sep[3*m+1], sep[3*m+2]])
        sep_distance=np.linalg.norm(distance)
        scalar_sep=np.append(scalar_sep, sep_distance)
                            
    return scalar_sep


def LJ_force_i(particles, i):
    """
    Finds the Leonard-Jones forces acting on particle i due to all closest 
    neighbour particles within a distance of 3.5*sigma (pbc and mic applied). 
    
    :param particles: numpy array of N Particle3D instances
    :param i: index of the chosen particle
    :return: numpy array containing all Leonard-Jones forces acting on
    particle i)
    """
    
    LJ_force=np.array([0, 0, 0])
    
    r_i=pbc(particles[i].pos, L)
    
    particle_sep=pair_sep(particles, i)
    
    for j in range(len(particles)):
        r_ij=r_i-pbc(particles[j].pos, L)
        mic_r_ij=mic(r_ij, L)
        r=particle_sep[j]
        
        if r!=0: #remove particle separation to itself
            F_j=(-48)*(r**(-14)-0.5*r**(-8))*mic_r_ij
            
            if r<3.5: #implement cut-off radius of 3.5*sigma
                LJ_force=LJ_force+F_j
         
    return LJ_force 

def update_LJ_all(particles):
    """
    Gives numpy array of all forces experienced by the N particles in the 
    particles array.
    
    :param particles: numpy array of N Particle3D instances
    :return: numpy array of forces acting on each particle.
    """    
    all_forces_list=[] #create empty list to hold force arrays
    
    for i in range(len(particles)):
        force_i=LJ_force_i(particles, i)
        all_forces=all_forces_list.append(force_i)
        
    all_forces=np.asarray(all_forces_list) #covert list to numpy array
    
    return all_forces
        

def update_vel(particles):
    """
    Updates the velocities of all particles in the particles array by
    calculating the total force acting on each particle.
    
    :param particles: numpy array of N Particle3D instances
    :return: numpy array of N Particle3D instances with updated velocities
    """    
    for i in range(len(particles)):
        F_i=LJ_force_i(particles, i)
        particles[i].update_vel(dt, F_i)
    
    return particles

def update_position(particles):
    """
    Updates the positions of all particles in the particles array by
    calculating the total force acting on each particle.
    
    :param particles: numpy array of N Particle3D instances
    :return: numpy array of N Particle3D instances with updated positions
    """  
    for i in range(len(particles)):
        F_i=LJ_force_i(particles, i)
        particles[i].update_pos_2nd(dt, F_i)
        particles[i].pos=pbc(particles[i].pos, L)
    
    return particles

def LJ_potential(particles, i):
    """
    Finds the Leonard-Jones potential experienced by particle i due to all 
    closest neighbour particles within a distance of 3.5*sigma (pbc and mic applied). 
    
    :param particles: numpy array of N Particle3D instances
    :param i: index of the chosen particle
    :return: float indicating potential energy of particle i
    """

    U_tot=0
    
    separations=pair_sep(particles, i)
    
    for j in range(len(particles)):

        r=separations[j]
        
        if r!=0: #condition to avoid potential due to particle i itself
        
           U_ij=4*(1/(r**12)-1/(r**6))
           
           U_tot=U_tot+U_ij #add to total potential energy
           
    return U_tot
        

def tot_potential_energy(particles):
    """
    Finds the total Leonard-Jones potential energy of the system.  
    
    :param particles: numpy array of N Particle3D instances
    :return: float indicating the total potential energy of the system
    """
    twice_tot_potential_energy=0
    
    for i in range(len(particles)):
        U_ij=LJ_potential(particles, i)
        twice_tot_potential_energy=twice_tot_potential_energy+U_ij
    
    tot_potential_energy=0.5*twice_tot_potential_energy
    
    return tot_potential_energy


def main():
    # Initialise data lists for plotting later
    time=0
    time_list = [time]
    pos_x_list = [particles[0].pos[0]]
    energy_list = [tot_potential_energy(particles)+Particle3D.sys_kinetic(particles)]

    # Start the time integration loop
    for i in range(numstep):
        
        # Update particle positions
        update_position(particles)
               
        # Update particle velocities 
        update_vel(particles)
        
        # Increase time
        time += dt
        
        energy=Particle3D.sys_kinetic(particles)+tot_potential_energy(particles)
        
        # Append information to data lists
        time_list.append(time)
        pos_x_list.append(particles[0].pos[0])
        energy_list.append(energy)
        
    # Plot particle trajectory to screen
    pyplot.title('Position vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('x position')
    pyplot.plot(time_list, pos_x_list)
    pyplot.show()

    # Plot particle energy to screen
    pyplot.title('Total energy vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('Energy')
    pyplot.plot(time_list, energy_list)
    pyplot.show()
    
    all_forces=update_LJ_all(particles)
    
    print(all_forces)

              
p1=Particle3D('p1', 1, np.array([2**(1/6)+0.1,0,0]), np.array([0.01, 0, 0]))
p2=Particle3D('p2', 1, np.array([0,0,0]), np.array([-0.01, 0, 0]))

    
particles=np.array([p1, p2])


        

# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()


    
    

