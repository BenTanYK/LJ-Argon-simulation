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

    sep_list=[] #make empty list to hold separation vector numpy arrays
    scalar_sep_list=[] #make empty list to hold distances
    
    for n in range(len(particles)):      
        r_ij=pbc(particles[i].pos, L)-pbc(particles[n].pos, L)
        mic_r_ij=mic(r_ij, L)
        sep_list.append(mic_r_ij)
        
    sep=np.asarray(sep_list)
            
    for m in range(len(particles)):
        sep_distance=np.linalg.norm(sep[m])
        scalar_sep_list.append(sep_distance)
        scalar_sep=np.array(scalar_sep_list)
                            
    return scalar_sep

def pair_separations(particles):
    """
    Returns array containing N arrays which give the separations for each of 
    the N particles.
    
    :param particles: numpy array of N Particle3D instances
    :return: numpy array of arrays containing particle separations
    """    
    
    #loop through all N particles and compute separation arrays for each instance
    pair_seps_list=[]
    for i in range(len(particles)):
        pair_sep_i=pair_sep(particles, i)
        pair_seps_list.append(pair_sep_i)
        
    pair_seps=np.array(pair_seps_list) #convert to np array of arrays
    
    return pair_seps

def LJ_force_i(particles, i, pair_seps):
    """
    Finds the Leonard-Jones forces acting on particle i due to all closest 
    neighbour particles within a distance of 3.5*sigma (pbc and mic applied). 
    
    :param particles: numpy array of N Particle3D instances
    :param i: index of the chosen particle
    :param pair_seps: np array of 1D numpy arrays containing the particle
    separations for each of the N particles 
    :return: numpy array containing all Leonard-Jones forces acting on
    particle i)
    """
    
    LJ_force=np.array([0, 0, 0])
    
    r_i=pbc(particles[i].pos, L)
    
    particle_sep=pair_seps[i]
    
    for j in range(len(particles)):
        r_ij=r_i-pbc(particles[j].pos, L)
        mic_r_ij=mic(r_ij, L)
        r=particle_sep[j]
        
        if r!=0: #remove particle separation to itself
            F_j=(-48)*(r**(-14)-0.5*r**(-8))*mic_r_ij
            
            if r<3.5: #implement cut-off radius of 3.5*sigma
                LJ_force=LJ_force+F_j
         
    return LJ_force 

def update_all_forces(particles, pair_seps):
    """
    Gives numpy array of all forces experienced by the N particles in the 
    particles array.
    
    :param particles: numpy array of N Particle3D instances
    :return: numpy array of forces acting on each particle.
    """    
    all_forces_list=[] #create empty list to hold force arrays
    
    for i in range(len(particles)):
        force_i=LJ_force_i(particles, i, pair_seps)
        all_forces=all_forces_list.append(force_i)
        
    all_forces=np.array(all_forces_list) #covert list to numpy array
    
    return all_forces
        

def update_vel(particles, force_array):
    """
    Updates the velocities of all particles in the particles array by
    calculating the total force acting on each particle.
    
    :param particles: numpy array of N Particle3D instances
    :param force_array: numpy array of N LJ forces acting on N particles
    :return: numpy array of N Particle3D instances with updated velocities
    """    
    
    for i in range(len(particles)):
        F_i=force_array[i]
        particles[i].update_vel(dt, F_i)
    
    return particles

def update_position(particles, force_array):
    """
    Updates the positions of all particles in the particles array by
    calculating the total force acting on each particle.
    
    :param particles: numpy array of N Particle3D instances
    :return: numpy array of N Particle3D instances with updated positions
    """  
    for i in range(len(particles)):
        particles[i].update_pos_2nd(dt, force_array[i])
        particles[i].pos=pbc(particles[i].pos, L)
    
    return particles


def tot_potential_energy(particles, pair_seps):
    """
    Finds the total Leonard-Jones potential energy of the system.  
    
    :param particles: numpy array of N Particle3D instances
    :return: float indicating the total potential energy of the system
    """
    
    twice_U_tot=0
    
    #create loop to account for each of the N particles 
    for i in range(len(particles)):
        
        seps=pair_seps[i]
                
        for j in range(len(seps)):
            if 1e-15<seps[j]<3.5:
                U_ij=4*(1/seps[j]**12-1/seps[j]**6)
                twice_U_tot+=U_ij
            elif seps[j]>3.5:
                U_ij=4*(1/3.5**12-1/3.5**6)
                twice_U_tot+=U_ij                 
        
    U_tot=0.5*twice_U_tot

    return U_tot    


def main():
    
    # Initial conditions
    time=0
    
    #compute initial pair separations for use in force and potential calculations
    pair_seps=pair_separations(particles)
    
    print(pair_seps)
    
    #compute initial forces
    forces=update_all_forces(particles, pair_seps)

    # Initialise data lists for plotting later
    time_list = [time]
    pos_x_list = [pbc(particles[0].pos, L)[0]]
    kin_energy_list = [Particle3D.sys_kinetic(particles)]
    pot_energy_list = [tot_potential_energy(particles, pair_seps)]
    energy_list=[tot_potential_energy(particles, pair_seps)+Particle3D.sys_kinetic(particles)]
    
    
    # Start the time integration loop
    for i in range(numstep):
        
        # Update particle positions
        update_position(particles, forces)
        
        #Update pair separations
        pair_seps=pair_separations(particles)
        
        # Update forces
        forces_new = update_all_forces(particles, pair_seps)
               
        # Update particle velocities by averaging current and new forces
        update_vel(particles, 0.5*(forces+forces_new))
        
        # Re-define force value
        forces = forces_new
        
        # Increase time
        time += dt
        
        kin_energy=Particle3D.sys_kinetic(particles)
        pot_energy=tot_potential_energy(particles, pair_seps)
        energy=kin_energy+pot_energy
        
        # Append information to data lists
        time_list.append(time)
        pos_x_list.append(particles[0].pos[0])
        kin_energy_list.append(kin_energy)
        pot_energy_list.append(pot_energy)
        energy_list.append(energy)
                
                
        
    # Plot particle trajectory to screen
    pyplot.title('Position vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('x position')
    pyplot.plot(time_list, pos_x_list)
    pyplot.show()

    # Plot particle energy to screen
    pyplot.title('Pot energy vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('Pot energy')
    pyplot.plot(time_list, pot_energy_list)
    pyplot.show()
    
    # Plot particle energy to screen
    pyplot.title('Kin energy vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('Kin energy')
    pyplot.plot(time_list, kin_energy_list)
    pyplot.show()    
    
    # Plot particle energy to screen
    pyplot.title('Tot energy vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('energy')
    pyplot.plot(time_list, energy_list)
    pyplot.show()  

              
p1=Particle3D('p1', 1, np.array([2**(1/6)+0.1,0,0]), np.array([0.01, 0, 0]))
p2=Particle3D('p2', 1, np.array([0,0,0]), np.array([-0.01, 0, 0]))

    
particles=np.array([p1, p2])

# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()


    
    

