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
numstep=500


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
            
            pair_seps[x,y]=mic(particles[x].pos-particles[y].pos, L)
            
    #Take lower left triangle elements as negative values of transpose element
        
        for y in range(0,x):
            
            pair_seps[x,y]=-1*mic(pair_seps[y,x], L)
    
    return pair_seps


def scalar_seps(pair_seps):
    
    """
    Computes NxN numpy array containing scalar separations between all N
    particles 
    
    :param pair_seps: NxNx3 numpy array of particle separations
    :return: numpy array containing scalar separations between all particles 
    """

    #create empty array to hold scalar separations
    scalar_seps = np.empty(shape=(len(particles), len(particles)), dtype=float) 
    
    #loop through all N particles, adding separations to each element
    for x in range(len(particles)):
        
    #Given scalar_seps is a symmetric matrix, only compute upper right triangle
        
        for y in range(x, len(particles)):
            
            scalar_seps[x,y]=np.linalg.norm(mic(pair_seps[x,y], L))
            
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
            
            #apply cutoff radius and avoid force due to particle itself
            if 1e-15<r<3.5: 
                forces[x,y]=-48*(1/r**14-1/(2*r**8))*mic(pair_seps[x,y], L)
           
            #add zero force for particle itself or for particle separation>3.5*sigma    
            else:
                forces[x,y]=np.zeros(3)
                        
            
    #Take lower left triangle elements as negative values of transpose element
        
        for y in range(0,x):            
            forces[x,y]=-1*forces[y,x]
    
    return forces   
  

def update_vel(particles, force_array):
    """
    Updates the velocities of all particles in the particles array by
    calculating the total force acting on each particle.
    
    :param particles: numpy array of N Particle3D instances
    :param force_array: NxNx3 numpy array giving LJ forces acting on N particles
    :return: numpy array of N Particle3D instances with updated velocities
    """    
    
    for y in range(len(particles)):
        F_y_total=np.zeros(3)
        for x in range(len(particles)):
            F_y_total=F_y_total+force_array[x,y]

        particles[y].update_vel(dt, F_y_total)
    
    return particles

def update_position(particles, force_array):
    """
    Updates the positions of all particles in the particles array by
    calculating the total force acting on each particle.
    
    :param particles: numpy array of N Particle3D instances
    :param force_array: NxNx3 numpy array giving LJ forces acting on N particles
    :return: numpy array of N Particle3D instances with updated positions
    """  
    for y in range(len(particles)):
        F_y_total=np.zeros(3)
        for x in range(len(particles)):
            F_y_total=F_y_total+force_array[x,y]
        
        particles[y].update_pos_2nd(dt, F_y_total)
    
    return particles


def tot_potential_energy(particles, scalar_seps):
    """
    Finds the total Leonard-Jones potential energy of the system.  
    
    :param particles: numpy array of N Particle3D instances
    :param scalar_seps: NxN numpy array of scalar particle separations
    :return: float indicating the total potential energy of the system
    """
    
    #create empty array to hold potential values
    potentials = np.empty(shape=(len(particles), len(particles)), dtype=float) 
    
    #loop through all N particles, adding separation vectors to each element
    for x in range(len(particles)):
        
    #Only compute upper right triangle of array to avoid double counting
        
        for y in range(x, len(particles)):
            
            r=scalar_seps[x,y]
            
            if 1e-15<r<3.5:
                potentials[x,y]=4*(1/r**12-1/r**6)
            
            elif r<1e-15:
                potentials[x,y]=0
                
            else:
                potentials[x,y]=-4*(1/3.5**12-1/3.5**6)
            
    for y in range(len(particles)):
        tot_pot=0
        for x in range(len(particles)):
            tot_pot=tot_pot+potentials[x,y]

    return tot_pot   
    
             
p1=Particle3D('p1', 1.0, np.array([(1.1+(2**(1/6))), 0, 0]), np.array([0.01, 0, 0]))
p2=Particle3D('p2', 1.0, np.array([1, 0, 0]), np.array([-0.01, 0, 0]))

particles=np.array([p1, p2])

def main():
    
    # Initial conditions
    time=0
    
    #compute initial pair separations for use in force and potential calculations
    ps=pair_seps(particles) 
    ss=scalar_seps(ps)

    #compute initial forces
    forces=update_all_forces(ps, ss)

    # Initialise data lists for plotting later
    time_list = [time]
    pos_x_list = [particles[0].pos[0]]
    kin_energy_list = [Particle3D.sys_kinetic(particles)]
    pot_energy_list = [tot_potential_energy(particles, ss)]
    energy_list=[kin_energy_list[0]+pot_energy_list[0]]
    mic_list=[mic(particles[0].pos-particles[1].pos, L)[0]]
    
    
    # Start the time integration loop
    for i in range(numstep):
        
        # Update particle positions
        update_position(particles, forces)
        
        #Update pair separations
        ps=pair_seps(particles)
        ss=scalar_seps(ps)

        # Update forces
        forces_new = update_all_forces(ps, ss)
               
        # Update particle velocities by averaging current and new forces
        update_vel(particles, 0.5*(forces+forces_new))
                
        # Re-define force value
        forces = forces_new
        
        # Increase time
        time += dt
        
        kin_energy=Particle3D.sys_kinetic(particles)
        pot_energy=tot_potential_energy(particles, ss)
        energy=kin_energy+pot_energy
        mic_=mic(particles[0].pos-particles[1].pos, L)[0]
 
        
        # Append information to data lists
        time_list.append(time)
        pos_x_list.append(particles[0].pos[0])
        kin_energy_list.append(kin_energy)
        pot_energy_list.append(pot_energy)
        energy_list.append(energy)
        mic_list.append(mic_)
    
           
    #convert all lists to numpy arrays

    times=np.array(time_list)

    
            
        
    
    
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
       
    
    # Plot particle energy to screen
    pyplot.title('mic vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('energy')
    pyplot.plot(time_list, mic_list)
    pyplot.show()  

# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()


    
    

