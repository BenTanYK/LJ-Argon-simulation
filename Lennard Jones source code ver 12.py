"""
Computer modelling project B: Lennard - Jones System

PROGRAMME DESCRIPTION GOES HERE

author: Benedict Tan s1934251 

"""

import sys
import math
import numpy as np
import matplotlib.pyplot as pyplot
from particle3D import Particle3D
from md_utils import set_initial_positions
from md_utils import set_initial_velocities
from PBC import pbc
from PBC import mic
import time


def pair_seps(particles, L):
    """
    Computes NxNx3 numpy array containing pair separation vectors between
    all N particles.
    
    :param particles: numpy array of N Particle3D instances
    :param L: scalar cubic side length of simulation unit cell
    :return: numpy array containing separations between the particles 
    """

    #create NxNx3 array of zeros to hold pair separation vectors
    pair_seps = np.zeros(shape=(len(particles), len(particles), 3), dtype=float) 
    
    #loop through all N particles, adding separation vectors to each element
    for x in range(len(particles)):
        
    #Given pair_seps is an antisymmetric matrix, only compute upper right triangle,
    #the values in the lower left triangle will be the negative values of their
    #respective transpose elements (pair_seps[x,y]=-pair_seps[y,x])
        
        for y in range(x, len(particles)):
            
            pair_seps[x,y]=mic(pbc(particles[x].pos, L)-pbc(particles[y].pos, L), L)
            
            pair_seps[y,x ]=-1*pair_seps[x,y]
            
    return pair_seps


def scalar_seps(pair_seps, L):
    
    """
    Computes NxN numpy array containing scalar separations between all N
    particles 
    
    :param pair_seps: NxNx3 numpy array of vector particle separations 
    :param L: scalar cubic side length of simulation unit cell
    :return: numpy array containing scalar separations between all particles 
    """

    #create NxN array of zeros to hold scalar separations
    scalar_seps = np.zeros(shape=(len(pair_seps), len(pair_seps)), dtype=float) 
    
    #loop through all N particles, adding separations to each element
    for x in range(len(pair_seps)):
        
    #Given scalar_seps is a symmetric matrix, only compute upper right triangle
        
        for y in range(x, len(pair_seps)):
            
            scalar_seps[x,y]=np.linalg.norm(mic(pair_seps[x,y], L))
            
            scalar_seps[y,x]=scalar_seps[x,y]
            
    
    return scalar_seps     
    

def update_all_forces(pair_seps, scalar_seps, L):
    """
    Computes NxNx3 numpy array containing force vectors acting on all N 
    particles.
    
    :param pair_seps: NxNx3 numpy array of vector particle separations
    :param scalar_seps: NxN numpy array of scalar particle separations
    :param L: scalar cubic side length of simulation unit cell
    :return: numpy array containing forces acting on all particles 
    """
    
    #create empty array to force vectors
    forces = np.zeros(shape=(len(pair_seps), len(pair_seps), 3), dtype=float) 
    
    #loop through all N particles, adding separation vectors to each element
    for x in range(len(pair_seps)):
        
    #Given forces is an antisymmetric matrix, only compute upper right triangle
        
        for y in range(x, len(pair_seps)):
            r=scalar_seps[x,y]
            
            #apply cutoff radius and avoid force due to particle itself
            if 1e-15<r<3.5: 
                forces[x,y]=-48*(1/r**14-1/(2*r**8))*mic(pair_seps[x,y], L)
                forces[y,x]=-1*forces[x,y]
           
            #add zero force for particle itself or for particle separation>3.5*sigma    
            else:
                forces[x,y]=np.zeros(3)
                forces[y,x]=np.zeros(3)
                       

    return forces   
 
    
def MSD(particles, initial_pos, L):
    """
    Returns the mean squared displacement for the system at a given time t.
    
    :param particles: 1D numpy array of N Particle3D instances
    :param initial_pos: NxNx3 numpy array containing initial positions of 
    :param L: scalar cubic side length of simulation unit cell
    :param numstep: 
    :return: float describing MSD at time t
    """     
    N=len(particles)
    
    MSD=0

    for i in range(N):
        MSD_i=(1/N)*(np.linalg.norm(mic(pbc(particles[i].pos, L)-pbc(initial_pos[i], L), L)))**2
        MSD+=MSD_i
        
    return MSD


def RDF(ss, L, rho, numstep, particles):
    

    """
    Creates a radial distribution function in the form of a histogram of the 
    average separations as a function of r.
        
    :param ss: 1D numpy array holding scalar separations between particles over
    the entire duration of the simulation
    :param L: cubic side length of unit cell used in simulation
    :param rho: density of the system
    :param particles: list of N Particle3D instances
    :return: two lists containing values of separation r_ij and the normalised 
    radial distribution function at each value of r_ij 
    """ 
    #due to pbc, mic, the longest distance two particles can be apart is
    #sqrt(3)/2 *L, so divide this area into 100 bins
    rdf, r_long=np.histogram(ss[ss>0], bins=100, range=(0,0.5*math.sqrt(3)*L))
    
    #bin size dr
    dr=r_long[1]-r_long[0]
    
    #convert list of bin edge points into halfwa
    r=r_long[1:]-0.5*dr

    rdf_normalised_list=[]
    
    #divide each value of rdf by 4*pi*rho*dr*r**2 to normalise
    #divide by numstep*len(particles) to account for time averaging
    #len(particles)=math.sqrt(len(ss))
    
    for n in range(len(rdf)):
        rdf_n=rdf[n]/(numstep*math.sqrt(len(ss))*4*math.pi*rho*dr*(r[n]**2))
        rdf_normalised_list.append(rdf_n)
        
    rdf_normalised=np.array(rdf_normalised_list)
    
        
    return rdf_normalised, r

def update_vel(particles, force_array, dt):
    """
    Updates the velocities of all particles in the particles array by
    calculating the total force acting on each particle.
    
    :param particles: numpy array of N Particle3D instances
    :param force_array: NxNx3 numpy array giving LJ forces acting on N particles
    :dt: timestep used in integration
    :return: numpy array of N Particle3D instances with updated velocities
    """    
    
    for y in range(len(particles)):
        F_y_total=np.zeros(3)
        for x in range(len(particles)):
            F_y_total=F_y_total+force_array[x,y]

        particles[y].update_vel(dt, F_y_total)
    
    return particles


def update_position(particles, force_array, dt, L):
    """
    Updates the positions of all particles in the particles array by
    calculating the total force acting on each particle.
    
    :param particles: numpy array of N Particle3D instances
    :param force_array: NxNx3 numpy array giving LJ forces acting on N particles
    :dt: timestep used in integration
    :return: numpy array of N Particle3D instances with updated positions
    """  
    for y in range(len(particles)):
        F_y_total=np.zeros(3)
        for x in range(len(particles)):
            F_y_total=F_y_total+force_array[x,y]
        
        particles[y].update_pos_2nd(dt, F_y_total, L)
    
    return particles


def tot_potential_energy(particles, scalar_seps):
    """
    Finds the total Leonard-Jones potential energy of the system.  
    
    :param particles: numpy array of N Particle3D instances
    :param scalar_seps: NxN numpy array of scalar particle separations
    :return: float indicating the total potential energy of the system
    """
    
    #create variable to hold potential
    tot_pot = 0 
    
    #loop through all N particles, adding the potential due to each pair 
    for x in range(len(particles)):
        
    #Only compute upper right triangle of array to avoid double counting
        
        for y in range(x, len(particles)):
            
            r=scalar_seps[x,y]
            
            if 1e-15<r<3.5:
                tot_pot+=4*(1/r**12-1/r**6)
            
            elif r<1e-15:
                tot_pot+=0
                
            else:
                tot_pot+=4*(1/3.5**12-1/3.5**6)

    return tot_pot   
    
             
def main():
        
    #read name of input file from command line
    filein=open(str(sys.argv[1]), 'r') 
   
    all_data = []
    #append each line in the file to the list all_data as a string
    for line in filein.readlines(): 
        all_data.append(str(line))
           
    sim_parameters=all_data[1] #find line containing simulation parameters
    sim_parameters_list=sim_parameters.split() #split line into list of indiv. elements
    dt=float(sim_parameters_list[0]) #find each parameter by index
    numstep=int(sim_parameters_list[1])
    time=0.0
    
    conditions=all_data[3] #find line containing number, temp. and density
    conditions_list=conditions.split() #split into indiv. parameters
    n_particles=int(conditions_list[0]) #find N, T and rho
    T=float(conditions_list[1])
    rho=float(conditions_list[2])
    
    # Close input file
    filein.close()

    #initialise initial particles list
    particles_list=[]
    for i in range(n_particles):
        particle_n=Particle3D('p'+str(i+1), 1.0, np.array([1,1,1]), np.array([1,1,1]))
        particles_list.append(particle_n)
        
    particles=np.array(particles_list)
    
    bs, _ = set_initial_positions(rho, particles)
    L=bs[0]
    set_initial_velocities(T, particles, seed=None)    
    
    #create initial positions array for use in MSD function
    initial_pos_list=[]
    for i in range(len(particles)):
        initial_pos_list.append(particles[i].pos)
        
    initial_pos=np.array(initial_pos_list)


    # Read name of output file from command line
    outfile_name = sys.argv[2]

    # Open output file
    outfile = open(outfile_name, "w")
        
    #compute initial pair separations for use in force and potential calculations
    ps=pair_seps(particles, L) 
    ss=scalar_seps(ps, L)
    
    #empty list to hold separations for use in rdf function
    rdf_list=[]    
    
    #compute initial forces
    forces=update_all_forces(ps, ss, L)

    # Initialise data lists for plotting later
    time_list = [time]
    kin_energy_list = [Particle3D.sys_kinetic(particles)]
    pot_energy_list = [tot_potential_energy(particles, ss)]
    energy_list=[kin_energy_list[0]+pot_energy_list[0]]
    MSD_list=[0]
    
    
    #add particle positions to XYZ file
    m=1
    outfile.write(str(len(particles))+'\n')
    outfile.write('Point='+str(m)+'\n')    
    for n in range(len(particles)):
        outfile.write(Particle3D.__str__(particles[n], L))   
        
    
    #create point integer for XYZ file
    m=2
    
    # Start the time integration loop
    for i in range(numstep):
        
        # Update particle positions
        update_position(particles, forces, dt, L)
        
        #Update pair separations
        ps=pair_seps(particles, L)
        ss=scalar_seps(ps, L)
        
        #Append scalar separations to rdf_list
        flat_ss=ss.flatten()
        flat_ss_list=list(flat_ss)
        rdf_list.append(flat_ss_list)

        # Update forces
        forces_new = update_all_forces(ps, ss, L)
               
        # Update particle velocities by averaging current and new forces
        update_vel(particles, 0.5*(forces+forces_new), dt)
                
        # Re-define force value
        forces = forces_new
        
        # Increase time
        time += dt
        
        kin_energy=Particle3D.sys_kinetic(particles)
        pot_energy=tot_potential_energy(particles, ss)
        energy=kin_energy+pot_energy
        
        #MSD
        MSD_time_t=MSD(particles, initial_pos, L)        
        
        # Append information to data lists
        time_list.append(time)
        kin_energy_list.append(kin_energy)
        pot_energy_list.append(pot_energy)
        energy_list.append(energy)
        MSD_list.append(MSD_time_t)
        
        
        #add particle positions to XYZ file
        outfile.write(str(len(particles))+'\n')
        outfile.write('Point='+str(m)+'\n')    
        for n in range(len(particles)):
            outfile.write(Particle3D.__str__(particles[n], L))           
        
        m+=1

    # Post-simulation:
    # Close output file
    outfile.close()
    
    #convert rdf list to array
    rdf_array=np.array(rdf_list)
    
    
    #apply rdf to 1D array of scalar separations
    rdf, r=RDF(rdf_array, L, rho, numstep, particles)
    

     
    # Plot rdf
    pyplot.title('Radial Distribution FUnction')
    pyplot.xlabel('r (σ)')
    pyplot.ylabel('RDF (r)')
    pyplot.plot(r, rdf)
    pyplot.show()
    
    #convert energy and time lists to numpy arrays
    times=np.array(time_list)
    kin_energies=np.array(kin_energy_list)
    pot_energies=np.array(pot_energy_list)
    energies=np.array(energy_list)
    MSD_array=np.array(MSD_list)
       
    # Plot particle potential energy to screen
    pyplot.title('System potential energy vs time')
    pyplot.xlabel('Time (σ * 1/Ɛ^0.5)')
    pyplot.ylabel('Energy (Ɛ)')
    pyplot.plot(times, pot_energies)
    pyplot.show()
    
    # Plot particle kinetic energy to screen
    pyplot.title('System kinetic energy vs time')
    pyplot.xlabel('Time (σ * 1/Ɛ^0.5)')
    pyplot.ylabel('Energy (Ɛ)')
    pyplot.plot(times, kin_energies)
    pyplot.show()
    
    # Plot particle energy to screen
    pyplot.title('Total system energy vs time')
    pyplot.xlabel('Time (σ * 1/Ɛ^0.5)')
    pyplot.ylabel('Energy (Ɛ)')
    pyplot.plot(times, energies)
    pyplot.show()  
    
    # Plot MSD as a function of time
    pyplot.title('Mean Square Displacement vs time')
    pyplot.xlabel('Time (σ * 1/Ɛ^0.5)')
    pyplot.ylabel('MSD (σ^2)')
    pyplot.plot(times, MSD_array)
    pyplot.show() 
    
    
    #output file containing kinetic, potential and total energies
    
    # Read name of output file from command line
    energy_file_name = sys.argv[3]

    # Open output file
    energy_file = open(energy_file_name, "w")
    
    #add energies for each timestep to file
    energy_file.write('Time(sigma * 1/epsilon**0.5) '+ 'Kinetic energy(epsilon) '+ 'Potential energy(epsilon) '+'Total energy(epsilon) '+'\n')
    for n in range(len(times)):
        energy_file.write(str(times[n])+' '+str(kin_energies[n])+' '+str(pot_energies[n])+' '+str(energies[n])+'\n')
    
    # Close output file
    energy_file.close()
    
    #output file containing rdf vs r values
    
    # Read name of output file from command line
    rdf_file_name = sys.argv[4]

    # Open output file
    rdf_file = open(rdf_file_name, "w")
    
    rdf_file.write('RDF '+'r(sigma) '+'\n')
    
    for n in range(len(rdf)):
        rdf_file.write(str(rdf[n])+' '+str(r[n])+'\n')
    
    # Close output file
    rdf_file.close()
    
    
    #output file containing msd vs time values
    
    # Read name of output file from command line
    msd_file_name = sys.argv[5]

    # Open output file
    msd_file = open(msd_file_name, "w")
    
    msd_file.write('MSD(sigma**2) '+'time(sigma*1/epsilon**0.5)'+'\n')
    
    for n in range(len(MSD_array)):
        msd_file.write(str(MSD_array[n])+' '+str(times[n])+'\n')
    
    # Close output file
    msd_file.close()
    
 

start_time = time.time()      

# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()

print('Programme run time is:' + "%s seconds" % (time.time() - start_time))

    
    

