"""
CMod Ex2: velocity Verlet time integration ofa particle moving in a morse 
potential.

Produces plots of the position of the particle and its energy, both as 
functions of time. Also saves both to file.

The potential is the Morse potential, V(r_1, r_2)=D_e{[1-exp(-a(R_12-r_e))]^2-1}, 
where the numpy array r_12=r_2-r_1, R_12 is the magnitude of r_12 and r_e, D_e 
and a are user  defined parameters that control control the position, depth and
curvature of the potential minimum. 

author: Benedict Tan s1934251 
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as pyplot
from particle3D import Particle3D
from scipy.signal import find_peaks

'''Define constants for later use'''
h=6.62607004e-34 #SI units
h_bar=h/(2*math.pi)
c=2.99792458e8 #SI units (ms^-1)
c_cm_per_sec=2.99792458e10 #speed of light (cm s^-1)

def force_morse(particle_1, particle_2, r_e, D_e, a):
    """
    Method to return the force on particle 1 at position r_1 in a Morse potential.
    Force is given by
    F(r_1, r_2) = -dV/dx = 2*a*D_e*[1-exp{-a(R_12-r_e)}]*exp[-a*(R_12-r_e)]*ro_12
    
    where ro_12 is the unit vector pointing in the direction of r_12

    :param particle_1: Particle3D instance located at r_1
    :param particle_2: Particle3D instance located at r_2
    :param r_e: parameter r_e that controls the position of the potential minimum
    :param D_e: parameter D_e that controls the depth of the potential minimum
    :param a: parameter a that controls the curvature of the potential minimum.
    :return: force acting on particle as Numpy array
    """
    r_12=particle_2.pos - particle_1.pos #return np array
    R_12=np.linalg.norm(r_12) #return float
    ro_12=(1/R_12)*r_12 #return np array
    
    force_1=2*a*D_e*(1-math.exp(-a*(R_12-r_e)))*math.exp(-a*(R_12-r_e))*ro_12
    
    return force_1


def pot_energy_morse(particle_1, particle_2, r_e, D_e, a):
    """
    Method to return potential energy 
    of particle in morse potential
    V(r_1, r_2)=D_e{[1-exp(-a(R_12-r_e))]^2-1}

    :param particle_1: Particle3D instance located at r_1
    :param particle_2: Particle3D instance located at r_2
    :param r_e: parameter r_e that controls the position of the potential minimum
    :param D_e: parameter D_e that controls the depth of the potential minimum
    :param a: parameter a that controls the curvature of the potential minimum.
    :return: potential energy of both particles as float
    """
    r_12=particle_2.pos-particle_1.pos #return np array
    R_12=np.linalg.norm(r_12) #return float
    potential=D_e*((1-math.exp(-a*(R_12-r_e)))**2-1) #return float
    return potential


# Begin main code
def main():
    #read name of input file from command line
    if len(sys.argv)!=3:
        print("Wrong number of arguments.")
        print("Usage: " + sys.argv[0] + " <output file>")
        quit()
    else:    
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
    
    morse_parameters=all_data[3] #find line containing morse potential parameters
    morse_parameters_list=morse_parameters.split() #split into indiv. parameters
    D_e=float(morse_parameters_list[0]) #find D_e, r_e and a
    r_e=float(morse_parameters_list[1])
    a=float(morse_parameters_list[2])
    
    p1_param=all_data[4] #find line containing particle 1 initial conditions
    list1=p1_param.split() #split into elements
    mass1=float(list1[1]) 
    pos1=np.array([float(list1[2]), float(list1[3]), float(list1[4])])
    vel1=np.array([float(list1[5]), float(list1[6]), float(list1[7])])
    p1=Particle3D('1', mass1, pos1, vel1)

    p2_param=all_data[5]
    list2=p2_param.split()
    mass2=float(list2[1])
    pos2=np.array([float(list2[2]), float(list2[3]), float(list2[4])])
    vel2=np.array([float(list2[5]), float(list2[6]), float(list2[7])])
    p2=Particle3D('2', mass2, pos2, vel2)
    
    filein.close()
        
    # Read name of output file from command line
    if len(sys.argv)!=3:
        print("Wrong number of arguments.")
        print("Usage: " + sys.argv[0] + " <output file>")
        quit()
    else:
        outfile_name = sys.argv[2]

    # Open output file
    outfile = open(outfile_name, "w")
    
    # Write out initial conditions
    energy = p1.kinetic_e() + pot_energy_morse(p1, p2, r_e, D_e, a)
    outfile.write("{0:f} {1:f} {2:12.8f}\n".format(time,p1.pos[0],energy))

    #Get initial force
    force=force_morse(p1, p2, r_e, D_e, a)

    # Initialise data lists for plotting later
    time_list = [time]
    pos_x_list = [p1.pos[0]]
    energy_list = [energy]

    # Start the time integration loop
    for i in range(numstep):
        # Update particle 1 position
        p1.update_pos_2nd(dt, force)
        
        # Update force
        force_new = force_morse(p1, p2, r_e, D_e, a)
        
        # Update particle velocity by averaging current and new forces
        p1.update_vel(dt, 0.5*(force+force_new))
        
        # Re-define force value
        force = force_new
        
        # Increase time
        time += dt
        
        # Output particle information
        energy = p1.kinetic_e() + pot_energy_morse(p1, p2, r_e, D_e, a)
        outfile.write("{0:f} {1:f} {2:12.8f}\n".format(time,p1.pos[0],energy))

        # Append information to data lists
        time_list.append(time)
        pos_x_list.append(p1.pos[0])
        energy_list.append(energy)

    # Post-simulation:
    # Close output file
    outfile.close()
    
    # Plot particle trajectory to screen
    pyplot.title('Velocity Verlet: position vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('x position')
    pyplot.plot(time_list, pos_x_list)
    pyplot.show()

    # Plot particle energy to screen
    pyplot.title('Velocity Verlet: total energy vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('Energy')
    pyplot.plot(time_list, energy_list)
    pyplot.show()
    
    pos_x_array=np.array(pos_x_list) #convert list of x positions to np array
    time_array=np.array(time_list) #convert list of times to np array
    
    """The next section of code will determine the period, frequency and 
    wavenumber of oscillations. The method I will use is to find peaks using
    scipy.signal.find_peaks and then measure the average time interval between
    peaks, which corresponds to the period of oscillations."""
    

    
    peaks_x_pos,_=find_peaks(pos_x_array) #determine indices of peaks
        
    period_list=[] #empty list to hold period values
    
    for n in range(0,(len(peaks_x_pos)-1)): #find difference in times between all the peaks
        #append period values to the list
        period_list.append(time_array[(peaks_x_pos[n+1])]-time_array[(peaks_x_pos[n])]) 
    
    period_array=np.array(period_list) #find average of period values
    T=np.average(period_array) #return average period
    print('The value of the period is: '+ str(T) + ' [T]')
    
    unit_T=1.018050571e-14 #units of T in seconds
    
    T_seconds=T*unit_T #convert period to seconds
    
    frequency=1/T_seconds #find frequency
    
    wavenumber=frequency/c_cm_per_sec #find wavenumber in cm^-1    
    
    print('The frequency of oxygen atom oscillations is ' + str(frequency) + ' s^-1')
    print('The wavenumber of oxygen atom oscillations is ' + str(wavenumber) + ' cm^-1')
    
    """The final part of the programme will determine the energy inaccuracy for a 
    simulation run. We can define delta_E as the difference between maximum and 
    minimum fluctuations. The energy inaccuracy is given by delta_E/E_o, where 
    E_o is the intial energy. delta E can be found using the same method as was 
    used to find the period in the x-position oscillations."""

    energy_array=np.array(energy_list) #create numpy array for energies
    
    peaks_energy,_=find_peaks(energy_array) #print indices of energy peaks
        
    delta_E_list=[] #empty list to hold delta_E values
       
    for n in range(0,(len(peaks_energy)-1)): #find difference in times between all the peaks
        delta_E_list.append(energy_list[(peaks_energy[n+1])]-energy_list[(peaks_energy[n])]) #append energies to list
            
    delta_E_array=np.array(delta_E_list) #find average of delta_E values
    delta_E=abs(np.average(delta_E_array)) #return average delta E
    print('The value of the delta_E is: '+ str(delta_E) + ' eV')
    
    initial_energy=energy_array[0] #find initial energy at t=0
    
    energy_inaccuracy=delta_E/initial_energy #find the value of the innacuracy
    
    print('The energy inaccuracy for this simulation run is ' + str(energy_inaccuracy))


# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()