# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:41:13 2022

@author: s1934251
"""

def main():
    # Initialise data lists for plotting later
    time=0
    time_list = [time]
    pos_x_list = [pbc(particles[0].pos, L)[0]]
    energy_list = [tot_potential_energy(particles)+Particle3D.sys_kinetic(particles)]

    # Start the time integration loop
    for i in range(numstep):
        
        #compute all forces
        all_forces=update_all_forces(particles)
        
        # Update particle positions
        update_position(particles, all_forces)
        
        # Compute force based on new positions
        all_forces_new=update_all_forces(particles)
               
        # Update particle velocities by averaging current and new forces
        update_vel(particles, 0.5*(all_forces+all_forces_new))
        
        #find total energy of system
        energy=Particle3D.sys_kinetic(list(particles))+tot_potential_energy(particles)
        
        # Redefine forces
        all_forces=all_forces_new        
        
        # Increase time
        time += dt
        
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