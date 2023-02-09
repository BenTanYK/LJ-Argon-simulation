"""
 CompMod Ex2: Particle3D, a class to describe point particles in 3D space

 An instance describes a particle in Euclidean 3D space: 
 velocity and position are [3] arrays. 
 
 pbc has been applied to every position

 author: Benedict Tan s1934251 

"""

import numpy as np
from PBC import pbc
from PBC import mic


class Particle3D(object):
    """
    Class to describe point-particles in 3D space

        Properties:
    label: name of the particle
    mass: mass of the particle
    pos: position of the particle
    vel: velocity of the particle

        Methods:
    __init__
    __str__
    kinetic_e  - computes the kinetic energy
    momentum - computes the linear momentum
    update_pos - updates the position to 1st order
    update_pos_2nd - updates the position to 2nd order
    update_vel - updates the velocity

        Static Methods:
    new_p3d - initializes a P3D instance from a file handle
    sys_kinetic - computes total K.E. of a p3d list
    com_velocity - computes total mass and CoM velocity of a p3d list
    """

    def __init__(self, label, mass, position, velocity):
        """
        Initialises a particle in 3D space

        :param label: String w/ the name of the particle
        :param mass: float, mass of the particle
        :param position: [3] float array w/ position
        :param velocity: [3] float array w/ velocity
        """
        self.label=str(label)
        self.mass=float(mass)
        self.pos=position
        self.vel=velocity
        


    def __str__(self, L):
        """
        XYZ-compliant string. The format is
        <label>    <x>  <y>  <z>
        """
        xyz_string=str(self.label) + '    ' + str(pbc(self.pos, L)[0]) + '  ' + str(pbc(self.pos, L)[1]) + '  ' + str(pbc(self.pos, L)[2]) + '\n'
        return xyz_string


    def kinetic_e(self):
        """
        Returns the kinetic energy of a Particle3D instance

        :return ke: float, 1/2 m v**2
        """
        ke=0.5*self.mass*(np.linalg.norm(self.vel))**2
        return ke


    def momentum(self):
        """
        Returns the momentum of a Particle3D instance

        :return p: [3] float array w/ momentum
        
        """
    
        p=self.mass*self.vel
        return p
    

    def update_pos(self, dt, L):
        """
        Updates the position of a Particle3D instance to first order
        
        :param dt: timestep used to update position 
        :param L: unit cell cubic side length
     
        """
        self.pos=self.pos+dt*self.vel


    def update_pos_2nd(self, dt, force, L):
        """
        Updates the position of a Particle3D instance to second order
        
        :param dt: timestep used to update position
        :param force: [3] float array w/ force acting on the particle 
        :param L: unit cell cubic side length
     
        """
        self.pos=pbc((self.pos+dt*self.vel+((dt**2)/(2*self.mass))*force), L)


    def update_vel(self, dt, force):
        """
        Updates the velocity of a Particle3D instance to second order

        :param dt: timestep used to update position
        :param force: [3] float array w/ force acting on the particle          
        
        """
        self.vel=self.vel+force*dt*(1/self.mass)


    @staticmethod
    def new_particle(file_handle):
        """
        Initialises a Particle3D instance given an input file handle.
        
        The input file should contain one line per planet in the following format:
        label   <mass>  <x> <y> <z>    <vx> <vy> <vz>
        
        :param inputFile: Readable file handle in the above format

        :return Particle3D instance
        """
        
        vals=file_handle.readline().split()
                          
        label=str(vals[0])
        mass=float(vals[1])
        position=np.array([float(vals[2]), float(vals[3]), float(vals[4])])
        velocity=np.array([float(vals[5]), float(vals[6]), float(vals[7])])
        
        return Particle3D(label, mass, position, velocity)


    @staticmethod
    def sys_kinetic(p3d_list):
        """
        Computes the total kinetic energy of a list of P3D's
        
        :param p3d_list: list in which each item is a P3D instance
        :return sys_ke: the total kinetic energy of the system     
              
        """
        
        sys_ke=0
        for n in range(0, len(p3d_list)):
            sys_ke=sys_ke+p3d_list[n].kinetic_e()           
                        
        return sys_ke


    @staticmethod
    def com_velocity(p3d_list):
        """
        Computes the total mass and CoM velocity of a list of P3D's

        :param p3d_list: list in which each item is a P3D instance
        :return total_mass: The total mass of the system 
        :return com_vel: Centre-of-mass velocity
        """
        
        total_mass=0
        for n in range(0, len(p3d_list)): #use for loop to sum masses
            total_mass=total_mass+p3d_list[n].mass 
            
        sum_mv=np.zeros(3)
        for n in range(0, len(p3d_list)): #for loop to to sum mv products
            sum_mv=sum_mv+(p3d_list[n].vel*p3d_list[n].mass)
        
        com_vel=(1/total_mass)*sum_mv
        
        return total_mass, com_vel
    

    
