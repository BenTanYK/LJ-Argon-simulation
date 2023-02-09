"""

CMod Ex1: programme exploring periodic boundary conditions (PBC)

Creates two methods, one of which, given x and l, returns the image of x 
inside the cube 0 ≤ x_i < l, and the other, given x and l, returns the image
of x closest to the origin

Author: B. Tan s1934251
Version: 10/2021

"""

import numpy as np

def pbc(x, l): 
    
    """
    Image of x
    
    :param x: position vector as a numpy array [x_1,x_2,x_3]
    :param l: side length of cube (must be a positive number)
    :return: image of x inside cube 0 ≤ x_i < l
    """
    
    l_vector=l*np.array([1,1,1])
    image_of_x=np.mod(x, l_vector)
    return image_of_x

def mic(x, l):
    
    """
    Closest image of x
    
    :param x: position vector as a numpy array [x_1,x_2,x_3]
    :param l: side length of cube (must be a positive number)
    :return: closest image of x to the origin
    """        
    
    """Use image_x function to find image of x in the cube with 0 ≤ x_i < l.
    Then take each component of this vector, and if it is further than l/2 
    from the origin, subtract l from the component. """
    
    positive_x=pbc(x, l) #find image in cube with 0 ≤ x_i < l
    
    if positive_x[0]>l/2: #x-component
        x_1=positive_x[0]-l
    else:
        x_1=positive_x[0]
    
    if positive_x[1]>l/2: #y-component
        x_2=positive_x[1]-l
    else:
        x_2=positive_x[1]
        
    if positive_x[2]>l/2: #z-component
        x_3=positive_x[2]-l
    else:
        x_3=positive_x[2]
        
    closest_image=np.array([x_1, x_2, x_3])
        
    return closest_image
    

def main(): #test method for pbc and mic function with user defined vector
    x_1=float(input('The x_1 component of position vector x is: '))
    x_2=float(input('The x_2 component of position vector x is: '))
    x_3=float(input('The x_3 component of position vector x is: '))
    l=float(input('The side length of the periodic box is:'))    
    
    x=np.array([x_1,x_2,x_3])
    
    print('The image of x inside the cube 0 ≤ x_i < l is: ', pbc(x, l))
    print('The closest image of x to the origin is: ', mic(x, l))

# Execute main method, but only if it is invoked directly    
if __name__ == "__main__":
    main()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    