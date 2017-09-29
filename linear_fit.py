#! /usr/bin/env python

#this is a script used for affine function fit 

import numpy as np
import matplotlib.pyplot as plt

def main():
    
    b, x = np.loadtxt("data_to_read.txt", unpack=True)

    #data
    b = b.reshape( ( len( b ), 1 ) )

    #measurements
    x = x.reshape( ( len( x ), 1 ) )
    A = np.hstack( ( x, np.ones( ( x.shape[0], 1 ) ) ) )
    a1, a0 = np.linalg.lstsq( A, b )[0]

    #plot
    plt.figure( 1 )
    plt.title( 'Fit' )
    plt.plot( x, b, 'o', label='Original data', markersize=5)
    plt.plot( x, a1*x + a0, 'r', label='Fitted line')
    plt.grid( True )
    plt.legend()

    plt.show()

if __name__=='__main__':

    main()
