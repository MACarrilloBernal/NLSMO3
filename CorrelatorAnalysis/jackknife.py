#!usr/bin/python3.9
# Filename: jackknife.py

import numpy as np

''' jackknife.ensemble '''
''' Takes a data sample 'sample' and a number of bins 'bins'
    and returns a jackknife resampled set of data '''

def ensemble( sample, bins ):

    bin_len = int( sample.size / bins )                                 # Number of elements contained in a bin

    jk_ensemble = np.array( [ np.sum( sample[ : i * bin_len ] )         # Sum of elements before the ith-bin
                              + np.sum( sample[ (i+1) * bin_len : ] )   # Sum of elements after the ith-bin
                              for i in range( bins ) ] )               # i Runs over the number of bins

    return jk_ensemble / ( sample.size - bin_len )                      # Resampled jk-ensemble

''' quickknife '''
''' For the specific case when the number of bins is equal to
    the number of elements within the sample, quickknife offers
    a more efficient way to genenerate a jackknife ensemble '''
def quickknife( sample ):

    N = sample.size
    jk_ensemble = np.zeros( N, dtype='float64' )+1j*np.zeros(N,dtype='float64')
    counter = np.arange( N )

    for i in range( N ):
        jk_ensemble[i] = np.sum( sample[ counter != i ] )

    return jk_ensemble / float( N - 1.0 )

''' jk_std '''
''' Takes a jk-ensemble and returns its jackknife-defined standard
    deviation, which is a measure of the error of the observable '''
def jk_std( jk_ensemble ):

    N = jk_ensemble.size                            # Number of elements in the jk-ensemble
    jk_mn = np.mean( jk_ensemble )                      # Mean value of the jk-ensemble
    square_sum = np.sum( ( jk_ensemble - jk_mn ) *
                         ( jk_ensemble - jk_mn ) )  # Sum over the squared differences between the jk_ensemble and its mean value

    return np.sqrt( ( float(N) - 1.0 ) * square_sum / float(N) )  # jk-standard deviation

''' jk_cov_element '''
''' Takes two jk-ensembles and returns their covariance as defined
    for jackknife statistics '''
def jk_cov( jk_ensemble_n, jk_ensemble_m ):

    N        = jk_ensemble_n.size                       # Number of elements in the jk-ensembles
    jk_mn_n  = np.mean( jk_ensemble_n )                 # mean values of the jk-ensembles
    jk_mn_m  = np.mean( jk_ensemble_m )
    prod_sum = np.sum( ( jk_ensemble_n - jk_mn_n )      # Sum over the products of the differences between
                       * ( jk_ensemble_m - jk_mn_m ) )  # each jk-ensemble and its corresponding mean value

    return ( N - 1. ) * prod_sum / float( N )           # jk-defined covariance

''' jk_cov_matrix '''
''' Takes all of the jk-ensembles and returns their covariance matrix,
    whose elements are given by the 'jk_cov' function '''
def jk_cov_matrix( all_jk_ensembles ):

    all_jk_ensembles = np.array( all_jk_ensembles )           # All jk-ensembles must be numpy arrays
    cov_matrix  = []                                          # Stores the jk-covariance matrix
    for jk_ensemble_n in all_jk_ensembles:                    # Runs over all jk-ensembles
        row = [ ]                                             # Stores one row of the matrix
        for jk_ensemble_m in all_jk_ensembles:                # Runs over all jk-ensembles
            col = jk_cov( jk_ensemble_n, jk_ensemble_m )      # Stores one element of the matrix
            row.append( col )                                 # Append the current element to the row
        cov_matrix.append( row )                              # Append row to the matrix

    return np.array( cov_matrix )                             # jk-covariance matrix
