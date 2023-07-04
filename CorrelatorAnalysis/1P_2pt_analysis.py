import numpy as np
import matplotlib.pyplot as plt
import jackknife as jk

from iminuit import Minuit

def observable( x_jk ):
    return x_jk

No_STATES = 4
T_MAX     = 64

t_range = np.arange( T_MAX )

re_data_bn  = []
re_data_jk  = []
re_data_mns = []
re_data_err = []

file = './Correlators/re_two_corr_n'
#file = './Correlators/bckp10K/re_two_corr_n'
type = '.bn'

for n in range( No_STATES ):

    aux = []

    for line in open( file+str(n)+type ):

        aux.append( np.fromstring( line, dtype=float, sep=' ' ) )

    aux = np.array( aux ).T

    re_data_bn.append( aux )

    aux_jk  = []
    for t in range( T_MAX ):
        aux_jk.append( jk.quickknife( aux[t] ) )

    aux_mns = []
    aux_err = []

    for t in range( T_MAX ):
        aux_mns.append( np.mean( aux_jk[t] ) )
        aux_err.append( jk.jk_std( aux_jk[t] ) )


    re_data_jk.append( aux_jk )
    re_data_mns.append( np.array( aux_mns ) )
    re_data_err.append( np.array( aux_err ) )

def hypth( pars, t ):
    return pars[1] * np.exp( - pars[0] * t )

ti = 6
tf = 16

guess_E = [ 0.145, 0.175, 0.244, 0.33 ]
guess_A = [ 365, 302, 218, 161 ]

t_fit = np.arange( ti, tf+1 )

bands_max = []
bands_min = []

for n in range( No_STATES ):

    jk_samples = [ re_data_jk[n][t] for t in t_fit ]

    cov_matrix = jk.jk_cov_matrix( jk_samples )
    InvCov     = np.linalg.inv( cov_matrix )

    mns = re_data_mns[n][ti:(tf+1)]

    def Chi2( En, An ):

        fit   = hypth( [ En, An ], t_fit )
        diffs = mns - fit

        return np.dot( diffs.T, np.dot( InvCov, diffs ) )

    dof = len( t_fit ) - 2

    Chi2_fit = Minuit( Chi2, En=guess_E[n], An=guess_A[n] )

    Chi2_fit.migrad()

    CovM = []
    for i in range( len( Chi2_fit.covariance ) ):
        aux = []
        for j in range( len( Chi2_fit.covariance ) ):
            aux.append( Chi2_fit.covariance[i][j] )
        CovM.append( aux )

    CovM = np.array( CovM )

    fit_vals = [ Chi2_fit.values[i] for i in range( len( Chi2_fit.values ) ) ]

    vPARs = np.random.multivariate_normal( fit_vals, CovM, 2**10 ).T

    t_band = np.arange( ti-0.5, tf+0.5, 0.025 )

    conf_max = np.zeros( len( t_band ) )
    conf_min = np.zeros( len( t_band ) )

    for i in range( len( t_band ) ):

        aux = hypth( [ vPARs[0], vPARs[1] ], t_band[i] )

        conf_max[i] = np.mean( aux ) + np.std( aux )
        conf_min[i] = np.mean( aux ) - np.std( aux )

    bands_max.append( conf_max )
    bands_min.append( conf_min )

    print( Chi2_fit.values )
    print( Chi2_fit.errors )
    print( Chi2_fit.fval / dof )

plt.rcParams.update({'font.size': 22})

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

#plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

fig, axs = plt.subplots( 1, 1, figsize=( 9, 6 ) )

right_side = axs.spines["right"]
right_side.set_visible(False)
top_side = axs.spines["top"]
top_side.set_visible(False)

for n in range( No_STATES ):

    the_alpha=1.0
    the_color='k'

    #if n>3:
    #    the_alpha=0.5
    #    the_color='b'

    axs.errorbar( t_range, re_data_mns[n].real, xerr=None,
                  yerr=re_data_err[n].real, ms=5, fmt='o', mfc='w', elinewidth=2.5,
                  capsize=4, mew=1.4, alpha=the_alpha, c=the_color, zorder=24 )

    axs.fill_between( t_band, bands_max[n], bands_min[n], facecolor='r',
                      alpha=0.85, interpolate=True, zorder=12 )

axs.set_ylim( 0.1499, 399.9999 )
axs.set_xlim( -0.999, 36.499 )
axs.set_yscale( 'log' )

axs.set_xlabel( r'$t$' )
axs.set_ylabel( r'$C_n^{\rm 2pt}(t)$' )

#name = './1P_2pt_Analysis_10K.pdf'
name = './1P_2pt_Analysis_100K.pdf'

plt.savefig( name, dpi=128, facecolor='w', edgecolor='w',
        format='pdf',
        transparent=True, bbox_inches='tight', pad_inches=0.1,
        metadata=None )

plt.show()
