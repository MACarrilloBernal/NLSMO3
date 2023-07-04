import numpy as np
import matplotlib.pyplot as plt
import jackknife as jk

def observable( x_jk ):
    return x_jk

No_STATES = 4
T_MAX     = 64
SHIFT     = 4

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

    for t in range( T_MAX-SHIFT ):
        aux_mns.append( np.mean( np.log( aux_jk[t] / aux_jk[t+SHIFT] )/SHIFT ) )
        aux_err.append( jk.jk_std( np.log( aux_jk[t] / aux_jk[t+SHIFT] )/SHIFT ) )


    re_data_jk.append( aux_jk )
    re_data_mns.append( np.array( aux_mns ) )
    re_data_err.append( np.array( aux_err ) )

## For 10K
E_mns = [ 0.14575, 0.17540, 0.24420, 0.32468 ]
E_err = [ 0.00149, 0.00125, 0.00209, 0.00360 ]

## For 100K
E_mns = [ 0.14529, 0.17492, 0.24474, 0.32791 ]
E_err = [ 0.00060, 0.00057, 0.00109, 0.00211 ]

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

    if n>3:
        the_alpha=0.5
        the_color='b'

    axs.errorbar( t_range[:-SHIFT], re_data_mns[n].real, xerr=None,
                  yerr=re_data_err[n].real, ms=5, fmt='o', mfc='w', elinewidth=2.5,
                  capsize=4, mew=1.4, alpha=the_alpha, c=the_color, zorder=24 )

    upper = ( E_mns[n] + E_err[n] ) * np.ones( len( t_range[:-SHIFT] ) )
    lower = ( E_mns[n] - E_err[n] ) * np.ones( len( t_range[:-SHIFT] ) )

    axs.fill_between( t_range[:-SHIFT], upper, lower, facecolor='r',
                      alpha=0.85, interpolate=True, zorder=12 )

axs.set_ylim( 0.1001, 0.3999 )
axs.set_xlim( 0.0001, 14.499 )

axs.set_xlabel( r'$t$' )
axs.set_ylabel( r'$m_{\rm eff}(t)$' )

name = './1P_Meff_10K.pdf'
name = './1P_Meff_100K.pdf'

plt.savefig( name, dpi=128, facecolor='w', edgecolor='w',
        format='pdf',
        transparent=True, bbox_inches='tight', pad_inches=0.1,
        metadata=None )

plt.show()
