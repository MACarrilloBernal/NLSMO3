/*******************************************************************************

Sigma_2P_corrs_iso0_Re.cpp

Analysis algorithm for the two-particle correlators of the non-linear O(3)
sigma-model in the isoscalar channel (real part).

References:

[1] Luscher and Wolff 1990
    Nucl.Phys.B 339 (1990) 222-252

7•12•22                                                       MA Carrillo-Bernal

*******************************************************************************/

/***** HEADERS ******/

#define _USE_MATH_DEFINES ; // for calling Pi

#include <iostream>   // cin, cout
#include <array>      // array<TYPE,SIZE>
#include <set>        // set<TYPE>
#include <vector>     // vector<TYPE>
#include <fstream>    // ifstream, ofstream, open, close
#include <string>     // string, getline
#include <sstream>    // stringstream
#include <iomanip>    // setprecision
#include <cmath>      // atan, acos, cos, sin, M_PI

using namespace std;

/*
/******************************************************************************/

/***** LATTICE VOLUME AND NUMBER OF CONFIGURATIONS ******/

// GLOBAL VARIABLES

const int TIME       =  128;      // Time extension of the lattice
const int LEN        =   64;      // Space extension of the lattice
const int ORDER      =    3;      // sigma O(n) order
const int CONF       = 10000;      // Number of configurations to be generated

const double INV_L    = 1.0/LEN;             // Inverse of the length
const double N_TOTAL  = 1.0;
const double UNIT     = 2.0 * M_PI * INV_L;

const int TMAX        =  16;
const int TAVG        =  32;
const int IN          =   0;
const int OUT         =   1;

const int N_REL_OUT = 1;
const int N_REL_IN = 1;

/***** JACKKNIFE PARAMETERS ******/

const int NOFBINS     = 1000;
const int BIN_LEN     = CONF / NOFBINS;
const double AVG_DENM = 1.0 / ( BIN_LEN * TAVG );

array<array<array<double,ORDER>,LEN>,TIME> field;

/*
/******************************************************************************/

//////
//
// METHOD:  Build_spin
// Takes:   Polar and azimuthal  angles
// Returns: A spin vector of magnitude 1
// Uses:    sin, cos

array<double,ORDER> Build_spin( array<double,ORDER-1> angles ){
  array<double,ORDER> spin;
  spin[0] = sin( angles[0] ) * cos( angles[1] );
  spin[1] = sin( angles[0] ) * sin( angles[1] );
  spin[2] = cos( angles[0] );
  return spin;
}

// This 'Build_spin' method reads the polar and azimuthal  angles to generate the
// components of a spin vector of magnitude 1.
//
//////

//////
//
// METHOD:  Re_one_particle
// Takes:   time, spin component
// Returns: One particle functional at a given lattice site (real part)
// Uses:    cos

double Re_one_particle( int t, int component, int n ) {
  double Re_FourierT = 0.0;
  for( int x = 0; x < LEN; x++ ) {
    Re_FourierT += field[t][x][component] * cos( x * n * UNIT );
  }
  return Re_FourierT;
}

// This 'Re_one_particle' method computes the real part of the discrete Fourier
// Transform of the field at a given time 't'.
//
//////

//////
//
// METHOD:  Im_one_particle
// Takes:   time, spin component
// Returns: One particle functional at a given lattice site (imaginary part)
// Uses:    sin

double Im_one_particle( int t, int component, int n ) {
  double Im_FourierT = 0.0;
  for( int x = 0; x < LEN; x++ ) {
    Im_FourierT -= field[t][x][component] * sin( x * n * UNIT );
  }
  return Im_FourierT;
}

// This 'Re_one_particle' method computes the imag part of the discrete Fourier
// Transform of the field at a given time 't'.
//
//////

//////
//
// METHOD:  Re_two_particle
// Takes:   time, spin components, momenta
// Returns: Two particle functional at a given lattice site (real part)
// Uses:    Re_one_particle, Im_one_particle

double Re_two_particle( int t, int a, int b, int n_rel ) {
  double Re_Sa = Re_one_particle( t, a, n_rel ),
         Im_Sa = Im_one_particle( t, a, n_rel ),
         Re_Sb = Re_one_particle( t, b, N_TOTAL-n_rel ),
         Im_Sb = Im_one_particle( t, b, N_TOTAL-n_rel );

  return Re_Sa * Re_Sb - Im_Sa * Im_Sb;
}

// This 'Re_two_particle' method computes the real part of the two-particle
// functional using the corresponding one-particle functions.
//
//////

//////
//
// METHOD:  Im_two_particle
// Takes:   time, spin components, momenta
// Returns: Two particle functional at a given lattice site (imaginary part)
// Uses:    Re_one_particle, Im_one_particle

double Im_two_particle( int t, int a, int b, int n_rel ) {
  double Re_Sa = Re_one_particle( t, a, n_rel ),
         Im_Sa = Im_one_particle( t, a, n_rel ),
         Re_Sb = Re_one_particle( t, b, N_TOTAL-n_rel ),
         Im_Sb = Im_one_particle( t, b, N_TOTAL-n_rel );

  return Re_Sa * Im_Sb + Im_Sa * Re_Sb;
}

// This 'Im_two_particle' method computes the imag part of the two-particle
// functional using the corresponding one-particle functions.
//
//////

//////
//
// METHOD:  Re_two_particle_iso_0
// Takes:   time, momenta
// Returns: Two particle functional with isospin 0 (real part)
// Uses:    Re_two_particle, Im_two_particle

double Re_two_particle_iso_0( int t, int n_rel ) {
  double val = 0.0;
  for ( int a = 0; a < ORDER; a++ ) {
    val += Re_two_particle( t, a, a, n_rel );
  }
  return val;
}

// This 'Re_two_particle_iso_0' method computes the real part of the
// two-particle functional with isospin I=0 and z-component I_z=0.
//
//////

//////
//
// METHOD:  Im_two_particle_iso_0
// Takes:   time, momenta
// Returns: Two particle functional with isospin 0 (imag part)
// Uses:    Re_two_particle, Im_two_particle

double Im_two_particle_iso_0( int t, int n_rel ) {
  double val = 0.0;
  for ( int a = 0; a < ORDER; a++ ) {
    val += Im_two_particle( t, a, a, n_rel );
  }
  return val;
}

// This 'Im_two_particle_iso_0' method computes the real part of the
// two-particle functional with isospin I=0 and z-component I_z=0.
//
//////

/*
/******************************************************************************/

int main() {

  double * re_corrs = new double[TMAX];

  // Define the name of the files
  // 'i', 'j' indicate the units of relative linear momenta of the outgoing and
  // incoming two-particle states respectively. 'N' indicates the units of total
  // linear momentum carried by the two-particle system.

  string re_name = "/scratch/mcarr020/L64/2pt/I0/N"
		           +to_string( int( N_TOTAL ) )
		           +"/re_two_corr_i"
		           +to_string( N_REL_OUT ),
         a       = "_j"+to_string( N_REL_IN ),
         b       = "_N"+to_string( int( N_TOTAL ) ),
         type    = ".bn";

  string re_file_path = re_name+a+b+type;

  ifstream input_field;              // Will store the input file.
  input_field.open( "./field.dat" );
                                     // Open the file with the configurations.
  string line_field;                 // Will store one of the configurations.

  for ( int m = 0; m < CONF; m++) {

    getline( input_field, line_field );  // Read the m-th configuration.
    stringstream ss_field( line_field ); // Parse the m-th configuration.
    string row_field;                    // Will store one time row of the m-th
                                         // configuration.
    for ( int t = 0; t < TIME; t++) {

      getline( ss_field, row_field, ';' );    // Read the t-th time row of the
                                              // m-th configuration.
      stringstream ss_row_field( row_field ); // Parse the t-th time row.
      string col_field;                       // Will store one site in the t-th
                                              // time row of the m-th
                                              // configuration.
      for ( int x = 0; x < LEN; x++) {

        getline( ss_row_field, col_field, ' ' ); // Read the x-th site at the
                                                 // t-th time row of the m-th
                                                 // configuration.
        stringstream ss_col_field( col_field );  // Parse the x-th site.
        string component;                        // Will store one component of
                                                 // the field at (t,x).
        array<double,ORDER-1> angles;            // Will store the angle
                                                 // variables of the spin at the
                                                 // given site.
        for ( int k = 0; k < ORDER-1; k++) {

          // k=0 corresponds to the polar angle.
          // k=1 corresponds to the azimuthal angle.
          getline( ss_col_field, component, ',' ); // Read the k-th angle.

          angles[k] = stod( component );           // Save the k-th angle.
        }

        field[t][x] = Build_spin( angles );  // Set the spin at (t,x) of the
                                             // m-th configuration.
      }
    }

    array<double,2> re_operators, im_operators;

    for ( int t = 0; t < TMAX; t++) {

      for ( int t_not = 0; t_not < TAVG; t_not++) {

            re_operators[OUT] = Re_two_particle_iso_0( t+t_not, N_REL_OUT );
            im_operators[OUT] = Im_two_particle_iso_0( t+t_not, N_REL_OUT );

            re_operators[IN]  = Re_two_particle_iso_0( t_not, N_REL_IN );
            im_operators[IN]  = - Im_two_particle_iso_0( t_not, N_REL_IN );

            // Compute the real part of the correlator
            re_corrs[t] += ( re_operators[OUT]*re_operators[IN]
                             - im_operators[OUT]*im_operators[IN] );

      } // Closes loop over time averages
    } // Closes loop over correlation times

    // If a number of configurations equal to the length of a bin has been
    // parsed, then save the binned correlator to the corresponding file and
    // reset its alllocated memory to zero.

    if ( !( (m+1)%BIN_LEN ) ) {

          // Set the variable to update the files

          ofstream write_re_corr;
          write_re_corr.open( re_file_path, ios_base::app );

          // Go over all the considered correlation times

          for ( int t = 0; t < TMAX; t++) {

            // Binned values for the correlator (real and imaginary parts)

            double bn_reC_2pt = re_corrs[t] * AVG_DENM;

            // Write the binned values on the files

            write_re_corr <<fixed<<setprecision(16)<< bn_reC_2pt << " ";

            // Reset the memory allocated for the correlators

            re_corrs[t] = 0.0;

          } // Closes loop over correlation times

          write_re_corr << "\n";

          write_re_corr.close();

    } // Closes condition for saving binned correlators

  } // Closes loop over field configurations

  delete[] re_corrs;

  return 0;
} // End of main()
