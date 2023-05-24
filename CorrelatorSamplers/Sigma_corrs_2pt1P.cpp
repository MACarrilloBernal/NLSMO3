/*******************************************************************************

Sigma_correlators.cpp

Analysis algorithm for the one-particle correlators of the non-linear O(3)
sigma-model.

References:

[1] Luscher and Wolff 1990
    Nucl.Phys.B 339 (1990) 222-252

5•18•21                                                       MA Carrillo-Bernal

Sigma_corrs_2pt1P.cpp

5•19•23                                                       MA Carrillo-Bernal

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

const int TIME        =  128;      // Time extension of the lattice
const int LEN         =   64;      // Space extension of the lattice
const int ORDER       =    3;      // sigma O(n) order
const int CONF        = 100000;    // Number of configurations to be generated

const double INV_L    = 1.0/LEN;             // Inverse of the length
const double UNIT     = 2.0 * M_PI * INV_L;  // Smallest, non-zero momentum

const int TMAX        =  64;  // Maximum correlation time to be considered
const int TAVG        =  50;  // Displacements in time dimension
const int IN          =   0;  // Label for incoming state
const int OUT         =   1;  // Label for outgoing state

const int MOMENTUM    =   8;  // Momentum of the particle

/***** JACKKNIFE PARAMETERS ******/

const int NOFBINS     = 1000;  // Number of subsets for the configurations
const int BIN_LEN     = CONF / NOFBINS;  // Number of configurations per subset
const double BIN_DENM = 1.0 / ( BIN_LEN );         // Useful denominator
const double AVG_DENM = 1.0 / ( BIN_LEN * TAVG );  // Useful denominator

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
// Returns: One particle functional at a given lattice site (real part)
// Uses:    sin

double Im_one_particle( int t, int component, int n ) {
  double Im_FourierT = 0.0;
  for( int x = 0; x < LEN; x++ ) {
    Im_FourierT -= field[t][x][component] * sin( x * n * UNIT );
  }
  return Im_FourierT;
}

// This 'Im_one_particle' method computes the imag part of the discrete Fourier
// Transform of the field at a given time 't'.
//
//////

int main() {

  double * re_corrs = new double[TMAX];
  for ( int t = 0; t < TMAX; i++) {
        re_corrs[t] = 0.0;
  }

  double * im_corrs = new double[TMAX];
  for ( int t = 0; t < TMAX; i++) {
        im_corrs[t] = 0.0;
  }

  ifstream input_field;              // Will store the input file.
  input_field.open( "/scratch/mcarr020/L64/field.dat" );
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

        re_operators[OUT] = 0.0;
        im_operators[OUT] = 0.0;


        // Compute the functionals for the given configuration and
        // correlation times t and t_not

        re_operators[IN]  = 0.0;
        im_operators[IN]  = 0.0;

  	    for ( int a = 0; a < ORDER; a++ ) {
          re_operators[OUT] += Re_one_particle( t+t_not, a, MOMENTUM );
          im_operators[OUT] += Im_one_particle( t+t_not, a, MOMENTUM );
          re_operators[IN]  += Re_one_particle( t_not, a, MOMENTUM );
          im_operators[IN]  += - Im_one_particle( t_not, a, MOMENTUM );
	      }

        // Compute the real part of the correlator

        re_corrs[t] += ( re_operators[OUT] * re_operators[IN]
                                       - im_operators[OUT] * im_operators[IN] );

        // Compute the imaginary part of the correlator

        im_corrs[t] += ( im_operators[OUT] * re_operators[IN]
                                       + re_operators[OUT] * im_operators[IN] );

      } // Closes loop over time averages
    } // Closes loop over correlation times

    // If a number of configurations equal to the length of a bin has been
    // parsed, then save the binned correlator to the corresponding file and
    // reset its alllocated memory to zero.

    if ( !( (m+1)%BIN_LEN ) ) {

      // Define the name of the files

      string re_name = "re_one_corr_n"+to_string( i_rel ),
             im_name = "im_one_corr_n"+to_string( i_rel ),
             type    = ".bn",
             folder  = "/scratch/mcarr020/L64/2pt/1P/";

      // Set the variable to update the files

      ofstream write_re_corr;
      write_re_corr.open( folder+re_name+type, ios_base::app );

      ofstream write_im_corr;
      write_im_corr.open( folder+im_name+type, ios_base::app );

      // Go over all the considered correlation times

      for ( int t = 0; t < TMAX; t++) {

        // Binned values for the correlator (real and imaginary parts)

        double bn_reC_2pt = re_corrs[t] * AVG_DENM,
               bn_imC_2pt = im_corrs[t] * AVG_DENM;

        // Write the binned values on the files

        write_re_corr <<fixed<<setprecision(18)<< bn_reC_2pt << " ";
        write_im_corr <<fixed<<setprecision(18)<< bn_imC_2pt << " ";

        // Reset the memory allocated for the correlators

        re_corrs[t] = 0.0;
        im_corrs[t] = 0.0;

      } // Closes loop over correlation times

      write_re_corr << "\n"; write_im_corr << "\n";
      write_re_corr.close(); write_im_corr.close();

    } // Closes condition for saving binned correlators
  } // Closes loop over field configurations

  return 0;
} // End of main()
