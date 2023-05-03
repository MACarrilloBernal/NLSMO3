/*******************************************************************************

Sigma_generator.cpp

Generating algorithm for the non-linear O(3) sigma-model based on the analysis
presented in [1].

References:

[1] Luscher and Wolff 1990
    Nucl.Phys.B 339 (1990) 222-252

6•04•21                                                       MA Carrillo-Bernal

*******************************************************************************/

/***** HEADERS ******/

#define _USE_MATH_DEFINES ; // for calling Pi

#include <iostream>   // cin, cout
#include <random>     // random_device, mt19937_64, uniform_real_distribution
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

/***** RNG: Mersenne Twister ******/

random_device rd{};         // Provides a seed (at random) for mt19937_64
mt19937_64 engine{rd()};    // Initializes the RNG with a given seed
//mt19937_64 engine{ 100 };    // Initializes the RNG with a given seed
uniform_real_distribution<double> dist{0.0, 1.0}; // RNG of real values in [0,1)

// Random integers in [0,N)
int randint( int N ){
  int rnd = round( dist( engine ) * ( N - 1 ) );
  return rnd;
}

/*
/******************************************************************************/

/***** LATTICE PARAMETERS ******/

// GLOBAL VARIABLES

const int TIME       = 256;     // Time extension of the lattice
const int LEN        = 128;     // Space extension of the lattice
const int ORDER      = 3;       // sigma O(n) order
const int CONF       = 1000;       // Number of configurations to be generated
const int SEP        = 1000;     // Separation between configurations
const int THERMALIZE = 75000;   // Number of updates for thermalization

array<array<array<double,ORDER>,LEN>,TIME> field;

// This 'field' variable defines a TIME*LEN lattice whose sites allocate an
// array of size ORDER defining the components of the simulated field at a
// given site. To denote the time slice we use 't', to denote the spatial
// position we use 'x', and to denote the component of the field we use 'n'.

array<array<int,LEN>,TIME> memo;

// This 'memo' variable defines a TIME*LEN lattice whose enetries are integer
// numbers indicating wether the spin in a given site (from the global variable
// 'field') is part of the cluster.

/*
/******************************************************************************/


/***** MODEL PARAMETERS ******/

// GLOBAL VARIABLE

const double coupling_beta = 1.54;  // Coupling constant

/*
/******************************************************************************/

/***** METHODS ******/

//////
//
// METHOD:  cold_start
// Takes:   Assumes global variable 'field'
// Returns: Modifies global variable 'field'

void cold_start() {
  double theta = M_PI * dist( engine );         // Polar angle
  double phi   = 2.0 * M_PI * dist( engine );   // Azimuthal angle
  for ( int t = 0; t < TIME; t++) {
    for ( int x = 0; x < LEN; x++) {
      for ( int n = 0; n < ORDER; n++) {
        field[t][x][0] = sin( theta ) * cos( phi );   // x
        field[t][x][1] = sin( theta ) * sin( phi );   // y
        field[t][x][2] = cos( theta );                // z
      }
    }
  }
}

// This 'cold_start' method initializes the lattice variable 'field' such that
// the field on every site has components (1,0,0).
//
//////

//////
//
// METHOD:  hot_start
// Takes:   Assumes global variable 'field'
// Returns: Modifies global variable 'field'
// Uses:    'dist( engine )' for calling the RNG

void hot_start() {
  for ( int t = 0; t < TIME; t++) {
    for ( int x = 0; x < LEN; x++) {
      double theta = M_PI * dist( engine );         // Polar angle
      double phi   = 2.0 * M_PI * dist( engine );   // Azimuthal angle
      field[t][x][0] = sin( theta ) * cos( phi );   // x
      field[t][x][1] = sin( theta ) * sin( phi );   // y
      field[t][x][2] = cos( theta );                // z
    }
  }
}

// This 'hot_start' method initializes the lattice variable 'field' such that
// the field on every site has random components (x,y,z) notmalized to 1.
//
//////

//////
//
// METHOD:  Build_spin
// Takes:   Polar and azimuthal  angles
// Returns: A spin vector of magnitude 1
// Uses:    sin, cos

array<double,ORDER> Build_spin( array<double,ORDER-1> angles ){

  array<double,ORDER> spin; // spin vector in terms of the angular variables

  spin[0] = sin( angles[0] ) * cos( angles[1] ); // 'x' component
  spin[1] = sin( angles[0] ) * sin( angles[1] ); // 'y' component
  spin[2] = cos( angles[0] );                    // 'z' component

  return spin;
}

// This 'Build_spin' method reads the polar and azimuthal  angles to generate the
// components of a spin vector of magnitude 1.
//
//////

//////
//
// METHOD:  Get_angles
// Takes:   The spin on a lattice site
// Returns: The angles of the unit spin vector
// Uses:    acos, atan

array<double,ORDER-1> Get_angles( array<double,ORDER> spin ) {

  double z;          // 'z' component of the spin
  double phi, theta; // angle variables for unit spin

  // Define 'z' in terms of the 'x', and 'y' components
  z = pow( abs( 1.0 - spin[0]*spin[0] - spin[1]*spin[1] ), 0.5 );

  // Identify the sign of the 'z' component
  if ( spin[2] >= 0 ){
    theta = acos( z );
  }else{
    theta = acos( -z );
  }

  // Identify the quadrant corresponding to the azimuthal angle
  if ( spin[1]>=0 ){
    if ( spin[0]>0  ){ phi = atan( spin[1] / spin[0] ); }else{
      if ( spin[0]==0 ){ phi = 0.5*M_PI; }
      else{ phi = M_PI + atan( spin[1] / spin[0] ); }
    }
  }else{
    if ( spin[0]<0  ){ phi = M_PI + atan( spin[1] / spin[0] ); }else{
      if ( spin[0]==0 ){ phi = 1.5*M_PI; }
      else{ phi = 2.0*M_PI + atan( spin[1] / spin[0] ); }
    }
  }

  return { theta, phi };
}

// This 'Get_angles' method takes the spin on a lattice site and computes its
// polar and azimuthal angles.
//
//////

//////
//
// METHOD:  Normalize
// Takes:   Assumes global variable 'field'
// Returns: None
// Uses:    pow, abs

void Normalize() {

  for ( int t = 0; t < TIME; t++) {
    for ( int x = 0; x < LEN; x++) {
      // Identify the sign of 'z'
      if ( field[t][x][2] >= 0 ){
        field[t][x][2] = pow( abs( 1.0 - field[t][x][0]*field[t][x][0]
                                       - field[t][x][1]*field[t][x][1] ), 0.5 );
      }else{
        field[t][x][2] =-pow( abs( 1.0 - field[t][x][0]*field[t][x][0]
                                       - field[t][x][1]*field[t][x][1] ), 0.5 );
      }
    }
  }
}

// This 'Normalize' method goes through the lattice and normalizes the spins on
// the site by defining the 'z' component of the fields in terms of the 'x', and
// 'y' components.
//
//////

//////
//
// METHOD:  dot
// Takes:   Two spins
// Returns: The scalar product of the spins

double dot( array<double,ORDER> s1, array<double,ORDER> s2 ) {
  double product = 0.0;
  for ( int n = 0; n < ORDER; n++ ) {
    product += s1[n] * s2[n];
  }
  return product;
}

// This 'dot' method computes and returns the scalar product (in spin space) of
// two given spins.
//
//////

//////
//
// METHOD:  P_bond
// Takes:   Two neighboring sites in the lattice
//          Probing spin for the update
//          Assumes global variable 'field'
// Returns: The probability of activating a bond between the neighbors
// Uses:    dist( engine ), dot

double P_bond( array<int,2> point,
               array<int,2> neighbor,
               array<double,ORDER> probe ) {

  array<double,ORDER> spin_point, spin_neighbor;
  double point_probe, neighbor_probe, aux;       // Projections of the spins in
                                                 // the direction of the probing
                                                 // spin

  // Spins at the given site  and the neighboring site
  spin_point     = field[point[0]][point[1]];
  spin_neighbor  = field[neighbor[0]][neighbor[1]];

  // Projection of the spins along the probing spin
  point_probe    = dot( probe, spin_point );
  neighbor_probe = dot( probe, spin_neighbor );

  // Compute the probability of bonding both spins
  aux = point_probe * neighbor_probe;
  if ( aux >= 0.0 ) {
    return 1.0 - exp( - 2.0 * coupling_beta * aux );
  }else{ return 0.0; }

}

// This 'P_bond' method computes and returns the probability of bonding the
// spins on two neighboring sites as a function of their projection along a
// given probing spin.
//
//////

//////
//
// METHOD:  Make_bonds
// Takes:   A lattice site
//          Probing spin for the update
//          Assumes global variable 'field'
// Returns: Set of bonded spins' sites
// Uses:    dist( engine ), dot, P_bond

set<array<int,2>> Make_bonds( set<array<int,2>> blob,
                              array<double,ORDER> probe ) {

  // Will store the sites to be added to the cluster.
  set<array<int,2>> expand = {};

  // Go over the sites added to the cluster during the last iteration, which are
  // referred as the 'blob'.
  for ( array<int,2> point : blob ) {

    // Neighboring directions
    int left  = ( LEN + point[1] - 1 ) % LEN,
        up    = ( TIME + point[0] - 1 ) % TIME,
        right = ( point[1] + 1 ) % LEN,
        down  = ( point[0] + 1 ) % TIME;

    // Coordinates of the neighboring sites
    array<int,2> neighbor_left  = { point[0], left },
                 neighbor_up    = { up, point[1] },
                 neighbor_right = { point[0], right },
                 neighbor_down  = { down, point[1] };

    // Make a MC step and check if the probability of bonding two spins is met
    if ( dist(engine) <= P_bond( point, neighbor_left, probe ) ) {
      // Provided that the spin was not part of the cluster, add it to the
      // 'expand' set
      if ( memo[ point[0] ][ left ]  == 0 ) {
        expand.insert( { point[0], left } );
      }
    }
    // Make a MC step and check if the probability of bonding two spins is met
    if ( dist(engine) <= P_bond( point, neighbor_up, probe ) ) {
      // Provided that the spin was not part of the cluster, add it to the
      // 'expand' set
      if ( memo[ up ][ point[1] ]    == 0 ) {
        expand.insert( { up, point[1] } );
      }
    }
    // Make a MC step and check if the probability of bonding two spins is met
    if ( dist(engine) <= P_bond( point, neighbor_right, probe ) ) {
      // Provided that the spin was not part of the cluster, add it to the
      // 'expand' set
      if ( memo[ point[0] ][ right ] == 0 ) {
        expand.insert( { point[0], right } );
      }
    }
    // Make a MC step and check if the probability of bonding two spins is met
    if ( dist(engine) <= P_bond( point, neighbor_down, probe ) ) {
      // Provided that the spin was not part of the cluster, add it to the
      // 'expand' set
      if ( memo[ down ][ point[1] ]  == 0 ) {
        expand.insert( { down, point[1] } );
      }
    }
    // YES! THIS WILL BE OPTIMIZED! <MACB,06.09.2021>
  }

  return expand;

}

// This 'Make_bonds' method performs four Monte Carlo steps in order to bond
// together two neighboring spins, thus one step per neighbor. If the proability
// to make the bond is satisfied, and the spins are parallel with respect to the
// Wolff line (the direction of the probe spin), then the neighboring spin is
// added to the cluster.
//
//////

//////
//
// METHOD:  Flip_cluster
// Takes:   The list of sites included in the cluster
//          The probing spin
//          Assumes global variable 'field'
// Returns: None
// Uses:    dot

void Flip_cluster( set<array<int,2>> will_flip, array<double,ORDER> probe ) {

  array<double,ORDER-1> angles;  // Angular variables of the spin

  // Go over the sites of the cluster, thus the 'will_flip' set
  for ( array<int,2> point : will_flip ) {

    // Spin at a given site
    array<double,ORDER> spin = field[ point[0] ][ point[1] ];

    // Projection of the spin along the probing spin
    double projection = dot( spin, probe );

    // Update the components of the spin at the given site
    for ( int n = 0; n < ORDER; n++) {
      field[ point[0] ][ point[1] ][ n ]
        = field[ point[0] ][ point[1] ][ n ] - 2.0 * projection * probe[ n ];
    }
  }
}

// This 'Flip_cluster' method goes over the sites included in the cluster and
// updates the spins at such sites by inverting them with respect to the Wolff
// line (the direction of the probing spin). See eq. 3.7 in reference [1].
//
//////

//////
//
// METHOD:  Grow_cluster
// Takes:   Assumes global variable 'field'
//          Assumes global variable 'memo'
// Returns: None
// Uses:    dist( engine ), dot, P_bond, Make_bonds, Flip_cluster

void Grow_cluster( ) {
  int m = randint( TIME ),
      n = randint( LEN );

  array<double,ORDER> probe_spin;

  double theta = M_PI * dist( engine );        // Polar angle
  double phi   = 2.0 * M_PI * dist( engine );  // Azimuthal angle
  probe_spin[0] = sin( theta ) * cos( phi );   // x
  probe_spin[1] = sin( theta ) * sin( phi );   // y
  probe_spin[2] = cos( theta );                // z

  set<array<int,2>> will_flip = { {m,n} },     // Sites contained in the cluster
                    blob      = { {m,n} };     // Expanding part of the cluster

  // Initialize global variable 'memo'.
  for ( int i = 0; i < TIME; i++) {
    for ( int j = 0; j < LEN; j++) {
      memo[i][j] = 0;
    }
  }

  // The initial site is the only non-zero entry in 'memo' at this point.
  memo[ m ][ n ] = 1;

  // As long as the expanding part of the cluster (thus the 'blob') is a
  // non-empty set, keep growing the cluster
  while ( blob.size() != 0 ) {

    // Get the sites to be added to the cluster by probing all possible bonds
    // with the sites inside the 'blob', refer to this set of sites as 'expand'.
    set<array<int,2>> expand = Make_bonds( blob, probe_spin );

    // Empty the 'blob' and exchange it with the 'expand' set.
    blob.clear(); blob.swap( expand );

    // Go over the sites of the expansion and add them to the cluster.
    for ( array<int,2> pair : blob ) {
      memo[ pair[0] ][ pair[1] ] = 1;
      will_flip.insert( { pair[0], pair[1] } );
    }
  }

  // Invert the spins within the cluster, with respect to the direction of the
  // probing spin.
  Flip_cluster( will_flip, probe_spin );
}

// This 'Grow_cluster' method builds a cluster around a random site of the
// lattice using a spin with random direction as the probing spin. Once the
// cluster stops growing, the spins in the cluster are inverted with respect to
// the Wolff line (the direction of the probing spin).
//
//////

//////
//
// METHOD:  New_config
// Takes:   None
// Returns: None
// Uses:    Grow_cluster

void New_config( ) {
  for ( int i = 0; i < SEP; i++) {
    Grow_cluster();
  }
}

// This 'New_config' method grows and inverts a cluster a number a number 'SEP'
// of times in order to diminish correlation between measurements.
//
//////

/*
/******************************************************************************/

int main(){

  // 01 - Check for an existing file of configurations. If there is not such a
  //      file, begin from a hot/cold start and themalize the system. Else, keep
  //      adding data to the existing file.

  bool status = false; // Assume there is no backup file for the last generated
                       // configuration.

  ifstream field_check( "./field_last.dat" ); // Call the file with the last
  if ( field_check.good() ){                // generated configuration and, if
    status = true;                          // it exists, then change 'status'
  }                                         // to 'true'.
  field_check.close();

  if( status ){

    ifstream field_last;                 // Stores the file
    field_last.open( "./field_last.dat" ); // Opens the file

    // A row in the 'field_last.dat' file represents a configuration of the
    // field. The fields at different times are separated by a ';' character.
    // For a given time, fiels at different positions are separated by a ' '
    // character, thus a blank space. For a given time and position, the
    // polar and azimuthal angles of the spin in a lattice site are separated by
    // a ',' character.

    string row_field; // Store the data of the field at a given time.

    for ( int t = 0; t < TIME; t++) {         // Go over the time extension.

      getline( field_last, row_field, ';' );  // Store the data of the field at
                                              // time 't' in 'row_field', this
                                              // is delimited by ';'.

      stringstream ss_row_field( row_field ); // Parse the data of the field at
                                              // the given time 't'.

      string col_field; // Store the data of the field at a given time, and
                        // position.

      for ( int x = 0; x < LEN; x++ ) {          // Go over the spatial length.

        getline( ss_row_field, col_field, ' ' ); // Store the data of the field
                                                 // at time 't' and position 'x'
                                                 // in 'col_field', this is
                                                 // delimited by ' '.

        stringstream ss_col_field( col_field );  // Pars the data of the field
                                                 // at the given time 't' and
                                                 // position 'x'.

        string component; // Store the data of the field at a given time,
                          // position, and component.

        array<double,ORDER-1> angles; // Store the angule values.

        for ( int i = 0; i < ORDER-1; i++ ){       // Go over the components.

          getline( ss_col_field, component, ',' ); // Store the data of the
                                                   // field at 't', 'x', and
                                                   // component 'n' in the
                                                   // variable 'component', this
                                                   // is delimietd by ','.

          angles[i] = stod( component ); // Set the value of the angles.
        }
        field[t][x] = Build_spin( angles ); // Set the components of the spin
                                            // at the site ( 't', 'x' ).
      }
    }
    field_last.close(); // Close the file 'field_last.dat'.
  }else{

    // B - Since there is no file containing the last saved configuration, it is
    //     necessary to start from a hot/cold configuration and thermalize the
    //     system.

    cold_start(); // Start from a cold configuration.
    //hot_start(); // Strt from a hot configuration.

    for ( int n = 0; n < THERMALIZE; n++) { // Repeat a number 'THERMALIZE' of
                                            // times.

      Grow_cluster(); // Update the field configuration.
    }
  }

  // 02 - Open the file containing the configurations of the fields and add a
  //      number 'CONFS' of configurations separated by a number 'SEP' of
  //      updates.

  ofstream field_confs;                             // Open the file that stores
  field_confs.open( "./field.dat", ios_base::app ); // the information of the
                                                    // field confifurations,
                                                    // 'ios_base::app' indicates
                                                    // that the data is added to
                                                    // an already existing file.

  array<double,ORDER-1> angles; // Store the angle variables of the spin

  for ( int i = 0; i < CONF; i++) { // Repeat a number 'CONF' of times.

    if ( !(i%100) ){ Normalize(); }

    New_config(); // Generate a new configuration.

    for ( int t = 0; t < TIME; t++) {        // Go over all times.
      for ( int x = 0; x < LEN; x++) {       // Go over all postions.
        angles = Get_angles( field[t][x] );  // Compute the angles of the spin.

        // Save the angular components of the spin into the 'field.dat' file
        // with the specified number of significant figures.
        field_confs <<fixed<<showpoint<<setprecision(16)<<angles[0]<<","
                    <<fixed<<showpoint<<setprecision(16)<<angles[1];

        if ( x+1 < LEN ) { field_confs << " "; } // If the maximum position has
                                                 // not been reached, separate
                                                 // the data using a ' '
                                                 // character.
      }
      if ( t+1 < TIME ) { field_confs << ";"; }  // If the maximum time has not
                                                 // been reached, sparate the
                                                 // data using a ';' character.
    }
    field_confs<<endl;                           // If all times, positions, and
                                                 // components have been of the
                                                 // current configuration have
                                                 // been covered, start a new
                                                 // line in 'field.dat'.
  }
  field_confs.close();

  // 03 - Save the last generated configuration so it can be used as the
  //      starting configuration following the steps given above for saving
  //      the data of a given configuration.

  ofstream field_bckp;
  field_bckp.open( "./field_last.dat" );
  for ( int t = 0; t < TIME; t++) {
    for ( int x = 0; x < LEN; x++) {
      angles = Get_angles( field[t][x] );
      field_bckp <<fixed<<showpoint<<setprecision(8)<<angles[0]<<","
                 <<fixed<<showpoint<<setprecision(8)<<angles[1];
      if ( x+1 < LEN ) { field_bckp << " "; }
    }
    if ( t+1 < TIME ) { field_bckp << ";"; }
  }
  field_bckp.close();

  return 0;
}
