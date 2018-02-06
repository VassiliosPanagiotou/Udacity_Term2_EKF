#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  
	is_initialized_ = false;
	previous_timestamp_ = 0;

  //Init matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //Measurement matrix - laser
  H_laser_ <<	1, 0, 0, 0,
				0, 1, 0, 0;

  //Covariance matrix - laser
  R_laser_ <<	0.0225, 0,
				0, 0.0225;

  //Covariance matrix - radar
  R_radar_ <<	0.09, 0, 0,
				0, 0.0009, 0,
				0, 0, 0.09;

  /**
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */

  // Init Timespan
  double dt=1.0,
         dt_2 = dt*dt,
         dt_3 = dt*dt_2,
         dt_4 = dt*dt_3;

  //Initial state
  VectorXd x_init = VectorXd(4);
  x_init << 1, 1, 1, 1;
  //Initial state covariance matrix
  MatrixXd P_init = MatrixXd(4, 4);
	P_init << 1, 0, 0, 0,
			  0, 1, 0, 0,
			  0, 0, 1000, 0,
			  0, 0, 0, 1000;

  //Initial state transition matrix
  MatrixXd F_init = MatrixXd(4, 4);
	F_init << 1, 0, dt, 0,
			  0, 1, 0, dt,
			  0, 0, 1, 0,
			  0, 0, 0, 1;

  //Initial process noise matrix
  double noise_ax = 9.0,
         noise_ay = 9.0;
  MatrixXd Q_init;
  Q_init = MatrixXd(4, 4);
	Q_init << (dt_4/4.0)*noise_ax,		0,					(dt_3/2.0)*noise_ax,	0,
			      0,					(dt_4/4.0)*noise_ay, 0,						(dt_2/2.0)*noise_ay,
			      (dt_3/2.0)*noise_ax,	0,                   dt_2 * noise_ax,     0,
			      0,					(dt_3/2.0)*noise_ay, 0,                   dt_2*noise_ay;

  ekf_.Init(x_init, P_init, F_init, H_laser_, R_laser_, Q_init);
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

	if (!is_initialized_) {
    //Initialize the state with the first measurement.
    //Create the covariance matrix.
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      cout << "EKF: first Radar measurement " << endl;
	  //Convert from polar to cartesian coordinates and initialize state
      double cos_phi = cos(measurement_pack.raw_measurements_[1]);
      double sin_phi = sin(measurement_pack.raw_measurements_[1]);
      ekf_.x_(0) = measurement_pack.raw_measurements_[0] * cos_phi;
      ekf_.x_(1) = measurement_pack.raw_measurements_[0] * sin_phi;
      ekf_.x_(2) = 0; 
      ekf_.x_(3) = 0; 
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      cout << "EKF: first Laser measurement " << endl;
      // Initialize state
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    previous_timestamp_ = measurement_pack.timestamp_;
	is_initialized_ = true;

    return;
  }
  
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  
  //Compute time difference
	double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
	previous_timestamp_ = measurement_pack.timestamp_;

  //Update state transition matrix
  ekf_.F_(0,2) = dt;
	ekf_.F_(1,3) = dt;

  //Update process noise covariance matrix
  double dt_2 = dt*dt,
         dt_3 = dt*dt_2,
         dt_4 = dt*dt_3;
  double noise_ax = 9.0,
         noise_ay = 9.0;
	ekf_.Q_(0,0) = dt_4/4.0 * noise_ax;
	ekf_.Q_(0,2) = dt_3/2.0 * noise_ax;
	ekf_.Q_(1,1) = dt_4/4.0 * noise_ay;
	ekf_.Q_(1,3) = dt_3/2.0 * noise_ay;
	ekf_.Q_(2,0) = dt_3/2.0 * noise_ax;
	ekf_.Q_(2,2) = dt_2     * noise_ax;
	ekf_.Q_(3,1) = dt_3/2.0 * noise_ay;
	ekf_.Q_(3,3) = dt_2     * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    cout << "EKF: measurement update RADAR" << endl;
	//Update state transition matrix and the covariance
	//Use the prediction to calculate jacobians
    // Radar updates
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }
  else {
    cout << "EKF: measurement update LASER" << endl;
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
