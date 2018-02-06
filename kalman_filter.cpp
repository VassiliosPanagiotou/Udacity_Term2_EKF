#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
	x_ = F_ * x_;
	P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  //Prediction and Measurement residual
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;

  //Kalman gain
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd K = P_ * Ht * S.inverse();
	
	//New estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  //Residual using h(x_)
	VectorXd z_pred = VectorXd(3);
  z_pred(0) = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
  if ( abs(z_pred(0)) >= 0.00000001 )
  {
    z_pred(1) = atan2(x_(1), x_(0));
    z_pred(2) = (x_(0)*x_(2) + x_(1)*x_(3)) / z_pred(0);
	  VectorXd y = z - z_pred;

    //Adjust measured angle
    static const double PI = 3.14159265359;
    while ( y(1) >  PI ) y(1) -= PI;
    while ( y(1) < -PI ) y(1) += PI;

	  //Kalman gain
	  MatrixXd Ht = H_.transpose();
	  MatrixXd S = H_ * P_ * Ht + R_;
	  MatrixXd K = P_ * Ht * S.inverse();

	  //new estimate
	  x_ = x_ + (K * y);
	  long x_size = x_.size();
	  MatrixXd I = MatrixXd::Identity(x_size, x_size);
	  P_ = (I - K * H_) * P_;
  }
  else
  {
    std::cout << "Skipping EKF update step because position norm is close to singular!\n";
  }
}
