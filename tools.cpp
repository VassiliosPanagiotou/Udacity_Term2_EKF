#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  VectorXd rmse;
  if ( estimations.size() == ground_truth.size() )
  {
    unsigned size =  estimations.size();
    if ( size != 0 )
    {
      if ( estimations[0].size() == ground_truth[0].size() )
      {
        unsigned dim = estimations[0].size();
        rmse = VectorXd(dim);
        //Error squares
        for ( unsigned d=0; d<dim; d++ )
        {
          rmse(d) = 0;
        }
        for ( unsigned s=0; s<size; s++ )
        {
          if ( (estimations[s].size() == dim) && (ground_truth[s].size() == dim) )
          {
		        VectorXd residual = estimations[s] - ground_truth[s];
		        residual = residual.array()*residual.array();
		        rmse += residual;
          }
          else
          {
            std::cout << "ERROR: CalculateRMSE called with unexpected dimensions!\n";
          }
        }
      
	      rmse = rmse/size;
	      rmse = rmse.array().sqrt();
      }
      else
      {
        std::cout << "ERROR: CalculateRMSE called with different dimensions\n";
      }
    }
    else
    {
      std::cout << "ERROR: CalculateRMSE called with sizes 0!\n";
    }
  }
  else
  {
    std::cout << "ERROR: CalculateRMSE called with different sizes\n";
  }
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
 
  double p2 = x_state(0)*x_state(0) + x_state(1)*x_state(1),
         p2_1o2 = sqrt(p2),
         p2_3o2 = p2 * p2_1o2,
         sp     = x_state(2)*x_state(1) - x_state(3)*x_state(0);
  MatrixXd jacobian;
  jacobian = MatrixXd(3,4);
  if ( abs(p2) >= 0.00000001 )
  {
    jacobian << x_state(0) / p2_1o2,       x_state(1) / p2_1o2,      0,                   0,
               -x_state(1) / p2,           x_state(0) / p2,          0,                   0,
                x_state(1) * sp / p2_3o2, -x_state(0) * sp / p2_3o2, x_state(0) / p2_1o2, x_state(1) / p2_1o2;
  }
  else
  {
    std::cout << "Jacobian not calculable!\n";
    jacobian << 0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0;
  }

  return jacobian;
}
