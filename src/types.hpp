
#pragma once

#include <Eigen/Dense>

namespace convexnmf {

using Scalar    = double;
using Matrix    = Eigen::MatrixXd;
using Vector    = Eigen::VectorXd;
using MatrixRef = Eigen::Ref<const Matrix>;
using VectorRef = Eigen::Ref<const Vector>;

using MatrixXf    = Eigen::MatrixXf;
using VectorXf    = Eigen::VectorXf;
using VectorXfRef = Eigen::Ref<const VectorXf>;
using MatrixXfRef = Eigen::Ref<const MatrixXf>;
} // namespace convexnmf
