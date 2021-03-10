
#pragma once

#include <Eigen/Dense>

#include "root_finding.hpp"
#include "types.hpp"

namespace convexnmf {

namespace norms {

inline Scalar L1_Infinity_MixedNorm(const MatrixRef &A) {
    return A.cwiseAbs().rowwise().maxCoeff().sum();
}

inline Scalar L1Norm(const MatrixRef &A) { return A.cwiseAbs().sum(); }

inline Scalar LInfinity_1_MixedNorm(const MatrixRef &A) {
    return A.cwiseAbs().rowwise().sum().maxCoeff();
}

} // namespace norms

// Returns a vector which is the projection of v onto the simplex
// with sum sum_target
Vector SimplexProjection(const VectorRef &v, Scalar sum_target);

Vector L1BallProjection(const VectorRef &v, Scalar radius);
Vector L2BallProjection(const VectorRef &v, Scalar radius);

// Computes the proximal operator
// u = argmin_B 0.5 ||A - B||_F^2 + \lambda ||B||_{1, \infty}
// This separates into computing the l_\infty proximal operator
// for each row, which, in turn, is equivalent to projection onto
// the l1 ball of radius \theta.
// (TODO) Add reference
Matrix L1InfinityProximalOperator(const MatrixRef &A, Scalar lambda);

// Computes the L1infinity projection, or
// u = argmin_B ||A - B||_F^2 subject to ||B||_{1,\infty} <= radius
// using the previous initial guess can be used.
Matrix L1InfinityProjection(const MatrixRef &A, Scalar radius, Scalar root_tolerance = 1.0e-10);

} // namespace convexnmf
