
#include "projections.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

namespace convexnmf {

namespace {

template <typename T> bool IsClose(T a, T b) {
    return std::abs(a - b) < std::numeric_limits<T>::epsilon() * std::max(1.0,std::abs(a+b));
};

} // namespace

// Algorithm is from
// ``Efficient Projections onto the l1-ball for Learning in High Dimensions."
// J.Duchi, S Shalev-Shwartz, Y. Singer, T Chandra. ICML '08.
//
// Sort v into mu, with mu in descending order (indices are from 1 to n)
// set rho = max{j \in n : mu_j-(sum_{r=1}^j mu_r-z)/j}
// theta = (sum_{r=1}^rho mu_r - z)/rho
// return max(v-theta, 0.0) elementwise
Vector SimplexProjection(const VectorRef &v, Scalar z) {
    if (IsClose(v.sum(), z) && IsClose(v.minCoeff(), 0.0))
        return v;
    if (z < std::numeric_limits<Scalar>::epsilon()) {
        return Vector::Zero(v.size());
    }

    Vector mu = v;
    std::sort(mu.data(), mu.data() + mu.size(), std::greater<Scalar>());
    Vector cumulative_sum(mu.size());
    std::partial_sum(mu.data(), mu.data() + mu.size(), cumulative_sum.data());

    int rho = 0, j = mu.size() - 1;
    while (true) {
        Scalar difference = (cumulative_sum(j) - z) / (j + 1);
        if (mu(j) - difference > 0) {
            rho = j;
            break;
        }
        j--;
    }

    Scalar theta  = (cumulative_sum(rho) - z) / (rho + 1);
    Vector retval = (v - theta * Vector::Ones(v.size())).cwiseMax(0.0);

    return retval;
}

// Algorithm from
// ``Efficient Projections onto the l1-ball for Learning in High Dimensions."
// J.Duchi, S Shalev-Shwartz, Y. Singer, T Chandra. ICML '08.
Vector L1BallProjection(const VectorRef &v, Scalar z) {
    if (IsClose(v.cwiseAbs().sum(), z))
        return v;
    if (v.cwiseAbs().sum() < z)
        return v;
    if (z < std::numeric_limits<Scalar>::epsilon()) {
        return Vector::Zero(v.size());
    }
    Vector beta   = SimplexProjection(v.cwiseAbs(), z);
    Vector retval = v.cwiseSign().cwiseProduct(beta);

    return retval;
}

Vector L2BallProjection(const VectorRef &src, Scalar radius) {
    Scalar normalizer = std::max(src.norm(), radius);
    return (src / normalizer);
}

// Uses the Moreau decomposition, implementation
// came from Sra.
Matrix L1InfinityProximalOperator(const MatrixRef &A, Scalar theta) {
    Matrix B = A;
    int    M = A.rows();
    for (int i = 0; i < M; ++i) {
        Vector row = A.row(i); // Have to do this assignment in the case theta>||row||_1
                               // and row-L1BallProjection(row, theta) = 0.
        B.row(i) = row - L1BallProjection(row, theta);
    }
    return B;
}

// Projection onto the mixed norm ball from Sra.
// (TODO) add a further description to this.
Matrix L1InfinityProjection(const MatrixRef &A, Scalar gamma, Scalar root_tolerance) {
    // g(theta) definition
    auto theta_func = [&A, &gamma](Scalar theta) {
        Matrix u = L1InfinityProximalOperator(A, theta);
        return -gamma + norms::L1_Infinity_MixedNorm(u);
    };

    if (norms::L1_Infinity_MixedNorm(A) <= gamma) {
        return A;
    }
    // Compute theta_root.
    Scalar theta_min = 0.0, theta_max = norms::L1Norm(A);
    Scalar initial_guess = (theta_max - theta_min) / 2.0;
    Scalar theta_root    = BisectionRootFind(theta_func, theta_min, theta_max, root_tolerance);

    // Result is proximal operator at \theta*
    return L1InfinityProximalOperator(A, theta_root);
}

} // namespace convexnmf
