
#include <algorithm>
#include <cmath>
#include <iostream>

#include <Eigen/Cholesky>

#include "convex_nmf.hpp"
#include "progress_reporter.hpp"
#include "projections.hpp"

namespace convexnmf {

void ConvexNMF::SetSparsifierType(const std::string &type_name, bool verbose) {

    if (type_name == "L2") {
        if (verbose)
            std::cout << "Set Sparsifier to L2." << std::endl;
        sparsifier_type = SparsifierType::L2;
    } else if (type_name == "LInfty") {
        if (verbose)
            std::cout << "Set Sparsifier to LInfty." << std::endl;
        sparsifier_type = SparsifierType::LInfty;
    } else {
        std::string current_setting = sparsifier_type == SparsifierType::L2 ? "L2" : "LInfty";
        std::cout << "Sparsifier type not recognized so not set, choose either 'L2' or 'LInfty'.";
        std::cout << " Currently set to " + current_setting + " \n";
    }
}

ConvexNMFResults ConvexNMF::Fit(const MatrixRef &X, Scalar lambda) {
    // TODO: Can move this to the Python front end.
    int    N      = X.cols();
    Matrix W_init = Matrix::Identity(N, N);
    return Fit(X, lambda, W_init);
}

ConvexNMFResults ConvexNMF::Fit(const MatrixRef &X, Scalar lambda, const MatrixRef &W_init) {

    ConvexNMFProgressReporter reporter;
    auto sparsifier = RowSparsifier::Create(sparsifier_type);
    // W Update set up
    int N = X.cols();
    int T = X.rows();

    Matrix XTXplusDeltaId = X.transpose() * X + rho * Matrix::Identity(N, N);
    Matrix XTX            = X.transpose() * X;

    std::cout << "Factoring .... ";
    Eigen::LLT<Matrix> cholesky_solver;
    cholesky_solver.compute(XTXplusDeltaId);
    std::cout << "Done. \n";

    // Set up W, Z, U
    const Scalar sqrt_size = static_cast<Scalar>(std::sqrt(N * N));
    Matrix       W         = W_init;
    Matrix       Z_1 = W_init, Z_2 = W_init, Z_3 = W_init;
    Matrix Y_1 = Matrix::Identity(N, N), Y_2 = Matrix::Identity(N, N), Y_3 = Matrix::Identity(N, N);

    if (report_progress) {
        reporter.PrintHeader();
    }

    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        // Updates

        Matrix previous_average = (Z_1 + Z_2 + Z_3) / 3.0; // Used in monitoring convergence.

        Z_1 = cholesky_solver.solve(XTX + rho * W - Y_1);
        Z_2 = sparsifier->Evaluate(W - Y_2 / rho, lambda, rho);
        Z_3 = (W - Y_3 / rho).cwiseMax(0.0);

        W = (Z_1 + Z_2 + Z_3 + (Y_1 + Y_2 + Y_3) / rho) / 3.0;

        Y_1 = Y_1 + rho * (Z_1 - W);
        Y_2 = Y_2 + rho * (Z_2 - W);
        Y_3 = Y_3 + rho * (Z_3 - W);

        // check primal and dual residual
        Scalar primal_squared =
            (W - Z_1).squaredNorm() + (W - Z_2).squaredNorm() + (W - Z_3).squaredNorm();
        Scalar primal = std::sqrt(primal_squared);

        Matrix current_average = (Z_1 + Z_2 + Z_3) / 3.0;
        Scalar dual            = 3.0 * rho * (current_average - previous_average).norm();

        Scalar z_norm     = std::sqrt(Z_1.squaredNorm() + Z_2.squaredNorm() + Z_3.squaredNorm());
        Scalar primal_tol = std::sqrt(3.0) * abs_tolerance * sqrt_size +
                            relative_tolerance * std::max(3 * W.norm(), z_norm);
        Scalar dual_tol = abs_tolerance * sqrt_size + relative_tolerance * (Y_1 + Y_2 + Y_3).norm();

        if (report_progress && (iteration % iteration_to_print == 0)) {
            reporter.PrintStatistics(iteration, primal, primal_tol, dual, dual_tol);
        }
        if ((primal < primal_tol) && (dual < dual_tol)) {

            ConvexNMFResults results{W, primal, dual, true};
            if (report_progress) {
                reporter.PrintConverged(iteration, primal, primal_tol, dual, dual_tol);
            }
            return results;
        } else if (iteration == max_iterations) {
            ConvexNMFResults results{W, primal, dual, false};
            if (report_progress) {
                reporter.PrintHitMaxIterations(iteration, primal, primal_tol, dual, dual_tol);
            }
            return results;
        }
    }
}

} // namespace convexnmf
