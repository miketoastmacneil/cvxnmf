
#include <iostream>

#include "progress_reporter.hpp"

namespace convexnmf {

void ConvexNMFProgressReporter::PrintHeader() {
    std::string out_message =
        "iter |" + space + "r_norm" + space + "eps_primal" + space + "s_norm" + space + "eps_dual";
    std::cout << out_message << std::endl;
}

void ConvexNMFProgressReporter::PrintStatistics(int iteration, Scalar primal, Scalar primal_tol,
                                                Scalar dual, Scalar dual_tol) {

    std::string s_primal = std::to_string(primal), s_primal_tol = std::to_string(primal_tol),
                s_dual = std::to_string(dual), s_dual_tol = std::to_string(dual_tol),
                s_iteration = std::to_string(iteration);
    std::string out_message = s_iteration + " | " + space + s_primal + space + s_primal_tol +
                              space + s_dual + space + s_dual_tol;
    std::cout << out_message << std::endl;
}

void ConvexNMFProgressReporter::PrintConverged(int iteration, Scalar primal, Scalar primal_tol,
                                               Scalar dual, Scalar dual_tol) {
    std::cout << "Converged at iteration:  " + std::to_string(iteration) + "\n";
    PrintStatistics(iteration, primal, primal_tol, dual, dual_tol);
}

void ConvexNMFProgressReporter::PrintHitMaxIterations(int iteration, Scalar primal,
                                                      Scalar primal_tol, Scalar dual,
                                                      Scalar dual_tol) {
    std::cout << "Exceed maximum iterations, returning best estimate.";
    PrintStatistics(iteration, primal, primal_tol, dual, dual_tol);
}
} // namespace convexnmf