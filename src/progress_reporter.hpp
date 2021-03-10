#pragma once

#include <string>

#include "types.hpp"

namespace convexnmf {

class ConvexNMFProgressReporter {
  public:
    ConvexNMFProgressReporter() : space{"    "} {}
    void PrintHeader();
    void PrintStatistics(int iteration, Scalar primal, Scalar primal_tol, Scalar dual,
                         Scalar dual_tol);
    void PrintHitMaxIterations(int iteration, Scalar primal, Scalar primal_tol, Scalar dual,
                               Scalar dual_tol);
    void PrintConverged(int iteration, Scalar primal, Scalar primal_tol, Scalar dual,
                        Scalar dual_tol);

    std::string space;
};

} // namespace convexnmf