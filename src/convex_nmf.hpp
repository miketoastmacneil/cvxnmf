#pragma once

#include <memory>
#include <string>

#include "progress_reporter.hpp"
#include "row_sparsifiers.hpp"
#include "types.hpp"

namespace convexnmf {

struct ConvexNMFResults {

    ConvexNMFResults(const MatrixRef &in_W, Scalar in_primal_residual, Scalar in_dual_residual,
                     bool in_converged)
        : W{in_W}, primal_residual{in_primal_residual},
          dual_residual{in_dual_residual}, converged{in_converged} {}
    Scalar primal_residual;
    Scalar dual_residual;
    bool   converged;

    Matrix W;
};

class ConvexNMF {
  public:
    ConvexNMF()
        : sparsifier_type{SparsifierType::L2}, rho{10.0}, relative_tolerance{1.0e-3},
          abs_tolerance{1.0e-4}, max_iterations{1000}, report_progress{false}, iteration_to_print{
                                                                                   100} {}

    ConvexNMFResults Fit(const MatrixRef &X, Scalar lambda, const MatrixRef &W_init);
    ConvexNMFResults Fit(const MatrixRef &X, Scalar lambda);

    void SetSparsifierType(const std::string &type_name, bool verbose);

    bool           report_progress;
    Scalar         rho, relative_tolerance, abs_tolerance;
    int            max_iterations, iteration_to_print;
    SparsifierType sparsifier_type;
};

} // namespace convexnmf
