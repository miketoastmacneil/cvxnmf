#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "convex_nmf.hpp"
#include "projections.hpp"
#include "types.hpp"

namespace py = pybind11;
using namespace convexnmf;

PYBIND11_MODULE(_convexnmf, m) {

    py::class_<ConvexNMFResults> convexNMFResults(m, "ConvexNMFResults");
    convexNMFResults.def(py::init<
                        const MatrixRef&,
                        Scalar, Scalar, bool>());
    convexNMFResults.def_readwrite("W", &ConvexNMFResults::W);
    convexNMFResults.def_readwrite("primal_residual", 
                                    &ConvexNMFResults::primal_residual);
    convexNMFResults.def_readwrite("dual_residual", 
                                    &ConvexNMFResults::dual_residual);
    convexNMFResults.def_readwrite("converged",
                                    &ConvexNMFResults::converged);

    py::class_<ConvexNMF> convexNMF(m,"ConvexNMF");

    convexNMF.def(py::init<>());
    convexNMF.def("Fit", py::overload_cast<const MatrixRef&, 
                                  Scalar,
                                  const MatrixRef&> 
        (&ConvexNMF::Fit), "Computes factorization with ADMM",
        py::arg("X"), py::arg("gamma"), py::arg("W_init"));
    convexNMF.def("Fit", py::overload_cast<const MatrixRef&, 
                                  Scalar>
        (&ConvexNMF::Fit), "Computes factorization with ADMM",
        py::arg("X"), py::arg("gamma"));
    convexNMF.def("SetSparsifierType", &ConvexNMF::SetSparsifierType);
    convexNMF.def_readwrite("report_progress", &ConvexNMF::report_progress);
    convexNMF.def_readwrite("rho", &ConvexNMF::rho);
    convexNMF.def_readwrite("relative_tolerance", &ConvexNMF::relative_tolerance);
    convexNMF.def_readwrite("abs_tolerance", &ConvexNMF::abs_tolerance);
    convexNMF.def_readwrite("max_iterations", &ConvexNMF::max_iterations);
    convexNMF.def_readwrite("iteration_to_print", &ConvexNMF::iteration_to_print);

}

