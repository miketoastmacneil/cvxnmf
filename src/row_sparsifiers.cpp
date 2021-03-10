
#include <exception>
#include <iostream>
#include <memory>

#include "projections.hpp"
#include "row_sparsifiers.hpp"

namespace convexnmf {

std::unique_ptr<RowSparsifier> RowSparsifier::Create(SparsifierType type) {
    switch (type) {
    case SparsifierType::L2: {
        return std::make_unique<L2RowSparsifier>();
    }
    case SparsifierType::LInfty: {
        return std::make_unique<LInftyRowSparsifier>();
    }
    default:
        throw std::invalid_argument("Sparsifier Type not known. Use either 'L2' or 'Linfty'.");
        break;
    }
}

// Proximal operator taken from Parikh and Boyd:
// https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf. See Section 6 on norms. Page 187, (69 of
// the pdf).
Matrix L2RowSparsifier::Evaluate(const MatrixRef &src, Scalar lambda, Scalar rho) const {
    int    M     = src.rows();
    Matrix out   = src;     // TODO, this makes a copy and kind of kills performance right now.
    Scalar theta = lambda / rho;
    for (int i = 0; i < M; ++i) {
        Vector row      = out.row(i);
        Scalar row_norm = row.norm();
        out.row(i)      = std::max((1.0 - theta / row_norm), 0.0) * row;
    }
    return out;
}

Matrix LInftyRowSparsifier::Evaluate(const MatrixRef &src, Scalar lambda, Scalar rho) const {
    return L1InfinityProximalOperator(src, lambda);
}

} // namespace convexnmf
