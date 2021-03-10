#pragma once

#include <memory>
#include <string>

#include "types.hpp"

namespace convexnmf {

enum class SparsifierType { L2, LInfty };

class RowSparsifier {
  public:
    static std::unique_ptr<RowSparsifier> Create(SparsifierType type);
    virtual Matrix Evaluate(const MatrixRef &src, Scalar lambda, Scalar rho) const = 0;
};

// Does row-wise projection onto the L2 Ball.
class L2RowSparsifier : public RowSparsifier {
  public:
    Matrix Evaluate(const MatrixRef &src, Scalar lambda, Scalar rho) const override;
};

// Projects onto the L1-Infty ball
class LInftyRowSparsifier : public RowSparsifier {
  public:
    Matrix Evaluate(const MatrixRef &src, Scalar lambda, Scalar rho) const override;
};

} // namespace convexnmf