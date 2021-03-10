
## Eigen is included in the top-level
set(CVX_NMF_SOURCES convex_nmf.cpp projections.cpp progress_reporter.cpp row_sparsifiers.cpp)
add_library(ConvexNMF STATIC ${CVX_NMF_SOURCES})
