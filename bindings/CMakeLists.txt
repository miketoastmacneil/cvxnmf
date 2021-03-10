add_subdirectory(pybind11)

set(CONVEX_NMF_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${CONVEX_NMF_DIR})
set(CONVEX_NMF_SOURCES ${CONVEX_NMF_DIR}/projections.cpp
                        ${CONVEX_NMF_DIR}/convex_nmf.cpp
                        ${CONVEX_NMF_DIR}/progress_reporter.cpp
                        ${CONVEX_NMF_DIR}/row_sparsifiers.cpp)
set(CONVEX_NMF_BINDINGS convex_nmf_bindings.cpp)
pybind11_add_module(_convexnmf ${CONVEX_NMF_SOURCES} ${CONVEX_NMF_BINDINGS})
