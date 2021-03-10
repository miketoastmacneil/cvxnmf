
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>

namespace convexnmf {

template <typename T> int sign(T val) { return (T(0) < val) - (T(0) > val); }

// Implements the Bisection method assuming
// Func f takes values T and returns values T.
// Comp returns whether or not the the value is less than or greater
// The number of max iterations has been checked for tolerances of
// 1.0e-8 and works fine.
template <typename T, class Func>
T BisectionRootFind(const Func &f, T minimum_val, T maximum_val, T tol) {
    T   x  = 0.0;
    T   lo = minimum_val, hi = maximum_val;
    int iteration      = 0;
    int max_iterations = 3.0 * std::log2(1.0 / tol);

    while (++iteration <= max_iterations) {
        x       = (lo + hi) / 2;
        T value = f(x);
        if (std::abs(value) < tol) {
            return x;
        }
        if (sign(f(x)) == sign(f(lo))) {
            lo = x;
        } else {
            hi = x;
        }
    }
    return x;
}
} // namespace convexnmf
