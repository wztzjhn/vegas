// SPDX-License-Identifier: BSD 3-Clause License
//
// Copyright (c) 2025, Zhentao Wang
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace vegas
{
    /** Configuration for VEGAS integration */
    struct Config {
        int ndim      = 2;         // Number of input dimensions
        int ncomp     = 1;         // Number of output components (integrands)
        int maxeval   = 1000000;   // Maximum total evaluations
        int niter     = 10;        // Number of iterations
        double α      = 1.5;       // Grid refinement parameter (0.5-2.0). α > 1: focuses on peaks, α < 1: more conservative
        uint64_t seed = 123456789; // Random seed for reproducibility
        double rtol   = 1e-3;      // Relative tolerance
        double atol   = 1e-10;     // Absolute tolerance
        int nbins     = 50;        // Number of bins per dimension
        int nstrat    = 0;         // Number of stratifications per dimension (0=auto)
        int verbose   = 0;         // Verbosity level (0=silent, 1=iterations, 2=detailed)
        std::vector<double> xmin;  // Lower bounds, whose size should match the input dimensions
        std::vector<double> xmax;  // Upper bounds, whose size should match the input dimensions
    };

    /** Result of VEGAS integration */
    struct Result {
        std::vector<double> integral; // Integral estimates for each component
        std::vector<double> error;    // Error estimates for each component
        std::vector<double> chi2;     // Reduced chi-squared (χ²/dof) for goodness of fit
        int neval = 0;                // Actual number of evaluations
        bool converged = true;        // True if all components converged

        Result() = default;
        explicit Result(int ncomp) : integral(ncomp), error(ncomp), chi2(ncomp) {}
    };

    /** Internal state for VEGAS algorithm */
    class VegasState {
    public:
        explicit VegasState(const Config &config) : ndim_(config.ndim),
                                                    ncomp_(config.ncomp),
                                                    nbins_(std::max(config.nbins, 2)),
                                                    nstrat_(config.nstrat),
                                                    α_(config.α),
                                                    iter_count_(0),
                                                    rng_(config.seed)
        {
            if (ndim_ <= 0 || ncomp_ <= 0) {
                throw std::invalid_argument("ndim and ncomp must be positive");
            }

            // Initialize grids
            xi_.resize(ndim_);
            d_.resize(ndim_);

            for (int i = 0; i < ndim_; ++i) {
                xi_[i].resize(nbins_ + 1);
                d_[i].resize(nbins_);

                // Initialize uniform grid
                for (int j = 0; j <= nbins_; ++j) {
                    xi_[i][j] = static_cast<double>(j) / nbins_;
                }

                // Initialize uniform weights
                std::fill(d_[i].begin(), d_[i].end(), 1.0);
            }

            // Initialize accumulation arrays
            sum_wgt_.assign(ncomp_, 0.0);
            sum_wgt2_.assign(ncomp_, 0.0);
            sum_integral_.assign(ncomp_, 0.0);
            sum_χ2_.assign(ncomp_, 0.0);
        }

        void refine_grid(const std::vector<std::vector<std::vector<double>>> &bin_accumulator)
        {
            if (static_cast<int>(bin_accumulator.size()) != ndim_) {
                throw std::invalid_argument("bin_accumulator dimension mismatch");
            }

            for (int dim = 0; dim < ndim_; ++dim) {
                if (static_cast<int>(bin_accumulator[dim].size()) != nbins_) {
                    throw std::invalid_argument("bin_accumulator bins mismatch");
                }

                std::vector<double> d_smooth(nbins_);

                // Accumulate across all components
                for (int i = 0; i < nbins_; ++i) {
                    d_smooth[i] = 0.0;
                    for (int comp = 0; comp < ncomp_; ++comp) {
                        d_smooth[i] += bin_accumulator[dim][i][comp];
                    }
                    d_smooth[i] = std::pow(std::abs(d_smooth[i]) + 1e-30, α_);
                }

                // Normalize
                double sum = std::accumulate(d_smooth.begin(), d_smooth.end(), 0.0);
                if (sum <= 0.0) sum = 1.0; // Prevent division by zero
                for (auto &val : d_smooth) val /= sum;

                // Compute cumulative distribution
                std::vector<double> cumulative(nbins_ + 1);
                cumulative[0] = 0.0;
                for (int i = 0; i < nbins_; ++i) {
                    cumulative[i + 1] = cumulative[i] + d_smooth[i];
                }

                // Ensure the last value is exactly 1.0
                cumulative[nbins_] = 1.0;

                // Compute a new grid by inverse transform sampling
                std::vector<double> xi_new(nbins_ + 1);
                xi_new[0]      = 0.0;
                xi_new[nbins_] = 1.0;

                for (int k = 1; k < nbins_; ++k) {
                    double target = static_cast<double>(k) / nbins_;

                    // Binary search for the bin containing target
                    // Find the largest i such that cumulative[i] <= target < cumulative[i+1]
                    const auto it = std::lower_bound(cumulative.begin(), cumulative.end(), target);
                    int bin = std::max(0, static_cast<int>(std::distance(cumulative.begin(), it)) - 1);
                    bin = std::clamp(bin, 0, nbins_ - 1);

                    // Linear interpolation within the bin
                    const double delta = cumulative[bin + 1] - cumulative[bin];
                    if (delta > 1e-10) {
                        const double frac = (target - cumulative[bin]) / delta;
                        xi_new[k] = xi_[dim][bin] + frac * (xi_[dim][bin + 1] - xi_[dim][bin]);
                    } else {
                        // Degenerate case: distribute uniformly
                        xi_new[k] = xi_[dim][bin];
                    }
                }

                // Enforce strict monotonicity
                for (int i = 1; i < nbins_; ++i) {
                    if (xi_new[i] <= xi_new[i - 1]) {
                        xi_new[i] = xi_new[i - 1] + (1.0 - xi_new[i - 1]) / (nbins_ - i + 1);
                    }
                }

                xi_[dim] = std::move(xi_new);
                d_[dim] = std::move(d_smooth);
            }
        }

        template <typename Integrand>
        Result integrate(Integrand &&integrand, const Config &config)
        {
            const int neval = config.maxeval / config.niter;
            if (neval < 1000) throw std::invalid_argument("maxeval/niter must be >= 1000");

            // Handle integration bounds
            std::vector<double> xmin = config.xmin;
            std::vector<double> xmax = config.xmax;

            // Validate bounds
            if (static_cast<int>(xmin.size()) != ndim_ ||
                static_cast<int>(xmax.size()) != ndim_) {
                throw std::invalid_argument("xmin/xmax size must match ndim");
            }

            // Compute volume Jacobian
            double volume_jacobian = 1.0;
            for (int dim = 0; dim < ndim_; ++dim) {
                double range = xmax[dim] - xmin[dim];
                if (range <= 0.0) {
                    throw std::invalid_argument("xmax must be > xmin for all dimensions");
                }
                volume_jacobian *= range;
            }

            // Calculate stratification if not specified
            int nstrat = nstrat_;
            if (nstrat <= 0) {
                nstrat = static_cast<int>(std::pow(neval / 2.0, 1.0 / ndim_));
                nstrat = std::clamp(nstrat, 1, 10);
            }

            // Calculate the total number of strata with overflow protection
            const int max_strat = static_cast<int>(std::sqrt(std::numeric_limits<int>::max()));
            long long nstrat_total_ll = 1;

            for (int i = 0; i < ndim_; ++i) {
                nstrat_total_ll *= nstrat;
                // Early exit if overflow is imminent
                if (nstrat_total_ll > max_strat) {
                    // Reduce nstrat to prevent overflow
                    nstrat = static_cast<int>(std::pow(max_strat, 1.0 / ndim_));
                    nstrat = std::max(1, nstrat);
                    nstrat_total_ll = 1;
                    for (int j = 0; j < ndim_; ++j) {
                        nstrat_total_ll *= nstrat;
                    }
                    break;
                }
            }

            // Ensure we don't have too many strata relative to samples
            if (nstrat_total_ll > neval / 2) {
                // Too many strata, reduce nstrat
                nstrat = static_cast<int>(std::pow(neval / 2.0, 1.0 / ndim_));
                nstrat = std::max(1, nstrat);
                nstrat_total_ll = 1;
                for (int i = 0; i < ndim_; ++i) {
                    nstrat_total_ll *= nstrat;
                }
            }

            // Final safety check
            if (nstrat_total_ll > std::numeric_limits<int>::max()) {
                throw std::runtime_error("Stratification overflow: reduce dimensions or nstrat");
            }

            int nstrat_total = static_cast<int>(nstrat_total_ll);
            int ncalls_per_strat = std::max(neval / nstrat_total, 2);

            Result result(ncomp_);

            // Working arrays
            std::vector<double> x_unit(ndim_);   // Point in [0,1]^ndim
            std::vector<double> x_actual(ndim_); // Point in [xmin,xmax]^ndim
            std::vector<double> f(ncomp_);

            // Main iteration loop
            for (int iter = 0; iter < config.niter; ++iter) {
                std::vector iter_sum(ncomp_, 0.0);
                std::vector iter_sum2(ncomp_, 0.0);

                // Bin accumulator for grid refinement (now properly 2D per dimension)
                std::vector bin_accumulator(
                    ndim_,
                    std::vector(nbins_, std::vector(ncomp_, 0.0))
                );

                // Loop over stratifications
                std::vector strat_idx(ndim_, 0);

                for (int istrat = 0; istrat < nstrat_total; ++istrat) {
                    // Welford's online algorithm for mean and variance
                    std::vector mean(ncomp_, 0.0);
                    std::vector m2(ncomp_, 0.0);

                    // Sample within this stratum
                    for (int icall = 0; icall < ncalls_per_strat; ++icall) {
                        double jacobian = 1.0;
                        std::vector<int> bin_idx(ndim_);

                        // Generate random point
                        std::uniform_real_distribution uniform_(0.0, 1.0);
                        for (int dim = 0; dim < ndim_; ++dim) {
                            double u = uniform_(rng_);
                            double bin_coord = (strat_idx[dim] + u) / nstrat;

                            // Find bin
                            int bin = static_cast<int>(bin_coord * nbins_);
                            bin = std::min(bin, nbins_ - 1);
                            bin_idx[dim] = bin;

                            // Map to [0,1] using importance sampling grid
                            double bin_width = xi_[dim][bin + 1] - xi_[dim][bin];
                            double u_in_bin = bin_coord * nbins_ - bin;
                            x_unit[dim] = xi_[dim][bin] + u_in_bin * bin_width;

                            // Transform to actual domain [xmin, xmax]
                            x_actual[dim] = xmin[dim] + x_unit[dim] * (xmax[dim] - xmin[dim]);

                            jacobian *= bin_width * nbins_;
                        }

                        // Apply volume Jacobian
                        jacobian *= volume_jacobian;

                        // Evaluate integrand on actual domain
                        integrand(x_actual, f);
                        ++result.neval;

                        // Check for invalid values
                        for (int comp = 0; comp < ncomp_; ++comp) {
                            if (!std::isfinite(f[comp])) {
                                f[comp] = 0.0; // Replace with zero
                            }
                        }

                        // Welford's algorithm for numerically stable variance
                        for (int comp = 0; comp < ncomp_; ++comp) {
                            double val = f[comp] * jacobian;
                            double delta = val - mean[comp];
                            mean[comp] += delta / (icall + 1);
                            double delta2 = val - mean[comp];
                            m2[comp] += delta * delta2;

                            // Accumulate for grid refinement
                            for (int dim = 0; dim < ndim_; ++dim) {
                                bin_accumulator[dim][bin_idx[dim]][comp] += std::abs(val);
                            }
                        }
                    }

                    // Compute variance from Welford's m2
                    for (int comp = 0; comp < ncomp_; ++comp) {
                        double variance = ncalls_per_strat > 1 ? m2[comp] / (ncalls_per_strat - 1) : 0.0;

                        // Ensure non-negative (should already be, but safety check)
                        if (variance < 0.0) variance = 0.0;

                        // Standard error of the mean
                        double sigma2 = variance / ncalls_per_strat;

                        iter_sum[comp] += mean[comp];
                        iter_sum2[comp] += sigma2;
                    }

                    // Update stratification indices
                    for (int dim = ndim_ - 1; dim >= 0; --dim) {
                        ++strat_idx[dim];
                        if (strat_idx[dim] < nstrat) break;
                        strat_idx[dim] = 0;
                    }
                }

                // Compute iteration results and update cumulative statistics
                for (int comp = 0; comp < ncomp_; ++comp) {
                    double iter_integral = iter_sum[comp] / nstrat_total;
                    double iter_variance = iter_sum2[comp] / (nstrat_total * nstrat_total);

                    // Use inverse variance weighting
                    double wgt = 0.0;
                    if (iter_variance > 0.0 && std::isfinite(iter_variance)) {
                        wgt = 1.0 / iter_variance;
                        // Cap the weight to prevent numerical overflow
                        constexpr double max_wgt = 1e30;
                        wgt = std::min(wgt, max_wgt);
                    } else {
                        // Fallback: use a weight proportional to number of samples
                        wgt = static_cast<double>(nstrat_total) * static_cast<double>(ncalls_per_strat);
                    }

                    // Final safety check
                    if (!std::isfinite(wgt) || wgt <= 0) {
                        wgt = 1.0;
                    }

                    double old_integral = sum_wgt_[comp] > 0 ? sum_integral_[comp] / sum_wgt_[comp] : 0.0;

                    sum_wgt_[comp] += wgt;
                    sum_wgt2_[comp] += wgt * wgt;
                    sum_integral_[comp] += wgt * iter_integral;

                    // Chi-squared accumulation
                    if (iter > 0 && sum_wgt_[comp] > wgt) {
                        const double delta = iter_integral - old_integral;
                        sum_χ2_[comp] += wgt * delta * delta;
                    }
                }

                ++iter_count_;

                // Refine grid
                if (iter < config.niter - 1) {
                    refine_grid(bin_accumulator);
                }

                // Verbose output
                if (config.verbose > 0) {
                    std::cout << "Iteration " << iter + 1 << ": ";
                    for (int comp = 0; comp < ncomp_; ++comp) {
                        if (sum_wgt_[comp] > 0) {
                            double current_integral = sum_integral_[comp] / sum_wgt_[comp];
                            double current_error = std::sqrt(1.0 / sum_wgt_[comp]);
                            std::cout << "I[" << comp << "]="
                                << std::scientific << std::setprecision(6)
                                << current_integral << "±" << current_error << " ";
                        } else {
                            std::cout << "I[" << comp << "]=N/A ";
                        }
                    }
                    std::cout << std::endl;
                }
            }

            // Final results
            result.converged = true;
            for (int comp = 0; comp < ncomp_; ++comp) {
                if (sum_wgt_[comp] > 0) {
                    result.integral[comp] = sum_integral_[comp] / sum_wgt_[comp];
                    result.error[comp] = std::sqrt(1.0 / sum_wgt_[comp]);

                    // Reduced chi-squared (χ²/dof)
                    if (iter_count_ > 1) {
                        double dof = iter_count_ - 1;
                        result.chi2[comp] = sum_χ2_[comp] / dof;
                    } else {
                        result.chi2[comp] = 1.0;
                    }

                    // Check convergence
                    double tolerance = config.atol + std::abs(result.integral[comp]) * config.rtol;
                    if (result.error[comp] > tolerance) {
                        result.converged = false;
                    }
                } else {
                    // No valid weight accumulated - integration failed
                    result.integral[comp] = 0.0;
                    result.error[comp] = std::numeric_limits<double>::infinity();
                    result.chi2[comp] = 0.0;
                    result.converged = false;
                }
            }

            return result;
        }

    private:
        int ndim_;
        int ncomp_;
        int nbins_;
        int nstrat_;
        double α_;

        // Grid boundaries and importance sampling weights
        std::vector<std::vector<double>> xi_; // [ndim][nbins+1]
        std::vector<std::vector<double>> d_;  // [ndim][nbins]

        // Accumulation for weighted averaging and chi-squared test
        std::vector<double> sum_wgt_;
        std::vector<double> sum_wgt2_;
        std::vector<double> sum_integral_;
        std::vector<double> sum_χ2_;
        int iter_count_;

        // Random number generator (thread-safe when each instance is independent)
        std::mt19937_64 rng_;
    };

    /**
     * Main integration function
     *
     * @param integrand Function with signature void(const std::vector<double>& x, std::vector<double>& f)
     *                  where x is the input point in [xmin,xmax]^ndim and f is the output vector of size ncomp
     * @param xmin Lower integration bounds, whose size should match the input dimensions
     * @param xmax Upper integration bounds, whose size should match the input dimensions
     * @param ncomp Number of output components (integrands)
     * @param maxeval Maximum total number of evaluations
     * @param niter Number of refinement iterations
     * @param alpha Grid refinement parameter (0.5-2.0): >1 focuses on peaks, <1 more conservative
     * @param seed Random seed for reproducibilityc
     * @param rtol Relative tolerance for convergence
     * @param atol Absolute tolerance for convergence
     * @param nbins Number of bins per dimension for importance sampling grid
     * @param nstrat Number of stratifications per dimension (0 = auto)
     * @param verbose Verbosity level (0=silent, 1=iterations, 2=detailed)
     * @return Result structure with integrals and errors
     */
    template <typename Integrand>
    Result integrate(Integrand &&integrand, const std::vector<double> &xmin, const std::vector<double> &xmax,
        int ncomp = 1,
        int maxeval = 1000000,
        int niter = 10,
        double alpha = 1.5,
        uint64_t seed = 123456789,
        double rtol = 1e-3,
        double atol = 1e-10,
        int nbins = 50,
        int nstrat = 0,
        int verbose = 0)
    {
        if (xmin.empty()) throw std::invalid_argument("xmin must not be empty");
        if (xmin.size() != xmax.size()) throw std::invalid_argument("xmin and xmax must have the same size");
        Config config;
        config.ndim = static_cast<int>(xmin.size());
        config.ncomp = ncomp;
        config.maxeval = maxeval;
        config.niter = niter;
        config.xmin = xmin;
        config.xmax = xmax;
        config.rtol = rtol;
        config.atol = atol;
        config.nbins = nbins;
        config.nstrat = nstrat;
        config.α = alpha;
        config.verbose = verbose;
        config.seed = seed;

        VegasState state(config);
        return state.integrate(std::forward<Integrand>(integrand), config);
    }

    /**
     * Integration function with userdata support
     *
     * @param integrand Function with signature void(const std::vector<double>& x,
     *                  std::vector<double>& f, void* userdata)
     * @param userdata Pointer to user data passed to integrand
     * @param xmin Lower integration bounds, whose size should match the input dimensions
     * @param xmax Upper integration bounds, whose size should match the input dimensions
     * @param ncomp Number of output components (integrands)
     * @param maxeval Maximum total number of evaluations
     * @param niter Number of refinement iterations\
     * @param alpha Grid refinement parameter (0.5-2.0): >1 focuses on peaks, <1 more conservative
     * @param seed Random seed for reproducibility
     * @param rtol Relative tolerance for convergence
     * @param atol Absolute tolerance for convergence
     * @param nbins Number of bins per dimension for importance sampling grid
     * @param nstrat Number of stratifications per dimension (0 = auto)
     * @param verbose Verbosity level (0=silent, 1=iterations, 2=detailed)
     * @return Result structure with integrals and errors
     */
    template <typename Integrand, typename UserData>
    Result integrate_with_data(Integrand &&integrand, UserData *userdata,
        const std::vector<double> &xmin, const std::vector<double> &xmax,
        int ncomp = 1,
        int maxeval = 1000000,
        int niter = 10,
        double alpha = 1.5,
        uint64_t seed = 123456789,
        double rtol = 1e-3,
        double atol = 1e-10,
        int nbins = 50,
        int nstrat = 0,
        int verbose = 0)
    {
        auto wrapped = [&integrand, userdata](const std::vector<double> &x, std::vector<double> &f)
        {
            integrand(x, f, userdata);
        };
        return integrate(wrapped, xmin, xmax, ncomp, maxeval, niter, alpha, seed,
                         rtol, atol, nbins, nstrat, verbose);
    }

    /**
     * Convenience function for integration over [0,1]^ndim
     *
     * @param integrand Function with signature void(const std::vector<double>& x, std::vector<double>& f)
     * @param ndim Number of dimensions
     * @param ncomp Number of output components (integrands)
     * @param maxeval Maximum total number of evaluations
     * @param niter Number of refinement iterations
     * @param alpha Grid refinement parameter (0.5-2.0): >1 focuses on peaks, <1 more conservative
     * @param seed Random seed for reproducibility
     * @param rtol Relative tolerance for convergence
     * @param atol Absolute tolerance for convergence
     * @param nbins Number of bins per dimension for importance sampling grid
     * @param nstrat Number of stratifications per dimension (0 = auto)
     * @param verbose Verbosity level (0=silent, 1=iterations, 2=detailed)
     * @return Result structure with integrals and errors
     */
    template <typename Integrand>
    Result integrate(Integrand &&integrand,
        int ndim,
        int ncomp = 1,
        int maxeval = 1000000,
        int niter = 10,
        double alpha = 1.5,
        uint64_t seed = 123456789,
        double rtol = 1e-3,
        double atol = 1e-10,
        int nbins = 50,
        int nstrat = 0,
        int verbose = 0)
    {
        std::vector xmin(ndim, 0.0);
        std::vector xmax(ndim, 1.0);
        return integrate(std::forward<Integrand>(integrand), xmin, xmax, ncomp, maxeval,
                        niter, alpha, seed, rtol, atol, nbins, nstrat, verbose);
    }

    /**
     * Convenience function for integration over [0,1]^ndim with userdata
     *
     * @param integrand Function with signature void(const std::vector<double>& x,
     *                  std::vector<double>& f, void* userdata)
     * @param userdata Pointer to user data passed to integrand
     * @param ndim Number of dimensions
     * @param ncomp Number of output components (integrands)
     * @param maxeval Maximum total number of evaluations
     * @param niter Number of refinement iterations
     * @param alpha Grid refinement parameter (0.5-2.0): >1 focuses on peaks, <1 more conservative
     * @param seed Random seed for reproducibility
     * @param rtol Relative tolerance for convergence
     * @param atol Absolute tolerance for convergence
     * @param nbins Number of bins per dimension for importance sampling grid
     * @param nstrat Number of stratifications per dimension (0 = auto)
     * @param verbose Verbosity level (0=silent, 1=iterations, 2=detailed)
     * @return Result structure with integrals and errors
     */
    template <typename Integrand, typename UserData>
    Result integrate_with_data(Integrand &&integrand, UserData *userdata,
        int ndim,
        int ncomp = 1,
        int maxeval = 1000000,
        int niter = 10,
        double alpha = 1.5,
        uint64_t seed = 123456789,
        double rtol = 1e-3,
        double atol = 1e-10,
        int nbins = 50,
        int nstrat = 0,
        int verbose = 0)
    {
        std::vector xmin(ndim, 0.0);
        std::vector xmax(ndim, 1.0);
        return integrate_with_data(std::forward<Integrand>(integrand), userdata, xmin, xmax,
                                   ncomp, maxeval, niter, alpha, seed, rtol, atol, nbins, nstrat, verbose);
    }
} // namespace vegas
