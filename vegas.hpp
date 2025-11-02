#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
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
        int neval     = 100000;    // Approximate number of evaluations per iteration
        int niter     = 10;        // Number of iterations
        int nbins     = 50;        // Number of bins per dimension
        int nstrat    = 0;         // Number of stratifications per dimension (0=auto)
        double α      = 1.5;       // Grid refinement parameter (0.5-2.0)
        double rtol   = 1e-3;      // Relative tolerance
        double atol   = 1e-10;     // Absolute tolerance
        int verbose   = 0;         // Verbosity level (0=silent, 1=iterations, 2=detailed)
        uint64_t seed = 123456789; // Random seed for reproducibility
    };

    /** Result of VEGAS integration */
    struct Result {
        std::vector<double> integral; // Integral estimates for each component
        std::vector<double> error;    // Error estimates for each component
        std::vector<double> prob;     // Chi-squared probabilities
        int neval = 0;                // Actual number of evaluations
        bool converged = true;        // True if all components converged

        Result() = default;
        explicit Result(int ncomp) : integral(ncomp), error(ncomp), prob(ncomp) {}
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

        void refine_grid(const std::vector<std::vector<double>> &bin_accumulator)
        {
            if (static_cast<int>(bin_accumulator.size()) != ndim_ * nbins_) throw std::invalid_argument("bin_accumulator size mismatch");
            for (const auto &ele : bin_accumulator) {
                if (static_cast<int>(ele.size()) != ncomp_) throw std::invalid_argument("bin_accumulator size mismatch");
            }
            const double δ = 1.0 / nbins_;

            for (int dim = 0; dim < ndim_; ++dim) {
                std::vector<double> d_smooth(nbins_);

                // Accumulate across all components
                for (int i = 0; i < nbins_; ++i) {
                    d_smooth[i] = 0.0;
                    for (int comp = 0; comp < ncomp_; ++comp) {
                        d_smooth[i] += bin_accumulator[dim * nbins_ + i][comp];
                    }
                    d_smooth[i] = std::pow(std::abs(d_smooth[i]) + 1e-30, α_);
                }

                // Normalize
                double sum = std::accumulate(d_smooth.begin(), d_smooth.end(), 0.0);
                // if (sum <= 0.0) sum = 1.0; // Prevent division by zero
                for (auto &val : d_smooth) val /= sum;

                // Compute new grid
                std::vector<double> xi_new(nbins_ + 1);
                xi_new[0]      = 0.0;
                xi_new[nbins_] = 1.0;

                int k = 1;
                double accum = 0.0;
                double target = δ;

                for (int i = 0; i < nbins_; ++i) {
                    accum += d_smooth[i];
                    while (accum >= target && k < nbins_) {
                        const double frac = (target - (accum - d_smooth[i])) / (d_smooth[i] + 1e-30);
                        xi_new[k] = xi_[dim][i] + frac * (xi_[dim][i + 1] - xi_[dim][i]);
                        ++k;
                        target += δ;
                    }
                }

                // Ensure monotonicity and fill remaining bins
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
            // Calculate stratification if not specified
            int nstrat = nstrat_;
            if (nstrat <= 0) {
                nstrat = static_cast<int>(std::pow(config.neval / 2.0, 1.0 / ndim_));
                nstrat = std::clamp(nstrat, 1, 10);
            }

            // Calculate the total number of strata
            long long nstrat_total_ll = 1;
            for (int i = 0; i < ndim_; ++i) {
                nstrat_total_ll *= nstrat;
            }

            // Make sure we don't overflow and have reasonable stratification
            if (nstrat_total_ll > config.neval / 2) {
                // Too many strata, reduce nstrat
                nstrat = static_cast<int>(std::pow(config.neval / 2.0, 1.0 / ndim_));
                nstrat = std::max(1, nstrat);
                nstrat_total_ll = 1;
                for (int i = 0; i < ndim_; ++i) {
                    nstrat_total_ll *= nstrat;
                }
            }

            int nstrat_total = static_cast<int>(nstrat_total_ll);
            int ncalls_per_strat = std::max(config.neval / nstrat_total, 2);

            Result result(ncomp_);

            // Working arrays
            std::vector<double> x(ndim_);
            std::vector<double> f(ncomp_);

            // Main iteration loop
            for (int iter = 0; iter < config.niter; ++iter) {
                std::vector iter_sum(ncomp_, 0.0);
                std::vector iter_sum2(ncomp_, 0.0);

                // Bin accumulator for grid refinement
                std::vector bin_accumulator(ndim_ * nbins_, std::vector(ncomp_, 0.0));

                // Loop over stratifications
                std::vector strat_idx(ndim_, 0);

                for (int istrat = 0; istrat < nstrat_total; ++istrat) {
                    std::vector strat_sum(ncomp_, 0.0);
                    std::vector strat_sum2(ncomp_, 0.0);

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

                            // Map to [0,1]^ndim using importance sampling grid
                            double bin_width = xi_[dim][bin + 1] - xi_[dim][bin];
                            double u_in_bin = bin_coord * nbins_ - bin;
                            x[dim] = xi_[dim][bin] + u_in_bin * bin_width;

                            jacobian *= bin_width * nbins_;
                        }

                        // Evaluate integrand
                        integrand(x, f);
                        ++result.neval;

                        // Check for invalid values
                        for (int comp = 0; comp < ncomp_; ++comp) {
                            if (!std::isfinite(f[comp])) {
                                f[comp] = 0.0; // Replace with zero
                            }
                        }

                        // Accumulate
                        for (int comp = 0; comp < ncomp_; ++comp) {
                            double val = f[comp] * jacobian;
                            strat_sum[comp] += val;
                            strat_sum2[comp] += val * val;

                            // Accumulate for grid refinement
                            for (int dim = 0; dim < ndim_; ++dim) {
                                bin_accumulator[dim * nbins_ + bin_idx[dim]][comp] += std::abs(val);
                            }
                        }
                    }

                    // Update iteration statistics with better variance handling
                    for (int comp = 0; comp < ncomp_; ++comp) {
                        double mean = strat_sum[comp] / ncalls_per_strat;
                        double mean2 = strat_sum2[comp] / ncalls_per_strat;
                        double variance = mean2 - mean * mean;

                        // Ensure non-negative variance
                        if (variance < 0.0) variance = 0.0;

                        // Standard error of the mean
                        double sigma2 = variance / ncalls_per_strat;

                        iter_sum[comp] += mean;
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
                        // Cast to double to avoid integer overflow
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

                    // Chi-squared probability (simplified)
                    if (iter_count_ > 1) {
                        double dof = iter_count_ - 1;
                        result.prob[comp] = sum_χ2_[comp] / dof;
                    } else {
                        result.prob[comp] = 1.0;
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
                    result.prob[comp] = 0.0;
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
     *                  where x is the input point in [0,1]^ndim and f is the output vector of size ncomp
     * @param config Configuration parameters
     * @return Result structure with integrals and errors
     */
    template <typename Integrand>
    Result integrate(Integrand &&integrand, const Config &config = Config{})
    {
        VegasState state(config);
        return state.integrate(std::forward<Integrand>(integrand), config);
    }

    /**
     * Integration function with userdata support
     *
     * @param integrand Function with signature void(const std::vector<double>& x,
     *                  std::vector<double>& f, void* userdata)
     * @param config Configuration parameters
     * @param userdata Pointer to user data passed to integrand
     * @return Result structure with integrals and errors
     */
    template <typename Integrand, typename UserData>
    Result integrate_with_data(Integrand &&integrand, const Config &config, UserData *userdata)
    {
        auto wrapped = [&integrand, userdata](const std::vector<double> &x, std::vector<double> &f)
        {
            integrand(x, f, userdata);
        };
        return integrate(wrapped, config);
    }
} // namespace vegas
