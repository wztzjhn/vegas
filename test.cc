#include "vegas.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <functional>

// Helper function to print test results
void print_result(const std::string& test_name,
                 const vegas::Result& result,
                 const std::vector<double>& expected,
                 bool show_details = true) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << test_name << "\n";
    std::cout << std::string(60, '=') << "\n";

    if (show_details) {
        for (size_t i = 0; i < result.integral.size(); ++i) {
            std::cout << "Component " << i << ":\n";
            std::cout << "  Result   = " << std::scientific << std::setprecision(8)
                      << result.integral[i] << " ± " << result.error[i] << "\n";
            if (i < expected.size()) {
                std::cout << "  Expected = " << expected[i] << "\n";
                double diff = std::abs(result.integral[i] - expected[i]);
                double sigma = diff / result.error[i];
                std::cout << "  Diff     = " << diff << " (" << sigma << " sigma)\n";
            }
            std::cout << "  χ²    = " << result.prob[i] << "\n";
        }
    }

    std::cout << "Total evaluations: " << result.neval << "\n";
    std::cout << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
}

// Diagnostic: Test smooth approximation of the same integral
void test_smooth_approximation() {
    std::cout << "\nDiagnostic: Smooth (regularized) version\n";
    std::cout << std::string(60, '-') << "\n";

    // Same integral but with regularization parameter
    double eps = 0.01;  // Removes singularity

    auto integrand = [eps](const std::vector<double>&x,std::vector<double>&f) {
        double k0 = M_PI * x[0];
        double k1 = M_PI * x[1];
        double k2 = M_PI * x[2];

        double A = 1.0 / (M_PI * M_PI * M_PI);
        double denom = 1.0 - cos(k0) * cos(k1) * cos(k2) + eps;

        f[0] = A / denom * (M_PI * M_PI * M_PI);
    };

    vegas::Config config;
    config.ndim = 3;
    config.ncomp = 1;
    config.neval = 100000;
    config.niter = 10;
    config.verbose = 0;

    auto result = vegas::integrate(integrand, config);

    std::cout << "Regularized result (eps=" << eps << "): "
              << result.integral[0] << " ± " << result.error[0] << "\n";
    std::cout << "χ² = " << result.prob[0] << "\n";
    std::cout << "Note: This should be lower than 1.393 due to regularization\n";
}

void test_gsl_ising_integral() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "GSL Benchmark: Ising Model Integral\n";
    std::cout << std::string(60, '=') << "\n";

    double exact = 1.3932039296856768591842462603255;

    // EXACT same integrand as GSL (no modifications!)
    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        double k0 = M_PI * x[0];
        double k1 = M_PI * x[1];
        double k2 = M_PI * x[2];

        double A = 1.0 / (M_PI * M_PI * M_PI);
        double denom = 1.0 - cos(k0) * cos(k1) * cos(k2);

        f[0] = A / denom * (M_PI * M_PI * M_PI); // Jacobian
    };

    // Test with different configurations
    std::vector<std::pair<std::string, vegas::Config>> configs;

    // Config 1: Original
    vegas::Config cfg1;
    cfg1.ndim = 3;
    cfg1.ncomp = 1;
    cfg1.neval = 100000;
    cfg1.niter = 10;
    cfg1.verbose = 0;
    cfg1.α = 1.5;
    cfg1.seed = 0;
    configs.push_back({"Original (seed=0)", cfg1});

    // Config 2: Different seed
    vegas::Config cfg2 = cfg1;
    cfg2.seed = 12345;
    configs.push_back({"Different seed", cfg2});

    // Config 3: More samples
    vegas::Config cfg3 = cfg1;
    cfg3.neval = 200000;
    cfg3.seed = 0;
    configs.push_back({"More samples", cfg3});

    // Config 4: Lower alpha (less aggressive adaptation)
    vegas::Config cfg4 = cfg1;
    cfg4.α = 0.5;
    cfg4.seed = 0;
    configs.push_back({"Lower alpha=0.5", cfg4});

    // Config 5: Higher alpha
    vegas::Config cfg5 = cfg1;
    cfg5.α = 2.0;
    cfg5.seed = 0;
    configs.push_back({"Higher alpha=2.0", cfg5});

    // Run all configurations
    for (const auto& [name, config] : configs) {
        std::cout << "\n" << name << ":\n";
        std::cout << std::string(40, '-') << "\n";

        auto result = vegas::integrate(integrand, config);

        double error = result.integral[0] - exact;
        double sigma = fabs(error) / result.error[0];

        std::cout << "Result = " << std::fixed << std::setprecision(6)
                  << result.integral[0] << " ± " << result.error[0] << "\n";
        std::cout << "Error  = " << error << " (" << sigma << " sigma)\n";
        std::cout << "χ²   = " << result.prob[0] << "\n";
    }

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "GSL reference: 1.393281 ± 0.000362\n";
    std::cout << std::string(60, '=') << "\n";
}

// Test 1: Simple polynomial x*y over [0,1]^2
void test_polynomial_2d() {
    vegas::Config config;
    config.ndim = 2;
    config.ncomp = 1;
    config.neval = 50000;
    config.niter = 5;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        f[0] = x[0] * x[1];
    };

    auto result = vegas::integrate(integrand, config);

    // Expected: ∫∫ x*y dx dy = 1/4
    print_result("Test 1: Polynomial x*y", result, {0.25});
}

// Test 2: Gaussian in 3D
void test_gaussian_3d() {
    vegas::Config config;
    config.ndim = 3;
    config.ncomp = 1;
    config.neval = 100000;
    config.niter = 5;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        double sum = 0.0;
        for (const auto& xi : x) {
            double z = 3.0 * (xi - 0.5);  // Center at 0.5, scale
            sum += z * z;
        }
        f[0] = std::exp(-sum);
    };

    auto result = vegas::integrate(integrand, config);

    // Expected: [∫_0^1 exp(-(3(x-0.5))^2) dx]^3
    // = [(1/3) * ∫_{-1.5}^{1.5} exp(-u^2) du]^3
    // = [(√π/3) * erf(1.5)]^3
    double expected = std::pow(std::sqrt(M_PI) / 3.0 * std::erf(1.5), 3.0);
    print_result("Test 2: Gaussian in 3D", result, {expected});
}

// Test 3: Corner peak (tests importance sampling)
void test_corner_peak() {
    vegas::Config config;
    config.ndim = 2;
    config.ncomp = 1;
    config.neval = 100000;
    config.niter = 10;
    config.verbose = 0;
    config.α = 1.5;  // More aggressive grid adaptation

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        // Peaks near origin
        double a = 0.1;
        f[0] = 1.0 / ((x[0] + a) * (x[1] + a));
    };

    auto result = vegas::integrate(integrand, config);

    // Expected: log((1+a)/a)^2
    double a = 0.1;
    double expected = std::pow(std::log((1.0 + a) / a), 2.0);
    print_result("Test 3: Corner Peak", result, {expected});
}

// Test 4: Oscillatory function
void test_oscillatory() {
    vegas::Config config;
    config.ndim = 2;
    config.ncomp = 1;
    config.neval = 200000;
    config.niter = 8;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        f[0] = std::cos(10.0 * M_PI * x[0]) * std::cos(10.0 * M_PI * x[1]);
    };

    auto result = vegas::integrate(integrand, config);

    // Expected: (sin(10π)/(10π))^2 ≈ 0
    double expected = std::pow(std::sin(10.0 * M_PI) / (10.0 * M_PI), 2.0);
    print_result("Test 4: Oscillatory Function", result, {expected});
}

// Test 5: Product of sines (separable integral)
void test_product_sines() {
    vegas::Config config;
    config.ndim = 4;
    config.ncomp = 1;
    config.neval = 100000;
    config.niter = 5;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        f[0] = 1.0;
        for (const auto& xi : x) {
            f[0] *= std::sin(M_PI * xi);
        }
    };

    auto result = vegas::integrate(integrand, config);

    // Expected: (2/π)^4
    double expected = std::pow(2.0 / M_PI, 4.0);
    print_result("Test 5: Product of Sines (4D)", result, {expected});
}

// Test 6: Discontinuous function
void test_discontinuous() {
    vegas::Config config;
    config.ndim = 2;
    config.ncomp = 1;
    config.neval = 100000;
    config.niter = 10;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        // Step function
        f[0] = (x[0] > 0.5 && x[1] > 0.5) ? 1.0 : 0.0;
    };

    auto result = vegas::integrate(integrand, config);

    // Expected: 0.25 (area of quarter square)
    print_result("Test 6: Discontinuous (Step)", result, {0.25});
}

// Test 7: Sphere volume in n dimensions
void test_sphere_volume() {
    vegas::Config config;
    config.ndim = 5;
    config.ncomp = 1;
    config.neval = 500000;
    config.niter = 8;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        // Map [0,1]^n to [-1,1]^n and check if inside unit sphere
        double r2 = 0.0;
        for (const auto& xi : x) {
            double z = 2.0 * xi - 1.0;
            r2 += z * z;
        }
        f[0] = (r2 <= 1.0) ? std::pow(2.0, x.size()) : 0.0; // Jacobian for [-1,1]^n
    };

    auto result = vegas::integrate(integrand, config);

    // Expected: π^(n/2) / Γ(n/2 + 1)
    // For n=5: π^2.5 / Γ(3.5) = π^2.5 / (2.5 * 1.5 * 0.5 * sqrt(π))
    int n = 5;
    double expected = std::pow(M_PI, n / 2.0) / std::tgamma(n / 2.0 + 1.0);
    print_result("Test 7: Sphere Volume (5D)", result, {expected});
}

// Test 8: Multiple components
void test_multiple_components() {
    vegas::Config config;
    config.ndim = 2;
    config.ncomp = 3;
    config.neval = 100000;
    config.niter = 5;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        f[0] = x[0] * x[0] + x[1] * x[1];                    // 2/3
        f[1] = std::exp(-x[0] - x[1]);                        // (1-1/e)^2
        f[2] = std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]); // 4/π^2
    };

    auto result = vegas::integrate(integrand, config);

    double e = std::exp(1.0);
    std::vector<double> expected = {
        2.0 / 3.0,
        std::pow(1.0 - 1.0 / e, 2.0),
        4.0 / (M_PI * M_PI)
    };

    print_result("Test 8: Multiple Components", result, expected);
}

// Test 9: Tsuda's function (difficult peak)
void test_tsuda() {
    vegas::Config config;
    config.ndim = 4;
    config.ncomp = 1;
    config.neval = 500000;
    config.niter = 10;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        double a = 0.1;
        double prod = 1.0;
        for (const auto& xi : x) {
            prod *= 1.0 / (a * a + (xi - 0.5) * (xi - 0.5));
        }
        f[0] = prod;
    };

    auto result = vegas::integrate(integrand, config);

    // Expected: (arctan(0.5/a) + arctan(0.5/a))^4 / a^4
    double a = 0.1;
    double arctan_sum = 2.0 * std::atan(0.5 / a);
    double expected = std::pow(arctan_sum / a, 4.0);
    print_result("Test 9: Tsuda's Function (peaked)", result, {expected});
}

// Test 10: Genz oscillatory test function
void test_genz_oscillatory() {
    vegas::Config config;
    config.ndim = 3;
    config.ncomp = 1;
    config.neval = 200000;
    config.niter = 8;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        // Genz "Oscillatory" test function
        std::vector<double> u = {1.0, 1.0, 1.0};
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            sum += u[i] * x[i];
        }
        f[0] = std::cos(2.0 * M_PI * u[0] + sum);
    };

    auto result = vegas::integrate(integrand, config);

    // Analytical solution is complex, just verify it runs
    print_result("Test 10: Genz Oscillatory", result, {}, false);
}

// Test 11: Camel function (multiple peaks)
void test_camel() {
    vegas::Config config;
    config.ndim = 2;
    config.ncomp = 1;
    config.neval = 200000;
    config.niter = 10;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        // Map to [-2, 2]
        double x1 = 4.0 * x[0] - 2.0;
        double x2 = 4.0 * x[1] - 2.0;

        // Six-hump camel function (negated to make peaks)
        double term1 = (4.0 - 2.1 * x1 * x1 + std::pow(x1, 4.0) / 3.0) * x1 * x1;
        double term2 = x1 * x2;
        double term3 = (-4.0 + 4.0 * x2 * x2) * x2 * x2;

        f[0] = std::exp(-(term1 + term2 + term3)) * 16.0; // Include Jacobian
    };

    auto result = vegas::integrate(integrand, config);

    print_result("Test 11: Six-Hump Camel (multiple peaks)", result, {}, false);
}

// Test 12: High-dimensional Gaussian with varying widths
void test_anisotropic_gaussian() {
    vegas::Config config;
    config.ndim = 6;
    config.ncomp = 1;
    config.neval = 1000000;
    config.niter = 10;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        // Different widths in different dimensions
        std::vector<double> sigma = {0.1, 0.2, 0.3, 0.1, 0.2, 0.3};
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double z = (x[i] - 0.5) / sigma[i];
            sum += z * z;
        }
        f[0] = std::exp(-0.5 * sum);
    };

    auto result = vegas::integrate(integrand, config);

    // Expected: ∏_i [∫_0^1 exp(-0.5*((x-0.5)/σ_i)^2) dx]
    // = ∏_i [σ_i * ∫_{-0.5/σ_i}^{0.5/σ_i} exp(-0.5*u^2) du]
    // = ∏_i [σ_i * √(2π) * (Φ(0.5/σ_i) - Φ(-0.5/σ_i))]
    // = ∏_i [σ_i * √(2π) * erf(0.5/(σ_i*√2))]
    std::vector<double> sigma = {0.1, 0.2, 0.3, 0.1, 0.2, 0.3};
    double expected = 1.0;
    for (auto s : sigma) {
        expected *= s * std::sqrt(2.0 * M_PI) * std::erf(0.5 / (s * std::sqrt(2.0)));
    }
    print_result("Test 12: Anisotropic Gaussian (6D)", result, {expected});
}

// Test 13: Function with near-singularity
void test_near_singularity() {
    vegas::Config config;
    config.ndim = 2;
    config.ncomp = 1;
    config.neval = 500000;
    config.niter = 10;
    config.verbose = 0;
    config.α = 2.0;  // More aggressive adaptation

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        double epsilon = 0.01;
        double r2 = x[0] * x[0] + x[1] * x[1];
        f[0] = 1.0 / std::sqrt(r2 + epsilon);
    };

    auto result = vegas::integrate(integrand, config);

    // Approximate expected value (numerical)
    print_result("Test 13: Near-Singularity", result, {}, false);
}

// Test 14: Exponential decay product
void test_exponential_product() {
    vegas::Config config;
    config.ndim = 3;
    config.ncomp = 1;
    config.neval = 50000;
    config.niter = 5;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        double lambda = 2.0;
        f[0] = 1.0;
        for (const auto& xi : x) {
            f[0] *= lambda * std::exp(-lambda * xi);
        }
    };

    auto result = vegas::integrate(integrand, config);

    // Expected: (1 - e^(-λ))^n
    double lambda = 2.0;
    double expected = std::pow(1.0 - std::exp(-lambda), 3.0);
    print_result("Test 14: Exponential Product", result, {expected});
}

// Test 15: Box function (tests stratification)
void test_box_function() {
    vegas::Config config;
    config.ndim = 3;
    config.ncomp = 1;
    config.neval = 100000;
    config.niter = 5;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        // 1 inside a box, 0 outside
        bool inside = true;
        for (const auto& xi : x) {
            inside = inside && (xi >= 0.25 && xi <= 0.75);
        }
        f[0] = inside ? 1.0 : 0.0;
    };

    auto result = vegas::integrate(integrand, config);

    // Expected: 0.5^3 = 0.125
    print_result("Test 15: Box Function (3D)", result, {0.125});
}

// Test 16: Very high dimensional (stress test)
void test_high_dimensional() {
    vegas::Config config;
    config.ndim = 10;
    config.ncomp = 1;
    config.neval = 2000000;
    config.niter = 8;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        // Simple product to avoid underflow
        f[0] = 1.0;
        for (const auto& xi : x) {
            f[0] *= 2.0 * xi;  // Expected value per dimension: 1
        }
    };

    auto result = vegas::integrate(integrand, config);

    // Expected: 1 (product of ten integrals of 2x from 0 to 1 = 1^10)
    print_result("Test 16: High Dimensional (10D)", result, {1.0});
}

// Test 17: Numerical stability test (very small values)
void test_small_values() {
    vegas::Config config;
    config.ndim = 2;
    config.ncomp = 1;
    config.neval = 100000;
    config.niter = 5;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        // Very small values to test numerical stability
        double r2 = (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5);
        f[0] = 1e-10 * std::exp(-100.0 * r2);
    };

    auto result = vegas::integrate(integrand, config);

    // Expected: approximately 1e-10 * π / 100
    double expected = 1e-10 * M_PI / 100.0;
    print_result("Test 17: Small Values (stability)", result, {expected});
}

// Test 18: Mixed scale function
void test_mixed_scale() {
    vegas::Config config;
    config.ndim = 3;
    config.ncomp = 2;
    config.neval = 200000;
    config.niter = 8;
    config.verbose = 0;

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f) {
        // One component is large, another is small
        f[0] = 1e6 * x[0] * x[1] * x[2];
        f[1] = 1e-6 * std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]) * std::sin(M_PI * x[2]);
    };

    auto result = vegas::integrate(integrand, config);

    std::vector<double> expected = {
        1e6 * 0.125,  // (1/2)^3
        1e-6 * std::pow(2.0 / M_PI, 3.0)
    };
    print_result("Test 18: Mixed Scale", result, expected);
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          VEGAS Monte Carlo Integration Test Suite          ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";

    try {
        test_polynomial_2d();
        test_gaussian_3d();
        test_corner_peak();
        test_oscillatory();
        test_product_sines();
        test_discontinuous();
        test_sphere_volume();
        test_multiple_components();
        test_tsuda();
        test_genz_oscillatory();
        test_camel();
        test_anisotropic_gaussian();
        test_near_singularity();
        test_exponential_product();
        test_box_function();
        test_high_dimensional();
        test_small_values();
        test_mixed_scale();

        test_gsl_ising_integral();

        test_smooth_approximation();

        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                  All tests completed!                      ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "\nException caught: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
