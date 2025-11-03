#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <functional>

#include "vegas.hpp"

// Helper function to print test results
void print_result(const std::string &test_name,
                  const vegas::Result &result,
                  const std::vector<double> &expected,
                  bool show_details = true)
{
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
            std::cout << "  χ²    = " << result.chi2[i] << "\n";
        }
    }

    std::cout << "Total evaluations: " << result.neval << "\n";
    std::cout << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
}

// Diagnostic: Test smooth approximation of the same integral
void test_smooth_approximation()
{
    std::cout << "\nDiagnostic: Smooth (regularized) version\n";
    std::cout << std::string(60, '-') << "\n";

    // Same integral but with regularization parameter
    double eps = 0.01; // Removes singularity

    auto integrand = [eps](const std::vector<double> &x, std::vector<double> &f)
    {
        double k0 = M_PI * x[0];
        double k1 = M_PI * x[1];
        double k2 = M_PI * x[2];

        double A = 1.0 / (M_PI * M_PI * M_PI);
        double denom = 1.0 - cos(k0) * cos(k1) * cos(k2) + eps;

        f[0] = A / denom * (M_PI * M_PI * M_PI);
    };

    auto result = vegas::integrate(integrand, 3);

    std::cout << "Regularized result (eps=" << eps << "): "
        << result.integral[0] << " ± " << result.error[0] << "\n";
    std::cout << "χ² = " << result.chi2[0] << "\n";
    std::cout << "Note: This should be lower than 1.393 due to regularization\n";
}

void test_gsl_ising_integral()
{
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "GSL Benchmark: Ising Model Integral\n";
    std::cout << std::string(60, '=') << "\n";

    double exact = 1.3932039296856768591842462603255;

    // Integration over [0, π]^3
    const std::vector xmin = {0.0, 0.0, 0.0};
    const std::vector xmax = {M_PI, M_PI, M_PI};

    // EXACT same integrand as GSL (no modifications!)
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        // x is already in [0, π]^3
        double k0 = x[0];
        double k1 = x[1];
        double k2 = x[2];

        double denom = 1.0 - cos(k0) * cos(k1) * cos(k2);
        f[0] = 1.0 / (M_PI * M_PI * M_PI * denom);
    };

    // Test with different configurations
    std::cout << "\n--- Configuration 1: Default (α=1.5) ---\n";
    auto result1 = vegas::integrate(integrand, xmin, xmax, 1, 1000000, 10, 1.5, 0);

    double error1 = result1.integral[0] - exact;
    double sigma1 = std::abs(error1) / result1.error[0];

    std::cout << "Result = " << std::fixed << std::setprecision(6)
              << result1.integral[0] << " ± " << result1.error[0] << "\n";
    std::cout << "Error  = " << error1 << " (" << sigma1 << " sigma)\n";
    std::cout << "χ²     = " << result1.chi2[0] << "\n";

    std::cout << "\n--- Configuration 2: Different seed ---\n";
    auto result2 = vegas::integrate(integrand, xmin, xmax, 1, 1000000, 10, 1.5, 12345);

    double error2 = result2.integral[0] - exact;
    double sigma2 = std::abs(error2) / result2.error[0];

    std::cout << "Result = " << std::fixed << std::setprecision(6)
              << result2.integral[0] << " ± " << result2.error[0] << "\n";
    std::cout << "Error  = " << error2 << " (" << sigma2 << " sigma)\n";
    std::cout << "χ²     = " << result2.chi2[0] << "\n";

    std::cout << "\n--- Configuration 3: More samples (2M) ---\n";
    auto result3 = vegas::integrate(integrand, xmin, xmax, 1, 2000000, 10, 1.5, 0);

    double error3 = result3.integral[0] - exact;
    double sigma3 = std::abs(error3) / result3.error[0];

    std::cout << "Result = " << std::fixed << std::setprecision(6)
              << result3.integral[0] << " ± " << result3.error[0] << "\n";
    std::cout << "Error  = " << error3 << " (" << sigma3 << " sigma)\n";
    std::cout << "χ²     = " << result3.chi2[0] << "\n";

    std::cout << "\n--- Configuration 4: Lower α=0.5 (conservative) ---\n";
    auto result4 = vegas::integrate(integrand, xmin, xmax, 1, 1000000, 10, 0.5, 0);

    double error4 = result4.integral[0] - exact;
    double sigma4 = std::abs(error4) / result4.error[0];

    std::cout << "Result = " << std::fixed << std::setprecision(6)
              << result4.integral[0] << " ± " << result4.error[0] << "\n";
    std::cout << "Error  = " << error4 << " (" << sigma4 << " sigma)\n";
    std::cout << "χ²     = " << result4.chi2[0] << "\n";

    std::cout << "\n--- Configuration 5: Higher α=2.0 (aggressive) ---\n";
    auto result5 = vegas::integrate(integrand, xmin, xmax, 1, 1000000, 10, 2.0, 0);

    double error5 = result5.integral[0] - exact;
    double sigma5 = std::abs(error5) / result5.error[0];

    std::cout << "Result = " << std::fixed << std::setprecision(6)
              << result5.integral[0] << " ± " << result5.error[0] << "\n";
    std::cout << "Error  = " << error5 << " (" << sigma5 << " sigma)\n";
    std::cout << "χ²     = " << result5.chi2[0] << "\n";

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "GSL reference: 1.393281 ± 0.000362\n";
    std::cout << "Exact value:   " << std::setprecision(16) << exact << "\n";
    std::cout << std::string(60, '=') << "\n";
}

// Test 1: Simple polynomial x*y over [0,1]^2
void test_polynomial_2d()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        f[0] = x[0] * x[1];
    };

    auto result = vegas::integrate(integrand, 2, 1, 250000, 5);

    // Expected: ∫∫ x*y dx dy = 1/4
    print_result("Test 1: Polynomial x*y", result, {0.25});
}

// Test 2: Gaussian in 3D
void test_gaussian_3d()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        double sum = 0.0;
        for (const auto &xi : x) {
            double z = 3.0 * (xi - 0.5); // Center at 0.5, scale
            sum += z * z;
        }
        f[0] = std::exp(-sum);
    };

    auto result = vegas::integrate(integrand, 3, 1, 500000, 5);

    // Expected: [∫_0^1 exp(-(3(x-0.5))^2) dx]^3
    // = [(1/3) * ∫_{-1.5}^{1.5} exp(-u^2) du]^3
    // = [(√π/3) * erf(1.5)]^3
    double expected = std::pow(std::sqrt(M_PI) / 3.0 * std::erf(1.5), 3.0);
    print_result("Test 2: Gaussian in 3D", result, {expected});
}

// Test 3: Corner peak (tests importance sampling)
void test_corner_peak()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        // Peaks near origin
        double a = 0.1;
        f[0] = 1.0 / ((x[0] + a) * (x[1] + a));
    };

    auto result = vegas::integrate(integrand, 2);

    // Expected: log((1+a)/a)^2
    double a = 0.1;
    double expected = std::pow(std::log((1.0 + a) / a), 2.0);
    print_result("Test 3: Corner Peak", result, {expected});
}

// Test 4: Oscillatory function
void test_oscillatory()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        f[0] = std::cos(10.0 * M_PI * x[0]) * std::cos(10.0 * M_PI * x[1]);
    };

    auto result = vegas::integrate(integrand, 2, 1, 1600000, 8);

    // Expected: (sin(10π)/(10π))^2 ≈ 0
    double expected = std::pow(std::sin(10.0 * M_PI) / (10.0 * M_PI), 2.0);
    print_result("Test 4: Oscillatory Function", result, {expected});
}

// Test 5: Product of sines (separable integral)
void test_product_sines()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        f[0] = 1.0;
        for (const auto &xi : x) {
            f[0] *= std::sin(M_PI * xi);
        }
    };

    auto result = vegas::integrate(integrand, 4, 1, 500000, 5);

    // Expected: (2/π)^4
    double expected = std::pow(2.0 / M_PI, 4.0);
    print_result("Test 5: Product of Sines (4D)", result, {expected});
}

// Test 6: Discontinuous function
void test_discontinuous()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        // Step function
        f[0] = (x[0] > 0.5 && x[1] > 0.5) ? 1.0 : 0.0;
    };

    auto result = vegas::integrate(integrand, 2);

    // Expected: 0.25 (area of quarter square)
    print_result("Test 6: Discontinuous (Step)", result, {0.25});
}

// Test 7: Sphere volume in n dimensions
void test_sphere_volume()
{
    const int n = 5;
    const std::vector xmin(n, -1.0);  // Native [-1,1]^n support
    const std::vector xmax(n, 1.0);

    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        // Check if inside unit sphere (x is already in [-1,1]^n)
        double r2 = 0.0;
        for (const auto &xi : x) {
            r2 += xi * xi;
        }
        f[0] = r2 <= 1.0 ? 1.0 : 0.0;
    };

    auto result = vegas::integrate(integrand, xmin, xmax, 1, 4000000, 8);

    // Expected: π^(n/2) / Γ(n/2 + 1)
    double expected = std::pow(M_PI, n / 2.0) / std::tgamma(n / 2.0 + 1.0);
    print_result("Test 7: Sphere Volume (5D)", result, {expected});
}

// Test 8: Multiple components
void test_multiple_components()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        f[0] = x[0] * x[0] + x[1] * x[1]; // 2/3
        f[1] = std::exp(-x[0] - x[1]); // (1-1/e)^2
        f[2] = std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]); // 4/π^2
    };

    auto result = vegas::integrate(integrand, 2, 3, 500000, 5);

    double e = std::exp(1.0);
    std::vector<double> expected = {
        2.0 / 3.0,
        std::pow(1.0 - 1.0 / e, 2.0),
        4.0 / (M_PI * M_PI)
    };

    print_result("Test 8: Multiple Components", result, expected);
}

// Test 9: Tsuda's function (difficult peak)
void test_tsuda()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        double a = 0.1;
        double prod = 1.0;
        for (const auto &xi : x) {
            prod *= 1.0 / (a * a + (xi - 0.5) * (xi - 0.5));
        }
        f[0] = prod;
    };

    auto result = vegas::integrate(integrand, 4, 1, 5000000, 10);

    // Expected: (arctan(0.5/a) + arctan(0.5/a))^4 / a^4
    double a = 0.1;
    double arctan_sum = 2.0 * std::atan(0.5 / a);
    double expected = std::pow(arctan_sum / a, 4.0);
    print_result("Test 9: Tsuda's Function (peaked)", result, {expected});
}

// Test 10: Genz oscillatory test function
void test_genz_oscillatory()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        // Genz "Oscillatory" test function
        std::vector<double> u = {1.0, 1.0, 1.0};
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            sum += u[i] * x[i];
        }
        f[0] = std::cos(2.0 * M_PI * u[0] + sum);
    };

    auto result = vegas::integrate(integrand, 3, 1, 1600000, 8);

    // Analytical solution is complex, just verify it runs
    print_result("Test 10: Genz Oscillatory", result, {}, false);
}

// Test 11: Camel function (multiple peaks)
void test_camel()
{
    const std::vector xmin = {-2.0, -2.0};  // Native [-2,2]^2 support
    const std::vector xmax = {2.0, 2.0};

    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        // x is already in [-2, 2]^2
        double x1 = x[0];
        double x2 = x[1];

        // Six-hump camel function (negated to make peaks)
        double term1 = (4.0 - 2.1 * x1 * x1 + std::pow(x1, 4.0) / 3.0) * x1 * x1;
        double term2 = x1 * x2;
        double term3 = (-4.0 + 4.0 * x2 * x2) * x2 * x2;

        f[0] = std::exp(-(term1 + term2 + term3));
    };

    auto result = vegas::integrate(integrand, xmin, xmax, 1, 2000000, 10);

    print_result("Test 11: Six-Hump Camel (multiple peaks)", result, {}, false);
}

// Test 12: High-dimensional Gaussian with varying widths
void test_anisotropic_gaussian()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        // Different widths in different dimensions
        std::vector<double> sigma = {0.1, 0.2, 0.3, 0.1, 0.2, 0.3};
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double z = (x[i] - 0.5) / sigma[i];
            sum += z * z;
        }
        f[0] = std::exp(-0.5 * sum);
    };

    auto result = vegas::integrate(integrand, 6, 1, 10000000, 10);

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
void test_near_singularity()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        double epsilon = 0.01;
        double r2 = x[0] * x[0] + x[1] * x[1];
        f[0] = 1.0 / std::sqrt(r2 + epsilon);
    };

    auto result = vegas::integrate(integrand, 2, 1, 5000000, 10, 2.0);

    // Approximate expected value (numerical)
    print_result("Test 13: Near-Singularity", result, {}, false);
}

// Test 14: Exponential decay product
void test_exponential_product()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        double lambda = 2.0;
        f[0] = 1.0;
        for (const auto &xi : x) {
            f[0] *= lambda * std::exp(-lambda * xi);
        }
    };

    auto result = vegas::integrate(integrand, 3, 1, 250000, 5);

    // Expected: (1 - e^(-λ))^n
    double lambda = 2.0;
    double expected = std::pow(1.0 - std::exp(-lambda), 3.0);
    print_result("Test 14: Exponential Product", result, {expected});
}

// Test 15: Box function (tests stratification)
void test_box_function()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        // 1 inside a box, 0 outside
        bool inside = true;
        for (const auto &xi : x) {
            inside = inside && (xi >= 0.25 && xi <= 0.75);
        }
        f[0] = inside ? 1.0 : 0.0;
    };

    auto result = vegas::integrate(integrand, 3, 1, 500000, 5);

    // Expected: 0.5^3 = 0.125
    print_result("Test 15: Box Function (3D)", result, {0.125});
}

// Test 16: Very high dimensional (stress test)
void test_high_dimensional()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        // Simple product to avoid underflow
        f[0] = 1.0;
        for (const auto &xi : x) {
            f[0] *= 2.0 * xi; // Expected value per dimension: 1
        }
    };

    auto result = vegas::integrate(integrand, 10, 1, 16000000, 8);

    // Expected: 1 (product of ten integrals of 2x from 0 to 1 = 1^10)
    print_result("Test 16: High Dimensional (10D)", result, {1.0});
}

// Test 17: Numerical stability test (very small values)
void test_small_values()
{
    const std::vector xmin(2, 0.0);
    const std::vector xmax(2, 1.0);

    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        // Very small values to test numerical stability
        double r2 = (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5);
        f[0] = 1e-10 * std::exp(-100.0 * r2);
    };

    auto result = vegas::integrate(integrand, xmin, xmax, 1, 500000, 5);

    // Expected: approximately 1e-10 * π / 100
    double expected = 1e-10 * M_PI / 100.0;
    print_result("Test 17: Small Values (stability)", result, {expected});
}

// Test 18: Mixed scale function
void test_mixed_scale()
{
    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        // One component is large, another is small
        f[0] = 1e6 * x[0] * x[1] * x[2];
        f[1] = 1e-6 * std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]) * std::sin(M_PI * x[2]);
    };

    auto result = vegas::integrate(integrand, 3, 2, 1600000, 8);

    std::vector<double> expected = {
        1e6 * 0.125, // (1/2)^3
        1e-6 * std::pow(2.0 / M_PI, 3.0)
    };
    print_result("Test 18: Mixed Scale", result, expected);
}

// Test 19: Custom boundaries - trigonometric integral
void test_custom_boundaries_trig()
{
    // Integrate over [0, π] × [0, 2π]
    const std::vector xmin(2, 0.0);
    const std::vector xmax = {M_PI, 2.0 * M_PI};

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f)
    {
        // x[0] ∈ [0, π], x[1] ∈ [0, 2π]
        // Integrate: ∫₀^π ∫₀^(2π) sin(x) * cos(y) dy dx
        f[0] = std::sin(x[0]) * std::cos(x[1]);
    };

    auto result = vegas::integrate(integrand, xmin, xmax);

    // Expected: ∫₀^π sin(x) dx * ∫₀^(2π) cos(y) dy
    //         = [-cos(x)]₀^π * [sin(y)]₀^(2π)
    //         = (-(-1) - (-1)) * (0 - 0) = 0
    double expected = 0.0;

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Test 19: Custom Boundaries [0,π]×[0,2π]\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Domain: [0, π] × [0, 2π]\n";
    std::cout << "Function: sin(x) * cos(y)\n";
    std::cout << "Component 0:\n";
    std::cout << "  Result   = " << std::scientific << std::setprecision(8)
              << result.integral[0] << " ± " << result.error[0] << "\n";
    std::cout << "  Expected = " << expected << "\n";
    double diff = std::abs(result.integral[0] - expected);
    double sigma = diff / result.error[0];
    std::cout << "  Diff     = " << diff << " (" << sigma << " sigma)\n";
    std::cout << "  χ²       = " << result.chi2[0] << "\n";
    std::cout << "Total evaluations: " << result.neval << "\n";
    std::cout << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
}

// Test 20: Custom boundaries - Gaussian in shifted domain
void test_custom_boundaries_gaussian()
{
    // Integrate over [-3, 3]^3
    const std::vector xmin = {-3.0, -3.0, -3.0};
    const std::vector xmax = {3.0, 3.0, 3.0};

    auto integrand = [](const std::vector<double>& x, std::vector<double>& f)
    {
        // x ∈ [-3, 3]^3
        // Integrate: exp(-x²/2 - y²/2 - z²/2) over [-3,3]^3
        double sum = 0.0;
        for (const auto& xi : x) {
            sum += xi * xi;
        }
        f[0] = std::exp(-0.5 * sum);
    };

    auto result = vegas::integrate(integrand, xmin, xmax, 1, 2000000);

    // Expected: ∏ᵢ ∫₋₃³ exp(-xᵢ²/2) dxᵢ
    //         = [√(2π) * (Φ(3) - Φ(-3))]³
    //         ≈ [√(2π) * 0.9973]³
    // Where Φ is the standard normal CDF
    double integral_1d = std::sqrt(2.0 * M_PI) * std::erf(3.0 / std::sqrt(2.0));
    double expected = std::pow(integral_1d, 3.0);

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Test 20: Custom Boundaries [-3,3]³\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Domain: [-3, 3]³\n";
    std::cout << "Function: exp(-0.5*(x²+y²+z²))\n";
    std::cout << "Component 0:\n";
    std::cout << "  Result   = " << std::scientific << std::setprecision(8)
              << result.integral[0] << " ± " << result.error[0] << "\n";
    std::cout << "  Expected = " << expected << "\n";
    double diff = std::abs(result.integral[0] - expected);
    double sigma = diff / result.error[0];
    std::cout << "  Diff     = " << diff << " (" << sigma << " sigma)\n";
    std::cout << "  χ²       = " << result.chi2[0] << "\n";
    std::cout << "Total evaluations: " << result.neval << "\n";
    std::cout << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
}

// Test 21: Large number of components (ncomp=2000) with automatic verification
void test_large_ncomp()
{
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Test 21: Large Number of Components (ncomp=2000)\n";
    std::cout << std::string(60, '=') << "\n";

    const int num_components = 2000;

    // Pre-compute expected values for all components
    std::vector<double> expected(num_components);

    // Components 0-999: Polynomials ∫₀¹ x^n * y^m dx dy
    for (int i = 0; i < 1000; ++i) {
        int n = i % 10;  // Powers 0-9 for x
        int m = i / 10;  // Powers 0-99 for y
        expected[i] = 1.0 / ((n + 1) * (m + 1));
    }

    // Components 1000-1999: Exponential decay ∫₀¹ exp(-λx) * exp(-μy) dx dy
    for (int i = 1000; i < 2000; ++i) {
        double lambda = 0.5 + 0.01 * (i - 1000);  // λ from 0.5 to 10.49
        double mu = 1.0 + 0.02 * (i - 1000);       // μ from 1.0 to 20.98
        expected[i] = (1.0 - std::exp(-lambda)) * (1.0 - std::exp(-mu));
    }

    auto integrand = [](const std::vector<double> &x, std::vector<double> &f)
    {
        // Components 0-999: Polynomials
        for (int i = 0; i < 1000; ++i) {
            int n = i % 10;
            int m = i / 10;
            f[i] = std::pow(x[0], n) * std::pow(x[1], m);
        }

        // Components 1000-1999: Exponential decay
        for (int i = 1000; i < 2000; ++i) {
            double lambda = 0.5 + 0.01 * (i - 1000);
            double mu = 1.0 + 0.02 * (i - 1000);
            f[i] = lambda * std::exp(-lambda * x[0]) * mu * std::exp(-mu * x[1]);
        }
    };

    std::cout << "Computing " << num_components << " integrals...\n";
    auto start = std::chrono::high_resolution_clock::now();

    auto result = vegas::integrate(integrand, 2, num_components, 4000000, 8);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Computation time: " << duration.count() << " ms\n";
    std::cout << "Total evaluations: " << result.neval << "\n\n";

    // Automatic verification with assertions
    std::cout << "Running automatic verification...\n";
    std::cout << std::string(60, '-') << "\n";

    int passed = 0;
    int failed = 0;
    int failed_3sigma = 0;
    int failed_5sigma = 0;
    double max_sigma = 0.0;
    int max_sigma_component = -1;
    double total_relative_error = 0.0;

    std::vector<int> failed_components;

    for (int comp = 0; comp < num_components; ++comp) {
        // Check for NaN or inf
        if (!std::isfinite(result.integral[comp]) || !std::isfinite(result.error[comp])) {
            std::cerr << "ERROR: Component " << comp << " has non-finite values!\n";
            std::cerr << "  Result = " << result.integral[comp]
                      << " ± " << result.error[comp] << "\n";
            ++failed;
            failed_components.push_back(comp);
            continue;
        }

        // Check for zero error (shouldn't happen)
        if (result.error[comp] <= 0.0) {
            std::cerr << "ERROR: Component " << comp << " has zero or negative error!\n";
            ++failed;
            failed_components.push_back(comp);
            continue;
        }

        // Compute deviation
        double diff = std::abs(result.integral[comp] - expected[comp]);
        double sigma = diff / result.error[comp];
        double rel_error = (expected[comp] != 0.0) ?
            std::abs(diff / expected[comp]) : diff;

        total_relative_error += rel_error;

        if (sigma > max_sigma) {
            max_sigma = sigma;
            max_sigma_component = comp;
        }

        // Count failures at different sigma levels
        if (sigma > 5.0) {
            ++failed_5sigma;
            ++failed;
            failed_components.push_back(comp);
        } else if (sigma > 3.0) {
            ++failed_3sigma;
        } else {
            ++passed;
        }
    }

    // Calculate statistics
    double pass_rate = 100.0 * passed / num_components;
    double avg_relative_error = total_relative_error / num_components;

    // Print summary
    std::cout << "\nVerification Results:\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << "✓ Passed (< 3σ):          " << passed << " / " << num_components
              << " (" << std::fixed << std::setprecision(2) << pass_rate << "%)\n";
    std::cout << "⚠ Warning (3-5σ):         " << failed_3sigma << "\n";
    std::cout << "✗ Failed (> 5σ):          " << failed_5sigma << "\n";
    std::cout << "✗ Failed (NaN/Inf/Zero):  " << failed - failed_5sigma << "\n";
    std::cout << "\nStatistics:\n";
    std::cout << "  Max deviation:          " << std::scientific << std::setprecision(4)
              << max_sigma << "σ (component " << max_sigma_component << ")\n";
    std::cout << "  Avg relative error:     " << avg_relative_error << "\n";
    std::cout << "  Overall converged:      " << (result.converged ? "Yes" : "No") << "\n";

    // Show some example results
    std::cout << "\nSample Results:\n";
    std::cout << std::string(60, '-') << "\n";
    std::vector<int> samples = {0, 100, 500, 999, 1000, 1500, 1999};
    for (int comp : samples) {
        double diff = std::abs(result.integral[comp] - expected[comp]);
        double sigma = diff / result.error[comp];
        std::cout << "Component " << std::setw(4) << comp << ": "
                  << std::scientific << std::setprecision(6)
                  << result.integral[comp] << " ± " << result.error[comp]
                  << " (expected: " << expected[comp] << ", "
                  << std::fixed << std::setprecision(2) << sigma << "σ)\n";
    }

    // Show worst failures if any
    if (!failed_components.empty() && failed_components.size() <= 10) {
        std::cout << "\nFailed Components:\n";
        std::cout << std::string(60, '-') << "\n";
        for (int comp : failed_components) {
            double diff = std::abs(result.integral[comp] - expected[comp]);
            double sigma = diff / result.error[comp];
            std::cout << "Component " << std::setw(4) << comp << ": "
                      << std::scientific << std::setprecision(6)
                      << result.integral[comp] << " ± " << result.error[comp]
                      << " (expected: " << expected[comp] << ", "
                      << std::fixed << std::setprecision(2) << sigma << "σ)\n";
        }
    } else if (failed_components.size() > 10) {
        std::cout << "\n" << failed_components.size() << " components failed (showing first 10):\n";
        std::cout << std::string(60, '-') << "\n";
        for (int i = 0; i < 10; ++i) {
            int comp = failed_components[i];
            double diff = std::abs(result.integral[comp] - expected[comp]);
            double sigma = diff / result.error[comp];
            std::cout << "Component " << std::setw(4) << comp << ": "
                      << std::scientific << std::setprecision(6)
                      << result.integral[comp] << " ± " << result.error[comp]
                      << " (expected: " << expected[comp] << ", "
                      << std::fixed << std::setprecision(2) << sigma << "σ)\n";
        }
    }

    std::cout << "\n" << std::string(60, '=') << "\n";

    // Assertions
    std::cout << "\nAssertion Checks:\n";
    std::cout << std::string(60, '-') << "\n";

    // Check 1: No NaN or Inf values
    bool check1 = (failed - failed_5sigma) == 0;
    std::cout << (check1 ? "✓" : "✗") << " No NaN/Inf/Zero errors\n";
    if (!check1) {
        throw std::runtime_error("ASSERTION FAILED: Found NaN/Inf/Zero values");
    }

    // Check 2: Pass rate > 95% (allowing for statistical fluctuations)
    // With 2000 components, expect ~5% to be > 2σ by chance
    bool check2 = pass_rate >= 95.0;
    std::cout << (check2 ? "✓" : "✗") << " Pass rate >= 95% (got "
              << std::fixed << std::setprecision(2) << pass_rate << "%)\n";
    if (!check2) {
        throw std::runtime_error("ASSERTION FAILED: Pass rate too low");
    }

    // Check 3: No catastrophic failures (> 10σ)
    bool check3 = max_sigma < 10.0;
    std::cout << (check3 ? "✓" : "✗") << " Max deviation < 10σ (got "
              << std::fixed << std::setprecision(2) << max_sigma << "σ)\n";
    if (!check3) {
        throw std::runtime_error("ASSERTION FAILED: Catastrophic deviation detected");
    }

    // Check 4: Average relative error is reasonable
    bool check4 = avg_relative_error < 0.01;  // < 1% average
    std::cout << (check4 ? "✓" : "✗") << " Avg relative error < 1% (got "
              << std::scientific << std::setprecision(4) << avg_relative_error << ")\n";
    if (!check4) {
        throw std::runtime_error("ASSERTION FAILED: Average relative error too high");
    }

    // Check 5: Less than 1% complete failures (> 5σ)
    double failure_rate = 100.0 * failed_5sigma / num_components;
    bool check5 = failure_rate < 1.0;
    std::cout << (check5 ? "✓" : "✗") << " Failure rate < 1% (got "
              << std::fixed << std::setprecision(2) << failure_rate << "%)\n";
    if (!check5) {
        throw std::runtime_error("ASSERTION FAILED: Too many 5-sigma failures");
    }

    std::cout << "\n✓ All assertions passed!\n";
    std::cout << std::string(60, '=') << "\n";
}

int main()
{
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
        test_custom_boundaries_trig();
        test_custom_boundaries_gaussian();

        // Stress test with many components
        test_large_ncomp();

        test_gsl_ising_integral();

        test_smooth_approximation();

        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                  All tests completed!                      ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
    } catch (const std::exception &e) {
        std::cerr << "\nException caught: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
