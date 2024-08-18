#include "BIM.hpp"
#include "QuantumAnalysisTools.hpp"
#include "Plotting.hpp"
#include <iostream>
#include <matplot/matplot.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/DenseGenMatProd.h>

using namespace matplot;

std::vector<std::pair<double, double>> findLocalMinima(const std::vector<std::pair<double, double>>& k_magnitude_pairs) {
    std::vector<std::pair<double, double>> local_minima;

    if (k_magnitude_pairs.size() < 3) {
        // Not enough points to have a local minimum
        return local_minima;
    }

    for (size_t i = 1; i < k_magnitude_pairs.size() - 1; ++i) {
        double prev_magnitude = k_magnitude_pairs[i - 1].second;
        double current_magnitude = k_magnitude_pairs[i].second;
        double next_magnitude = k_magnitude_pairs[i + 1].second;

        if (current_magnitude < prev_magnitude && current_magnitude < next_magnitude) {
            local_minima.push_back(k_magnitude_pairs[i]);
        }
    }

    return local_minima;
}

void integralTesting() { // GIVES CORRECT RESULTS - OK
    try {
        // Define the function to integrate: f(x) = x^2
        auto func = [](const double x) {
            return x * x * x + x * x;
        };

        // Compute the integral from 0 to 1
        constexpr double lower = 0.0;
        constexpr double upper = 1.0;
        const double result = Boundary::Integral::computeIntegral(func, lower, upper);
        std::cout << "Integral result: " << result << std::endl;

        // Expected result
        constexpr double expected = 1.0 / 4.0 + 1.0 / 3.0;
        std::cout << "Expected result: " << expected << std::endl;

        if (std::abs(result - expected) < 1e-6) {
            std::cout << "Test passed." << std::endl;
        } else {
            std::cout << "Test failed." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

}

/**
 * @brief Plots the real part, imaginary part, and magnitude of the Hankel function H0^{(1)}(z) for real inputs.
 *
 * This function generates plots for the real part, imaginary part, and magnitude of the Hankel function H0^{(1)}(z)
 * for real inputs over a specified range.
 */
void plotHankelRealInput() { // CORRECT
    constexpr int num_points = 200;
    constexpr double start = 0.1;
    constexpr double end = 10.0;
    std::vector<double> x_values(num_points);
    std::vector<double> real_part(num_points);
    std::vector<double> imag_part(num_points);
    std::vector<double> magnitude(num_points);

    constexpr double step = (end - start) / (num_points - 1);
    for (int i = 0; i < num_points; ++i) {
        const double x = start + i * step;
        x_values[i] = x;

        gsl_sf_result result_J1;
        gsl_sf_result result_Y1;
        gsl_sf_bessel_J1_e(x, &result_J1);
        gsl_sf_bessel_Y1_e(x, &result_Y1);

        std::complex<double> hankel(result_J1.val, result_Y1.val);
        real_part[i] = std::real(hankel);
        imag_part[i] = std::imag(hankel);
        magnitude[i] = std::abs(hankel);
    }

    const auto fig = figure(true);
    fig->size(800, 600);

    const auto ax1 = fig->add_subplot(3, 1, 0);
    ax1->plot(x_values, real_part);
    ax1->title("Real part of H1^{(1)}(x)");
    ax1->xlabel("x");
    ax1->ylabel("Re(H1^{(1)}(x))");
    hold(ax1, true);
    ax1->plot(x_values, std::vector<double>(num_points, 0), "--k");  // Plot y=0 axis

    const auto ax2 = fig->add_subplot(3, 1, 1);
    ax2->plot(x_values, imag_part);
    ax2->title("Imaginary part of H1^{(1)}(x)");
    ax2->xlabel("x");
    ax2->ylabel("Im(H1^{(1)}(x))");
    hold(ax2, true);
    ax2->plot(x_values, std::vector<double>(num_points, 0), "--k");  // Plot y=0 axis

    const auto ax3 = fig->add_subplot(3, 1, 2);
    ax3->plot(x_values, magnitude);
    ax3->title("Magnitude of H1^{(1)}(x)");
    ax3->xlabel("x");
    ax3->ylabel("|H1^{(1)}(x)|");
    hold(ax3, true);
    ax3->plot(x_values, std::vector<double>(num_points, 0), "--k");  // Plot y=0 axis

    // Find intersections with the x-axis and add markers
    for (int i = 1; i < num_points; ++i) {
        if ((real_part[i - 1] <= 0 && real_part[i] > 0) || (real_part[i - 1] >= 0 && real_part[i] < 0)) {
            const double x_intercept = x_values[i];
            ax1->hold(true);
            ax1->text(x_intercept, 0, std::to_string(x_intercept));
        }
        if ((imag_part[i - 1] <= 0 && imag_part[i] > 0) || (imag_part[i - 1] >= 0 && imag_part[i] < 0)) {
            const double x_intercept = x_values[i];
            ax2->hold(true);
            ax2->text(x_intercept, 0, std::to_string(x_intercept));
        }
    }

    show();
}

/**
 * @brief Plots the mesh and contour plots for the real part, imaginary part, and magnitude of the Hankel function of the first kind H_0^{(1)}(z)
 * for complex inputs over a grid.
 *
 * This function generates mesh and contour plots of the real part, imaginary part, and magnitude of the Hankel function of the first kind
 * H_0^{(1)}(z) for a grid of complex inputs ranging from 0.1 to 10 in both real and imaginary parts.
 */
void plotHankelMeshAndContour() { // CORRECT
    constexpr int num_points = 200;
    constexpr double start = -10.0;
    constexpr double end = 10.0;

    std::vector<std::vector<double>> ReZ(num_points, std::vector<double>(num_points));
    std::vector<std::vector<double>> ImZ(num_points, std::vector<double>(num_points));
    std::vector<std::vector<double>> MagZ(num_points, std::vector<double>(num_points));
    std::vector<std::vector<double>> x_values(num_points, std::vector<double>(num_points));
    std::vector<std::vector<double>> y_values(num_points, std::vector<double>(num_points));

    constexpr double step = (end - start) / (num_points - 1);
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < num_points; ++j) {
            x_values[i][j] = start + i * step;
            y_values[i][j] = start + j * step;
        }
    }

    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < num_points; ++j) {
            std::complex<double> z(x_values[i][j], y_values[i][j]);
            gsl_sf_result real_part;
            gsl_sf_result imag_part;
            gsl_sf_bessel_J1_e(std::abs(z), &real_part);
            gsl_sf_bessel_Y1_e(std::abs(z), &imag_part);
            std::complex<double> hankel(real_part.val, imag_part.val);
            ReZ[i][j] = real(hankel);
            ImZ[i][j] = imag(hankel);
            MagZ[i][j] = std::abs(hankel);
        }
    }
    const auto fig = figure(true);
    fig->size(800, 600);
    const auto ax1 = fig->add_subplot(2, 3, 0);
    ax1->mesh(x_values, y_values, ReZ);
    ax1->title("Re(H1^{(1)}(z))");
    ax1->xlabel("x");
    ax1->ylabel("y");

    const auto ax2 = fig->add_subplot(2, 3, 1);
    ax2->mesh(x_values, y_values, ImZ);
    ax2->title("Im(H1^{(1)}(z))");
    ax2->xlabel("x");
    ax2->ylabel("y");

    const auto ax3 = fig->add_subplot(2, 3, 2);
    ax3->mesh(x_values, y_values, MagZ);
    ax3->title("|H1^{(1)}(z)|");
    ax3->xlabel("x");
    ax3->ylabel("y");

    const auto ax4 = fig->add_subplot(2, 3, 3);
    ax4->contour(x_values, y_values, ReZ);
    ax4->title("Re(H1^{(1)}(z))");
    ax4->xlabel("x");
    ax4->ylabel("y");

    const auto ax5 = fig->add_subplot(2, 3, 4);
    ax5->contour(x_values, y_values, ImZ);
    ax5->title("Im(H1^{(1)}(z))");
    ax5->xlabel("x");
    ax5->ylabel("y");

    const auto ax6 = fig->add_subplot(2, 3, 5);
    ax6->contour(x_values, y_values, MagZ);
    ax6->title("|H1^{(1)}(z)|");
    ax6->xlabel("x");
    ax6->ylabel("y");

    show();
}

void executeBIMCircleSVD(const bool showCircle = true) {
    using namespace BIM;
    double radius = 1.0;
    Point center(0.0, 0.0);
    auto circleBoundary = std::make_shared<Circle>(center, radius);

    if (showCircle) {
        // Plot the boundary using matplot
        const auto fig = figure(true);
        const auto ax = fig->add_axes();
        circleBoundary->plot(ax, 50, true, false);
        show();
    }

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(circleBoundary);
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 20.0;
    constexpr int SIZE_K = 200000;
    constexpr int scalingFactor = 15;

    // Create and use the KRangeSolver
    KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, circleBoundary, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);
    solver.printLocalMinimaOfSingularValues(0.1);
    auto localMinima = solver.findLocalMinima();
    auto secondLocalMinima = solver.findLocalMinima(2);
    auto thirdLocalMinima = solver.findLocalMinima(3);
    auto fourthLocalMinima = solver.findLocalMinima(4);

    auto analytical_eigenvalues = AnalysisTools::computeCircleAnalyticalEigenvalues(k_min, k_max, radius);

    // Function to print differences
    auto printDifferences = [](const auto& minima, const std::vector<std::tuple<double, int, int>>& analytical_eigenvalues_comp, const std::string& label) {
        std::cout << "\nDifferences between numerical and analytical k values for " << label << ":\n";
        for (const auto& [numerical_k, smallest_singular_value] : minima) {
            // Find the closest analytical eigenvalue
            auto closest_analytical = *std::ranges::min_element(analytical_eigenvalues_comp,
                                                                [numerical_k](const auto& a, const auto& b) {
                                                                    return std::abs(std::get<0>(a) - numerical_k) < std::abs(std::get<0>(b) - numerical_k);
                                                                });

            double closest_analytical_k = std::get<0>(closest_analytical);
            int m = std::get<1>(closest_analytical);
            int n = std::get<2>(closest_analytical);

            double absolute_difference = std::abs(numerical_k - closest_analytical_k);
            double relative_difference = absolute_difference / closest_analytical_k;

            if (smallest_singular_value < 0.05) {
                std::cout << "Numerical k: " << numerical_k
                      << ", Analytical k: " << closest_analytical_k
                      << " (m: " << m << ", n: " << n << ")"
                      << ", Smallest Singular Value: " << smallest_singular_value
                      << ", Absolute Difference: " << absolute_difference
                      << ", Relative Difference: " << relative_difference << std::endl;
            }
        }
    };

    // Print differences for the first, second, and third local minima
    printDifferences(localMinima, analytical_eigenvalues, "first local minima");
    printDifferences(secondLocalMinima, analytical_eigenvalues, "second local minima");
    printDifferences(thirdLocalMinima, analytical_eigenvalues, "third local minima");
    printDifferences(fourthLocalMinima, analytical_eigenvalues, "fourth local minima");

    // Compare numerical k values between the first, second, and third local minima
    std::cout << "\nComparing numerical k values between first, second, and third local minima:\n";

    constexpr double tolerance = 1e-2; // Define a tolerance for comparing k values

    // ReSharper disable once CppDeclarationHidesUncapturedLocal
    auto compareMinima = [](const auto& minima1, const auto& minima2, const std::string& label1, const std::string& label2, double tolerance) {
        std::cout << "\nDifferences between " << label1 << " and " << label2 << " (within tolerance):\n";
        for (const auto& [numerical_k1, smallest_singular_value1] : minima1) {
            for (const auto& [numerical_k2, smallest_singular_value2] : minima2) {
                if (std::abs(numerical_k1 - numerical_k2) <= tolerance) {
                    double absolute_difference = std::abs(numerical_k1 - numerical_k2);
                    double relative_difference = absolute_difference / numerical_k1;

                    std::cout << "Numerical k (" << label1 << "): " << numerical_k1
                          << ", Numerical k (" << label2 << "): " << numerical_k2
                          << ", Absolute Difference: " << absolute_difference
                          << ", Relative Difference: " << relative_difference << std::endl;
                }
            }
        }
    };

    // Compare first vs second, first vs third, and second vs third within tolerance
    compareMinima(localMinima, secondLocalMinima, "first local minima", "second local minima", tolerance);
    compareMinima(localMinima, thirdLocalMinima, "first local minima", "third local minima", tolerance);
    compareMinima(secondLocalMinima, thirdLocalMinima, "second local minima", "third local minima", tolerance);
    compareMinima(thirdLocalMinima, fourthLocalMinima, "third local minima", "fourth local minima", tolerance);

    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    // Plotting the smallest singular values
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, -0.003, 0.042);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, 0.0, 1.0);

    // Plotting the second smallest singular values
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 2, -0.003, 0.042);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 2, 0.0, 1.0);

    // Plotting the third smallest singular values
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 3, -0.003, 0.042);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 3, 0.0, 1.0);

    // Plotting the fourth smallest singular values
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 4, -0.003, 0.042);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 4, 0.0, 1.0);

    // Plot analytical solutions as purple diamonds
    const std::vector<double> analytical_x(analytical_eigenvalues.size(), 0);
    std::vector<double> analytical_k_values;
    analytical_k_values.reserve(analytical_eigenvalues.size());
    for (const auto& [k_mn, nu, zero_index] : analytical_eigenvalues) {
        analytical_k_values.push_back(k_mn);
    }

    const auto l = ax->scatter(analytical_k_values, analytical_x);
    l->display_name("Analytical");
    l->marker_style(line_spec::marker_style::diamond);
    l->marker_size(15);
    l->marker_color("purple");

    const auto l2 = ax2->scatter(analytical_k_values, analytical_x);
    l2->display_name("Analytical");
    l2->marker_style(line_spec::marker_style::diamond);
    l2->marker_size(15);
    l2->marker_color("purple");

    // Add vertical dashed asymptotes at analytical k values
    for (const auto& [k_mn, nu, l] : analytical_eigenvalues) {
        if (nu % 2 == 0) {
            ax->plot({k_mn, k_mn}, {ax->y_axis().limits().at(0), ax->y_axis().limits().at(1)}, "k--");
            // Add label for (nu,l) parallel to the dashed line
            double offset_y;
            if ((nu / 2) % 2 == 0) {
                offset_y = ax->y_axis().limits().at(1) * 0.95; // Slightly below the top of the y-axis
            } else {
                offset_y = ax->y_axis().limits().at(1) * 0.75; // Slightly more below the top of the y-axis
            }
            auto txt = ax->text(k_mn, offset_y, "(" + std::to_string(nu) + "," + std::to_string(l) + ")");
            txt->font_size(10);
            txt->color("black");
        }
    }

    // Add labels for (k,l) pairs below the diamonds in ax2
    for (const auto& [k_mn, nu, zero_index] : analytical_eigenvalues) {
        double offset_y = (nu % 2 == 0) ? 0.001 : -0.001; // Alternate offset for even and odd values
        if (nu == 0) {
            offset_y *= 2;
        }
        auto txt = ax->text(k_mn, offset_y, "(" + std::to_string(nu) + "," + std::to_string(zero_index) + ")");
        txt->font_size(10);
        if (nu == 0) {
            txt->color("red");
        } else {
            txt->color("black");
        }
    }
    matplot::legend(ax, false);

    auto degeneraciesResult = solver.calculateDegeneracies(0.05, 1e-5); // for k=40 we choose 0.007 for no missing levels
    KRangeSolver::printDegeneracies(std::get<0>(degeneraciesResult));
    const auto numerical_eigenvalues = std::get<0>(degeneraciesResult);

    const auto figComparison = figure(true);
    figComparison->size(800, 800);
    auto axComparison = figComparison->add_axes();
    axComparison->xlabel("k");
    axComparison->ylabel("Number of Eigenvalues");
    axComparison->title("Comparing theoretical vs. numerical eigenvalue counting functions");
    matplot::legend(axComparison, true);
    AnalysisTools::plotTheoreticalEigenvalues(axComparison, AnalysisTools::computeCircleAnalyticalEigenvalues(k_min, k_max, radius), k_min, k_max);
    AnalysisTools::plotNumericalEigenvalues(axComparison, numerical_eigenvalues, k_min, k_max);

    // The other plots concerning level spacings and oscillating part of the number of states
    auto fig_spacingsOscillatoty = figure(true);
    fig_spacingsOscillatoty->size(1500, 1000);
    auto ax_upper = fig_spacingsOscillatoty->add_subplot(2, 2, 1);
    auto ax_lower = fig_spacingsOscillatoty->add_subplot(2, 2, 2);
    auto ax_oscillatory = fig_spacingsOscillatoty->add_subplot(2, 2, 3);

    // The level spacing distribution
    auto numerical_results = std::get<0>(degeneraciesResult);
    auto unfolded_energies = AnalysisTools::unfoldEnergyEigenvalues(numerical_results, circleBoundary, circleBoundary->calculateArcLength(), 0.25);
    //auto _ = AnalysisTools::plotSpacingDistribution(ax_upper, ax_lower, unfolded_energies, 0.5);
    // The oscillating part of the number of energy eigenvalues
    //auto u = AnalysisTools::plotOscillatoryPartWeyl(ax_oscillatory, std::get<0>(degeneraciesResult), circleBoundary, circleBoundary->calculateArcLength());

    //show(fig);
    save(fig, "Circle_SVD_plot.png");
    //show(figComparison);
    save(figComparison, "Circle_Comparison_plot.png");
    //show(fig_spacingsOscillatoty);
    save(fig_spacingsOscillatoty, "Circle_spacings_oscillatory.png");
}

void executeBIMQuarterCircleSVD(const bool showCircle = true) {
    using namespace BIM;
    double radius = 1.0;
    Point center(0.0, 0.0);
    auto circleBoundary = std::make_shared<QuarterCircle>(center, radius, Point{1.0,0.0});

    if (showCircle) {
        // Plot the boundary using matplot
        const auto fig = figure(true);
        const auto ax = fig->add_axes();
        circleBoundary->plot(ax, 50, true, false);
        show();
    }

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<XYReflectionSymmetryNNStandard>(circleBoundary);
    // KRangeSolver parameters
    constexpr double k_min = 5.0;
    constexpr double k_max = 20.0;
    constexpr int SIZE_K = 200000;
    constexpr int scalingFactor = 10;

    // Create and use the KRangeSolver
    KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, circleBoundary, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);
    solver.printLocalMinimaOfSingularValues(0.1);
    auto localMinima = solver.findLocalMinima(0.15);

    // Analytical eigenvalues using GSL for Bessel function zeros
    std::vector<std::tuple<double, int, int>> analytical_eigenvalues;
    std::cout << "Analytical eigenvalues for the circle:" << std::endl;

    // Iterate over orders nu (m)
    for (int nu = 0; nu <= 100; ++nu) {
        // Start with the first zero and increment until we exceed k_max * radius
        int zero_index = 1;
        while (true) {
            // Compute the zero of the Bessel function of order nu
            const double zero = gsl_sf_bessel_zero_Jnu(nu, zero_index);
            // Calculate the corresponding eigenvalue k_mn
            double k_mn = zero / radius;
            // Break the loop if k_mn exceeds k_max
            if (k_mn > k_max) break;
            // Add k_mn to the list if it is within the specified range
            if (k_min <= k_mn) {
                analytical_eigenvalues.emplace_back(k_mn, nu, zero_index);
                std::cout << "k(" << nu << "," << zero_index << ") = " << k_mn << std::endl;
            }
            ++zero_index;
        }
    }

    // Calculate and print the differences between numerical and analytical eigenvalues
    std::cout << "\nDifferences between numerical and analytical k values:\n";
    for (const auto& [numerical_k, smallest_singular_value] : localMinima) {
        // Find the closest analytical eigenvalue
        auto closest_analytical = *std::ranges::min_element(analytical_eigenvalues,
                                                            [numerical_k](const auto& a, const auto& b) {
                                                                return std::abs(std::get<0>(a) - numerical_k) < std::abs(std::get<0>(b) - numerical_k);
                                                            });

        double closest_analytical_k = std::get<0>(closest_analytical);
        int nu = std::get<1>(closest_analytical);
        int zero_index = std::get<2>(closest_analytical);

        double absolute_difference = std::abs(numerical_k - closest_analytical_k);
        double relative_difference = absolute_difference / closest_analytical_k;

        if (smallest_singular_value < 0.2) {
            std::cout << "Numerical k: " << numerical_k
                  << ", Analytical k: " << closest_analytical_k
                  << " (k: " << nu << ", l: " << zero_index << ")"
                  << ", Smallest Singular Value: " << smallest_singular_value
                  << ", Absolute Difference: " << absolute_difference
                  << ", Relative Difference: " << relative_difference << std::endl;
        }
    }

    const auto fig = figure(true);
    fig->size(1800, 1200);
    const auto ax = fig->add_subplot(1, 1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, -0.05, 0.5);

    // Plot analytical solutions as purple diamonds
    const std::vector<double> analytical_x(analytical_eigenvalues.size(), 0);
    std::vector<double> analytical_k_values;
    analytical_k_values.reserve(analytical_eigenvalues.size());
    for (const auto& [k_mn, nu, zero_index] : analytical_eigenvalues) {
        analytical_k_values.push_back(k_mn);
    }

    const auto l = ax->scatter(analytical_k_values, analytical_x);
    l->display_name("Analytical");
    l->marker_style(line_spec::marker_style::diamond);
    l->marker_size(15);
    l->marker_color("purple");

    // Add vertical dashed asymptotes at analytical k values
    for (const auto& [k_mn, nu, l] : analytical_eigenvalues) {
        if (nu % 2 == 0) {
            ax->plot({k_mn, k_mn}, {ax->y_axis().limits().at(0), ax->y_axis().limits().at(1)}, "k--");
            // Add label for (nu,l) parallel to the dashed line
            double offset_y;
            if ((nu / 2) % 2 == 0) {
                offset_y = ax->y_axis().limits().at(1) * 0.95; // Slightly below the top of the y-axis
            } else {
                offset_y = ax->y_axis().limits().at(1) * 0.75; // Slightly more below the top of the y-axis
            }
            auto txt = ax->text(k_mn, offset_y, "(" + std::to_string(nu) + "," + std::to_string(l) + ")");
            txt->font_size(10);
            txt->color("black");
        }
    }

    // Add labels for (k,l) pairs below the diamonds in ax2
    for (const auto& [k_mn, nu, zero_index] : analytical_eigenvalues) {
        double offset_y = (nu % 2 == 0) ? 0.02 : -0.02; // Alternate offset for even and odd values
        if (nu == 0) {
            offset_y *= 2;
        }
        auto txt = ax->text(k_mn, offset_y, "(" + std::to_string(nu) + "," + std::to_string(zero_index) + ")");
        txt->font_size(10);
        if (nu == 0) {
            txt->color("red");
        } else {
            txt->color("black");
        }
    }
    matplot::legend(ax, false);
    save(fig, "Quarter_Circle_SVD_NN.png");
    show();
}

void executeBIMHalfCircleSVD(const bool showCircle = true) {
    using namespace BIM;
    double radius = 1.0;
    Point center(0.0, 0.0);
    auto circleBoundary = std::make_shared<SemiCircle>(center, radius, Point{1.0,0.0});

    if (showCircle) {
        // Plot the boundary using matplot
        const auto fig = figure(true);
        const auto ax = fig->add_axes();
        ax->xlim({-1.5, 1.5});
        ax->ylim({-1.5, 1.5});
        circleBoundary->plot(ax, 50, true, true);
        show();
    }

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<YReflectionSymmetryDStandard>(circleBoundary);
    // KRangeSolver parameters
    constexpr double k_min = 20.0;
    constexpr double k_max = 30.0;
    constexpr int SIZE_K = 10000;
    constexpr int scalingFactor = 40;

    // Create and use the KRangeSolver
    KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, circleBoundary, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);
    solver.printLocalMinimaOfSingularValues(0.1);
    auto localMinima = solver.findLocalMinima();

    // Analytical eigenvalues using GSL for Bessel function zeros
    std::vector<std::tuple<double, int, int>> analytical_eigenvalues;
    std::cout << "Analytical eigenvalues for the circle:" << std::endl;

    // Iterate over orders nu (m)
    for (int nu = 0; nu <= 100; ++nu) {
        // Start with the first zero and increment until we exceed k_max * radius
        int zero_index = 1;
        while (true) {
            // Compute the zero of the Bessel function of order nu
            const double zero = gsl_sf_bessel_zero_Jnu(nu, zero_index);
            // Calculate the corresponding eigenvalue k_mn
            double k_mn = zero / radius;
            // Break the loop if k_mn exceeds k_max
            if (k_mn > k_max) break;
            // Add k_mn to the list if it is within the specified range
            if (k_min <= k_mn) {
                analytical_eigenvalues.emplace_back(k_mn, nu, zero_index);
                std::cout << "k(" << nu << "," << zero_index << ") = " << k_mn << std::endl;
            }
            ++zero_index;
        }
    }

    // Calculate and print the differences between numerical and analytical eigenvalues
    std::cout << "\nDifferences between numerical and analytical k values:\n";
    for (const auto& [numerical_k, smallest_singular_value] : localMinima) {
        // Find the closest analytical eigenvalue
        auto closest_analytical = *std::ranges::min_element(analytical_eigenvalues,
                                                            [numerical_k](const auto& a, const auto& b) {
                                                                return std::abs(std::get<0>(a) - numerical_k) < std::abs(std::get<0>(b) - numerical_k);
                                                            });

        double closest_analytical_k = std::get<0>(closest_analytical);
        int nu = std::get<1>(closest_analytical);
        int zero_index = std::get<2>(closest_analytical);

        double absolute_difference = std::abs(numerical_k - closest_analytical_k);
        double relative_difference = absolute_difference / closest_analytical_k;

        std::cout << "Numerical k: " << numerical_k
                  << ", Analytical k: " << closest_analytical_k
                  << " (k: " << nu << ", l: " << zero_index << ")"
                  << ", Smallest Singular Value: " << smallest_singular_value
                  << ", Absolute Difference: " << absolute_difference
                  << ", Relative Difference: " << relative_difference << std::endl;
    }


    const auto fig = figure(true);
    fig->size(1800, 1200);
    const auto ax = fig->add_subplot(1, 1, 1);
    //const auto ax2 = fig->add_subplot(2, 1, 2);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, -0.07, 0.5);
    //solver.plotSmallestSingularValues(ax2, -0.002, 0.005);

    // Plot analytical solutions as purple diamonds
    const std::vector<double> analytical_x(analytical_eigenvalues.size(), 0);
    std::vector<double> analytical_k_values;
    analytical_k_values.reserve(analytical_eigenvalues.size());
    for (const auto& [k_mn, nu, zero_index] : analytical_eigenvalues) {
        analytical_k_values.push_back(k_mn);
    }

    const auto l = ax->scatter(analytical_k_values, analytical_x);
    l->display_name("Analytical");
    l->marker_style(line_spec::marker_style::diamond);
    l->marker_size(15);
    l->marker_color("purple");

    //const auto l2 = ax2->scatter(analytical_k_values, analytical_x);
    //l2->display_name("Analytical");
    //l2->marker_style(line_spec::marker_style::diamond);
    //l2->marker_size(15);
    //l2->marker_color("purple");

    // Add vertical dashed asymptotes at analytical k values
    for (const auto& [k_mn, nu, l] : analytical_eigenvalues) {
        if (nu % 2 == 0) {
            ax->plot({k_mn, k_mn}, {ax->y_axis().limits().at(0), ax->y_axis().limits().at(1)}, "k--");
            // Add label for (nu,l) parallel to the dashed line
            double offset_y;
            if ((nu / 2) % 2 == 0) {
                offset_y = ax->y_axis().limits().at(1) * 0.95; // Slightly below the top of the y-axis
            } else {
                offset_y = ax->y_axis().limits().at(1) * 0.75; // Slightly more below the top of the y-axis
            }
            auto txt = ax->text(k_mn, offset_y, "(" + std::to_string(nu) + "," + std::to_string(l) + ")");
            txt->font_size(10);
            txt->color("black");
        }
    }

    // Add labels for (k,l) pairs below the diamonds in ax2
    for (const auto& [k_mn, nu, zero_index] : analytical_eigenvalues) {
        double offset_y = (nu % 2 == 0) ? 0.02 : -0.02; // Alternate offset for even and odd values
        if (nu == 0) {
            offset_y *= 2;
        }
        auto txt = ax->text(k_mn, offset_y, "(" + std::to_string(nu) + "," + std::to_string(zero_index) + ")");
        txt->font_size(10);
        if (nu == 0) {
            txt->color("red");
        } else {
            txt->color("black");
        }
    }

    // Example usage: print Fredholm matrix and derivatives for a specific analytical eigenvalue
    //int selected_m = 1;
    //int selected_n = 1;
    //Debugging::printFredholmMatrixAndDerivativesForEigenvalue(circleBoundary, kernelStrategy, analytical_eigenvalues, selected_m, selected_n, scalingFactor, false, Debugging::PRINT_TYPES::FREDHOLM);

    matplot::legend(ax, false);
    show();
}

void executeBIMCircleDeterminant(const bool showCircle = true) {
    using namespace BIM;
    double radius = 1.0;
    Point center(0.0, 0.0);
    auto circleBoundary = std::make_shared<Circle>(center, radius);

    if (showCircle) {
        // Plot the boundary using matplot
        const auto fig = figure(true);
        const auto ax = fig->add_axes();
        circleBoundary->plot(ax, 50, true, false);
        show();
    }

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(circleBoundary);
    // KRangeSolver parameters
    constexpr double k_min = 40.0;
    constexpr double k_max = 50.0;
    constexpr int SIZE_K = 111000;
    constexpr int scalingFactor = 15;

    // Create and use the KRangeSolver
    KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, circleBoundary, kernelStrategy);
    solver.computeDeterminants(false);

    // Analytical eigenvalues using GSL for Bessel function zeros
    auto analytical_eigenvalues = AnalysisTools::computeCircleAnalyticalEigenvalues(k_min, k_max, radius);

    const auto fig = figure(true);
    fig->size(1800, 1200);
    const auto ax = fig->add_subplot(3, 1, 0);
    const auto ax2 = fig->add_subplot(3, 1, 1);
    const auto ax3 = fig->add_subplot(3, 1, 2);
    auto _ = solver.plotDeterminants(ax, -1.0, 1.0, false);
    auto ____ = solver.plotDeterminants(ax2, -0.005, 0.05, false); // NOLINT(*-reserved-identifier)
    auto v = solver.plotMagnitudeDeterminants(ax3, -0.005,
        1.0, false);
    auto r = findLocalMinima(v);
    std::cout << "Local minima:\n";
    for (const auto& [k, magnitude] : r) {
        std::cout << "k: " << k << ", magnitude: " << magnitude << '\n';
    }

    // Calculate and print the differences between numerical and analytical eigenvalues
    std::cout << "\nDifferences between numerical and analytical k values:\n";
    std::cout << std::fixed << std::setprecision(16);
    for (const auto& [numerical_k, magnitude] : r) {
        // Find the closest analytical eigenvalue
        auto closest_analytical = *std::ranges::min_element(analytical_eigenvalues,
                                                            [numerical_k](const auto& a, const auto& b) {
                                                                return std::abs(std::get<0>(a) - numerical_k) < std::abs(std::get<0>(b) - numerical_k);
                                                            });

        double closest_analytical_k = std::get<0>(closest_analytical);
        int nu = std::get<1>(closest_analytical);
        int zero_index = std::get<2>(closest_analytical);

        double absolute_difference = std::abs(numerical_k - closest_analytical_k);
        double relative_difference = absolute_difference / closest_analytical_k;

        std::cout << "Numerical k: " << numerical_k
                  << ", Analytical k: " << closest_analytical_k
                  << " (k: " << nu << ", l: " << zero_index << ")"
                  << ", Smallest Determinant Value: " << magnitude
                  << ", Absolute Difference: " << absolute_difference
                  << ", Relative Difference: " << relative_difference << std::endl;
    }

    std::vector<double> analytical_values;
    for (const auto& val : analytical_eigenvalues) {
        if (auto it = std::ranges::find_if(analytical_values, [&val](double x) {
            return std::abs(x - std::get<0>(val)) < 1e-8;
        }); it == analytical_values.end()) {
            analytical_values.push_back(std::get<0>(val));
            std::cout << "Analytical value: " << std::get<0>(val) << std::endl;
        }
    }

    // Plot analytical solutions as purple diamonds
    std::vector<double> analytical_x(analytical_eigenvalues.size(), 0);
    auto l = ax->scatter(analytical_values, analytical_x);
    l->display_name("Analytical");
    l->marker_style(line_spec::marker_style::diamond);
    l->marker_size(5);
    l->marker_color("purple");

    auto l2 = ax2->scatter(analytical_values, analytical_x);
    l2->display_name("Analytical");
    l2->marker_style(line_spec::marker_style::diamond);
    l2->marker_size(5);
    l2->marker_color("purple");

    auto l3 = ax3->scatter(analytical_values, analytical_x);
    l3->display_name("Analytical");
    l3->marker_style(line_spec::marker_style::diamond);
    l3->marker_size(5);
    l3->marker_color("purple");

    show(fig);
}

void executeBIMRectangleDeterminants(const bool showRectangle = true) {   // Standard BIM - OK, but precision problems
    using namespace BIM;

    // Define the rectangle in the first quadrant
    double width = 2.0; // Preverba narejena za width = 2.0
    double height = 1.0; // Preverba narejena za 1.0
    Point bottomLeft(0.0, 0.0); // Preverba narejena za Point bottomLeft(0.0)
    auto rectangleBoundary = std::make_shared<Rectangle>(bottomLeft, width, height);

    if (showRectangle) {
        // Plot the boundary using matplot
        const auto fig = figure(true);
        const auto ax = fig->add_axes();
        rectangleBoundary->plot(ax, 20, true, false);
        show();
    }

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(rectangleBoundary);
    // KRangeSolver parameters
    constexpr double k_min = 9.0; // Preverba narejena za k_min = 0.5
    constexpr double k_max = 10.0; // Preverba narejena za k_max = 10.0
    constexpr int SIZE_K = 20000; // Preverba narejena za SIZE_K = 2000
    constexpr int scalingFactor = 10; // Preverba narejena za scalingFactor = 10

    // Create and use the KRangeSolver
    KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, rectangleBoundary, kernelStrategy);
    solver.computeDeterminants(false);

    // Analytical eigenvalues
    std::vector<double> analytical_eigenvalues;
    std::cout << "Analytical eigenvalues for the rectangle:" << std::endl;
    for (int m = 1; m <= 100; ++m) {
        for (int n = 1; n <= 100; ++n) {
            if (const double k_mn_squared = std::pow(m * M_PI / width, 2) + std::pow(n * M_PI / height, 2); k_min * k_min <= k_mn_squared && k_mn_squared <= k_max * k_max) {
                double k_mn = sqrt(k_mn_squared);
                analytical_eigenvalues.push_back(k_mn);
                std::cout << "k(" << m << "," << n << ") = " << k_mn << std::endl;
            }
        }
    }

    // Plot determinants and get roots
    // Plotting
    const auto fig = figure(true);
    fig->size(1800, 1200);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);
    auto [real_roots, imag_roots] = solver.plotDeterminants(ax, -10.0, 10.0, false);
    auto _ = solver.plotDeterminants(ax2, -0.1, 0.5, false);

    // Plot analytical solutions as purple boxes
    std::vector<double> analytical_x(analytical_eigenvalues.size(), 0);
    auto l = ax->scatter(analytical_eigenvalues, analytical_x);
    l->display_name("Analytical");
    l->marker_style(line_spec::marker_style::diamond);
    l->marker_size(15);
    l->marker_color("purple");

    auto l2 = ax2->scatter(analytical_eigenvalues, analytical_x);
    l2->display_name("Analytical");
    l2->marker_style(line_spec::marker_style::diamond);
    l2->marker_size(15);
    l2->marker_color("purple");
    show(fig);
}

void executeBIMRectangleSVD(const bool showRectangle = true, const int scalingFactor = 10) {
    using namespace BIM;

    // Define the rectangle in the first quadrant
    double width = M_PI / 3.0;
    double height = 1.0;
    Point bottomLeft(0.0, 0.0);
    auto rectangleBoundary = std::make_shared<Rectangle>(bottomLeft, width, height);

    if (showRectangle) {
        // Plot the boundary using matplot
        const auto fig = figure(true);
        const auto ax = fig->add_axes();
        rectangleBoundary->plot(ax, 50, true, false);
        show();
    }

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(rectangleBoundary);
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 20.0; // k=100 also 60 also 50
    constexpr int SIZE_K = 200000; // 26000 also 15000 also 60000

    // Create and use the KRangeSolver
    KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, rectangleBoundary, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);
    auto localMinima = solver.findLocalMinima(0.01);
    auto secondLocalMinima = solver.findLocalMinima(2, 0.005);
    auto thirdLocalMinima = solver.findLocalMinima(3, 0.005);
    auto fourthLocalMinima = solver.findLocalMinima(4, 0.005);

    const auto& newlocalMinima = localMinima;
    const auto& newSecondlocalMinima = secondLocalMinima;
    const auto& newThirdlocalMinima = thirdLocalMinima;
    const auto& newFourthlocalMinima = fourthLocalMinima;

    // Analytical eigenvalues
    std::vector<std::tuple<double, int, int>> analytical_eigenvalues = AnalysisTools::computeRectangleAnalyticalEigenvalues(k_min, k_max, width, height);

    // Print differences for the first, second, and third local minima
    AnalysisTools::printDifferences(newlocalMinima, analytical_eigenvalues, "first local minima", false, false, 0.05);
    AnalysisTools::printDifferences(newSecondlocalMinima, analytical_eigenvalues, "second local minima", false, false, 0.05);
    AnalysisTools::printDifferences(newThirdlocalMinima, analytical_eigenvalues, "third local minima", false, false, 0.05);
    AnalysisTools::printDifferences(newFourthlocalMinima, analytical_eigenvalues, "fourth local minima", false, false, 0.05);

    // Compare numerical k values between the first, second, and third local minima
    std::cout << "\nComparing numerical k values between first, second, and third local minima:\n";

    // Compare first vs second, first vs third, and second vs third within tolerance
    AnalysisTools::compareMinima(newlocalMinima, newSecondlocalMinima, "first local minima", "second local minima", false, true, 0.05);
    AnalysisTools::compareMinima(newSecondlocalMinima, newThirdlocalMinima, "first local minima", "third local minima", false, true, 0.05);
    AnalysisTools::compareMinima(newThirdlocalMinima, newFourthlocalMinima, "second local minima", "third local minima", false, true, 0.05);
    AnalysisTools::compareMinima(newFourthlocalMinima, newlocalMinima, "third local minima", "fourth local minima", false, true, 0.05);

    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);
    // Plot SVD and get roots
    AnalysisTools::plotSVDResults(ax, ax2, solver, analytical_eigenvalues, -0.5, 1.0, -0.0, 0.01);

    auto degeneraciesResult = solver.calculateDegeneracies(0.005, 0.005);
    const auto numerical_results = std::get<0>(degeneraciesResult);

    const auto figComparison = figure(true);
    figComparison->size(800, 800);
    auto axComparison = figComparison->add_axes();
    axComparison->xlabel("k");
    axComparison->ylabel("Number of Eigenvalues");
    axComparison->title("Comparing theoretical vs. numerical eigenvalue counting functions");
    matplot::legend(axComparison, true);
    AnalysisTools::plotTheoreticalEigenvalues(axComparison, AnalysisTools::computeRectangleAnalyticalEigenvalues(k_min, k_max, width, height), k_min, k_max);
    AnalysisTools::plotNumericalEigenvalues(axComparison, numerical_results, k_min, k_max);

    const auto figWeyl2 = figure(true);
    figWeyl2->size(800, 800);
    auto axWeyl2 = figWeyl2->add_axes();
    AnalysisTools::compareEigenvalueCountsWithWeylDefault(axWeyl2, numerical_results, rectangleBoundary, rectangleBoundary->calculateArcLength(), 0.25);

    // The other plots concerning level spacings and oscillating part of the number of states
    auto fig_spacingsOscillatoty = figure(true);
    fig_spacingsOscillatoty->size(1500, 1000);
    auto ax_upper = fig_spacingsOscillatoty->add_subplot(2, 2, 0);
    auto ax_lower = fig_spacingsOscillatoty->add_subplot(2, 2, 1);
    auto ax_oscillatory = fig_spacingsOscillatoty->add_subplot(2, 2, 2);
    auto ax_cdf = fig_spacingsOscillatoty->add_subplot(2, 2, 3);

    auto fig_statistics = figure(true);
    fig_statistics->size(1500, 1000);
    auto ax_rigidity = fig_statistics->add_subplot(1, 2, 0);
    auto ax_number_variance = fig_statistics->add_subplot(1, 2, 1);

    auto extracted_numerical_results = std::get<0>(degeneraciesResult);
    auto res = AnalysisTools::plotUnfoldedEnergiesAndAnalyses(extracted_numerical_results, rectangleBoundary, ax_upper, ax_lower, ax_oscillatory, ax_cdf, ax_rigidity, ax_number_variance, 0.25);

    show(fig);
    save(fig, "Rectangle_SVD_to_20.png");
    show(figComparison);
    save(fig, "Rectangle_Comparison_to_20.png");
    show(figWeyl2);
    save(fig, "Rectangle_Weyl_to_20.png");
    show(fig_spacingsOscillatoty);
    save(fig, "Rectangle_spacings_oscillatory_to_20.png");
    show(fig_statistics);
}

void plotBVariationOfNumericalVsAnalyticalKDifferences_RECTANGLE() {
    using namespace BIM;

    // Define the rectangle in the first quadrant
    double width = 2.0;
    double height = 1.0;
    Point bottomLeft(0.0, 0.0);
    auto rectangleBoundary = std::make_shared<Rectangle>(bottomLeft, width, height);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(rectangleBoundary);
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 50.0;
    // ReSharper disable once CppTooWideScope
    constexpr int SIZE_K = 10000;

    // Analytical eigenvalues
    std::vector<std::tuple<double, int, int>> analytical_eigenvalues;
    std::cout << "Analytical eigenvalues for the rectangle:" << std::endl;
    for (int m = 1; m <= 10000; ++m) {
        for (int n = 1; n <= 10000; ++n) {
            if (const double k_mn_squared = std::pow(m * M_PI / width, 2) + std::pow(n * M_PI / height, 2); k_min * k_min <= k_mn_squared && k_mn_squared <= k_max * k_max) {
                double k_mn = sqrt(k_mn_squared);
                analytical_eigenvalues.emplace_back(k_mn, m, n);
                std::cout << "k(" << m << "," << n << ") = " << k_mn << std::endl;
            }
        }
    }

    const auto figWeyl = figure(true);
    figWeyl->size(800, 800);
    auto axWeyl = figWeyl->add_axes();
    std::vector<matplot::color> colors = {matplot::color::red, matplot::color::green, matplot::color::blue, matplot::color::yellow, matplot::color::cyan, matplot::color::magenta, matplot::color::black};

    size_t color_index = 0; // Initialize color index
    size_t marker_index = 0; // Initialize marker index

    for (const std::vector<int> b_values = {6, 7, 8, 9, 10, 12, 13}; const auto b_value: b_values) {
        // Create and use the KRangeSolver
        KRangeSolver solver(k_min, k_max, SIZE_K, b_value, rectangleBoundary, kernelStrategy);
        solver.computeSingularValueDecomposition(Eigen::ComputeThinU);
        auto localMinima = solver.findLocalMinima();
        auto degeneraciesResult = solver.calculateDegeneracies(1e-2, 0.01);
        //KRangeSolver::printDegeneracies(std::get<0>(degeneraciesResult));

        // Get only the numerical k values
        std::vector<double> numerical_k_vector;
        for (const auto& [numerical_k, _] : std::get<0>(degeneraciesResult)) {
            numerical_k_vector.emplace_back(numerical_k);
        }

        // Get the analytical k values
        std::vector<double> analytical_k_vector;
        analytical_k_vector.reserve(analytical_eigenvalues.size());
        for (const auto& [k_mn, m, n] : analytical_eigenvalues) {
            analytical_k_vector.emplace_back(k_mn);
        }

        // Sort the analytical k vector
        std::ranges::sort(analytical_k_vector);

        // Remove duplicates within the tolerance of sqrt(machine_epsilon)
        analytical_k_vector.erase(
            std::ranges::unique(analytical_k_vector,
                                [](const double l, const double r) {
                                    return std::abs(l - r) < std::sqrt(std::numeric_limits<double>::epsilon());
                                }).begin(),
            analytical_k_vector.end()
        );
        auto result = AnalysisTools::calculateAndPlotKDifferencesUsingWeyl(axWeyl, numerical_k_vector, analytical_k_vector, rectangleBoundary->calculateArcLength(), rectangleBoundary, false);
        auto sct = std::get<0>(result);
        sct->color(colors[color_index % colors.size()]); // Assign color from the colors vector
        sct->marker_size(15.0);
        // Assign marker style using a switch statement
        switch (marker_index % 7) { // Assuming there are 7 markers
            case 0: sct->marker(matplot::line_spec::marker_style::asterisk); break; // NOLINT(*-branch-clone)
            case 1: sct->marker(matplot::line_spec::marker_style::circle); break;
            case 2: sct->marker(matplot::line_spec::marker_style::cross); break;
            case 3: sct->marker(matplot::line_spec::marker_style::hexagram); break;
            case 4: sct->marker(matplot::line_spec::marker_style::square); break;
            case 5: sct->marker(matplot::line_spec::marker_style::downward_pointing_triangle); break;
            case 6: sct->marker(matplot::line_spec::marker_style::plus_sign); break;
            default: ;
        }
        ++marker_index; // Increment marker index
        ++color_index; // Increment color index
    }
    std::vector<std::string> names;
    for (const std::vector<int> b_values = {6, 7, 8, 9, 10, 12, 13}; auto b_value: b_values) {
        names.push_back("b=" + std::to_string(b_value)); // NOLINT(*-inefficient-vector-operation)
    }
    axWeyl->legend(names);
    matplot::legend(axWeyl, on);
    show(figWeyl);
}

void plotBVariationOfNumericalVsAnalyticalEnergyDifferences_RECTANGLE() {
    using namespace BIM;

    // Define the rectangle in the first quadrant
    double width = 2.0;
    double height = 1.0;
    Point bottomLeft(0.0, 0.0);
    auto rectangleBoundary = std::make_shared<Rectangle>(bottomLeft, width, height);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(rectangleBoundary);
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 20.0;
    // ReSharper disable once CppTooWideScope
    constexpr int SIZE_K = 10000;

    // Analytical eigenvalues
    std::vector<std::tuple<double, int, int>> analytical_eigenvalues;
    std::cout << "Analytical eigenvalues for the rectangle:" << std::endl;
    for (int m = 1; m <= 10000; ++m) {
        for (int n = 1; n <= 10000; ++n) {
            if (const double k_mn_squared = std::pow(m * M_PI / width, 2) + std::pow(n * M_PI / height, 2); k_min * k_min <= k_mn_squared && k_mn_squared <= k_max * k_max) {
                double k_mn = sqrt(k_mn_squared);
                analytical_eigenvalues.emplace_back(k_mn, m, n);
                std::cout << "k(" << m << "," << n << ") = " << k_mn << std::endl;
            }
        }
    }

    const auto figWeyl = figure(true);
    figWeyl->size(800, 800);
    auto axWeyl = figWeyl->add_subplot(2, 1, 0);
    std::vector<matplot::color> colors = {matplot::color::red, matplot::color::green, matplot::color::blue, matplot::color::yellow, matplot::color::cyan, matplot::color::magenta, matplot::color::black};

    size_t color_index = 0; // Initialize color index
    size_t marker_index = 0; // Initialize marker index

    // Vector for holding the results of the b variation
    std::vector<std::vector<double>> b_var_results;

    for (const std::vector<int> b_values = {6, 7, 8, 9, 10, 12, 15, 17}; const auto b_value: b_values) {
        // Create and use the KRangeSolver
        KRangeSolver solver(k_min, k_max, SIZE_K, b_value, rectangleBoundary, kernelStrategy);
        solver.computeSingularValueDecomposition(Eigen::ComputeThinU);
        auto localMinima = solver.findLocalMinima();
        std::cout << "Number of local minima is: " << localMinima.size() << std::endl;
        auto degeneraciesResult = solver.calculateDegeneracies(1e-2, 0.01);
        //KRangeSolver::printDegeneracies(std::get<0>(degeneraciesResult));

        // Get only the numerical k values
        std::vector<double> numerical_k_vector;
        for (const auto& [numerical_k, _] : std::get<0>(degeneraciesResult)) {
            numerical_k_vector.emplace_back(numerical_k);
        }

        // Get the analytical k values
        std::vector<double> analytical_k_vector;
        analytical_k_vector.reserve(analytical_eigenvalues.size());
        for (const auto& [k_mn, m, n] : analytical_eigenvalues) {
            analytical_k_vector.emplace_back(k_mn);
        }

        // Sort the analytical k vector
        std::ranges::sort(analytical_k_vector);

        // Remove duplicates within the tolerance of sqrt(machine_epsilon)
        analytical_k_vector.erase(
            std::ranges::unique(analytical_k_vector,
                                [](const double l, const double r) {
                                    return std::abs(l - r) < std::sqrt(std::numeric_limits<double>::epsilon());
                                }).begin(),
            analytical_k_vector.end()
        );
        auto result = AnalysisTools::calculateAndPlotEnergyDifferencesUsingWeyl(axWeyl, numerical_k_vector, analytical_k_vector, rectangleBoundary->calculateArcLength(), rectangleBoundary, true);
        auto sct = std::get<0>(result);
        b_var_results.emplace_back(std::get<1>(result));
        sct->color(colors[color_index % colors.size()]); // Assign color from the colors vector
        if (marker_index > 7) {
            sct->marker_size(30.0);
        } else {
            sct->marker_size(15.0);
        }
        // Assign marker style using a switch statement
        switch (marker_index % 7) { // Assuming there are 7 markers
            case 0: sct->marker(matplot::line_spec::marker_style::asterisk); break; // NOLINT(*-branch-clone)
            case 1: sct->marker(matplot::line_spec::marker_style::circle); break;
            case 2: sct->marker(matplot::line_spec::marker_style::cross); break;
            case 3: sct->marker(matplot::line_spec::marker_style::hexagram); break;
            case 4: sct->marker(matplot::line_spec::marker_style::square); break;
            case 5: sct->marker(matplot::line_spec::marker_style::downward_pointing_triangle); break;
            case 6: sct->marker(matplot::line_spec::marker_style::plus_sign); break;
            default: ;
        }
        ++marker_index; // Increment marker index
        ++color_index; // Increment color index
    }
    std::vector<std::string> names;
    for (const std::vector<int> b_values = {6, 7, 8, 9, 10, 12, 15, 17}; auto b_value: b_values) {
        names.push_back("b=" + std::to_string(b_value)); // NOLINT(*-inefficient-vector-operation)
    }
    axWeyl->legend(names);
    matplot::legend(axWeyl, on);

    auto axErrors = figWeyl->add_subplot(2, 1, 1);
    AnalysisTools::plotLogLogAverageErrorsWithLinearFit(axErrors, b_var_results, {6, 7, 8, 9, 10, 12, 15, 17});
    show(figWeyl);
}

void plotBVariationOfNumericalVsAnalyticalKDifferences_CIRCLE() {
    using namespace BIM;

    double radius = 1.0;
    Point center(0.0, 0.0);
    auto circleBoundary = std::make_shared<Circle>(center, radius);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(circleBoundary);
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 50.0;
    // ReSharper disable once CppTooWideScope
    constexpr int SIZE_K = 20000;

    auto analytical_eigenvalues = AnalysisTools::computeCircleAnalyticalEigenvalues(k_min, k_max, radius);

    const auto figWeyl = figure(true);
    figWeyl->size(800, 800);
    auto axWeyl = figWeyl->add_axes();
    std::vector<matplot::color> colors = {matplot::color::red, matplot::color::green, matplot::color::blue, matplot::color::yellow, matplot::color::cyan, matplot::color::magenta, matplot::color::black};

    size_t color_index = 0; // Initialize color index
    size_t marker_index = 0; // Initialize marker index

    for (const std::vector<int> b_values = {6, 7, 8, 9, 10, 12, 13}; const auto b_value: b_values) {
        // Create and use the KRangeSolver
        KRangeSolver solver(k_min, k_max, SIZE_K, b_value, circleBoundary, kernelStrategy);
        solver.computeSingularValueDecomposition(Eigen::ComputeThinU);
        auto localMinima = solver.findLocalMinima();
        auto degeneraciesResult = solver.calculateDegeneracies(1e-2, 0.01);
        //KRangeSolver::printDegeneracies(std::get<0>(degeneraciesResult));

        // Get only the numerical k values
        std::vector<double> numerical_k_vector;
        for (const auto& [numerical_k, _] : std::get<0>(degeneraciesResult)) {
            numerical_k_vector.emplace_back(numerical_k);
        }

        // Get the analytical k values
        std::vector<double> analytical_k_vector;
        analytical_k_vector.reserve(analytical_eigenvalues.size());
        for (const auto& [k_mn, m, n] : analytical_eigenvalues) {
            analytical_k_vector.emplace_back(k_mn);
        }

        // Sort the analytical k vector
        std::ranges::sort(analytical_k_vector);

        // Remove duplicates within the tolerance of sqrt(machine_epsilon)
        analytical_k_vector.erase(
            std::ranges::unique(analytical_k_vector,
                                [](const double l, const double r) {
                                    return std::abs(l - r) < std::sqrt(std::numeric_limits<double>::epsilon());
                                }).begin(),
            analytical_k_vector.end()
        );
        auto result = AnalysisTools::calculateAndPlotKDifferencesUsingWeyl(axWeyl, numerical_k_vector, analytical_k_vector, circleBoundary->calculateArcLength(), circleBoundary, false);
        auto sct = std::get<0>(result);
        sct->color(colors[color_index % colors.size()]); // Assign color from the colors vector
        sct->marker_size(15.0);
        // Assign marker style using a switch statement
        switch (marker_index % 7) { // Assuming there are 7 markers
            case 0: sct->marker(matplot::line_spec::marker_style::asterisk); break; // NOLINT(*-branch-clone)
            case 1: sct->marker(matplot::line_spec::marker_style::circle); break;
            case 2: sct->marker(matplot::line_spec::marker_style::cross); break;
            case 3: sct->marker(matplot::line_spec::marker_style::hexagram); break;
            case 4: sct->marker(matplot::line_spec::marker_style::square); break;
            case 5: sct->marker(matplot::line_spec::marker_style::downward_pointing_triangle); break;
            case 6: sct->marker(matplot::line_spec::marker_style::plus_sign); break;
            default: ;
        }
        ++marker_index; // Increment marker index
        ++color_index; // Increment color index
    }
    std::vector<std::string> names;
    for (const std::vector<int> b_values = {6, 7, 8, 9, 10, 12, 13}; auto b_value: b_values) {
        names.push_back("b=" + std::to_string(b_value)); // NOLINT(*-inefficient-vector-operation)
    }
    axWeyl->legend(names);
    matplot::legend(axWeyl, on);
    show(figWeyl);
}

void plotBVariationOfNumericalVsAnalyticalEnergyDifferences_CIRCLE() {
    using namespace BIM;

    double radius = 1.0;
    Point center(0.0, 0.0);
    auto circleBoundary = std::make_shared<Circle>(center, radius);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(circleBoundary);
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 50.0;
    // ReSharper disable once CppTooWideScope
    constexpr int SIZE_K = 20000;

    auto analytical_eigenvalues = AnalysisTools::computeCircleAnalyticalEigenvalues(k_min, k_max, radius);

    const auto figWeyl = figure(true);
    figWeyl->size(800, 800);
    auto axWeyl = figWeyl->add_axes();
    std::vector<matplot::color> colors = {matplot::color::red, matplot::color::green, matplot::color::blue, matplot::color::yellow, matplot::color::cyan, matplot::color::magenta, matplot::color::black};

    size_t color_index = 0; // Initialize color index
    size_t marker_index = 0; // Initialize marker index

    for (const std::vector<int> b_values = {6, 7, 8, 9, 10, 12, 13}; const auto b_value: b_values) {
        // Create and use the KRangeSolver
        KRangeSolver solver(k_min, k_max, SIZE_K, b_value, circleBoundary, kernelStrategy);
        solver.computeSingularValueDecomposition(Eigen::ComputeThinU);
        auto localMinima = solver.findLocalMinima();
        auto degeneraciesResult = solver.calculateDegeneracies(1e-2, 0.01);
        //KRangeSolver::printDegeneracies(std::get<0>(degeneraciesResult));

        // Get only the numerical k values
        std::vector<double> numerical_k_vector;
        for (const auto& [numerical_k, _] : std::get<0>(degeneraciesResult)) {
            numerical_k_vector.emplace_back(numerical_k);
        }

        // Get the analytical k values
        std::vector<double> analytical_k_vector;
        analytical_k_vector.reserve(analytical_eigenvalues.size());
        for (const auto& [k_mn, m, n] : analytical_eigenvalues) {
            analytical_k_vector.emplace_back(k_mn);
        }

        // Sort the analytical k vector
        std::ranges::sort(analytical_k_vector);

        // Remove duplicates within the tolerance of sqrt(machine_epsilon)
        analytical_k_vector.erase(
            std::ranges::unique(analytical_k_vector,
                                [](const double l, const double r) {
                                    return std::abs(l - r) < std::sqrt(std::numeric_limits<double>::epsilon());
                                }).begin(),
            analytical_k_vector.end()
        );
        auto result = AnalysisTools::calculateAndPlotEnergyDifferencesUsingWeyl(axWeyl, numerical_k_vector, analytical_k_vector, circleBoundary->calculateArcLength(), circleBoundary, false);
        auto sct = std::get<0>(result);
        sct->color(colors[color_index % colors.size()]); // Assign color from the colors vector
        sct->marker_size(15.0);
        // Assign marker style using a switch statement
        switch (marker_index % 7) { // Assuming there are 7 markers
            case 0: sct->marker(matplot::line_spec::marker_style::asterisk); break; // NOLINT(*-branch-clone)
            case 1: sct->marker(matplot::line_spec::marker_style::circle); break;
            case 2: sct->marker(matplot::line_spec::marker_style::cross); break;
            case 3: sct->marker(matplot::line_spec::marker_style::hexagram); break;
            case 4: sct->marker(matplot::line_spec::marker_style::square); break;
            case 5: sct->marker(matplot::line_spec::marker_style::downward_pointing_triangle); break;
            case 6: sct->marker(matplot::line_spec::marker_style::plus_sign); break;
            default: ;
        }
        ++marker_index; // Increment marker index
        ++color_index; // Increment color index
    }
    std::vector<std::string> names;
    for (const std::vector<int> b_values = {6, 7, 8, 9, 10, 12, 13}; auto b_value: b_values) {
        names.push_back("b=" + std::to_string(b_value)); // NOLINT(*-inefficient-vector-operation)
    }
    axWeyl->legend(names);
    matplot::legend(axWeyl, on);
    show(figWeyl);
}

// Returns a vector of tupled results: numerical_k, analytical_k, m, n, smallest_singular_value, absolute and relative difference
std::vector<std::tuple<double, double, int, int, double, double, double>> executeBIMRectangle_b_variation(const int scalingFactor) {
    using namespace BIM;

    // Define the rectangle in the first quadrant
    double width = 2.0;
    double height = 1.0;
    Point bottomLeft(0.0, 0.0);
    auto rectangleBoundary = std::make_shared<Rectangle>(bottomLeft, width, height);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(rectangleBoundary);
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 20.0;
    constexpr int SIZE_K = 2000;

    // Create and use the KRangeSolver
    KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, rectangleBoundary, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);
    auto localMinima = solver.findLocalMinima();

    // Analytical eigenvalues
    std::vector<std::tuple<double, int, int>> analytical_eigenvalues;
    std::vector<std::tuple<double, int, int>> problematic_analytical_eigenvalues;
    std::cout << "Analytical eigenvalues for the rectangle:" << std::endl;
    for (int m = 0; m <= 10000; ++m) {
        for (int n = 0; n <= 10000; ++n) {
            if (const double k_mn_squared = std::pow(m * M_PI / width, 2) + std::pow(n * M_PI / height, 2); k_min * k_min <= k_mn_squared && k_mn_squared <= k_max * k_max) {
                double k_mn = sqrt(k_mn_squared);
                if (m == 0 || n == 0) {
                    problematic_analytical_eigenvalues.emplace_back(k_mn, m, n);
                } else {
                    analytical_eigenvalues.emplace_back(k_mn, m, n);
                }
                std::cout << "k(" << m << "," << n << ") = " << k_mn << std::endl;
            }
        }
    }

    // Calculate and print the differences between numerical and analytical eigenvalues
    std::vector<std::tuple<double, double, int, int, double, double, double>> returning_tuples;
    std::cout << "\nDifferences between numerical and analytical k values:\n";
    for (const auto& [numerical_k, smallest_singular_value] : localMinima) {
        // Find the closest analytical eigenvalue
        auto closest_analytical = *std::ranges::min_element(analytical_eigenvalues,
                                                            [numerical_k](const auto& a, const auto& b) {
                                                                return std::abs(std::get<0>(a) - numerical_k) < std::abs(std::get<0>(b) - numerical_k);
                                                            });

        const double closest_analytical_k = std::get<0>(closest_analytical);
        const int m = std::get<1>(closest_analytical);
        const int n = std::get<2>(closest_analytical);

        const double absolute_difference = std::abs(numerical_k - closest_analytical_k);
        const double relative_difference = absolute_difference / closest_analytical_k;

        if (smallest_singular_value < 0.01) {
            std::cout << "Numerical k: " << numerical_k
                  << ", Analytical k: " << closest_analytical_k
                  << " (m: " << m << ", n: " << n << ")"
                  << ", Smallest Singular Value: " << smallest_singular_value
                  << ", Absolute Difference: " << absolute_difference
                  << ", Relative Difference: " << relative_difference << std::endl;

            returning_tuples.emplace_back(numerical_k, closest_analytical_k, m, n, smallest_singular_value, absolute_difference, relative_difference);
        }
    }
    return returning_tuples;
}

void executeBIMQuarterRectangleSVD(const bool showQuarterRectangle = true) {
    using namespace BIM;

    // Define the rectangle in the first quadrant
    double width = 2.0;
    double height = 1.0;
    Point bottomLeft(0.0, 0.0);
    auto rectangleBoundary = std::make_shared<QuarterRectangle>(bottomLeft, width, height);

    if (showQuarterRectangle) {
        // Plot the boundary using matplot
        const auto fig = figure(true);
        const auto ax = fig->add_axes();
        rectangleBoundary->plot(ax, 50, true, false);
        show();
    }

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<XYReflectionSymmetryNNStandard>(rectangleBoundary);
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 30.0;
    constexpr int SIZE_K = 100000;
    constexpr int scalingFactor = 10;

    // Create and use the KRangeSolver
    KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, rectangleBoundary, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Analytical eigenvalues
    std::vector<double> analytical_eigenvalues;
    std::vector<std::tuple<double, int, int>> analytical_eigenvalues_with_info;
    std::vector<std::tuple<double, int, int>> problematic_analytical_eigenvalues;
    std::cout << "Analytical eigenvalues for the rectangle:" << std::endl;
    for (int m = 1; m <= 10000; ++m) {
        for (int n = 1; n <= 10000; ++n) {
            if (const double k_mn_squared = std::pow(m * M_PI / width, 2) + std::pow(n * M_PI / height, 2); k_min * k_min <= k_mn_squared && k_mn_squared <= k_max * k_max) {
                double k_mn = sqrt(k_mn_squared);
                if (m == 0 || n == 0) {
                    problematic_analytical_eigenvalues.emplace_back(k_mn, m, n);
                } else {
                    analytical_eigenvalues.emplace_back(k_mn);
                    analytical_eigenvalues_with_info.emplace_back(k_mn, m, n);

                }
                std::cout << "k(" << m << "," << n << ") = " << k_mn << std::endl;
            }
        }
    }

    auto localMinima = solver.findLocalMinima(0.15);
    // Calculate and print the differences between numerical and analytical eigenvalues
    std::cout << "\nDifferences between numerical and analytical k values:\n";
    for (const auto& [numerical_k, smallest_singular_value] : localMinima) {
        // Find the closest analytical eigenvalue
        auto closest_analytical = *std::ranges::min_element(analytical_eigenvalues_with_info,
                                                            [numerical_k](const auto& a, const auto& b) {
                                                                return std::abs(std::get<0>(a) - numerical_k) < std::abs(std::get<0>(b) - numerical_k);
                                                            });

        double closest_analytical_k = std::get<0>(closest_analytical);
        int m = std::get<1>(closest_analytical);
        int n = std::get<2>(closest_analytical);

        double absolute_difference = std::abs(numerical_k - closest_analytical_k);
        double relative_difference = absolute_difference / closest_analytical_k;

        if (smallest_singular_value < 0.15) {
            std::cout << "Numerical k: " << numerical_k
                  << ", Analytical k: " << closest_analytical_k
                  << " (m: " << m << ", n: " << n << ")"
                  << ", Smallest Singular Value: " << smallest_singular_value
                  << ", Absolute Difference: " << absolute_difference
                  << ", Relative Difference: " << relative_difference << std::endl;
        }
    }

    // Plot SVD and get roots
    // Plotting
    const auto fig = figure(true);
    fig->size(1800, 1200);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, -1.0, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.05, 0.15);

    // Plot analytical solutions as purple diamonds
    std::vector<double> analytical_x(analytical_eigenvalues.size(), 0);
    auto l = ax->scatter(analytical_eigenvalues, analytical_x);
    l->display_name("Analytical");
    l->marker_style(line_spec::marker_style::diamond);
    l->marker_size(15);
    l->marker_color("purple");

    auto l2 = ax2->scatter(analytical_eigenvalues, analytical_x);
    l2->display_name("Analytical");
    l2->marker_style(line_spec::marker_style::diamond);
    l2->marker_size(15);
    l2->marker_color("purple");

    // Plot problematic analytical solutions as asterisks and add text annotations
    std::vector<double> problematic_analytical_x(problematic_analytical_eigenvalues.size(), 0);
    std::vector<double> problematic_k_values;
    problematic_k_values.reserve(problematic_analytical_eigenvalues.size());
    std::map<double, int> k_mn_count;
    for (const auto& [k_mn, m, n] : problematic_analytical_eigenvalues) {
        problematic_k_values.push_back(k_mn);
        k_mn_count[k_mn]++;
    }

    // Add vertical dashed asymptotes at analytical k values
    for (const auto& [k_mn, m, n] : analytical_eigenvalues_with_info) {
        if (m % 2 && n % 2) {
            ax->plot({k_mn, k_mn}, {ax->y_axis().limits().at(0), ax->y_axis().limits().at(1)}, "k--");
            ax2->plot({k_mn, k_mn}, {ax2->y_axis().limits().at(0), ax2->y_axis().limits().at(1)}, "k--");
            // Add label for (nu,l) parallel to the dashed line
            double offset_y;
            double offset_y2;
            offset_y = ax->y_axis().limits().at(1) * 0.95; // Slightly below the top of the y-axis
            offset_y2 = ax2->y_axis().limits().at(1) * 0.95; // Slightly below the top of the y-axis
            auto txt = ax->text(k_mn, offset_y, "(" + std::to_string(m) + "," + std::to_string(n) + ")");
            txt->font_size(10);
            txt->color("black");

            auto txt2 = ax2->text(k_mn, offset_y2, "(" + std::to_string(m) + "," + std::to_string(n) + ")");
            txt2->font_size(10);
            txt2->color("black");
        }
    }

    // Add labels for (m,n) pairs without adding to the legend
    std::map<double, int> label_count;
    for (const auto& [k_mn, m, n] : analytical_eigenvalues_with_info) {
        double offset = 0.01;
        double offset2 = 0.1;
        double offset_y;
        double offset_y2;
        int count = label_count[k_mn];
        if (m % 2 != 0) {
            offset_y = 0.01 +count * offset; // Adjust text position to avoid overlap
            offset_y2 = 0.1 +count * offset2; // Adjust text position to avoid overlap
        } else {
            offset_y = -0.01 -count * offset; // Adjust text position to avoid overlap
            offset_y2 = -0.1 -count * offset2; // Adjust text position to avoid overlap
        }

        // Adjust text position for bottom plot (ax2)
        auto txt2 = ax2->text(k_mn, offset_y, "(" + std::to_string(m) + "," + std::to_string(n) + ")");
        txt2->font_size(10);
        txt2->color("black");

        auto txt1 = ax->text(k_mn, offset_y2, "(" + std::to_string(m) + "," + std::to_string(n) + ")");
        txt1->font_size(10);
        txt1->color("black");

        // Update label count for next iteration
        label_count[k_mn]++;
    }

    matplot::legend(ax, false);
    matplot::legend(ax2, false);
    show();
}

void executeBIMHalfRectangleSVD(const bool showQuarterRectangle = true) {
    using namespace BIM;

    // Define the rectangle in the first quadrant
    double width = 2.0;
    double height = 1.0;
    Point center(0.0, 0.0);
    auto rectangleBoundary = std::make_shared<YSymmetryHalfRectangle>(center, width, height);

    if (showQuarterRectangle) {
        // Plot the boundary using matplot
        const auto fig = figure(true);
        const auto ax = fig->add_axes();
        rectangleBoundary->plot(ax, 50, true, false);
        show();
    }

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<YReflectionSymmetryNStandard>(rectangleBoundary);
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 30.0;
    constexpr int SIZE_K = 100000;
    constexpr int scalingFactor = 10;

    // Create and use the KRangeSolver
    KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, rectangleBoundary, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Analytical eigenvalues
    std::vector<double> analytical_eigenvalues;
    std::vector<std::tuple<double, int, int>> analytical_eigenvalues_with_info;
    std::vector<std::tuple<double, int, int>> problematic_analytical_eigenvalues;
    std::cout << "Analytical eigenvalues for the rectangle:" << std::endl;
    for (int m = 1; m <= 10000; ++m) {
        for (int n = 1; n <= 10000; ++n) {
            if (const double k_mn_squared = std::pow(m * M_PI / width, 2) + std::pow(n * M_PI / height, 2); k_min * k_min <= k_mn_squared && k_mn_squared <= k_max * k_max) {
                double k_mn = sqrt(k_mn_squared);
                if (m == 0 || n == 0) {
                    problematic_analytical_eigenvalues.emplace_back(k_mn, m, n);
                } else {
                    analytical_eigenvalues.emplace_back(k_mn);
                    analytical_eigenvalues_with_info.emplace_back(k_mn, m, n);

                }
                std::cout << "k(" << m << "," << n << ") = " << k_mn << std::endl;
            }
        }
    }

    auto localMinima = solver.findLocalMinima(0.15);
    // Calculate and print the differences between numerical and analytical eigenvalues
    std::cout << "\nDifferences between numerical and analytical k values:\n";
    for (const auto& [numerical_k, smallest_singular_value] : localMinima) {
        // Find the closest analytical eigenvalue
        auto closest_analytical = *std::ranges::min_element(analytical_eigenvalues_with_info,
                                                            [numerical_k](const auto& a, const auto& b) {
                                                                return std::abs(std::get<0>(a) - numerical_k) < std::abs(std::get<0>(b) - numerical_k);
                                                            });

        double closest_analytical_k = std::get<0>(closest_analytical);
        int m = std::get<1>(closest_analytical);
        int n = std::get<2>(closest_analytical);

        double absolute_difference = std::abs(numerical_k - closest_analytical_k);
        double relative_difference = absolute_difference / closest_analytical_k;

        if (smallest_singular_value < 0.15) {
            std::cout << "Numerical k: " << numerical_k
                  << ", Analytical k: " << closest_analytical_k
                  << " (m: " << m << ", n: " << n << ")"
                  << ", Smallest Singular Value: " << smallest_singular_value
                  << ", Absolute Difference: " << absolute_difference
                  << ", Relative Difference: " << relative_difference << std::endl;
        }
    }

    // Plot SVD and get roots
    // Plotting
    const auto fig = figure(true);
    fig->size(1800, 1200);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, -1.0, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.05, 0.15);

    // Plot analytical solutions as purple diamonds
    std::vector<double> analytical_x(analytical_eigenvalues.size(), 0);
    auto l = ax->scatter(analytical_eigenvalues, analytical_x);
    l->display_name("Analytical");
    l->marker_style(line_spec::marker_style::diamond);
    l->marker_size(15);
    l->marker_color("purple");

    auto l2 = ax2->scatter(analytical_eigenvalues, analytical_x);
    l2->display_name("Analytical");
    l2->marker_style(line_spec::marker_style::diamond);
    l2->marker_size(15);
    l2->marker_color("purple");

    // Plot problematic analytical solutions as asterisks and add text annotations
    std::vector<double> problematic_analytical_x(problematic_analytical_eigenvalues.size(), 0);
    std::vector<double> problematic_k_values;
    problematic_k_values.reserve(problematic_analytical_eigenvalues.size());
    std::map<double, int> k_mn_count;
    for (const auto& [k_mn, m, n] : problematic_analytical_eigenvalues) {
        problematic_k_values.push_back(k_mn);
        k_mn_count[k_mn]++;
    }

    // Add vertical dashed asymptotes at analytical k values
    for (const auto& [k_mn, m, n] : analytical_eigenvalues_with_info) {
        if (m % 2 && n % 2) {
            ax->plot({k_mn, k_mn}, {ax->y_axis().limits().at(0), ax->y_axis().limits().at(1)}, "k--");
            ax2->plot({k_mn, k_mn}, {ax2->y_axis().limits().at(0), ax2->y_axis().limits().at(1)}, "k--");
            // Add label for (nu,l) parallel to the dashed line
            double offset_y;
            double offset_y2;
            offset_y = ax->y_axis().limits().at(1) * 0.95; // Slightly below the top of the y-axis
            offset_y2 = ax2->y_axis().limits().at(1) * 0.95; // Slightly below the top of the y-axis
            auto txt = ax->text(k_mn, offset_y, "(" + std::to_string(m) + "," + std::to_string(n) + ")");
            txt->font_size(10);
            txt->color("black");

            auto txt2 = ax2->text(k_mn, offset_y2, "(" + std::to_string(m) + "," + std::to_string(n) + ")");
            txt2->font_size(10);
            txt2->color("black");
        }
    }

    // Add labels for (m,n) pairs without adding to the legend
    std::map<double, int> label_count;
    for (const auto& [k_mn, m, n] : analytical_eigenvalues_with_info) {
        double offset = 0.01;
        double offset2 = 0.1;
        double offset_y;
        double offset_y2;
        int count = label_count[k_mn];
        if (m % 2 != 0) {
            offset_y = 0.01 +count * offset; // Adjust text position to avoid overlap
            offset_y2 = 0.1 +count * offset2; // Adjust text position to avoid overlap
        } else {
            offset_y = -0.01 -count * offset; // Adjust text position to avoid overlap
            offset_y2 = -0.1 -count * offset2; // Adjust text position to avoid overlap
        }

        // Adjust text position for bottom plot (ax2)
        auto txt2 = ax2->text(k_mn, offset_y, "(" + std::to_string(m) + "," + std::to_string(n) + ")");
        txt2->font_size(10);
        txt2->color("black");

        auto txt1 = ax->text(k_mn, offset_y2, "(" + std::to_string(m) + "," + std::to_string(n) + ")");
        txt1->font_size(10);
        txt1->color("black");

        // Update label count for next iteration
        label_count[k_mn]++;
    }

    matplot::legend(ax, false);
    matplot::legend(ax2, false);
    show();
}

void executeBIMRectangleSVDWithBetaVariation(const std::vector<double>& beta_values, const bool showRectangle = true) {
    using namespace BIM;

    // Define the rectangle in the first quadrant
    double width = 2.0;
    double height = 1.0;
    Point bottomLeft(0.0, 0.0);
    auto rectangleBoundary = std::make_shared<Rectangle>(bottomLeft, width, height);

    if (showRectangle) {
        // Plot the boundary using matplot
        const auto fig = figure(true);
        const auto ax = fig->add_axes();
        rectangleBoundary->plot(ax, 20, true, false);
        show();
    }

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(rectangleBoundary);
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 20.0;
    constexpr int SIZE_K = 2000;
    constexpr int scalingFactor = 10;

    // Create and use the KRangeSolver
    KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, rectangleBoundary, kernelStrategy);
    solver.computeSingularValueDecompositionWithBetaVariation(beta_values, 0);

    // Analytical eigenvalues
    std::vector<double> analytical_eigenvalues;
    std::vector<std::tuple<double, int, int>> problematic_analytical_eigenvalues;
    std::cout << "Analytical eigenvalues for the rectangle:" << std::endl;
    for (int m = 0; m <= 10000; ++m) {
        for (int n = 0; n <= 10000; ++n) {
            if (const double k_mn_squared = std::pow(m * M_PI / width, 2) + std::pow(n * M_PI / height, 2); k_min * k_min <= k_mn_squared && k_mn_squared <= k_max * k_max) {
                double k_mn = sqrt(k_mn_squared);
                if (m == 0 || n == 0) {
                    problematic_analytical_eigenvalues.emplace_back(k_mn, m, n);
                } else {
                    analytical_eigenvalues.push_back(k_mn);
                }
                std::cout << "k(" << m << "," << n << ") = " << k_mn << std::endl;
            }
        }
    }

    // Plotting
    const auto fig = figure(true);
    fig->size(1800, 1200);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);
    solver.plotSmallestSingularValuesWithBeta(ax, -1.0, 1.0, beta_values);
    solver.plotSmallestSingularValuesWithBeta(ax2, -0.05, 0.15, beta_values);

    // Plot analytical solutions as purple diamonds
    std::vector<double> analytical_x(analytical_eigenvalues.size(), 0);
    auto l = ax->scatter(analytical_eigenvalues, analytical_x);
    l->display_name("Analytical");
    l->marker_style(line_spec::marker_style::diamond);
    l->marker_size(15);
    l->marker_color("purple");

    auto l2 = ax2->scatter(analytical_eigenvalues, analytical_x);
    l2->display_name("Analytical");
    l2->marker_style(line_spec::marker_style::diamond);
    l2->marker_size(15);
    l2->marker_color("purple");

    // Plot problematic analytical solutions as asterisks and add text annotations
    std::vector<double> problematic_analytical_x(problematic_analytical_eigenvalues.size(), 0);
    std::vector<double> problematic_k_values;
    problematic_k_values.reserve(problematic_analytical_eigenvalues.size());
    std::map<double, int> k_mn_count;
    for (const auto& [k_mn, m, n] : problematic_analytical_eigenvalues) {
        problematic_k_values.push_back(k_mn);
        k_mn_count[k_mn]++;
    }

    auto l3 = ax->scatter(problematic_k_values, problematic_analytical_x);
    l3->display_name("Analytical");
    l3->marker_style(line_spec::marker_style::asterisk);
    l3->marker_size(15);
    l3->marker_color("green");

    auto l4 = ax2->scatter(problematic_k_values, problematic_analytical_x);
    l4->display_name("Analytical");
    l4->marker_style(line_spec::marker_style::asterisk);
    l4->marker_size(15);
    l4->marker_color("green");

    // Add labels for (m,n) pairs without adding to the legend
    std::map<double, int> label_count;
    for (const auto& [k_mn, m, n] : problematic_analytical_eigenvalues) {
        double offset = 0.01;
        int count = label_count[k_mn];
        double offset_y = -0.01 -count * offset; // Adjust text position to avoid overlap

        // Adjust text position for bottom plot (ax2)
        auto txt2 = ax2->text(k_mn, offset_y, "(" + std::to_string(m) + "," + std::to_string(n) + ")");
        txt2->font_size(10);
        txt2->color("black");

        // Update label count for next iteration
        label_count[k_mn]++;
    }

    matplot::legend(ax, false);
    matplot::legend(ax2, false);
    show();
}

void executeMushroomBilliardSVD(const bool plotSketch = false) {
    using namespace BIM;

    // Define the MushroomBilliard
    Point center(0.0, 0.0);
    double radius = 3.0/2.0;
    double height = 1.0;
    double width = 1.0;
    const auto mushroom = std::make_shared<MushroomBilliard>(center, radius, height, width);

    // Plot the boundary using matplot
    if (plotSketch) {
        const auto figMushroom = figure(true);
        const auto axMushroom = figMushroom->add_axes();
        mushroom->plot(axMushroom, 25, true, true);
        show();
    }

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(mushroom);
    // KRangeSolver parameters
    constexpr double k_min = 0.5; // Interesting for k_min = 25.8
    constexpr double k_max = 20.0; // Interesting for k_max = 26.2
    constexpr int SIZE_K = 20000;
    constexpr int scalingFactor = 10;

    // Create and use the KRangeSolver
    KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, mushroom, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Plotting
    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, 0, 0.02);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.01, 1.00);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 2, 0, 0.02);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 2, -0.01, 1.00);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 3, 0, 0.02);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 3, -0.01, 1.00);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 4, 0, 0.02);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 4, -0.01, 1.00);

    hold(ax, on);
    // Add dashed line at y = 0 in both ax and ax2
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax->plot({k_min, k_max}, {0.0, 0.0}, "k--");
    hold(ax2, on);
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax2->plot({k_min, k_max}, {0.0, 0.0}, "k--");

    matplot::legend(ax, true);
    matplot::legend(ax2, true);

    solver.printLocalMinimaOfSingularValues(0.005);

    show();
}

void executeQuarterBunimovichSVD() {
    // Testing Quarter Bunimovich Stadium
    Boundary::Point TopLeftQB(0.0, 0.0);
    double widthQB = 2.0;
    double heightQB = 1.0;
    const auto qbunimovich = std::make_shared<Boundary::QuarterBunimovichStadium>(TopLeftQB, heightQB, widthQB);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::XYReflectionSymmetryNNStandard>(qbunimovich);
    // KRangeSolver parameters
    constexpr double k_min = 0.5; // Interesting for k_min = 25.8
    constexpr double k_max = 30.0; // Interesting for k_max = 26.2
    constexpr int SIZE_K = 50000;
    constexpr int scalingFactor = 10;

    // Create and use the KRangeSolver
    BIM::KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, qbunimovich, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Plotting
    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, 0, 0.02);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 2, 0, 0.02);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 3, 0, 0.02);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 3, -0.01, 0.07);

    hold(ax, on);
    // Add dashed line at y = 0 in both ax and ax2
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax->plot({k_min, k_max}, {0.0, 0.0}, "k--");
    hold(ax2, on);
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax2->plot({k_min, k_max}, {0.0, 0.0}, "k--");

    matplot::legend(ax, true);
    matplot::legend(ax2, true);

    solver.printLocalMinimaOfSingularValues(0.02);
    show();
}

void executeHalfMushroomBilliardSVD_Dirichlet(const bool plotSketch = false) {
    using namespace BIM;

    // Define the MushroomBilliard
    Point center(0.0, 0.0);
    double radius = 3.0/2.0;
    double height = 1.0;
    double width = 1.0;
    const auto mushroom = std::make_shared<HalfMushroomBilliardWithDirichletEdge>(center, radius, height, width);

    // Plot the boundary using matplot
    if (plotSketch) {
        const auto figMushroom = figure(true);
        const auto axMushroom = figMushroom->add_axes();
        mushroom->plot(axMushroom, 25, false, true);
        show();
    }

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(mushroom);
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 20.0;
    constexpr int SIZE_K = 300000;
    constexpr int scalingFactor = 15;

    // Create and use the KRangeSolver
    KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, mushroom, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Plotting
    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, 0, 1e-2);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.01, 1.00);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 2, 0, 1e-2);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 2, -0.01, 1.00);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 3, 0, 1e-2);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 3, -0.01, 1.00);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 4, 0, 1e-2);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 4, -0.01, 1.00);

    hold(ax, on);
    // Add dashed line at y = 0 in both ax and ax2
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax->plot({k_min, k_max}, {0.0, 0.0}, "k--");
    hold(ax2, on);
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax2->plot({k_min, k_max}, {0.0, 0.0}, "k--");

    matplot::legend(ax, true);
    matplot::legend(ax2, true);

    solver.printLocalMinimaOfSingularValues(0.05);
    show();
}

// Comapre the eigenvalues of
void HalfMushroom_ND_ToFullMushroomComparison() {
    // Define the MushroomBilliard
    Boundary::Point center(0.0, 0.0);
    double radius = 3.0/2.0;
    double height = 0.9;
    double width = 0.7;
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 15.0;
    constexpr int SIZE_K = 50000;
    constexpr int scalingFactor = 15;
    // 2 geometries
    const auto mushroomH = std::make_shared<Boundary::HalfMushroomBilliard>(center, radius, height, width);
    const auto mushroomF = std::make_shared<Boundary::MushroomBilliard>(center, radius, height, width);
    // 3 integration kernels
    const auto kernelHN = std::make_shared<KernelIntegrationStrategies::YReflectionSymmetryNStandard>(mushroomH);
    const auto kernelHD = std::make_shared<KernelIntegrationStrategies::YReflectionSymmetryDStandard>(mushroomH);
    const auto kernelF = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(mushroomF);
    // 3 solvers
    BIM::KRangeSolver solverHN(k_min, k_max, SIZE_K, scalingFactor, mushroomH, kernelHN);
    BIM::KRangeSolver solverHD(k_min, k_max, SIZE_K, scalingFactor, mushroomH, kernelHD);
    BIM::KRangeSolver solverF(k_min, k_max, SIZE_K, scalingFactor, mushroomF, kernelF);
    solverHN.computeSingularValueDecomposition(Eigen::ComputeThinU);
    solverHD.computeSingularValueDecomposition(Eigen::ComputeThinU);
    solverF.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Plotting
    const auto fig = figure(true);
    fig->size(1500, 1000);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    auto pl1 = solverHN.plotSmallestSingularValues(ax, 0, 1e-6);
    pl1->color("red");
    auto pl2 = solverHD.plotSmallestSingularValues(ax, 0, 1e-6);
    pl2->color("green");
    auto pl3 = solverF.plotSmallestSingularValues(ax, 0, 0.1);
    pl3->color("blue");
    auto pl4 = solverHN.plotSingularValues(ax2, 2, -0.01, 0.50);
    pl4->color("red");
    auto pl5 = solverHD.plotSingularValues(ax2, 2, -0.01, 0.50);
    pl5->color("green");
    auto pl6 = solverF.plotSingularValues(ax2, 2, -0.01, 0.50);
    pl6->color("blue");

    hold(ax, on);
    // Add dashed line at y = 0 in both ax and ax2
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax->plot({k_min, k_max}, {0.0, 0.0}, "k--");
    hold(ax2, on);
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax2->plot({k_min, k_max}, {0.0, 0.0}, "k--");

    matplot::legend(ax, true);
    matplot::legend(ax2, true);
    show(fig);
}

void executeHalfMushroomBilliardSVD_Neumann(const bool plotSketch = false) {
    using namespace BIM;

    // Define the MushroomBilliard
    Point center(0.0, 0.0);
    double radius = 3.0/2.0;
    double height = 1.0;
    double width = 1.0;
    const auto mushroom = std::make_shared<HalfMushroomBilliard>(center, radius, height, width);

    // Plot the boundary using matplot
    if (plotSketch) {
        const auto figMushroom = figure(true);
        const auto axMushroom = figMushroom->add_axes();
        mushroom->plot(axMushroom, 25, false, true);
        show();
    }

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<YReflectionSymmetryNStandard>(mushroom);
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 20.0;
    constexpr int SIZE_K = 50000;
    constexpr int scalingFactor = 15;

    // Create and use the KRangeSolver
    KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, mushroom, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Plotting
    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, 0, 1e-6);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.01, 1.00);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 2, 0, 1e-6);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 2, -0.01, 1.00);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 3, 0, 1e-6);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 3, -0.01, 1.00);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 4, 0, 1e-6);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 4, -0.01, 1.00);

    hold(ax, on);
    // Add dashed line at y = 0 in both ax and ax2
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax->plot({k_min, k_max}, {0.0, 0.0}, "k--");
    hold(ax2, on);
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax2->plot({k_min, k_max}, {0.0, 0.0}, "k--");

    matplot::legend(ax, true);
    matplot::legend(ax2, true);

    solver.printLocalMinimaOfSingularValues(1e-5);
    show();
}

void executeRightTriangle() {
    constexpr double k1 = 1.0;
    constexpr double k2 = 2.0;
    auto triangle = std::make_shared<Boundary::RightTriangle>(k1, k2);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(triangle);

    // KRangeSolver parameters
    constexpr double k_min = 50.0;
    constexpr double k_max = 52.0;
    constexpr int SIZE_K = 100000;
    constexpr int scalingFactor = 10;

    // Create and use the KRangeSolver
    BIM::KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, triangle, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Plotting
    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, 0, 0.001);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 2, 0, 0.001);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 3, 0, 0.001);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 3, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 4, 0, 0.001);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 4, -0.01, 0.07);

    hold(ax, on);
    // Add dashed line at y = 0 in both ax and ax2
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax->plot({k_min, k_max}, {0.0, 0.0}, "k--");
    hold(ax2, on);
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax2->plot({k_min, k_max}, {0.0, 0.0}, "k--");

    matplot::legend(ax, true);
    matplot::legend(ax2, true);
    solver.printLocalMinimaOfSingularValues(0.01);

    show();
}

void executeEllipse() {
    Boundary::Point center(0.0, 0.0);
    constexpr double a = 2.0;
    constexpr double b = 1.0;
    auto ellipse = std::make_shared<Boundary::Ellipse>(center, a, b);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(ellipse);

    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 20.0;
    constexpr int SIZE_K = 50000;
    constexpr int scalingFactor = 10;

    // Create and use the KRangeSolver
    BIM::KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, ellipse, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Plotting
    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 2, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 3, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 3, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 4, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 4, -0.01, 0.07);

    hold(ax, on);
    // Add dashed line at y = 0 in both ax and ax2
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax->plot({k_min, k_max}, {0.0, 0.0}, "k--");
    hold(ax2, on);
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax2->plot({k_min, k_max}, {0.0, 0.0}, "k--");

    matplot::legend(ax, true);
    matplot::legend(ax2, true);

    solver.printLocalMinimaOfSingularValues(0.05);
    show();
}

void executeProsenBilliard() {
    constexpr double a = 0.2;
    auto prosen = std::make_shared<Boundary::ProsenBilliard>(a);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(prosen);

    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 20.0;
    constexpr int SIZE_K = 200000;
    constexpr int scalingFactor = 15;

    // Create and use the KRangeSolver
    BIM::KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, prosen, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Plotting
    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 2, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 3, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 3, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 4, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 4, -0.01, 0.07);

    hold(ax, on);
    // Add dashed line at y = 0 in both ax and ax2
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax->plot({k_min, k_max}, {0.0, 0.0}, "k--");
    hold(ax2, on);
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax2->plot({k_min, k_max}, {0.0, 0.0}, "k--");

    matplot::legend(ax, true);
    matplot::legend(ax2, true);

    solver.printLocalMinimaOfSingularValues(0.05);
    show();
}

void executeQuarterProsenBilliard() {
    constexpr double a = 0.2;
    auto prosen = std::make_shared<Boundary::QuarterProsenBilliard>(a);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::XYReflectionSymmetryStandardDD>(prosen);

    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 20.0;
    constexpr int SIZE_K = 20000;
    constexpr int scalingFactor = 10;

    // Create and use the KRangeSolver
    BIM::KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, prosen, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Plotting
    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 2, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 3, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 3, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 4, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 4, -0.01, 0.07);

    hold(ax, on);
    // Add dashed line at y = 0 in both ax and ax2
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax->plot({k_min, k_max}, {0.0, 0.0}, "k--");
    hold(ax2, on);
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax2->plot({k_min, k_max}, {0.0, 0.0}, "k--");

    matplot::legend(ax, true);
    matplot::legend(ax2, true);

    solver.printLocalMinimaOfSingularValues(0.05);
    show();
}

void executeC3Curve() {
    constexpr double a = 0.2;
    auto c3 = std::make_shared<Boundary::C3CurveFull>(a);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(c3);

    // KRangeSolver parameters
    constexpr double k_min = 20.0;
    constexpr double k_max = 30.0;
    constexpr int SIZE_K = 80000;
    constexpr int scalingFactor = 12;

    // Create and use the KRangeSolver
    BIM::KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, c3, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Plotting
    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 2, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 3, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 3, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 4, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 4, -0.01, 0.07);

    hold(ax, on);
    // Add dashed line at y = 0 in both ax and ax2
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax->plot({k_min, k_max}, {0.0, 0.0}, "k--");
    hold(ax2, on);
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax2->plot({k_min, k_max}, {0.0, 0.0}, "k--");

    matplot::legend(ax, true);
    matplot::legend(ax2, true);

    solver.printLocalMinimaOfSingularValues(0.05);
    show();
}

void executeC3DesymmetrizedCurve() {
    constexpr double a = 0.2;
    auto c3 = std::make_shared<Boundary::C3DesymmetrizedBilliard>(a);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(c3);

    // KRangeSolver parameters
    constexpr double k_min = 100.0;
    constexpr double k_max = 101.0;
    constexpr int SIZE_K = 100000;
    constexpr int scalingFactor = 15;

    // Create and use the KRangeSolver
    BIM::KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, c3, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Plotting
    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 2, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 3, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 3, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 4, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 4, -0.01, 0.07);

    hold(ax, on);
    // Add dashed line at y = 0 in both ax and ax2
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax->plot({k_min, k_max}, {0.0, 0.0}, "k--");
    hold(ax2, on);
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax2->plot({k_min, k_max}, {0.0, 0.0}, "k--");

    matplot::legend(ax, true);
    matplot::legend(ax2, true);

    solver.printLocalMinimaOfSingularValues(0.05);
    show();
}

void executeRobnikFull() {
    constexpr double eps = 0.9;
    const auto robnik = std::make_shared<Boundary::RobnikBilliard>(eps);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(robnik);

    // KRangeSolver parameters
    constexpr double k_min = 40.0;
    constexpr double k_max = 41.0;
    constexpr int SIZE_K = 200000;
    constexpr int scalingFactor = 15;

    // Create and use the KRangeSolver
    BIM::KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, robnik, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Plotting
    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 2, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 3, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 3, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 4, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 4, -0.01, 0.07);

    hold(ax, on);
    // Add dashed line at y = 0 in both ax and ax2
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax->plot({k_min, k_max}, {0.0, 0.0}, "k--");
    hold(ax2, on);
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax2->plot({k_min, k_max}, {0.0, 0.0}, "k--");

    matplot::legend(ax, true);
    matplot::legend(ax2, true);

    solver.printLocalMinimaOfSingularValues(0.2);
    const std::string nameFile = "Robnik_Billiard_kmin_" + std::to_string(k_min) +  "_kmax_" + std::to_string(k_max) + ".png";
    save(fig, nameFile);
    show();
}

void executePolygonalBilliards() {
    std::vector<Boundary::Point> vertices = {
        {0, 0},   // Bottom-left
        {1, 3},   // Top-mid
        {3, 2},   // Top-right
        {4, 0},   // Bottom-right
        {2, -1}   // Bottom-mid
    };
    const auto polygonBilliard = std::make_shared<Boundary::PolygonBilliard>(vertices);
    const auto figPoly = figure(true);
    const auto axPoly = figPoly->add_axes();
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(polygonBilliard);
    // Calculate the bounding box to set the limits
    double minX, maxX, minY, maxY;
    polygonBilliard->getBoundingBox(minX, maxX, minY, maxY);
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 20.0;
    constexpr int SIZE_K = 250000;
    constexpr int scalingFactor = 15;

    // Create and use the KRangeSolver
    BIM::KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, polygonBilliard, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Plotting
    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 2, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 3, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 3, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 4, -0.1, 1);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 4, -0.01, 0.07);

    hold(ax, on);
    // Add dashed line at y = 0 in both ax and ax2
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax->plot({k_min, k_max}, {0.0, 0.0}, "k--");
    hold(ax2, on);
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax2->plot({k_min, k_max}, {0.0, 0.0}, "k--");

    matplot::legend(ax, true);
    matplot::legend(ax2, true);

    solver.printLocalMinimaOfSingularValues(0.01);
    show();
}

void executeBoundaryTesting() {
    using namespace Boundary;

    // Testing HalfMushroomBilliardNoStemWithDirichletEdge
    double radiusHMNS = 3.0/2.0;
    Point centerHMNS{0.0,0.0};
    const auto hmns = std::make_shared<HalfMushroomBilliardNoStemWithDirichletEdge>(centerHMNS, radiusHMNS);
    std::cout << "Half Mushroom billiard with no stem boundary Info: " << std::endl;
    hmns->printBoundaryInfo(50);
    const auto figHMNS = figure(true);
    figHMNS->size(1000, 1000);
    const auto axHMNS = figHMNS->add_axes();
    double minX, maxX, minY, maxY;
    hmns->getBoundingBox(minX, maxX, minY, maxY);

    double xRange = maxX - minX;
    double yRange = maxY - minY;
    double xCenter = (maxX + minX) / 2.0;
    double yCenter = (maxY + minY) / 2.0;
    double limit = 1.2 * std::max(xRange, yRange);

    axHMNS->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axHMNS->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axHMNS->axes_aspect_ratio(1.0);
    hmns->plot(axHMNS, 30, false, true);
    show();
    save(figHMNS, "Halh_Mushroom_Dirichlet_Edge.png");

    // Testinf the desymmetrized C3 curve
    constexpr auto ac3desymm = 0.2;
    const auto c3_desymm = std::make_shared<C3CurveDesymmetrized>(ac3desymm);
    const auto figc3desymm = figure(true);
    const auto axc3desymm = figc3desymm->add_axes();
    c3_desymm->getBoundingBox(minX, maxX, minY, maxY);
    c3_desymm->printBoundaryInfo(50);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axc3desymm->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axc3desymm->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axc3desymm->axes_aspect_ratio(1.0);
    c3_desymm->plot(axc3desymm, 100, true, true);
    show();
    save(figc3desymm, "C3Desymmetrized.png");


    // The full desymmetrized C3 billiard
    constexpr auto ac3desymmbilliard = 0.2;
    const auto c3_desymmbilliard = std::make_shared<C3DesymmetrizedBilliard>(ac3desymmbilliard);
    const auto figc3desymmbilliard = figure(true);
    figc3desymmbilliard->size(1000, 1000);
    const auto axc3desymmbilliard = figc3desymmbilliard->add_axes();
    c3_desymmbilliard->getBoundingBox(minX, maxX, minY, maxY);
    std::cout << "c3 desymmetrized billiard boundary Info: " << std::endl;
    c3_desymmbilliard->printBoundaryInfo(75);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axc3desymmbilliard->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axc3desymmbilliard->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axc3desymmbilliard->axes_aspect_ratio(1.0);
    c3_desymmbilliard->plot(axc3desymmbilliard, 50, true, true);
    show();

    // Testing C3 curve
    double aC3 = 0.2;
    const auto c3 = std::make_shared<C3CurveFull>(aC3);
    std::cout << "C3 full curve Boundary Info" << std::endl;
    c3->printBoundaryInfo(50);
    const auto figC3 = figure(true);
    figC3->size(1000, 1000);
    const auto axC3 = figC3->add_axes();
    // Calculate the bounding box to set the limits
    c3->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axC3->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axC3->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axC3->axes_aspect_ratio(1.0);
    c3->plot(axC3, 100, true, true);
    show();
    save(figC3, "C3_full.png");

    // Testing Square with parabolic top boundary
    const auto squareWTop = std::make_shared<SquareWithParabolicTop>(Point(0.0, 0.0), 1.0, -1.5);
    const auto figSqr = figure(true);
    figSqr->size(1000, 1000);
    const auto axSqr = figSqr->add_axes();
    // Calculate the bounding box to set the limits
    squareWTop->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axSqr->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axSqr->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axSqr->axes_aspect_ratio(1.0);
    squareWTop->plot(axSqr, 25, true, true);
    std::cout << "Square with parabolic top boundary info" << std::endl;
    squareWTop->printBoundaryInfo(50);
    squareWTop->printSegmentsBoundingBox();
    squareWTop->printCompositeBoundingBox();
    show();
    save(figSqr, "Square_Parabolic_Top.png");

    // Testing HalfMUshroomBilliardFullStem class
    double radiusFS = 3.0/2.0;
    double stemHeightFS = 1.0;
    Point centerFS(0.0, 0.0);
    const auto fs = std::make_shared<HalfMushroomBilliardFullStemWidthWithDirichletEdge>(centerFS, radiusFS, stemHeightFS);
    std::cout << "Half Mushroom billiard with full stem boundary Info: " << std::endl;
    fs->printBoundaryInfo(50);
    const auto figFS = figure(true);
    figFS->size(1000, 1000);
    const auto axFS= figFS->add_axes();
    // Calculate the bounding box to set the limits
    fs->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axFS->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axFS->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axFS->axes_aspect_ratio(1.0);
    fs->plot(axFS, 30, false, true);
    show();
    save(figFS, "Half_Mushroom_Full_Stem.png");

    // Testing Robnik billiard
    double eps = 0.9;
    const auto robnik = std::make_shared<RobnikBilliard>(eps);
    std::cout << "Robnik Boundary Info" << std::endl;
    robnik->printBoundaryInfo(50);
    const auto figRob = figure(true);
    figRob->size(1000, 1000);
    const auto axRob = figRob->add_axes();
    // Calculate the bounding box to set the limits
    robnik->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axRob->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axRob->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axRob->axes_aspect_ratio(1.0);
    robnik->plot(axRob, 50, true, true);
    show();
    save(figRob, "Robnik.png");

    // Testing Circle
    Point center(0.0, 0.0);
    double radius = 1.0;
    const auto circle = std::make_shared<Circle>(center, radius);
    std::cout << "Circle Boundary Info" << std::endl;
    circle->printBoundaryInfo(50);
    const auto figCir = figure(true);
    figCir->size(1000, 1000);
    const auto axCir = figCir->add_axes();
    // Calculate the bounding box to set the limits
    circle->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axCir->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axCir->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axCir->axes_aspect_ratio(1.0);
    circle->plot(axCir, 50, true, true);
    show();
    save(figCir, "Circle.png");

    // Testing Ellipse
    Point centerE(0.0, 0.0);
    double a = 2.0;
    double b = 1.0;
    const auto ellipse = std::make_shared<Boundary::Ellipse>(centerE, a, b);
    std::cout << "Print ellipse info" << std::endl;
    ellipse->printBoundaryInfo(50);
    const auto figEll = figure(true);
    figEll->size(1000, 1000);
    const auto axEll = figEll->add_axes();
    // Calculate the bounding box to set the limits
    ellipse->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axEll->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axEll->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axEll->axes_aspect_ratio(1.0);
    ellipse->plot(axEll, 100, true, true);
    show();
    save(figEll, "Ellipse.png");

    // Testing Half Ellipse
    Point centerHE(0.0, 0.0);
    double aH = 2.0;
    double bH = 1.0;
    const auto ellipseH = std::make_shared<Boundary::SemiEllipse>(centerHE, aH, bH, Point{0.0, 1.0});
    std::cout << "Print ellipse info" << std::endl;
    ellipseH->printBoundaryInfo(50);
    const auto figEllH = figure(true);
    const auto axEllH = figEllH->add_axes();
    // Calculate the bounding box to set the limits
    ellipseH->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axEllH->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axEllH->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axEllH->axes_aspect_ratio(0.5);
    ellipseH->plot(axEllH, 100, true, true);
    show();
    save(figEllH, "HalfEllipse.png");

    // Testing Rectangle
    Point TopLeft(0.0, 0.0);
    double width = 2.0;
    double height = 1.0;
    const auto rectangle = std::make_shared<Rectangle>(TopLeft, width, height);
    std::cout << "Rectangle Boundary Info" << std::endl;
    rectangle->printBoundaryInfo(50);
    const auto figRec = figure(true);
    figRec->size(1000, 1000);
    const auto axRec = figRec->add_axes();
    // Calculate the bounding box to set the limits
    rectangle->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axRec->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axRec->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axRec->axes_aspect_ratio(1.0);
    rectangle->plot(axRec, 25, true, true);
    show();
    rectangle->printSegmentsBoundingBox();
    rectangle->printCompositeBoundingBox();
    save(figRec, "Rectangle.png");

    // Testing Quarter Rectangle
    Point bottomLeftq(0.0, 0.0);
    double widthq = 2.0;
    double heightq = 1.0;
    const auto qrectangle = std::make_shared<QuarterRectangle>(bottomLeftq, widthq, heightq);
    std::cout << "Quarter Rectangle Boundary Info" << std::endl;
    qrectangle->printBoundaryInfo(50);
    const auto qfigRec = figure(true);
    qfigRec->size(1000, 1000);
    const auto qaxRec = qfigRec->add_axes();
    // Calculate the bounding box to set the limits
    qrectangle->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    qaxRec->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    qaxRec->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    qaxRec->axes_aspect_ratio(1.0);
    qrectangle->plot(qaxRec, 25, true, true);
    show();
    qrectangle->printSegmentsBoundingBox();
    qrectangle->printCompositeBoundingBox();
    save(qfigRec, "Quarter_Rectangle.png");


    // Testing Y Half Rectangle
    Point centerYHR(0.0, 0.0);
    double widthYHR = 2.0;
    double heightYHR = 1.0;
    const auto half_rectangle_Y = std::make_shared<YSymmetryHalfRectangle>(centerYHR, widthYHR, heightYHR);
    std::cout << "Rectangle Boundary Info" << std::endl;
    half_rectangle_Y->printBoundaryInfo(50);
    const auto figRecYHR = figure(true);
    figRecYHR->size(1000, 1000);
    const auto axRecYHR = figRecYHR->add_axes();
    // Calculate the bounding box to set the limits
    half_rectangle_Y->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axRecYHR->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axRecYHR->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axRecYHR->axes_aspect_ratio(1.0);
    half_rectangle_Y->plot(axRecYHR, 25, true, true);
    show();
    half_rectangle_Y->printSegmentsBoundingBox();
    half_rectangle_Y->printCompositeBoundingBox();
    save(figRecYHR, "Half_Rectangle_Y.png");

    // Testing X Half Rectangle
    Point centerXHR(0.0, 0.0);
    double widthXHR = 2.0;
    double heightXHR = 1.0;
    const auto half_rectangle_X = std::make_shared<XSymmetryHalfRectangle>(centerXHR, widthXHR, heightXHR);
    std::cout << "Rectangle Boundary Info" << std::endl;
    half_rectangle_X->printBoundaryInfo(50);
    const auto figRecXHR = figure(true);
    figRecXHR->size(1000, 1000);
    const auto axRecXHR = figRecXHR->add_axes();
    // Calculate the bounding box to set the limits
    half_rectangle_X->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axRecXHR->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axRecXHR->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axRecXHR->axes_aspect_ratio(1.0);
    half_rectangle_X->plot(axRecXHR, 25, true, true);
    show();
    half_rectangle_X->printSegmentsBoundingBox();
    half_rectangle_X->printCompositeBoundingBox();
    save(figRecXHR, "Half_Rectangle_X.png");


    // Testing Quarter Prosen Billiard
    double aPH = 0.2;
    const auto prosenH = std::make_shared<QuarterProsenBilliard>(aPH);
    std::cout << "Quarter Prosen Boundary Info" << std::endl;
    prosenH->printBoundaryInfo(20);
    const auto figPH = figure(true);
    figPH->size(1000, 1000);
    const auto axPH = figPH->add_axes();
    // Calculate the bounding box to set the limits
    prosenH->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axPH->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axPH->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axPH->axes_aspect_ratio(1.0);
    prosenH->plot(axPH, 100, true, true);
    show();
    save(figPH, "Quarter_Prosen.png");

    // Testing Prosen Billiard
    double aP = 0.2;
    const auto prosen = std::make_shared<ProsenBilliard>(aP);
    std::cout << "Prosen Boundary Info" << std::endl;
    prosen->printBoundaryInfo(80);
    const auto figP = figure(true);
    figP->size(1000, 1000);
    const auto axP = figP->add_axes();
    // Calculate the bounding box to set the limits
    prosen->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axP->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axP->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axP->axes_aspect_ratio(1.0);
    prosen->plot(axP, 100, true, true);
    show();
    save(figP, "Prosen.png");


    // Testing Mushroom Billiard
    Point CenterM(0.0, 0.0);
    double capM = 2.0;
    double widthM = 1.0;
    double heightM = 1.0;
    const auto mushroom = std::make_shared<MushroomBilliard>(CenterM, capM, heightM, widthM);
    std::cout << "Mushroom Boundary Info" << std::endl;
    mushroom->printBoundaryInfo(50);
    const auto figMush = figure(true);
    figMush->size(1000, 1000);
    const auto axMush = figMush->add_axes();
    // Calculate the bounding box to set the limits
    mushroom->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axMush->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axMush->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axMush->axes_aspect_ratio(1.0);
    mushroom->plot(axMush, 25, true, true);
    show();
    mushroom->printSegmentsBoundingBox();
    mushroom->printCompositeBoundingBox();
    save(figMush, "Mushroom.png");

    // Testing HalfMushroom Billiard
    Point CenterMH(0.0, 0.0);
    double capMH = 2.0;
    double widthMH = 1.0;
    double heightMH = 1.0;
    const auto mushroomH = std::make_shared<HalfMushroomBilliard>(CenterMH, capMH, heightMH, widthMH);
    std::cout << "Half Mushroom Boundary Info" << std::endl;
    mushroomH->printBoundaryInfo(50);
    const auto figMushH = figure(true);
    figMushH->size(1000, 1000);
    const auto axMushH = figMushH->add_axes();
    // Calculate the bounding box to set the limits
    mushroomH->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axMushH->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axMushH->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axMushH->axes_aspect_ratio(1.0);
    mushroomH->plot(axMushH, 25, true, true);
    show();
    mushroomH->printSegmentsBoundingBox();
    mushroomH->printCompositeBoundingBox();
    save(figMushH, "Half_Mushroom.png");

    // Testing Bunimovich Stadium
    Point TopLeftB(0.0, 0.0);
    double widthB = 2.0;
    double heightB = 1.0;
    const auto bunimovich = std::make_shared<BunimovichStadium>(TopLeftB, widthB, heightB);
    std::cout << "Bunimovich full Boundary Info" << std::endl;
    bunimovich->printBoundaryInfo(50);
    const auto figBuni = figure(true);
    figBuni->size(1000, 1000);
    const auto axBuni = figBuni->add_axes();
    // Calculate the bounding box to set the limits
    bunimovich->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axBuni->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axBuni->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axBuni->axes_aspect_ratio(1.0);
    bunimovich->plot(axBuni, 25, true, true);
    show();
    bunimovich->printSegmentsBoundingBox();
    bunimovich->printCompositeBoundingBox();
    save(figBuni, "Bunimovich.png");

    // Testing Quarter Bunimovich Stadium
    Point TopLeftQB(0.0, 0.0);
    double widthQB = 2.0;
    double heightQB = 1.0;
    const auto qbunimovich = std::make_shared<QuarterBunimovichStadium>(TopLeftQB, heightQB, widthQB);
    std::cout << "Quarter Bunimovich full Boundary Info" << std::endl;
    qbunimovich->printBoundaryInfo(50);
    const auto figQBuni = figure(true);
    figQBuni->size(1000, 1000);
    const auto axQBuni = figQBuni->add_axes();
    // Calculate the bounding box to set the limits
    qbunimovich->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axQBuni->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axQBuni->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axQBuni->axes_aspect_ratio(1.0);
    qbunimovich->plot(axQBuni, 25, true, true);
    show();
    qbunimovich->printSegmentsBoundingBox();
    qbunimovich->printCompositeBoundingBox();
    save(figQBuni, "QuarterBunimovich.png");


    // Testing Polygon Billiard
    std::vector<Point> vertices = {
        {0, 0},   // Bottom-left
        {1, 3},   // Top-mid
        {3, 2},   // Top-right
        {4, 0},   // Bottom-right
        {2, -1}   // Bottom-mid
    };
    const auto polygonBilliard = std::make_shared<PolygonBilliard>(vertices);
    const auto figPoly = figure(true);
    figPoly->size(1000, 1000);
    const auto axPoly = figPoly->add_axes();
    // Calculate the bounding box to set the limits
    polygonBilliard->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axPoly->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axPoly->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axPoly->axes_aspect_ratio(1.0);
    polygonBilliard->plot(axPoly, 25, true, true);
    show();
    std::cout << "Polygonal billiard info" << std::endl;
    polygonBilliard->printBoundaryInfo(50);
    polygonBilliard->printSegmentsBoundingBox();
    polygonBilliard->printCompositeBoundingBox();
    save(figPoly, "Polygon.png");

    // Testing Right Triangle
    // One side is k1 = 1.0 and the other side is k2 = 2.0
    const auto triangle = std::make_shared<RightTriangle>(1.0, 2.0);
    const auto figTriangle = figure(true);
    figTriangle->size(1000, 1000);
    const auto axTriangle = figTriangle->add_axes();
    // Calculate the bounding box to set the limits
    triangle->getBoundingBox(minX, maxX, minY, maxY);

    xRange = maxX - minX;
    yRange = maxY - minY;
    xCenter = (maxX + minX) / 2.0;
    yCenter = (maxY + minY) / 2.0;
    limit = 1.2 * std::max(xRange, yRange);

    axTriangle->xlim({xCenter - limit / 2.0, xCenter + limit / 2.0});
    axTriangle->ylim({yCenter - limit / 2.0, yCenter + limit / 2.0});
    axTriangle->axes_aspect_ratio(1.0);
    triangle->plot(axTriangle, 25, true, true);
    std::cout << "Triangle info" << std::endl;
    triangle->printBoundaryInfo(50);
    triangle->printSegmentsBoundingBox();
    triangle->printCompositeBoundingBox();
    show();
    save(figTriangle, "Right_Triangle.png");
    // Testing partial parametric boundary

}

// Function to compare and plot the results
void compareAndPlot(const std::vector<std::vector<std::tuple<double, double, int, int, double, double, double>>>& results_by_scaling) {
    using namespace matplot;
    for (const auto& result : results_by_scaling) {
        std::cout << result.size() << std::endl;
    }

    // Create a new figure
    const auto fig = figure(true);
    fig->title("Rectangle SVD abs. and rel. differences vs. b");
    fig->size(800, 1200);

    const auto ax1 = subplot(1, 2, 0);
    const auto ax2 = subplot(1, 2, 1);

    // Define colors for different scaling factors
    const std::vector<std::string> colors = {"b", "r", "g", "c", "m", "y", "k"};

    // Iterate over each scaling factor's results
    for (size_t i = 0; i < results_by_scaling.size(); ++i) {
        const auto& results = results_by_scaling[i];
        std::vector<double> numerical_ks, analytical_ks, smallest_singular_values, absolute_diffs, relative_diffs;

        for (const auto& [numerical_k, analytical_k, m, n, smallest_singular_value, absolute_diff, relative_diff] : results) {
            numerical_ks.push_back(numerical_k);
            analytical_ks.push_back(analytical_k);
            smallest_singular_values.push_back(smallest_singular_value);
            absolute_diffs.push_back(absolute_diff);
            relative_diffs.push_back(relative_diff);
        }

        std::string color = colors[i % colors.size()];

        // Plot absolute differences
        auto pl1 = plot(ax1, numerical_ks, absolute_diffs, color);
        hold(ax1, on);

        // Plot relative differences
        plot(ax2, numerical_ks, relative_diffs, color);
        hold(ax2, on);
    }

    xlabel(ax1, "Numerical k");
    ylabel(ax1, "Absolute Difference");
    title(ax1, "Absolute Differences");
    ylim(ax1, {0.008, 1e-6});
    const auto l1 = matplot::legend(ax1, {"b=5", "b=7", "b=7", "b=8", "b=9", "b=10", "b=15"});
    l1->location(matplot::legend::general_alignment::bottomright);

    xlabel(ax2, "Numerical k");
    ylabel(ax2, "Relative Difference");
    title(ax2, "Relative Differences");
    ylim(ax2, {0.0008, 1e-7});
    const auto l2 = matplot::legend(ax2, {"b=5", "b=7", "b=7", "b=8", "b=9", "b=10", "b=15"});
    l2->location(matplot::legend::general_alignment::bottomright);

    // Show the plot
    show();
}

void executePlotAndCompare() {
    auto res1 = executeBIMRectangle_b_variation(5);
    auto res2 = executeBIMRectangle_b_variation(6);
    auto res3 = executeBIMRectangle_b_variation(7);
    auto res4 = executeBIMRectangle_b_variation(8);
    auto res5 = executeBIMRectangle_b_variation(9);
    auto res6 = executeBIMRectangle_b_variation(10);
    auto res7 = executeBIMRectangle_b_variation(15);
    std::vector<std::vector<std::tuple<double, double, int, int, double, double, double>>> combined_results;

    combined_results.emplace_back(res1);
    combined_results.emplace_back(res2);
    combined_results.emplace_back(res3);
    combined_results.emplace_back(res4);
    combined_results.emplace_back(res5);
    combined_results.emplace_back(res6);
    combined_results.emplace_back(res7);

    for (const auto& result : combined_results) {
        std::cout << "We are doing the next result in combined results" << std::endl;
        for (const auto& elems : result) {
            std::cout << "Numerical k is: " << std::get<0>(elems) << ", the pair is: " << "(m=" << std::get<2>(elems) << ",n=" << std::get<3>(elems) << ")" << "the asbolute difference is: " << std::get<5>(elems) << ", the realtive difference is " << std::get<6>(elems) << std::endl;
        }
    }
    compareAndPlot(combined_results);
}

void compareSingularValues(const Eigen::MatrixXcd& A, int num_singular_values) {
    // Helper function to convert a complex matrix to its real representation
    auto complexToRealMatrix = [](const Eigen::MatrixXcd& A) {
        const int m = A.rows(); // NOLINT(*-narrowing-conversions)
        const int n = A.cols(); // NOLINT(*-narrowing-conversions)
        Eigen::MatrixXd B(2 * m, 2 * n);

        B.topLeftCorner(m, n) = A.real();
        B.topRightCorner(m, n) = -A.imag();
        B.bottomLeftCorner(m, n) = A.imag();
        B.bottomRightCorner(m, n) = A.real();

        return B;
    };

    // Perform full SVD using Eigen and return the smallest singular values
    auto computeFullSVDEigen = [](const Eigen::MatrixXcd& A, int num_singular_values) {
        const auto start = std::chrono::high_resolution_clock::now();
        const Eigen::BDCSVD<Eigen::MatrixXcd> full_svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> full_svd_duration = end - start;
        std::cout << "Full SVD (Eigen) duration: " << full_svd_duration.count() << " seconds" << std::endl;

        Eigen::VectorXd singular_values = full_svd.singularValues();
        Eigen::VectorXd smallest_singular_values = singular_values.tail(num_singular_values);
        std::sort(smallest_singular_values.data(), smallest_singular_values.data() + num_singular_values);
        return smallest_singular_values;
    };

    // Perform SVD using Spectra and return the smallest singular values
    auto computeSmallestSVDSpectra = [complexToRealMatrix](const Eigen::MatrixXcd& A, int num_singular_values) {
        Eigen::MatrixXd B = complexToRealMatrix(A);

        const auto start = std::chrono::high_resolution_clock::now();
        Spectra::DenseGenMatProd<double> op(B);
        Spectra::GenEigsSolver<Spectra::DenseGenMatProd<double>> eigs(op, num_singular_values, 2 * num_singular_values);
        eigs.init();
        eigs.compute(Spectra::SortRule::SmallestMagn, 100000, 1e-6, Spectra::SortRule::SmallestMagn);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> spectra_duration = end - start;
        std::cout << "Spectra duration: " << spectra_duration.count() << " seconds" << std::endl;

        if (eigs.info() == Spectra::CompInfo::Successful) {
            return eigs.eigenvalues();
        } else {
            std::cerr << "Spectra computation did not converge!" << std::endl;
            return Eigen::VectorXcd();
        }
    };

    // Compute the smallest singular values using full SVD (Eigen)
    const Eigen::VectorXd smallest_singular_values_A = computeFullSVDEigen(A, num_singular_values);
    std::cout << "Smallest singular values from full SVD:\n" << smallest_singular_values_A << std::endl;

    // Compute the smallest singular values using Spectra
    const Eigen::VectorXcd smallest_singular_values_B = computeSmallestSVDSpectra(A, num_singular_values);
    std::cout << "Smallest singular values from Spectra:\n" << smallest_singular_values_B << std::endl;
}

void testTimeSVD() {
    // Define the size of the matrix
    constexpr int size = 10000;
    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr))); // NOLINT(*-msc51-cpp)
    // Create a random complex matrix of the specified size
    const Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Random(size, size);
    // Start timing
    const auto start = std::chrono::high_resolution_clock::now();
    // Compute the SVD using the divide-and-conquer algorithm
    const Eigen::BDCSVD<Eigen::MatrixXcd> svd(matrix, Eigen::ComputeThinU);
    // End timing
    const auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    const std::chrono::duration<double> duration = end - start;
    // Output the time taken to perform the SVD
    std::cout << "Time taken for SVD: " << duration.count() << " seconds" << std::endl;
    // Optionally, output the singular values
    std::cout << "Singular values: " << svd.singularValues().transpose() << std::endl;
}

void plottingTestingRectangle() {
    double width = M_PI / 3.0;
    double height = 1.0;
    Boundary::Point bottomLeft(-width/2, -height/2);
    auto rectangleBoundary = std::make_shared<Boundary::Rectangle>(bottomLeft, width, height);
    const auto fig = matplot::figure(true);
    fig->size(1500, 1000);
    const auto ax1 = fig->add_subplot(2, 4, 0);
    const auto ax2 = fig->add_subplot(2, 4, 1);
    const auto ax3 = fig->add_subplot(2, 4, 2);
    const auto ax4 = fig->add_subplot(2, 4, 3);
    //rectangleBoundary->printBoundaryInfo(50);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(rectangleBoundary);
    const auto analytical_rectangle = AnalysisTools::computeRectangleAnalyticalEigenvalues(0.0, 100, width, height);

    const double val1 = std::get<0>(analytical_rectangle.at(std::ceil(analytical_rectangle.size() - 5 ))); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi1(val1, 30, rectangleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1, 1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500);

    const double val2 = std::get<0>(analytical_rectangle.at(std::ceil(analytical_rectangle.size() - 2 ))); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi2(val2, 30, rectangleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500);

    const double val3 = std::get<0>(analytical_rectangle.at(std::ceil(analytical_rectangle.size() - 10 ))); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi3(val3, 30, rectangleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3, 1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500);

    const double val4 = std::get<0>(analytical_rectangle.at(std::ceil(analytical_rectangle.size() -3  ))); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi4(val4, 30, rectangleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4, 1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500);

    const auto ax1Husimi = fig->add_subplot(2, 4, 4);
    const auto ax2Husimi = fig->add_subplot(2, 4, 5);
    const auto ax3Husimi = fig->add_subplot(2, 4, 6);
    const auto ax4Husimi = fig->add_subplot(2, 4, 7);
    plotting1.plotPoincareHusimiHeatmap(ax1Husimi, 300, 1.0, std::vector<std::string>{"T", "R", "B", "L"});
    plotting2.plotPoincareHusimiHeatmap(ax2Husimi, 300, 1.0, std::vector<std::string>{"T", "R", "B", "L"});
    plotting3.plotPoincareHusimiHeatmap(ax3Husimi, 300, 1.0, std::vector<std::string>{"T", "R", "B", "L"});
    plotting4.plotPoincareHusimiHeatmap(ax4Husimi, 300, 1.0, std::vector<std::string>{"T", "R", "B", "L"});

    auto fig2 = figure(true);
    const auto ax22 = fig2->add_axes();
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotWavefunctionOnBoundary(ax22, 0, 1.0, true, std::vector<std::string>{"T", "R", "B", "L"});
    show(fig);
    show(fig2);
}

void plottingRectangle_Detail() {
    double width = M_PI / 3.0;
    double height = 1.0;
    Boundary::Point bottomLeft(-width/2, -height/2);
    auto rectangleBoundary = std::make_shared<Boundary::Rectangle>(bottomLeft, width, height);
    const auto fig = matplot::figure(true);
    fig->size(2000, 1500);
    const auto ax = fig->add_subplot(2, 3, 0);
    const auto axH = fig->add_subplot(2, 3, 1);
    const auto axRad = fig->add_subplot(2, 3, 2);
    const auto axAng = fig->add_subplot(2, 3, 3);
    const auto axM = fig->add_subplot(2, 3, 4);
    const auto axNormalDerWave = fig->add_subplot(2, 3, 5);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(rectangleBoundary);
    const auto analytical_rectangle = AnalysisTools::computeRectangleAnalyticalEigenvalues(0.0, 35, width, height);  // Normaluse use k_max = 100
    const double val = std::get<0>(analytical_rectangle.at(std::ceil(analytical_rectangle.size()-1))); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi(val, 30, rectangleBoundary, kernelStrategy);
    std::cout << "k value: " << val << std::endl;

    std::cout << "The s values are: " << std::endl;
    for (const auto s_values = bi.getArclengthValues(); auto s: s_values) {
        std::cout << s << " , ";
    }

    std::cout << "The differences between s values are: " << std::endl;
    for (const auto s_diff = bi.getArcDiffLengthValues(); auto s: s_diff) {
        std::cout << s << " , ";
    }

    const EigenfunctionsAndPlotting plotting(bi,  1);
    plotting.plotWavefunctionDensityHeatmap(ax, 500, 500, height/width, 0, true, true); // NOLINT(*-narrowing-conversions)
    plotting.plotPoincareHusimiHeatmap(axH, 500, 1.0, std::vector<std::string>{"T", "R", "B", "L"});
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotRadiallyIntegratedMomentumDensity(axRad, 0, 500, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotAngularIntegratedMomentumDensity(axAng, 0, 500, 1.0);
    plotting.plotMomentumDensityHeatmapPolar(axM, 500, 500, 1.0, 0);
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotNormalDerivativeOfWavefunction(axNormalDerWave, 0, 1.0, true, std::vector<std::string>{"T", "R", "B", "L"});
    std::cout << "Processed all" << std::endl;

    const auto fig2 = figure(true);
    const auto ax22 = fig2->add_subplot(1, 1, 0);
    // ReSharper disable once CppNoDiscardExpression
    //plotting.plotAngularDifferenceOfNormalsVsArcLength(ax22, 1.0, true, std::vector<std::string>{"T", "R", "B", "L"});
    std::cout << "Max value of wavefunction is: " << plotting.calculateLargestWavefunctionValue(1000, 1000, 0) << std::endl;
    std::cout << "Normalization is: " << plotting.calculateWavefunctionNormalization(1000, 1000, 0) << std::endl;
    show(fig);
    show(fig2);
}

void plottingCircle_Detail() {
    using namespace BIM;
    double radius = 1.0;
    Point center(0.0, 0.0);
    auto circleBoundary = std::make_shared<Circle>(center, radius);
    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(circleBoundary);
    const auto fig = matplot::figure(true);
    fig->size(1500, 1000);
    const auto ax1 = fig->add_subplot(2, 3, 0);
    const auto ax2 = fig->add_subplot(2, 3, 1);
    const auto ax3 = fig->add_subplot(2, 3, 2);
    const auto ax4 = fig->add_subplot(2, 3, 3);
    const auto ax5 = fig->add_subplot(2, 3, 4);
    const auto ax6 = fig->add_subplot(2, 3, 5);
    const auto analytical_circle = AnalysisTools::computeCircleAnalyticalEigenvalues(0.5, 100, radius);

    const double val = std::get<0>(analytical_circle.at(std::ceil(analytical_circle.size()/2))); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi(val, 30, circleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting(bi,  1);
    plotting.plotWavefunctionDensityHeatmap(ax1, 500, 500, 1.0, 0, true, true);
    plotting.plotPoincareHusimiHeatmap(ax2, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotRadiallyIntegratedMomentumDensity(ax3, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotAngularIntegratedMomentumDensity(ax4, 0, 400, 1.0);
    plotting.plotMomentumDensityHeatmapPolar(ax5, 500, 500, 1.0, 0);
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotNormalDerivativeOfWavefunction(ax6, 0, 1.0, true);
    std::cout << "val: " << val << std::endl;
    const auto fig2 = figure(true);
    const auto ax22 = fig2->add_axes();
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotWavefunctionOnBoundary(ax22, 0, 1.0, true);
    std::cout << "Max value of wavefunction is: " << plotting.calculateLargestWavefunctionValue(1000, 1000, 0) << std::endl;
    std::cout << "Normalization is: " << plotting.calculateWavefunctionNormalization(1000, 1000, 0) << std::endl;

    show(fig);
    show(fig2);
}

void plottingTestingCircle() {
    using namespace BIM;
    double radius = 1.0;
    Point center(0.0, 0.0);
    auto circleBoundary = std::make_shared<Circle>(center, radius);
    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(circleBoundary);
    const auto fig = matplot::figure(true);
    fig->size(1500, 1000);
    const auto ax1 = fig->add_subplot(6, 3, 0);
    const auto ax2 = fig->add_subplot(6, 3, 1);
    const auto ax3 = fig->add_subplot(6, 3, 2);
    //circleBoundary->printBoundaryInfo(50);

    // Define the kernel strategy
    const auto analytical_rectangle = AnalysisTools::computeCircleAnalyticalEigenvalues(0.5, 100, radius);

    std::cout << std::fixed << std::setprecision(16) << std::endl;
    const double val1 = std::get<0>(analytical_rectangle.at(std::ceil(analytical_rectangle.size() / 2))); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi1(val1, 10, circleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500);
    std::cout << "val1: " << val1 << std::endl;

    const double val2 = std::get<0>(analytical_rectangle.at(400)); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi2(val2, 10, circleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500);
    std::cout << "val2: " << val2 << std::endl;

    const double val3 = std::get<0>(analytical_rectangle.at(100)); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi3(val3, 10, circleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500);
    std::cout << "val3: " << val3 << std::endl;

    const auto axHusimi1 = fig->add_subplot(6, 3, 3);
    const auto axHusimi2 = fig->add_subplot(6, 3, 4);
    const auto axHusimi3 = fig->add_subplot(6, 3, 5);
    plotting1.plotPoincareHusimiHeatmap(axHusimi1, 1000);
    plotting2.plotPoincareHusimiHeatmap(axHusimi2, 1000);
    plotting3.plotPoincareHusimiHeatmap(axHusimi3, 1000);
    const auto axRadInt1 = fig->add_subplot(6, 3, 6);
    const auto axRadInt2 = fig->add_subplot(6, 3, 7);
    const auto axRadInt3 = fig->add_subplot(6, 3, 8);
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotRadiallyIntegratedMomentumDensity(axRadInt1, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotRadiallyIntegratedMomentumDensity(axRadInt2, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotRadiallyIntegratedMomentumDensity(axRadInt3, 0, 400, 1.0);
    const auto axAngInt1 = fig->add_subplot(6, 3, 9);
    const auto axAngInt2 = fig->add_subplot(6, 3, 10);
    const auto axAngInt3 = fig->add_subplot(6, 3, 11);
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotAngularIntegratedMomentumDensity(axAngInt1, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotAngularIntegratedMomentumDensity(axAngInt2, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotAngularIntegratedMomentumDensity(axAngInt3, 0, 400, 1.0);
    const auto axMom1 = fig->add_subplot(6, 3, 12);
    const auto axMom2 = fig->add_subplot(6, 3, 13);
    const auto axMom3 = fig->add_subplot(6, 3, 14);
    plotting1.plotMomentumDensityHeatmapPolar(axMom1, 400, 400, 1.0, 0);
    plotting2.plotMomentumDensityHeatmapPolar(axMom2, 400, 400, 1.0, 0);
    plotting3.plotMomentumDensityHeatmapPolar(axMom3, 400, 400, 1.0, 0);
    const auto ax_angles_diff1 = fig->add_subplot(6, 3, 15);
    const auto ax_angles_diff2 = fig->add_subplot(6, 3, 16);
    const auto ax_angles_diff3 = fig->add_subplot(6, 3, 17);
    // ReSharper disable once CppExpressionWithoutSideEffects
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotAngularDifferenceOfNormalsVsArcLength(ax_angles_diff1, 1.0);
    // ReSharper disable once CppExpressionWithoutSideEffects
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotAngularDifferenceOfNormalsVsArcLength(ax_angles_diff2, 1.0);
    // ReSharper disable once CppExpressionWithoutSideEffects
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotAngularDifferenceOfNormalsVsArcLength(ax_angles_diff3, 1.0);

    show(fig);
}

void plottingTestingCircle_lowE() {
    using namespace BIM;
    double radius = 1.0;
    Point center(0.0, 0.0);
    auto circleBoundary = std::make_shared<Circle>(center, radius);
    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(circleBoundary);
    const auto fig = matplot::figure(true);
    const auto ax1 = fig->add_subplot(4, 3, 0);
    const auto ax2 = fig->add_subplot(4, 3, 1);
    const auto ax3 = fig->add_subplot(4, 3, 2);
    //circleBoundary->printBoundaryInfo(50);

    const auto analytical_circle = AnalysisTools::computeCircleAnalyticalEigenvalues(0.5, 50, radius);

    std::cout << std::fixed << std::setprecision(16) << std::endl;
    const double val1 = std::get<0>(analytical_circle.at(0)); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi1(val1, 10, circleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500);
    std::cout << "val1: " << val1 << std::endl;

    const double val2 = std::get<0>(analytical_circle.at(10)); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi2(val2, 10, circleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500);
    std::cout << "val2: " << val2 << std::endl;

    const double val3 = std::get<0>(analytical_circle.at(23)); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi3(val3, 10, circleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500);
    std::cout << "val3: " << val3 << std::endl;

    const auto axHusimi1 = fig->add_subplot(4, 3, 3);
    const auto axHusimi2 = fig->add_subplot(4, 3, 4);
    const auto axHusimi3 = fig->add_subplot(4, 3, 5);
    plotting1.plotPoincareHusimiHeatmap(axHusimi1, 1000);
    plotting2.plotPoincareHusimiHeatmap(axHusimi2, 1000);
    plotting3.plotPoincareHusimiHeatmap(axHusimi3, 1000);
    const auto axRadInt1 = fig->add_subplot(4, 3, 6);
    const auto axRadInt2 = fig->add_subplot(4, 3, 7);
    const auto axRadInt3 = fig->add_subplot(4, 3, 8);
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotRadiallyIntegratedMomentumDensity(axRadInt1, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotRadiallyIntegratedMomentumDensity(axRadInt2, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotRadiallyIntegratedMomentumDensity(axRadInt3, 0, 400, 1.0);
    const auto axAngInt1 = fig->add_subplot(4, 3, 9);
    const auto axAngInt2 = fig->add_subplot(4, 3, 10);
    const auto axAngInt3 = fig->add_subplot(4, 3, 11);
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotAngularIntegratedMomentumDensity(axAngInt1, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotAngularIntegratedMomentumDensity(axAngInt2, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotAngularIntegratedMomentumDensity(axAngInt3, 0, 400, 1.0);

    std::vector<axes_handle> axes = {ax1, ax2, ax3, axHusimi1, axHusimi2, axHusimi3, axRadInt1, axRadInt2, axRadInt3, axAngInt1, axAngInt2, axAngInt3};
    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 4, 3, 1000);
    show(fig);
}

void plottingTestingCircle_HighE() {
    using namespace BIM;
    double radius = 1.0;
    Point center(0.0, 0.0);
    auto circleBoundary = std::make_shared<Circle>(center, radius);
    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<DefaultKernelIntegrationStrategy>(circleBoundary);
    const auto fig = matplot::figure(true);
    const auto ax1 = fig->add_subplot(4, 3, 0);
    const auto ax2 = fig->add_subplot(4, 3, 1);
    const auto ax3 = fig->add_subplot(4, 3, 2);
    //circleBoundary->printBoundaryInfo(50);

    // Define the kernel strategy
    const auto analytical_circle = AnalysisTools::computeCircleAnalyticalEigenvalues(0.5, 200, radius);

    std::cout << std::fixed << std::setprecision(16) << std::endl;
    const double val1 = std::get<0>(analytical_circle.at(analytical_circle.size()- 1)); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi1(val1, 10, circleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500);
    std::cout << "val1: " << val1 << std::endl;

    const double val2 = std::get<0>(analytical_circle.at(analytical_circle.size()- 2)); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi2(val2, 10, circleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500);
    std::cout << "val2: " << val2 << std::endl;

    const double val3 = std::get<0>(analytical_circle.at(analytical_circle.size() - 3)); // NOLINT(*-narrowing-conversions)
    BIM::BoundaryIntegral bi3(val3, 10, circleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500);
    std::cout << "val3: " << val3 << std::endl;

    const auto axHusimi1 = fig->add_subplot(4, 3, 3);
    const auto axHusimi2 = fig->add_subplot(4, 3, 4);
    const auto axHusimi3 = fig->add_subplot(4, 3, 5);
    plotting1.plotPoincareHusimiHeatmap(axHusimi1, 1000);
    plotting2.plotPoincareHusimiHeatmap(axHusimi2, 1000);
    plotting3.plotPoincareHusimiHeatmap(axHusimi3, 1000);
    const auto axRadInt1 = fig->add_subplot(4, 3, 6);
    const auto axRadInt2 = fig->add_subplot(4, 3, 7);
    const auto axRadInt3 = fig->add_subplot(4, 3, 8);
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotRadiallyIntegratedMomentumDensity(axRadInt1, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotRadiallyIntegratedMomentumDensity(axRadInt2, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotRadiallyIntegratedMomentumDensity(axRadInt3, 0, 400, 1.0);
    const auto axAngInt1 = fig->add_subplot(4, 3, 9);
    const auto axAngInt2 = fig->add_subplot(4, 3, 10);
    const auto axAngInt3 = fig->add_subplot(4, 3, 11);
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotAngularIntegratedMomentumDensity(axAngInt1, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotAngularIntegratedMomentumDensity(axAngInt2, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotAngularIntegratedMomentumDensity(axAngInt3, 0, 400, 1.0);

    std::vector<axes_handle> axes = {ax1, ax2, ax3, axHusimi1, axHusimi2, axHusimi3, axRadInt1, axRadInt2, axRadInt3, axAngInt1, axAngInt2, axAngInt3};
    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 4, 3, 1200);
    show(fig);
}

void plottingTestingQuarterCircle() {
    using namespace BIM;
    double radius = 1.0;
    Point center(0.0, 0.0);
    auto circleBoundary = std::make_shared<QuarterCircle>(center, radius, Point{1.0,0.0});
    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<XYReflectionSymmetryNNStandard>(circleBoundary);
    const auto fig = matplot::figure(true);
    fig->size(1000, 1000);
    const auto ax1 = fig->add_subplot(4, 3, 0);
    const auto ax2 = fig->add_subplot(4, 3, 1);
    const auto ax3 = fig->add_subplot(4, 3, 2);
    //circleBoundary->printBoundaryInfo(50);

    // Define the kernel strategy
    const auto analytical_rectangle = AnalysisTools::computeCircleAnalyticalEigenvalues(0.5, 100, radius);

    BIM::BoundaryIntegral bi1(11.7915879158791590, 10, circleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500);

    BIM::BoundaryIntegral bi2(13.5889208892088931, 10, circleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500);

    BIM::BoundaryIntegral bi3(14.3720487204872054, 10, circleBoundary, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500);

    const auto axHusimi1 = fig->add_subplot(4, 3, 3);
    const auto axHusimi2 = fig->add_subplot(4, 3, 4);
    const auto axHusimi3 = fig->add_subplot(4, 3, 5);
    plotting1.plotPoincareHusimiHeatmap(axHusimi1, 500);
    plotting2.plotPoincareHusimiHeatmap(axHusimi2, 500);
    plotting3.plotPoincareHusimiHeatmap(axHusimi3, 500);
    const auto axRad1 = fig->add_subplot(4, 3, 6);
    const auto axRad2 = fig->add_subplot(4, 3, 7);
    const auto axRad3 = fig->add_subplot(4, 3, 8);
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotRadiallyIntegratedMomentumDensity(axRad1, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotRadiallyIntegratedMomentumDensity(axRad2, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotRadiallyIntegratedMomentumDensity(axRad3, 0, 400, 1.0);
    const auto axAng1 = fig->add_subplot(4, 3, 9);
    const auto axAng2 = fig->add_subplot(4, 3, 10);
    const auto axAng3 = fig->add_subplot(4, 3, 11);
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotAngularIntegratedMomentumDensity(axAng1, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotAngularIntegratedMomentumDensity(axAng2, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotAngularIntegratedMomentumDensity(axAng3, 0, 400, 1.0);
    show(fig);
}

void plottingTestingMushroomFull() {
    Boundary::Point center(0.0, 0.0);
    double radius = 3.0/2.0;
    double height = 1.0;
    double width = 1.0;
    const auto mushroom = std::make_shared<Boundary::MushroomBilliard>(center, radius, height, width);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(mushroom);
    const auto fig = matplot::figure(true);
    fig->size(1200, 1000);
    const auto ax = fig->add_subplot(2, 3, 0);
    const auto axHusimi = fig->add_subplot(2, 3, 1);
    const auto axRadInt = fig->add_subplot(2, 3, 2);
    const auto axAngInt = fig->add_subplot(2, 3, 3);
    const auto axMom = fig->add_subplot(2, 3, 4);
    const auto axNormWaveDer = fig->add_subplot(2, 3, 5);
    const auto fig2 = figure(true);
    fig2->size(1200, 800);
    const auto ax2 = fig2->add_subplot(1, 3, 0);
    const auto ax21 = fig2->add_subplot(1, 3, 1);
    const auto ax22 = fig2->add_subplot(1, 3, 2);

    // Define the kernel strategy
    BIM::BoundaryIntegral bi(6.62267, 30, mushroom, kernelStrategy);

    std::cout << "The s values are: " << std::endl;
    for (const auto s_values = bi.getArclengthValues(); auto s: s_values) {
        std::cout << s << " , ";
    }

    std::cout << "The differences between s values are: " << std::endl;
    for (const auto s_diff = bi.getArcDiffLengthValues(); auto s: s_diff) {
        std::cout << s << " , ";
    }

    const EigenfunctionsAndPlotting plotting(bi,  1);
    plotting.plotWavefunctionDensityHeatmap(ax, 500, 500);
    plotting.plotPoincareHusimiHeatmap(axHusimi, 400, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotRadiallyIntegratedMomentumDensity(axRadInt, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotAngularIntegratedMomentumDensity(axAngInt, 0, 400, 1.0);
    plotting.plotMomentumDensityHeatmapPolar(axMom, 500, 500, 1.0, 0);
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotNormalDerivativeOfWavefunction(axNormWaveDer, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotWavefunctionOnBoundary(ax2, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    plotting.plotCurvature(ax21, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotAngularDifferenceOfNormalsVsArcLength(ax22, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    std::cout << "Max value of wavefunction is: " << plotting.calculateLargestWavefunctionValue(1000, 1000, 0) << std::endl;
    std::cout << "Normalization is: " << plotting.calculateWavefunctionNormalization(1000, 1000, 0) << std::endl;

    show(fig);
    show(fig2);
}

void plottingTestingMushroomFull_Semicircle() {
    Boundary::Point center(0.0, 0.0);
    double radius = 3.0/2.0;
    double height = 1.0;
    double width = 1.0;
    const auto mushroom = std::make_shared<Boundary::MushroomBilliard>(center, radius, height, width);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(mushroom);
    const auto fig = matplot::figure(true);
    fig->size(1200, 1000);
    const auto ax = fig->add_subplot(2, 3, 0);
    const auto axHusimi = fig->add_subplot(2, 3, 1);
    const auto axRadInt = fig->add_subplot(2, 3, 2);
    const auto axAngInt = fig->add_subplot(2, 3, 3);
    const auto axMom = fig->add_subplot(2, 3, 4);

    BIM::BoundaryIntegral bi(19.8069480694806970, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting(bi,  1);
    plotting.plotWavefunctionDensityHeatmap(ax, 500, 500);
    plotting.plotPoincareHusimiHeatmap(axHusimi, 400, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotRadiallyIntegratedMomentumDensity(axRadInt, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotAngularIntegratedMomentumDensity(axAngInt, 0, 400, 1.0);
    plotting.plotMomentumDensityHeatmapPolar(axMom, 500, 500, 1.0, 0);
    show(fig);
}

void plotTestingEllipseFull() {
    Boundary::Point center(0.0, 0.0);
    constexpr double a = 2.0;
    constexpr double b = 1.0;
    auto ellipse = std::make_shared<Boundary::Ellipse>(center, a, b);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(ellipse);
    const auto fig = matplot::figure(true);
    const auto fig2 = figure(true);
    const auto ax21 = fig2->add_subplot(1, 2, 0);
    const auto ax22 = fig2->add_subplot(1, 2, 1);
    const auto ax1 = fig->add_subplot(2, 3, 0);
    const auto ax2 = fig->add_subplot(2, 3, 1);
    const auto ax3 = fig->add_subplot(2, 3, 2);
    const auto ax4 = fig->add_subplot(2, 3, 3);
    const auto ax5 = fig->add_subplot(2, 3, 4);
    const auto ax6 = fig->add_subplot(2, 3, 5);
    const std::vector<axes_handle> axes = {ax1, ax2, ax3, ax4, ax5, ax6};

    // Define the kernel strategy
    BIM::BoundaryIntegral bi(9.9014200710035514, 30, ellipse, kernelStrategy);
    const EigenfunctionsAndPlotting plotting(bi,  1);
    plotting.plotWavefunctionDensityHeatmap(ax1, 500, 500, 0.5, 0, true, true);
    plotting.plotPoincareHusimiHeatmap(ax2, 300);
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotRadiallyIntegratedMomentumDensity(ax3, 0, 500, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotAngularIntegratedMomentumDensity(ax4, 0, 500, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotNormalDerivativeOfWavefunction(ax5, 0, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotMomentumDensityHeatmapPolar(ax6);
    plotting.plotCurvature(ax21, 1.0);
    // ReSharper disable once CppExpressionWithoutSideEffects
    // ReSharper disable once CppNoDiscardExpression
    //plotting.plotAngularDifferenceOfNormalsVsArcLength(ax22, 1.0);
    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 2, 3, 1500);
    std::cout << "Largest value: " << plotting.calculateLargestWavefunctionValue(500, 500, 0) << std::endl;
    std::cout << "Normalization is: " << plotting.calculateWavefunctionNormalization(500, 500, 0) << std::endl;;
    show(fig);
    show(fig2);
}

void plotTestingEllipseHalf() {
    Boundary::Point center(0.0, 0.0);
    constexpr double a = 2.0;
    constexpr double b = 1.0;
    auto ellipseH = std::make_shared<Boundary::SemiEllipse>(center, a, b, Boundary::Point{0.0, 1.0});
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::XReflectionSymmetryNStandard>(ellipseH);
    const auto fig = matplot::figure(true);
    const auto ax = fig->add_subplot(1, 2, 0);
    const auto axH = fig->add_subplot(1, 2, 1);

    // Define the kernel strategy
    BIM::BoundaryIntegral bi(9.9014200710035514, 10, ellipseH, kernelStrategy);
    const EigenfunctionsAndPlotting plotting(bi,  1);
    plotting.plotWavefunctionDensityHeatmap(ax, 500, 500, 0.5);
    plotting.plotPoincareHusimiHeatmap(axH, 300);
    show(fig);
}

void plotTestingProsenFull() {
    constexpr double a = 0.2;
    auto prosen = std::make_shared<Boundary::ProsenBilliard>(a);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(prosen);
    const auto fig = matplot::figure(true);
    const auto fig2 = matplot::figure(true);

    const auto ax1 = fig->add_subplot(4, 4, 0);
    const auto ax2 = fig->add_subplot(4, 4, 1);
    const auto ax3 = fig->add_subplot(4, 4, 2);
    const auto ax4 = fig->add_subplot(4, 4, 3);
    const auto axHus1 = fig->add_subplot(4, 4, 4);
    const auto axHus2 = fig->add_subplot(4, 4, 5);
    const auto axHus3 = fig->add_subplot(4, 4, 6);
    const auto axHus4 = fig->add_subplot(4, 4, 7);
    const auto axNorm1 = fig->add_subplot(4, 4, 8);
    const auto axNorm2 = fig->add_subplot(4, 4, 9);
    const auto axNorm3 = fig->add_subplot(4, 4, 10);
    const auto axNorm4 = fig->add_subplot(4, 4, 11);
    const auto axAng1 = fig->add_subplot(4, 4, 12);
    const auto axAng2 = fig->add_subplot(4, 4, 13);
    const auto axAng3 = fig->add_subplot(4, 4, 14);
    const auto axAng4 = fig->add_subplot(4, 4, 15);
    const std::vector<axes_handle> axes = {ax1, ax2, ax3, ax4, axHus1, axHus2, axHus3, axHus4, axNorm1, axNorm2, axNorm3, axNorm4, axAng1, axAng2, axAng3, axAng4};

    // Define the kernel strategy
    BIM::BoundaryIntegral bi1(39.6920846042302102, 15, prosen, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 1.0, 0, true, true);
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotNormalDerivativeOfWavefunction(axNorm1, 0, 1.0);
    plotting1.plotPoincareHusimiHeatmap(axHus1, 500, 1.0, std::nullopt, 0);
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotRadiallyIntegratedMomentumDensity(axAng1, 0, 500, 1.0);

    BIM::BoundaryIntegral bi2(39.2334616730836530, 15, prosen, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 1.0, 0, true, true);
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotNormalDerivativeOfWavefunction(axNorm2, 0, 1.0);
    plotting2.plotPoincareHusimiHeatmap(axHus2, 500, 1.0, std::nullopt, 0);
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotRadiallyIntegratedMomentumDensity(axAng2, 0, 500, 1.0);

    BIM::BoundaryIntegral bi3(39.2728636431821627, 15, prosen, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 1.0, 0, true, true);
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotNormalDerivativeOfWavefunction(axNorm3, 0, 1.0);
    plotting3.plotPoincareHusimiHeatmap(axHus3, 500, 1.0, std::nullopt, 0);
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotRadiallyIntegratedMomentumDensity(axAng3, 0, 500, 1.0);

    BIM::BoundaryIntegral bi4(38.2181109055452808, 15, prosen, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 1.0, 0, true, true);
    // ReSharper disable once CppNoDiscardExpression
    plotting4.plotNormalDerivativeOfWavefunction(axNorm4, 0, 1.0);
    plotting4.plotPoincareHusimiHeatmap(axHus4, 500, 1.0, std::nullopt, 0);
    // ReSharper disable once CppNoDiscardExpression
    plotting4.plotRadiallyIntegratedMomentumDensity(axAng4, 0, 500, 1.0);

    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 4, 4, 1500, 0.02);
    const auto ax_diff_norm1 = fig2->add_subplot(2, 2, 0);
    const auto ax_diff_norm2 = fig2->add_subplot(2, 2, 1);
    const auto ax_diff_norm3 = fig2->add_subplot(2, 2, 2);
    const auto ax_diff_norm4 = fig2->add_subplot(2, 2, 3);
    // ReSharper disable once CppExpressionWithoutSideEffects
    // ReSharper disable once CppNoDiscardExpression
    //plotting1.plotAngularDifferenceOfNormalsVsArcLength(ax_diff_norm1, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    //plotting2.plotAngularDifferenceOfNormalsVsArcLength(ax_diff_norm2, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    //plotting3.plotAngularDifferenceOfNormalsVsArcLength(ax_diff_norm3, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    //plotting4.plotAngularDifferenceOfNormalsVsArcLength(ax_diff_norm4, 1.0);
    show(fig);
    show(fig2);
}

void plotTestingQuarterProsen() {
    constexpr double a = 0.2;
    auto prosen = std::make_shared<Boundary::QuarterProsenBilliard>(a);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::XYReflectionSymmetryStandardDD>(prosen);
    const auto fig = matplot::figure(true);
    const auto ax = fig->add_axes();

    // Define the kernel strategy
    BIM::BoundaryIntegral bi(15.8423921196059805, 10, prosen, kernelStrategy);
    const EigenfunctionsAndPlotting plotting(bi,  1);
    plotting.plotWavefunctionDensityHeatmap(ax, 500, 500, 1.0);
    show(fig);
}

void plotTestingC3Full() {
    constexpr double a = 0.2;
    auto c3 = std::make_shared<Boundary::C3CurveFull>(a);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(c3);
    const auto fig = matplot::figure(true);
    const auto ax1 = fig->add_subplot(4, 4, 0);
    const auto ax2 = fig->add_subplot(4, 4, 1);
    const auto ax3 = fig->add_subplot(4, 4, 2);
    const auto ax4 = fig->add_subplot(4, 4, 3);
    const auto ax5 = fig->add_subplot(4, 4, 4);
    const auto ax6 = fig->add_subplot(4, 4, 5);
    const auto ax7 = fig->add_subplot(4, 4, 6);
    const auto ax8 = fig->add_subplot(4, 4, 7);
    const auto ax9 = fig->add_subplot(4, 4, 8);
    const auto ax10 = fig->add_subplot(4, 4, 9);
    const auto ax11 = fig->add_subplot(4, 4, 10);
    const auto ax12 = fig->add_subplot(4, 4, 11);
    const auto ax13 = fig->add_subplot(4, 4, 12);
    const auto ax14 = fig->add_subplot(4, 4, 13);
    const auto ax15 = fig->add_subplot(4, 4, 14);
    const auto ax16 = fig->add_subplot(4, 4, 15);

    const std::vector<axes_handle> axes = {ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16};

    // Define the kernel strategy
    BIM::BoundaryIntegral bi1(13.4954497724886249, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi2(15.7916895844792240, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi3(15.8979698984949263, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi4(17.4931496574828742, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi5(17.5497024851242571, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    plotting5.plotWavefunctionDensityHeatmap(ax5, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi6(19.85276763838191982, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    plotting6.plotWavefunctionDensityHeatmap(ax6, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi7(21.5998949986874820, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting7(bi7,  1);
    plotting7.plotWavefunctionDensityHeatmap(ax7, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi8(21.6145201815022681, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting8(bi8,  1);
    plotting8.plotWavefunctionDensityHeatmap(ax8, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi9(22.2336529206615090, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting9(bi9,  1);
    plotting9.plotWavefunctionDensityHeatmap(ax9, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi10(23.2639157989474867, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting10(bi10,  1);
    plotting10.plotWavefunctionDensityHeatmap(ax10, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi11(25.3325666570832126, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting11(bi11,  1);
    plotting11.plotWavefunctionDensityHeatmap(ax11, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi12( 25.8261978274728428, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting12(bi12,  1);
    plotting12.plotWavefunctionDensityHeatmap(ax12, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi13(27.0024625307816351, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting13(bi13,  1);
    plotting13.plotWavefunctionDensityHeatmap(ax13, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi14(27.1254640683008539, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting14(bi14,  1);
    plotting14.plotWavefunctionDensityHeatmap(ax14, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi15(27.7899723746546812, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting15(bi15,  1);
    plotting15.plotWavefunctionDensityHeatmap(ax15, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi16(28.0118501481268538, 30, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting16(bi16,  1);
    plotting16.plotWavefunctionDensityHeatmap(ax16, 500, 500, 1.0, 0, true, true);

    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 4, 4, 1500, 0.01);
    show(fig);
}

void plotTestingC3Desymmetrized() {
    constexpr double a = 0.2;
    auto c3 = std::make_shared<Boundary::C3DesymmetrizedBilliard>(a);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(c3);
    const auto fig = matplot::figure(true);
    fig->size(1500, 1200);
    const auto ax1 = fig->add_subplot(2, 2, 0);
    const auto ax2 = fig->add_subplot(2, 2, 1);
    const auto ax3 = fig->add_subplot(2, 2, 2);
    const auto ax4 = fig->add_subplot(2, 2, 3);

    const std::vector<axes_handle> axes = {ax1, ax2, ax3, ax4};

    // Define the kernel strategy
    BIM::BoundaryIntegral bi1(49.5879658796587961, 15, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi2(37.0293101465507348, 15, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi3(49.5879658796587961, 15, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi4(39.7544237721188622, 15, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 1.0, 0, true, true);


    //EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 2, 2, 1500, 0.01);
    show(fig);
}

void plotTestingC3Desymmetrized_HighE() {
    constexpr double a = 0.2;
    auto c3 = std::make_shared<Boundary::C3DesymmetrizedBilliard>(a);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(c3);
    const auto fig = matplot::figure(true);
    fig->size(1500, 1200);
    const auto ax1 = fig->add_subplot(2, 3, 0);
    const auto ax2 = fig->add_subplot(2, 3, 1);
    const auto ax3 = fig->add_subplot(2, 3, 2);
    const auto ax4 = fig->add_subplot(2, 3, 3);
    const auto ax5 = fig->add_subplot(2, 3, 4);
    const auto ax6 = fig->add_subplot(2, 3, 5);

    // Define the kernel strategy
    BIM::BoundaryIntegral bi1(100.1424114241142433, 15, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi2(100.4485944859448665, 15, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi3(100.5668056680566735, 15, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi4(100.6684266842668478, 15, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi5(100.6684266842668478, 15, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    plotting5.plotWavefunctionDensityHeatmap(ax5, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi6(100.6684266842668478, 15, c3, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    plotting6.plotWavefunctionDensityHeatmap(ax6, 500, 500, 1.0, 0, true, true);

    show(fig);
}

void plotTestingSquareWithParabolicTop() {
    const auto squareWTop = std::make_shared<Boundary::SquareWithParabolicTop>(Boundary::Point(0.0, 0.0), 1.0, -1.5);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(squareWTop);
    const auto fig = matplot::figure(true);
    const auto ax1 = fig->add_subplot(5, 4, 0);
    const auto ax2 = fig->add_subplot(5, 4, 1);
    const auto ax3 = fig->add_subplot(5, 4, 2);
    const auto ax4 = fig->add_subplot(5, 4, 3);
    const auto ax5 = fig->add_subplot(5, 4, 4);
    const auto ax6 = fig->add_subplot(5, 4, 5);
    const auto ax7 = fig->add_subplot(5, 4, 6);
    const auto ax8 = fig->add_subplot(5, 4, 7);
    const auto ax9 = fig->add_subplot(5, 4, 8);
    const auto ax10 = fig->add_subplot(5, 4, 9);
    const auto ax11 = fig->add_subplot(5, 4, 10);
    const auto ax12 = fig->add_subplot(5, 4, 11);
    const auto ax13 = fig->add_subplot(5, 4, 12);
    const auto ax14 = fig->add_subplot(5, 4, 13);
    const auto ax15 = fig->add_subplot(5, 4, 14);
    const auto ax16 = fig->add_subplot(5, 4, 15);

    const std::vector<axes_handle> axes = {ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16};

    // Define the kernel strategy
    BIM::BoundaryIntegral bi1(12.4816399081995417, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 1.0);

    BIM::BoundaryIntegral bi2(12.8654018270091353, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 1.0);

    BIM::BoundaryIntegral bi3(13.5795928979644902, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 1.0);

    BIM::BoundaryIntegral bi4(13.9196745983729926, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 1.0);

    BIM::BoundaryIntegral bi5(13.9842199210996068, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    plotting5.plotWavefunctionDensityHeatmap(ax5, 500, 500, 1.0);

    BIM::BoundaryIntegral bi6(15.7620438102190530, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    plotting6.plotWavefunctionDensityHeatmap(ax6, 500, 500, 1.0);

    BIM::BoundaryIntegral bi7(15.7823239116195584, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting7(bi7,  1);
    plotting7.plotWavefunctionDensityHeatmap(ax7, 500, 500, 1.0);

    BIM::BoundaryIntegral bi8(15.7909039545197736, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting8(bi8,  1);
    plotting8.plotWavefunctionDensityHeatmap(ax8, 500, 500, 1.0);

    BIM::BoundaryIntegral bi9(16.0070625353126772, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting9(bi9,  1);
    plotting9.plotWavefunctionDensityHeatmap(ax9, 500, 500, 1.0);

    BIM::BoundaryIntegral bi10(17.6893359466797335, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting10(bi10,  1);
    plotting10.plotWavefunctionDensityHeatmap(ax10, 500, 500, 1.0);

    BIM::BoundaryIntegral bi11(18.0458952294761481, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting11(bi11,  1);
    plotting11.plotWavefunctionDensityHeatmap(ax11, 500, 500, 1.0);

    BIM::BoundaryIntegral bi12( 18.5673278366391834, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting12(bi12,  1);
    plotting12.plotWavefunctionDensityHeatmap(ax12, 500, 500, 1.0);

    BIM::BoundaryIntegral bi13(19.0610703053515280, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting13(bi13,  1);
    plotting13.plotWavefunctionDensityHeatmap(ax13, 500, 500, 1.0);

    BIM::BoundaryIntegral bi14(19.4943624718123587, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting14(bi14,  1);
    plotting14.plotWavefunctionDensityHeatmap(ax14, 500, 500, 1.0);

    BIM::BoundaryIntegral bi15(19.7794538972694873, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting15(bi15,  1);
    plotting15.plotWavefunctionDensityHeatmap(ax15, 500, 500, 1.0);

    BIM::BoundaryIntegral bi16(19.8119215596078000, 15, squareWTop, kernelStrategy);
    const EigenfunctionsAndPlotting plotting16(bi16,  1);
    plotting16.plotWavefunctionDensityHeatmap(ax16, 500, 500, 1.0);

    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 5, 4, 1650, 0.005, 0.01);
    show(fig);
}

void plotTestingRobnikFull_LowE() {
    constexpr double eps = 0.9;
    const auto robnik = std::make_shared<Boundary::RobnikBilliard>(eps);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(robnik);
    const auto fig = matplot::figure(true);
    const auto ax1 = fig->add_subplot(4, 4, 0);
    const auto ax2 = fig->add_subplot(4, 4, 1);
    const auto ax3 = fig->add_subplot(4, 4, 2);
    const auto ax4 = fig->add_subplot(4, 4, 3);
    const auto axHusimi1 = fig->add_subplot(4, 4, 4);
    const auto axHusimi2 = fig->add_subplot(4, 4, 5);
    const auto axHusimi3 = fig->add_subplot(4, 4, 6);
    const auto axHusimi4 = fig->add_subplot(4, 4, 7);

    // Define the kernel strategy
    BIM::BoundaryIntegral bi1(12.0217670294504906, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 1.0);
    plotting1.plotPoincareHusimiHeatmap(axHusimi1, 300);

    BIM::BoundaryIntegral bi2(13.8931482191369859, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 1.0);
    plotting2.plotPoincareHusimiHeatmap(axHusimi2, 300);

    BIM::BoundaryIntegral bi3(14.5083834730578847, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 1.0);
    plotting3.plotPoincareHusimiHeatmap(axHusimi3, 300);

    BIM::BoundaryIntegral bi4(18.6141769029483832, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 1.0);
    plotting4.plotPoincareHusimiHeatmap(axHusimi4, 300);

    const auto axRad1 = fig->add_subplot(4, 4, 8);
    const auto axRad2 = fig->add_subplot(4, 4, 9);
    const auto axRad3 = fig->add_subplot(4, 4, 10);
    const auto axRad4 = fig->add_subplot(4, 4, 11);
    const auto axAng1 = fig->add_subplot(4, 4, 12);
    const auto axAng2 = fig->add_subplot(4, 4, 13);
    const auto axAng3 = fig->add_subplot(4, 4, 14);
    const auto axAng4 = fig->add_subplot(4, 4, 15);
    const std::vector<axes_handle> axes = {ax1, ax2, ax3, ax4, axHusimi1, axHusimi2, axHusimi3, axHusimi4, axRad1, axRad2, axRad3, axRad4, axAng1, axAng2, axAng3, axAng4};

    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotRadiallyIntegratedMomentumDensity(axRad1, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotRadiallyIntegratedMomentumDensity(axRad2, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotRadiallyIntegratedMomentumDensity(axRad3, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting4.plotRadiallyIntegratedMomentumDensity(axRad4, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotAngularIntegratedMomentumDensity(axAng1, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotAngularIntegratedMomentumDensity(axAng2, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotAngularIntegratedMomentumDensity(axAng3, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting4.plotAngularIntegratedMomentumDensity(axAng4, 0, 400, 1.0);

    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 4, 4, 1500, 0.05);
    show(fig);
};

void plotTestingRobnikFull_MiddleE() {
    constexpr double eps = 0.9;
    const auto robnik = std::make_shared<Boundary::RobnikBilliard>(eps);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(robnik);
    const auto fig = matplot::figure(true);
    fig->size(1500, 1200);
    const auto fig2 = matplot::figure(true);
    const auto ax1 = fig->add_subplot(2, 2, 0);
    const auto ax2 = fig->add_subplot(2, 2, 1);
    const auto ax3 = fig->add_subplot(2, 2, 2);
    const auto ax4 = fig->add_subplot(2, 2, 3);

    BIM::BoundaryIntegral bi1(40.0015000300005994, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    BIM::BoundaryIntegral bi2(40.9039380787615769, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    BIM::BoundaryIntegral bi3(40.9444388887777748, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    BIM::BoundaryIntegral bi4(40.9823796475929498, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 1.0, 0, true, true);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 1.0, 0, true, true);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 1.0, 0, true, true);
    // ReSharper disable once CppNoDiscardExpression
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 1.0, 0, true, true);
    save(fig, "robnik_40_41.png");
    // ReSharper disable once CppNoDiscardExpression
    //plotting1.plotAngularDifferenceOfNormalsVsArcLength(axCurv, 1.0);
    show(fig);
    show(fig2);
}

void plotTestingRobnikFull_MiddleEMore() {
    constexpr double eps = 0.9;
    const auto robnik = std::make_shared<Boundary::RobnikBilliard>(eps);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(robnik);

    const auto fig = matplot::figure(true);
    fig->size(1500, 1500);
    const auto ax1 = fig->add_subplot(3, 3, 0);
    const auto ax2 = fig->add_subplot(3, 3, 1);
    const auto ax3 = fig->add_subplot(3, 3, 2);
    const auto ax4 = fig->add_subplot(3, 3, 3);
    const auto ax5 = fig->add_subplot(3, 3, 4);
    const auto ax6 = fig->add_subplot(3, 3, 5);
    const auto ax7 = fig->add_subplot(3, 3, 6);
    const auto ax8 = fig->add_subplot(3, 3, 7);

    BIM::BoundaryIntegral bi1(40.0015000300005994, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    BIM::BoundaryIntegral bi2(40.9039380787615769, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    BIM::BoundaryIntegral bi3(40.9444388887777748, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    BIM::BoundaryIntegral bi4(40.9823796475929498, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    BIM::BoundaryIntegral bi5(40.7830589152945748, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    BIM::BoundaryIntegral bi6(40.7111885559427833, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    BIM::BoundaryIntegral bi7(40.6741083705418518, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting7(bi7,  1);
    BIM::BoundaryIntegral bi8(40.4952324761623785, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting8(bi8,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 1.0, 0, true, true);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 1.0, 0, true, true);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 1.0, 0, true, true);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 1.0, 0, true, true);
    plotting5.plotWavefunctionDensityHeatmap(ax5, 500, 500, 1.0, 0, true, true);
    plotting6.plotWavefunctionDensityHeatmap(ax6, 500, 500, 1.0, 0, true, true);
    plotting7.plotWavefunctionDensityHeatmap(ax7, 500, 500, 1.0, 0, true, true);
    plotting8.plotWavefunctionDensityHeatmap(ax8, 500, 500, 1.0, 0, true, true);

    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, {ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8}, 3, 3, 0.005, 0.005);
    save(fig, "robnik_40_41_more.png");
    show(fig);
}

void plotTestingRobnik_MiddleEHusimi() {
    constexpr double eps = 0.9;
    const auto robnik = std::make_shared<Boundary::RobnikBilliard>(eps);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(robnik);

    const auto fig2 = matplot::figure(true);
    fig2->size(1000, 1000);
    const auto ax12 = fig2->add_subplot(3, 3, 0);
    const auto ax22 = fig2->add_subplot(3, 3, 1);
    const auto ax32 = fig2->add_subplot(3, 3, 2);
    const auto ax42 = fig2->add_subplot(3, 3, 3);
    const auto ax52 = fig2->add_subplot(3, 3, 4);
    const auto ax62 = fig2->add_subplot(3, 3, 5);
    const auto ax72 = fig2->add_subplot(3, 3, 6);
    const auto ax82 = fig2->add_subplot(3, 3, 7);

    BIM::BoundaryIntegral bi1(40.0015000300005994, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    BIM::BoundaryIntegral bi2(40.9039380787615769, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    BIM::BoundaryIntegral bi3(40.9444388887777748, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    BIM::BoundaryIntegral bi4(40.9823796475929498, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    BIM::BoundaryIntegral bi5(40.7830589152945748, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    BIM::BoundaryIntegral bi6(40.7111885559427833, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    BIM::BoundaryIntegral bi7(40.6741083705418518, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting7(bi7,  1);
    BIM::BoundaryIntegral bi8(40.4952324761623785, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting8(bi8,  1);

    plotting1.plotPoincareHusimiHeatmap(ax12, 500, 1.0);
    plotting2.plotPoincareHusimiHeatmap(ax22, 500, 1.0);
    plotting3.plotPoincareHusimiHeatmap(ax32, 500, 1.0);
    plotting4.plotPoincareHusimiHeatmap(ax42, 500, 1.0);
    plotting5.plotPoincareHusimiHeatmap(ax52, 500, 1.0);
    plotting6.plotPoincareHusimiHeatmap(ax62, 500, 1.0);
    plotting7.plotPoincareHusimiHeatmap(ax72, 500, 1.0);
    plotting8.plotPoincareHusimiHeatmap(ax82, 500, 1.0);

    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig2, {ax12, ax22, ax32, ax42, ax52, ax62, ax72, ax82}, 3, 3, 0.005, 0.005);
    save(fig2, "robnik_40_41_more_husimi.png");
    show(fig2);
}

void plotTestingRobnik_MiddleENormalDer() {
    constexpr double eps = 0.9;
    const auto robnik = std::make_shared<Boundary::RobnikBilliard>(eps);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(robnik);

    const auto fig2 = matplot::figure(true);
    fig2->size(1000, 1000);
    const auto ax12 = fig2->add_subplot(3, 3, 0);
    const auto ax22 = fig2->add_subplot(3, 3, 1);
    const auto ax32 = fig2->add_subplot(3, 3, 2);
    const auto ax42 = fig2->add_subplot(3, 3, 3);
    const auto ax52 = fig2->add_subplot(3, 3, 4);
    const auto ax62 = fig2->add_subplot(3, 3, 5);
    const auto ax72 = fig2->add_subplot(3, 3, 6);
    const auto ax82 = fig2->add_subplot(3, 3, 7);

    BIM::BoundaryIntegral bi1(40.0015000300005994, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    BIM::BoundaryIntegral bi2(40.9039380787615769, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    BIM::BoundaryIntegral bi3(40.9444388887777748, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    BIM::BoundaryIntegral bi4(40.9823796475929498, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    BIM::BoundaryIntegral bi5(40.7830589152945748, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    BIM::BoundaryIntegral bi6(40.7111885559427833, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    BIM::BoundaryIntegral bi7(40.6741083705418518, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting7(bi7,  1);
    BIM::BoundaryIntegral bi8(40.4952324761623785, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting8(bi8,  1);

    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotNormalDerivativeOfWavefunction(ax12, 0, 1.0, false);
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotNormalDerivativeOfWavefunction(ax22, 0, 1.0, false);
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotNormalDerivativeOfWavefunction(ax32, 0, 1.0, false);
    // ReSharper disable once CppNoDiscardExpression
    plotting4.plotNormalDerivativeOfWavefunction(ax42, 0, 1.0, false);
    // ReSharper disable once CppNoDiscardExpression
    plotting5.plotNormalDerivativeOfWavefunction(ax52, 0, 1.0, false);
    // ReSharper disable once CppNoDiscardExpression
    plotting6.plotNormalDerivativeOfWavefunction(ax62, 0, 1.0, false);
    // ReSharper disable once CppNoDiscardExpression
    plotting7.plotNormalDerivativeOfWavefunction(ax72, 0, 1.0, false);
    // ReSharper disable once CppNoDiscardExpression
    plotting8.plotNormalDerivativeOfWavefunction(ax82, 0, 1.0, false);

    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig2, {ax12, ax22, ax32, ax42, ax52, ax62, ax72, ax82}, 3, 3, 1200, 0.05, 0.08);
    save(fig2, "robnik_40_41_more_normal_der.png");
    show(fig2);
}

void plotTestingRobnik_MiddleEMomentumDensity() {
    constexpr double eps = 0.9;
    const auto robnik = std::make_shared<Boundary::RobnikBilliard>(eps);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(robnik);

    const auto fig2 = matplot::figure(true);
    fig2->size(1000, 1000);
    const auto ax12 = fig2->add_subplot(3, 3, 0);
    const auto ax22 = fig2->add_subplot(3, 3, 1);
    const auto ax32 = fig2->add_subplot(3, 3, 2);
    const auto ax42 = fig2->add_subplot(3, 3, 3);
    const auto ax52 = fig2->add_subplot(3, 3, 4);
    const auto ax62 = fig2->add_subplot(3, 3, 5);
    const auto ax72 = fig2->add_subplot(3, 3, 6);
    const auto ax82 = fig2->add_subplot(3, 3, 7);

    BIM::BoundaryIntegral bi1(40.0015000300005994, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    BIM::BoundaryIntegral bi2(40.9039380787615769, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    BIM::BoundaryIntegral bi3(40.9444388887777748, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    BIM::BoundaryIntegral bi4(40.9823796475929498, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    BIM::BoundaryIntegral bi5(40.7830589152945748, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    BIM::BoundaryIntegral bi6(40.7111885559427833, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    BIM::BoundaryIntegral bi7(40.6741083705418518, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting7(bi7,  1);
    BIM::BoundaryIntegral bi8(40.4952324761623785, 15, robnik, kernelStrategy);
    const EigenfunctionsAndPlotting plotting8(bi8,  1);

    plotting1.plotMomentumDensityHeatmapPolar(ax12, 500, 500, 1.0, 0);
    plotting2.plotMomentumDensityHeatmapPolar(ax22, 500, 500, 1.0, 0);
    plotting3.plotMomentumDensityHeatmapPolar(ax32, 500, 500, 1.0, 0);
    plotting4.plotMomentumDensityHeatmapPolar(ax42, 500, 500, 1.0, 0);
    plotting5.plotMomentumDensityHeatmapPolar(ax52, 500, 500, 1.0, 0);
    plotting6.plotMomentumDensityHeatmapPolar(ax62, 500, 500, 1.0, 0);
    plotting7.plotMomentumDensityHeatmapPolar(ax72, 500, 500, 1.0, 0);
    plotting8.plotMomentumDensityHeatmapPolar(ax82, 500, 500, 1.0, 0);

    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig2, {ax12, ax22, ax32, ax42, ax52, ax62, ax72, ax82}, 3, 3, 1200, 0.05, 0.08);
    save(fig2, "robnik_40_41_more_momemtum.png");
    show(fig2);
}

void plotTestingPolygonalBilliards() {
    std::vector<Boundary::Point> vertices = {
        {0, 0},   // Bottom-left
        {1, 3},   // Top-mid
        {3, 2},   // Top-right
        {4, 0},   // Bottom-right
        {2, -1}   // Bottom-mid
    };
    const auto polygonBilliard = std::make_shared<Boundary::PolygonBilliard>(vertices);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(polygonBilliard);
    const auto fig = matplot::figure(true);
    fig->size(1500, 1000);
    const auto ax1 = fig->add_subplot(2, 3, 0);
    const auto ax2 = fig->add_subplot(2, 3, 1);
    const auto ax3 = fig->add_subplot(2, 3, 2);
    const auto ax4 = fig->add_subplot(2, 3, 3);
    const auto ax5 = fig->add_subplot(2, 3, 4);
    const auto ax6 = fig->add_subplot(2, 3, 5);
    //const std::vector<axes_handle> axes = {ax1, ax2, ax3, ax4, ax5 ,ax6};

    // Define the kernel strategy
    BIM::BoundaryIntegral bi1(19.8568694274777116, 15, polygonBilliard, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi2(19.8730934923739717, 15, polygonBilliard, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi3(19.9306577226308903, 15, polygonBilliard, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi4(19.9611558446233808, 15, polygonBilliard, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi5(19.9761319045276196, 15, polygonBilliard, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    plotting5.plotWavefunctionDensityHeatmap(ax5, 500, 500, 1.0, 0, true, true);

    BIM::BoundaryIntegral bi6(19.9997659990639960, 15, polygonBilliard, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    plotting6.plotWavefunctionDensityHeatmap(ax6, 500, 500, 1.0, 0, true, true);

    //EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 2, 3, 1000, 0.005, 0.005);
    save(fig, "polygonal_18_20.png");
    show(fig);
}

void plotTestingQuarterBunimovich() {
    // Testing Quarter Bunimovich Stadium
    Boundary::Point TopLeftQB(0.0, 0.0);
    double widthQB = 2.0;
    double heightQB = 1.0;
    const auto qbunimovich = std::make_shared<Boundary::QuarterBunimovichStadium>(TopLeftQB, heightQB, widthQB);
    qbunimovich->printBoundaryInfo(50);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::XYReflectionSymmetryNNStandard>(qbunimovich);
    const auto fig = matplot::figure(true);
    fig->size(1000, 500);

    const auto ax1 = fig->add_axes();

    BIM::BoundaryIntegral bi1(9.18698, 10, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 0.5);
    show(fig);
}

void executeSquareWithParabolicTop() {
    // Define the kernel strategy
    const auto squareWTop = std::make_shared<Boundary::SquareWithParabolicTop>(Boundary::Point(0.0, 0.0), 1.0, -1.5);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(squareWTop);
    // KRangeSolver parameters
    constexpr double k_min = 0.5;
    constexpr double k_max = 20.0;
    constexpr int SIZE_K = 200000;
    constexpr int scalingFactor = 15;

    // Create and use the KRangeSolver
    BIM::KRangeSolver solver(k_min, k_max, SIZE_K, scalingFactor, squareWTop, kernelStrategy);
    solver.computeSingularValueDecomposition(Eigen::ComputeThinU);

    // Plotting
    const auto fig = figure(true);
    fig->size(1000, 800);
    const auto ax = fig->add_subplot(2, 1, 1);
    const auto ax2 = fig->add_subplot(2, 1, 2);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax, 0, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSmallestSingularValues(ax2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 2, 0, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 2, -0.01, 0.07);

    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax, 3, 0, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    solver.plotSingularValues(ax2, 3, -0.01, 0.07);

    hold(ax, on);
    // Add dashed line at y = 0 in both ax and ax2
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax->plot({k_min, k_max}, {0.0, 0.0}, "k--");
    hold(ax2, on);
    // ReSharper disable once CppZeroConstantCanBeReplacedWithNullptr
    ax2->plot({k_min, k_max}, {0.0, 0.0}, "k--");

    matplot::legend(ax, true);
    matplot::legend(ax2, true);

    solver.printLocalMinimaOfSingularValues(0.02);
    show();
}

void plot_six_consecutive_mushroomFull_Dirichlet() {
    Boundary::Point center(0.0, 0.0);
    double radius = 3.0/2.0;
    double height = 1.0;
    double width = 1.0;
    const auto mushroom = std::make_shared<Boundary::MushroomBilliard>(center, radius, height, width);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(mushroom);
    const auto fig = matplot::figure(true);
    const auto fig2 = figure(true);
    const auto fig3 = figure(true);

    const auto ax1 = fig->add_subplot(2, 4, 0);
    const auto ax2 = fig->add_subplot(2, 4, 1);
    const auto ax3 = fig->add_subplot(2, 4, 2);
    const auto ax4 = fig->add_subplot(2, 4, 3);
    const auto ax5 = fig->add_subplot(2, 4, 4);
    const auto ax6 = fig->add_subplot(2, 4, 5);
    const auto ax7 = fig->add_subplot(2, 4, 6);
    const auto ax8 = fig->add_subplot(2, 4, 7);

    const auto ax12 = fig2->add_subplot(2, 4, 0);
    const auto ax22 = fig2->add_subplot(2, 4, 1);
    const auto ax32 = fig2->add_subplot(2, 4, 2);
    const auto ax42 = fig2->add_subplot(2, 4, 3);
    const auto ax52 = fig2->add_subplot(2, 4, 4);
    const auto ax62 = fig2->add_subplot(2, 4, 5);
    const auto ax72 = fig2->add_subplot(2, 4, 6);
    const auto ax82 = fig2->add_subplot(2, 4, 7);

    const auto ax13 = fig3->add_subplot(2, 4, 0);
    const auto ax23 = fig3->add_subplot(2, 4, 1);
    const auto ax33 = fig3->add_subplot(2, 4, 2);
    const auto ax43 = fig3->add_subplot(2, 4, 3);
    const auto ax53 = fig3->add_subplot(2, 4, 4);
    const auto ax63 = fig3->add_subplot(2, 4, 5);
    const auto ax73 = fig3->add_subplot(2, 4, 6);
    const auto ax83 = fig3->add_subplot(2, 4, 7);

    BIM::BoundaryIntegral bi1(19.2463224877416259, 15, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 1, 0, true, true);
    plotting1.plotPoincareHusimiHeatmap(ax12, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotNormalDerivativeOfWavefunction(ax13, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});

    BIM::BoundaryIntegral bi2(19.3034576781922596, 15, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 1, 0, true, true);
    plotting2.plotPoincareHusimiHeatmap(ax22, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotNormalDerivativeOfWavefunction(ax23, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});


    BIM::BoundaryIntegral bi3(19.4127230424101391, 15, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 1, 0, true, true);
    plotting3.plotPoincareHusimiHeatmap(ax32, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotNormalDerivativeOfWavefunction(ax33, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});


    BIM::BoundaryIntegral bi4(19.5273184243947462, 15, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 1, 0, true, true);
    plotting4.plotPoincareHusimiHeatmap(ax42, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting4.plotNormalDerivativeOfWavefunction(ax43, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});


    BIM::BoundaryIntegral bi5(19.7622292074306891, 30, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    plotting5.plotWavefunctionDensityHeatmap(ax5, 500, 500, 1, 0, true, true);
    plotting5.plotPoincareHusimiHeatmap(ax52, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting5.plotNormalDerivativeOfWavefunction(ax53, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});


    BIM::BoundaryIntegral bi6(19.8070793569311867, 15, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    plotting6.plotWavefunctionDensityHeatmap(ax6, 500, 500, 1, 0, true, true);
    plotting6.plotPoincareHusimiHeatmap(ax62, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting6.plotNormalDerivativeOfWavefunction(ax63, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});


    BIM::BoundaryIntegral bi7(19.9743899146330470, 15, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting7(bi7,  1);
    plotting7.plotWavefunctionDensityHeatmap(ax7, 500, 500, 1, 0, true, true);
    plotting7.plotPoincareHusimiHeatmap(ax72, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting7.plotNormalDerivativeOfWavefunction(ax73, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});


    BIM::BoundaryIntegral bi8(19.9968149893832958, 15, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting8(bi8,  1);
    plotting8.plotWavefunctionDensityHeatmap(ax8, 500, 500, 1, 0, true, true);
    plotting8.plotPoincareHusimiHeatmap(ax82, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting8.plotNormalDerivativeOfWavefunction(ax83, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});

    std::vector<axes_handle> axes = {ax1, ax2, ax3, ax4, ax5, ax6, ax7 ,ax8};
    std::vector<axes_handle> axesH = {ax12, ax22, ax32, ax42, ax52, ax62, ax72 ,ax82};
    std::vector<axes_handle> axesN = {ax13, ax23, ax33, ax43, ax53, ax63, ax73 ,ax83};
    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 2, 4, 800);
    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig2, axesH, 2, 4, 800);
    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig3, axesN, 2, 4, 800, 0.05);

    show(fig);
    show(fig2);
    show(fig3);
}

void plot_six_consecutive_mushroomHalf_Dirichlet() {
    Boundary::Point center(0.0, 0.0);
    double radius = 3.0/2.0;
    double height = 1.0;
    double width = 1.0;
    const auto mushroom = std::make_shared<Boundary::HalfMushroomBilliard>(center, radius, height, width);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::YReflectionSymmetryDStandard>(mushroom);
    const auto fig = matplot::figure(true);
    const auto fig2 = figure(true);
    const auto fig3 = figure(true);

    const auto ax1 = fig->add_subplot(2, 4, 0);
    const auto ax2 = fig->add_subplot(2, 4, 1);
    const auto ax3 = fig->add_subplot(2, 4, 2);
    const auto ax4 = fig->add_subplot(2, 4, 3);
    const auto ax5 = fig->add_subplot(2, 4, 4);
    const auto ax6 = fig->add_subplot(2, 4, 5);
    const auto ax7 = fig->add_subplot(2, 4, 6);
    const auto ax8 = fig->add_subplot(2, 4, 7);

    const auto ax12 = fig2->add_subplot(2, 4, 0);
    const auto ax22 = fig2->add_subplot(2, 4, 1);
    const auto ax32 = fig2->add_subplot(2, 4, 2);
    const auto ax42 = fig2->add_subplot(2, 4, 3);
    const auto ax52 = fig2->add_subplot(2, 4, 4);
    const auto ax62 = fig2->add_subplot(2, 4, 5);
    const auto ax72 = fig2->add_subplot(2, 4, 6);
    const auto ax82 = fig2->add_subplot(2, 4, 7);

    const auto ax13 = fig3->add_subplot(2, 4, 0);
    const auto ax23 = fig3->add_subplot(2, 4, 1);
    const auto ax33 = fig3->add_subplot(2, 4, 2);
    const auto ax43 = fig3->add_subplot(2, 4, 3);
    const auto ax53 = fig3->add_subplot(2, 4, 4);
    const auto ax63 = fig3->add_subplot(2, 4, 5);
    const auto ax73 = fig3->add_subplot(2, 4, 6);
    const auto ax83 = fig3->add_subplot(2, 4, 7);

    BIM::BoundaryIntegral bi1(19.0655506555065557, 30, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500);
    plotting1.plotPoincareHusimiHeatmap(ax12, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotNormalDerivativeOfWavefunction(ax13, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});

    BIM::BoundaryIntegral bi2(19.2449524495244972, 30, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500);
    plotting2.plotPoincareHusimiHeatmap(ax22, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotNormalDerivativeOfWavefunction(ax23, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});


    BIM::BoundaryIntegral bi3(19.4052440524405263, 30, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500);
    plotting3.plotPoincareHusimiHeatmap(ax32, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotNormalDerivativeOfWavefunction(ax33, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});


    BIM::BoundaryIntegral bi4(19.5271202712027119, 30, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500);
    plotting4.plotPoincareHusimiHeatmap(ax42, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting4.plotNormalDerivativeOfWavefunction(ax43, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});


    BIM::BoundaryIntegral bi5(19.7601476014760138, 30, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    plotting5.plotWavefunctionDensityHeatmap(ax5, 500, 500);
    plotting5.plotPoincareHusimiHeatmap(ax52, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting5.plotNormalDerivativeOfWavefunction(ax53, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});


    BIM::BoundaryIntegral bi6(19.8069480694806970, 30, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    plotting6.plotWavefunctionDensityHeatmap(ax6, 500, 500);
    plotting6.plotPoincareHusimiHeatmap(ax62, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting6.plotNormalDerivativeOfWavefunction(ax63, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});


    BIM::BoundaryIntegral bi7(19.9742597425974253, 30, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting7(bi7,  1);
    plotting7.plotWavefunctionDensityHeatmap(ax7, 500, 500);
    plotting7.plotPoincareHusimiHeatmap(ax72, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting7.plotNormalDerivativeOfWavefunction(ax73, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});


    BIM::BoundaryIntegral bi8(19.9962949629496300, 30, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting8(bi8,  1);
    plotting8.plotWavefunctionDensityHeatmap(ax8, 500, 500);
    plotting8.plotPoincareHusimiHeatmap(ax82, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting8.plotNormalDerivativeOfWavefunction(ax83, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});

    std::vector<axes_handle> axes = {ax1, ax2, ax3, ax4, ax5, ax6, ax7 ,ax8};
    std::vector<axes_handle> axesH = {ax12, ax22, ax32, ax42, ax52, ax62, ax72 ,ax82};
    std::vector<axes_handle> axesN = {ax13, ax23, ax33, ax43, ax53, ax63, ax73 ,ax83};
    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 2, 4, 800);
    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig2, axesH, 2, 4, 800);
    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig3, axesN, 2, 4, 800, 0.05);

    show(fig);
    show(fig2);
    show(fig3);
}

void plot_six_consecutive_mushroomFull_Neumann() {
    Boundary::Point center(0.0, 0.0);
    double radius = 3.0/2.0;
    double height = 1.0;
    double width = 1.0;
    const auto mushroom = std::make_shared<Boundary::MushroomBilliard>(center, radius, height, width);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(mushroom);
    const auto fig = matplot::figure(true);
    const auto figHusimi = figure(true);
    const auto fig3 = figure(true);

    const auto ax1 = fig->add_subplot(2, 4, 0);
    const auto ax2 = fig->add_subplot(2, 4, 1);
    const auto ax3 = fig->add_subplot(2, 4, 2);
    const auto ax4 = fig->add_subplot(2, 4, 3);
    const auto ax5 = fig->add_subplot(2, 4, 4);
    const auto ax6 = fig->add_subplot(2, 4, 5);
    const auto ax7 = fig->add_subplot(2, 4, 6);
    const auto ax8 = fig->add_subplot(2, 4, 7);

    const auto axH1 = figHusimi->add_subplot(2, 4, 0);
    const auto axH2 = figHusimi->add_subplot(2, 4, 1);
    const auto axH3 = figHusimi->add_subplot(2, 4, 2);
    const auto axH4 = figHusimi->add_subplot(2, 4, 3);
    const auto axH5 = figHusimi->add_subplot(2, 4, 4);
    const auto axH6 = figHusimi->add_subplot(2, 4, 5);
    const auto axH7 = figHusimi->add_subplot(2, 4, 6);
    const auto axH8 = figHusimi->add_subplot(2, 4, 7);

    const auto ax13 = fig3->add_subplot(2, 4, 0);
    const auto ax23 = fig3->add_subplot(2, 4, 1);
    const auto ax33 = fig3->add_subplot(2, 4, 2);
    const auto ax43 = fig3->add_subplot(2, 4, 3);
    const auto ax53 = fig3->add_subplot(2, 4, 4);
    const auto ax63 = fig3->add_subplot(2, 4, 5);
    const auto ax73 = fig3->add_subplot(2, 4, 6);
    const auto ax83 = fig3->add_subplot(2, 4, 7);

    BIM::BoundaryIntegral bi1(18.3627472549450985, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 1.0, 0, true, true);
    plotting1.plotPoincareHusimiHeatmap(axH1, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting1.plotNormalDerivativeOfWavefunction(ax13, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});

    BIM::BoundaryIntegral bi2(18.6201524030480634, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 1.0, 0, true, true);
    plotting2.plotPoincareHusimiHeatmap(axH2, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting2.plotNormalDerivativeOfWavefunction(ax23, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});

    BIM::BoundaryIntegral bi3(18.7348146962939275, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 1.0, 0, true, true);
    plotting3.plotPoincareHusimiHeatmap(axH3, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting3.plotNormalDerivativeOfWavefunction(ax33, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});

    BIM::BoundaryIntegral bi4(18.9723294465889332, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 1.0, 0, true, true);
    plotting4.plotPoincareHusimiHeatmap(axH4, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting4.plotNormalDerivativeOfWavefunction(ax43, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});

    BIM::BoundaryIntegral bi5(19.0936218724374491, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    plotting5.plotWavefunctionDensityHeatmap(ax5, 500, 500, 1.0, 0, true, true);
    plotting5.plotPoincareHusimiHeatmap(axH5, 500, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting5.plotNormalDerivativeOfWavefunction(ax53, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});

    BIM::BoundaryIntegral bi6(19.2192043840876821, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    plotting6.plotWavefunctionDensityHeatmap(ax6, 500, 500, 1.0, 0, true, true);
    plotting6.plotPoincareHusimiHeatmap(axH6, 300, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting6.plotNormalDerivativeOfWavefunction(ax63, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});

    BIM::BoundaryIntegral bi7(19.5128802576051541, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting7(bi7,  1);
    plotting7.plotWavefunctionDensityHeatmap(ax7, 500, 500, 1.0, 0, true, true);
    plotting7.plotPoincareHusimiHeatmap(axH7, 300, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting7.plotNormalDerivativeOfWavefunction(ax73, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});

    BIM::BoundaryIntegral bi8(19.6536730734614693, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting8(bi8,  1);
    plotting8.plotWavefunctionDensityHeatmap(ax8, 500, 500, 1.0, 0, true, true);
    plotting8.plotPoincareHusimiHeatmap(axH8, 300, 1.0, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});
    // ReSharper disable once CppNoDiscardExpression
    plotting8.plotNormalDerivativeOfWavefunction(ax83, 0, 1.0, true, std::vector<std::string>{"SC", "JR", "HR", "W", "HL", "JL"});

    std::vector<axes_handle> axes = {ax1, ax2, ax3, ax4, ax5, ax6, ax7 ,ax8};
    std::vector<axes_handle> axesH = {axH1, axH2, axH3, axH4, axH5, axH6, axH7 ,axH8};
    std::vector<axes_handle> axesN = {ax13, ax23, ax33, ax43, ax53, ax63, ax73 ,ax83};
    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 2, 4, 800);
    EigenfunctionsAndPlotting::arrangeAxesInFigure(figHusimi, axesH, 2, 4, 800);
    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig3, axesN, 2, 4, 800, 0.05);

    show(fig);
    show(figHusimi);
    show(fig3);

}

void plot_six_consecutive_mushroomHalf_Neumann() {
    Boundary::Point center(0.0, 0.0);
    double radius = 3.0/2.0;
    double height = 1.0;
    double width = 1.0;
    const auto mushroom = std::make_shared<Boundary::HalfMushroomBilliard>(center, radius, height, width);
    //auto f = figure(true);
    //auto ax_plt = f->add_axes();
    //mushroom->plot(ax_plt, 1000, true, true);
    //show(f);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::YReflectionSymmetryNStandard>(mushroom);
    const auto fig = matplot::figure(true);
    fig->size(1500, 1200);

    const auto ax1 = fig->add_subplot(2, 4, 0);
    const auto ax2 = fig->add_subplot(2, 4, 1);
    const auto ax3 = fig->add_subplot(2, 4, 2);
    const auto ax4 = fig->add_subplot(2, 4, 3);
    const auto ax5 = fig->add_subplot(2, 4, 4);
    const auto ax6 = fig->add_subplot(2, 4, 5);
    const auto ax7 = fig->add_subplot(2, 4, 6);
    const auto ax8 = fig->add_subplot(2, 4, 7);

    BIM::BoundaryIntegral bi1(18.3627472549450985, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500);

    BIM::BoundaryIntegral bi2(18.6201524030480634, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500);

    BIM::BoundaryIntegral bi3(18.7348146962939275, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500);

    BIM::BoundaryIntegral bi4(18.9723294465889332, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500);

    BIM::BoundaryIntegral bi5(19.0936218724374491, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    plotting5.plotWavefunctionDensityHeatmap(ax5, 500, 500);

    BIM::BoundaryIntegral bi6(19.2192043840876821, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    plotting6.plotWavefunctionDensityHeatmap(ax6, 500, 500);

    BIM::BoundaryIntegral bi7(19.5128802576051541, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting7(bi7,  1);
    plotting7.plotWavefunctionDensityHeatmap(ax7, 500, 500);

    BIM::BoundaryIntegral bi8(19.6536730734614693, 20, mushroom, kernelStrategy);
    const EigenfunctionsAndPlotting plotting8(bi8,  1);
    plotting8.plotWavefunctionDensityHeatmap(ax8, 500, 500);

    show(fig);
}

void plot_six_consecutive_quarterBunimovich() {
    // Testing Quarter Bunimovich Stadium
    Boundary::Point TopLeftQB(0.0, 0.0);
    double widthQB = 2.0;
    double heightQB = 1.0;
    const auto qbunimovich = std::make_shared<Boundary::QuarterBunimovichStadium>(TopLeftQB, heightQB, widthQB);
    qbunimovich->printCompositeBoundingBox();

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::XYReflectionSymmetryNNStandard>(qbunimovich);
    const auto fig = matplot::figure(true);
    fig->size(1500, 1200);

    const auto ax1 = fig->add_subplot(4, 2, 0);
    const auto ax2 = fig->add_subplot(4, 2, 1);
    const auto ax3 = fig->add_subplot(4, 2, 2);
    const auto ax4 = fig->add_subplot(4, 2, 3);
    const auto ax5 = fig->add_subplot(4, 2, 4);
    const auto ax6 = fig->add_subplot(4, 2, 5);
    const auto ax7 = fig->add_subplot(4, 2, 6);
    const auto ax8 = fig->add_subplot(4, 2, 7);

    BIM::BoundaryIntegral bi1(28.8724874497489949, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 0.5);

    BIM::BoundaryIntegral bi2(29.0123202464049292, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 0.5);

    BIM::BoundaryIntegral bi3(29.2087941758835186, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 0.5);

    BIM::BoundaryIntegral bi4(29.4394887897757975, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 0.5);

    BIM::BoundaryIntegral bi5(29.5787315746314938, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    plotting5.plotWavefunctionDensityHeatmap(ax5, 500, 500, 0.5);

    BIM::BoundaryIntegral bi6(29.6849336986739729, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    plotting6.plotWavefunctionDensityHeatmap(ax6, 500, 500, 0.5);

    BIM::BoundaryIntegral bi7(29.8188663773275451, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting7(bi7,  1);
    plotting7.plotWavefunctionDensityHeatmap(ax7, 500, 500, 0.5);

    BIM::BoundaryIntegral bi8(29.8719674393487864, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting8(bi8,  1);
    plotting8.plotWavefunctionDensityHeatmap(ax8, 500, 500, 0.5);

    show(fig);
}

void plot_six_consecutiveFullBunimovich() {
    // Testing Quarter Bunimovich Stadium
    Boundary::Point TopLeftQB(0.0, 0.0);
    double widthQB = 2.0;
    double heightQB = 1.0;
    const auto qbunimovich = std::make_shared<Boundary::BunimovichStadium>(TopLeftQB, heightQB, widthQB);
    qbunimovich->printCompositeBoundingBox();

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(qbunimovich);
    const auto fig = matplot::figure(true);
    fig->size(2000, 1200);
    const auto figH = figure(true);
    figH->size(2000, 1200);

    const auto ax1 = fig->add_subplot(4, 2, 0);
    const auto ax2 = fig->add_subplot(4, 2, 1);
    const auto ax3 = fig->add_subplot(4, 2, 2);
    const auto ax4 = fig->add_subplot(4, 2, 3);
    const auto ax5 = fig->add_subplot(4, 2, 4);
    const auto ax6 = fig->add_subplot(4, 2, 5);
    const auto ax7 = fig->add_subplot(4, 2, 6);
    const auto ax8 = fig->add_subplot(4, 2, 7);
    const std::vector<axes_handle> axes = {ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8};

    const auto ax1H = figH->add_subplot(4, 2, 0);
    const auto ax2H = figH->add_subplot(4, 2, 1);
    const auto ax3H = figH->add_subplot(4, 2, 2);
    const auto ax4H = figH->add_subplot(4, 2, 3);
    const auto ax5H = figH->add_subplot(4, 2, 4);
    const auto ax6H = figH->add_subplot(4, 2, 5);
    const auto ax7H = figH->add_subplot(4, 2, 6);
    const auto ax8H = figH->add_subplot(4, 2, 7);

    BIM::BoundaryIntegral bi1(28.8724874497489949, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 0.5, 0, true, true);
    plotting1.plotPoincareHusimiHeatmap(ax1H, 400, 1.0, std::vector<std::string>{"RSC", "B", "LSC", "T"});

    BIM::BoundaryIntegral bi2(29.0123202464049292, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 0.5, 0, true, true);
    plotting2.plotPoincareHusimiHeatmap(ax2H, 400, 1.0, std::vector<std::string>{"RSC", "B", "LSC", "T"});

    BIM::BoundaryIntegral bi3(29.2087941758835186, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 0.5, 0, true, true);
    plotting3.plotPoincareHusimiHeatmap(ax3H, 400, 1.0, std::vector<std::string>{"RSC", "B", "LSC", "T"});

    BIM::BoundaryIntegral bi4(29.4394887897757975, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 0.5, 0, true, true);
    plotting4.plotPoincareHusimiHeatmap(ax4H, 400, 1.0, std::vector<std::string>{"RSC", "B", "LSC", "T"});

    BIM::BoundaryIntegral bi5(29.5787315746314938, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    plotting5.plotWavefunctionDensityHeatmap(ax5, 500, 500, 0.5, 0, true, true);
    plotting5.plotPoincareHusimiHeatmap(ax5H, 400, 1.0, std::vector<std::string>{"RSC", "B", "LSC", "T"});

    BIM::BoundaryIntegral bi6(29.6849336986739729, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    plotting6.plotWavefunctionDensityHeatmap(ax6, 500, 500, 0.5, 0, true, true);
    plotting6.plotPoincareHusimiHeatmap(ax6H, 400, 1.0, std::vector<std::string>{"RSC", "B", "LSC", "T"});

    BIM::BoundaryIntegral bi7(29.8188663773275451, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting7(bi7,  1);
    plotting7.plotWavefunctionDensityHeatmap(ax7, 500, 500, 0.5, 0, true, true);
    plotting7.plotPoincareHusimiHeatmap(ax7H, 400, 1.0, std::vector<std::string>{"RSC", "B", "LSC", "T"});

    BIM::BoundaryIntegral bi8(29.8719674393487864, 30, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting8(bi8,  1);
    plotting8.plotWavefunctionDensityHeatmap(ax8, 500, 500, 0.5, 0, true, true);
    plotting8.plotPoincareHusimiHeatmap(ax8H, 400, 1.0, std::vector<std::string>{"RSC", "B", "LSC", "T"});

    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 4, 2, 1500, 0.005, 0.005);
    save(fig, "Bunimovich_25_30.png");
    save(figH, "Bunimovich_25_30_Husimi.png");
    show(fig);
    show(figH);
}

void plot_Bowtie_Orbit_Bunimovich() {
    // Testing Quarter Bunimovich Stadium
    Boundary::Point TopLeftQB(0.0, 0.0);
    double widthQB = 2.0;
    double heightQB = 1.0;
    const auto qbunimovich = std::make_shared<Boundary::BunimovichStadium>(TopLeftQB, heightQB, widthQB);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(qbunimovich);
    const auto fig = matplot::figure(true);
    fig->size(1500, 1200);
    const auto ax = fig->add_subplot(2, 3, 0);
    const auto axH = fig->add_subplot(2, 3, 1);
    const auto axRad = fig->add_subplot(2, 3, 2);
    const auto axAng = fig->add_subplot(2, 3, 3);
    const auto axMom = fig->add_subplot(2, 3, 4);

    BIM::BoundaryIntegral bi(29.4394887897757975, 20, qbunimovich, kernelStrategy);
    const EigenfunctionsAndPlotting plotting(bi,  1);
    plotting.plotWavefunctionDensityHeatmap(ax, 500, 500, 0.5);
    plotting.plotPoincareHusimiHeatmap(axH, 400, 1.0, std::vector<std::string>{"RSC", "B", "LSC", "T"});
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotRadiallyIntegratedMomentumDensity(axRad, 0, 400, 1.0);
    // ReSharper disable once CppNoDiscardExpression
    plotting.plotAngularIntegratedMomentumDensity(axAng, 0, 400, 1.0);
    plotting.plotMomentumDensityHeatmapPolar(axMom);
    show(fig);
}

void plot_eight_consecutive_rightTriangle() {
    constexpr double k1 = 1.0;
    constexpr double k2 = 2.0;
    auto triangle = std::make_shared<Boundary::RightTriangle>(k1, k2);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(triangle);
    const auto triangleFig = figure(true);
    const auto axTriangle = triangleFig->add_axes();
    triangle->plot(axTriangle, 50, true, true);
    axTriangle->axes_aspect_ratio(2.0);
    //show(triangleFig);

    const auto fig = matplot::figure(true);
    //fig->size(2000, 2000);

    const auto ax1 = fig->add_subplot(5, 2, 0);
    const auto ax2 = fig->add_subplot(5, 2, 1);
    const auto ax3 = fig->add_subplot(5, 2, 2);
    const auto ax4 = fig->add_subplot(5, 2, 3);
    const auto ax5 = fig->add_subplot(5, 2, 4);
    const auto ax6 = fig->add_subplot(5, 2, 5);
    const auto ax7 = fig->add_subplot(5, 2, 6);
    const auto ax8 = fig->add_subplot(5, 2, 7);
    const auto ax9 = fig->add_subplot(5, 2, 8);
    const auto ax10 = fig->add_subplot(5, 2, 9);

    const std::vector<axes_handle> axes = {ax1, ax2, ax3, ax4, ax5, ax6 ,ax7, ax8, ax9, ax10};

    BIM::BoundaryIntegral bi1(29.9309662064413757, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi2(29.7104647364315753, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi3(29.6395642637617591, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi4(29.6318975459836409, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi5(29.4570963806425361, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    plotting5.plotWavefunctionDensityHeatmap(ax5, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi6(28.9875599170661147, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    plotting6.plotWavefunctionDensityHeatmap(ax6, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi7(28.7971919812798767, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting7(bi7,  1);
    plotting7.plotWavefunctionDensityHeatmap(ax7, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi8(28.5372902486016571, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting8(bi8,  1);
    plotting8.plotWavefunctionDensityHeatmap(ax8, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi9(28.3901226008173388, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting9(bi9,  1);
    plotting9.plotWavefunctionDensityHeatmap(ax9, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi10(28.0452869685797914, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting10(bi10,  1);
    plotting10.plotWavefunctionDensityHeatmap(ax10, 500, 500, 0.5, 0, true, true);

    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 5, 2, 3000);
    save(fig, "triangle_25_30.png");
    show(fig);
}

void plot_eight_consecutive_rightTriangleHigh() {
    constexpr double k1 = 1.0;
    constexpr double k2 = 2.0;
    auto triangle = std::make_shared<Boundary::RightTriangle>(k1, k2);
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(triangle);
    const auto triangleFig = figure(true);
    const auto axTriangle = triangleFig->add_axes();
    triangle->plot(axTriangle, 50, true, true);
    axTriangle->axes_aspect_ratio(2.0);
    //show(triangleFig);

    const auto fig = matplot::figure(true);
    //fig->size(2000, 2000);

    const auto ax1 = fig->add_subplot(5, 2, 0);
    const auto ax2 = fig->add_subplot(5, 2, 1);
    const auto ax3 = fig->add_subplot(5, 2, 2);
    const auto ax4 = fig->add_subplot(5, 2, 3);
    const auto ax5 = fig->add_subplot(5, 2, 4);
    const auto ax6 = fig->add_subplot(5, 2, 5);
    const auto ax7 = fig->add_subplot(5, 2, 6);
    const auto ax8 = fig->add_subplot(5, 2, 7);
    const auto ax9 = fig->add_subplot(5, 2, 8);
    const auto ax10 = fig->add_subplot(5, 2, 9);

    const std::vector<axes_handle> axes = {ax1, ax2, ax3, ax4, ax5, ax6 ,ax7, ax8, ax9, ax10};

    BIM::BoundaryIntegral bi1(50.0392203922039229, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting1(bi1,  1);
    plotting1.plotWavefunctionDensityHeatmap(ax1, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi2(50.0423004230042281, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting2(bi2,  1);
    plotting2.plotWavefunctionDensityHeatmap(ax2, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi3(50.1106411064110659, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting3(bi3,  1);
    plotting3.plotWavefunctionDensityHeatmap(ax3, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi4(50.2905629056290593, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting4(bi4,  1);
    plotting4.plotWavefunctionDensityHeatmap(ax4, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi5(50.3831638316383135, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting5(bi5,  1);
    plotting5.plotWavefunctionDensityHeatmap(ax5, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi6(50.4454244542445451, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting6(bi6,  1);
    plotting6.plotWavefunctionDensityHeatmap(ax6, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi7(50.7314273142731409, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting7(bi7,  1);
    plotting7.plotWavefunctionDensityHeatmap(ax7, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi8(50.9839098390983878, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting8(bi8,  1);
    plotting8.plotWavefunctionDensityHeatmap(ax8, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi9(51.0305103051030500, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting9(bi9,  1);
    plotting9.plotWavefunctionDensityHeatmap(ax9, 500, 500, 0.5, 0, true, true);

    BIM::BoundaryIntegral bi10(51.1810718107181088, 10, triangle, kernelStrategy);
    const EigenfunctionsAndPlotting plotting10(bi10,  1);
    plotting10.plotWavefunctionDensityHeatmap(ax10, 500, 500, 0.5, 0, true, true);

    EigenfunctionsAndPlotting::arrangeAxesInFigure(fig, axes, 5, 2, 3000);
    save(fig, "triangle_50_55.png");
    show(fig);
}

void printingFredholmDerivativesRectangle() {
    double width = M_PI / 3.0;
    double height = 1.0;
    Boundary::Point bottomLeft(0.0, 0.0);
    auto rectangleBoundary = std::make_shared<Boundary::Rectangle>(bottomLeft, width, height);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(rectangleBoundary);
    const BIM::BoundaryIntegral bi(25.05214, 10, rectangleBoundary, kernelStrategy);
    printFredholmMatrixAndDerivatives(bi, true, BIM::Debugging::PRINT_TYPES::FREDHOLM_COMBINED_DERIVATIVE);
    const auto fig = figure(true);
    const auto ax = fig->add_axes();
    rectangleBoundary->plot(ax, 20, true, true);
    show(fig);
}

void printingFredholmDerivativesCircle() {
    double radius = 1.0;
    Boundary::Point center(0.0, 0.0);
    auto circle = std::make_shared<Boundary::Circle>(center,radius);

    // Define the kernel strategy
    const auto kernelStrategy = std::make_shared<KernelIntegrationStrategies::DefaultKernelIntegrationStrategy>(circle);
    const BIM::BoundaryIntegral bi(25.0521, 10, circle, kernelStrategy);
    printFredholmMatrixAndDerivatives(bi, true, BIM::Debugging::PRINT_TYPES::FREDHOLM_COMBINED_DERIVATIVE);
    const auto fig = figure(true);
    const auto ax = fig->add_axes();
    circle->plot(ax, 50, true, true);
    show(fig);
}

void plot_Test_Number_Variance_Rectangle() {
    constexpr double width = M_PI/3;
    constexpr double height = 1.0;
    constexpr double perimeter = 2*(width + height);
    const auto rectangle = std::make_shared<Boundary::Rectangle>(Boundary::Point{0.0, 0.0}, width, height);

    const auto fig = figure(true);
    const auto ax = fig->add_axes();
    auto plts = AnalysisTools::plotNumberVarianceSigma(ax, AnalysisTools::computeRectangleEigenvaluesWithoutIndices(0.5, 2000, width, height), rectangle, perimeter, 0.25, 0.1, 5, 50);
    show(fig);
}

void plot_Test_Spectral_Rigidity_Rectangle() {
    constexpr double width = M_PI/3;
    constexpr double height = 1.0;
    constexpr double perimeter = 2*(width + height);
    const auto rectangle = std::make_shared<Boundary::Rectangle>(Boundary::Point{0.0, 0.0}, width, height);

    const auto fig = figure(true);
    const auto ax = fig->add_axes();
    AnalysisTools::plotSpectralRigidityDelta(ax, AnalysisTools::computeRectangleEigenvaluesWithoutIndices(0.5, 400, width, height), rectangle, perimeter, 0.25, 0.1, 5, 50);
    show(fig);
}

// Testing Bessel functions from GSL package
// Function to compute higher-order Bessel functions using recurrence relation
double expected_bessel_Jn(const int n, const double x) {
    if (n == 0) return gsl_sf_bessel_J0(x);
    if (n == 1) return gsl_sf_bessel_J1(x);

    double Jm_minus2 = gsl_sf_bessel_J0(x);
    double Jm_minus1 = gsl_sf_bessel_J1(x);
    double Jm = 0.0;

    for (int m = 2; m <= n; ++m) {
        Jm = (2.0 * (m - 1) / x) * Jm_minus1 - Jm_minus2;
        Jm_minus2 = Jm_minus1;
        Jm_minus1 = Jm;
    }
    return Jm;
}

// Function to test the values of the J Bessel function for different indices using GSL
void test_gsl_bessels(const int max_order = 10) {
    constexpr double x_values[] = {0.1, 1.0, 10.0, 100.0, 300.0}; // Values to test

    std::cout << std::fixed << std::setprecision(15);
    std::cout << "Testing GSL Bessel function values for numerical discrepancies:\n";

    for (const double x : x_values) {
        for (int n = 0; n <= max_order; ++n) {
            constexpr double tol = 1e-10;
            const double gsl_value = gsl_sf_bessel_Jn(n, x);
            const double expected_value = expected_bessel_Jn(n, x);

            const double discrepancy = std::abs(gsl_value - expected_value);
            const bool is_within_tolerance = (discrepancy < tol);

            std::cout << "J_" << n << "(" << x << "): GSL = " << gsl_value
                      << ", Expected = " << expected_value
                      << ", Discrepancy = " << discrepancy
                      << ", Within Tolerance = " << (is_within_tolerance ? "Yes" : "No")
                      << "\n";

            if (!is_within_tolerance) {
                std::cout << "Warning: Numerical discrepancy detected for J_" << n << "(" << x << ")\n";
            }
        }
        std::cout << "\n";
    }
}

int main() {
    //integralTesting();
    //executeBIMCircleSVD(false);
    //executeBIMHalfCircleSVD(true);
    //executeBIMQuarterCircleSVD(false);
    //executeBIMCircleDeterminant(false);
    //executeBIMRectangleDeterminants(false);
    //executeBIMRectangleSVD(false);
    //executeExpandedBIM();
    //executeMushroomBilliardSVD(false);
    //executeHalfMushroomBilliardSVD_Dirichlet(false);
    //HalfMushroom_ND_ToFullMushroomComparison();
    //executeHalfMushroomBilliardSVD_Neumann(true);
    //executeHalfMushroomBilliardSVD_Dirichlet(false);
    //executeBIMQuarterRectangleSVD(false);
    //executeQuarterBunimovichSVD();
    //executeBIMRectangleSVDWithBetaVariation({0.0, 0.1, 0.3, 0.5}, false);
    //executeConformalTestingCircle();
    //executeConformalTestingRobnikNumerical();
    //executeExpandedBIM();
    //executeBoundaryTesting();
    //executeRightTriangle();
    //executeEllipse();
    //executeProsenBilliard();
    //executeQuarterProsenBilliard();
    //executeC3Curve();
    //executeC3DesymmetrizedCurve();
    //executeRobnikFull();
    //executePolygonalBilliards();
    //executeSquareWithParabolicTop();
    //executeBIMHalfRectangleSVD(true);

    // Boyd's approach
    //executeBIMCircleDeterminantBoyd(false);
    //test_fft();

    // Classical Billiards
    //executeClassicalBilliardsMushroom();
    //executeClassicalBilliardsCircle();
    //executeClassicalBilliardsRobnik();
    //executeClassicalBilliardsEllipse();
    //executeClassicalBilliardsProsen();
    //executeClassicalBilliardsRectangle();

    // Vergini Saraceno
    //executeVerginiSaracenoRPWBasisTest();
    //executeVerginiSaracenoReducedEigenProblemTesting();
    //executeVerginiSaraceno();
    //test_gsl_bessels(100);

    // Comparison for the rectangle b variation
    //executePlotAndCompare();

    // b variation - k
    //plotBVariationOfNumericalVsAnalyticalKDifferences_RECTANGLE();
    //plotBVariationOfNumericalVsAnalyticalKDifferences_CIRCLE();

    // b variation - E
    //plotBVariationOfNumericalVsAnalyticalEnergyDifferences_RECTANGLE();
    //plotBVariationOfNumericalVsAnalyticalEnergyDifferences_CIRCLE();

    // IPFMS
    //plot_theta_m_vs_ratio();
    //plot_transition();

    // Saving and analyzing
    //saveRectangle("Rectangle_kmin_0.5_kmax_100_stepsize_0.0001_b_10.csv");
    //analyzeRectangle("Rectangle_kmin_0.5_kmax_100_stepsize_0.00005_b_10.csv");

    // Testing Lancsoz SVD vs full SVD
    // Define a complex matrix A of size 50x50
    /**
    constexpr int size = 500;
    constexpr int num_singular_values = 4;
    const Eigen::MatrixXcd A = Eigen::MatrixXcd::Random(size, size);
    // Compare the singular values
    compareSingularValues(A, num_singular_values);*/
    //testTimeSVD();

    // Testing the nullspace
    //testNullspace();
    //testMATLABEngine();

    //Testing EBIM
    //testingExpandedBIMRectangle();
    //testingExpandedBIMCircle();
    //printingFredholmDerivativesRectangle(); // OK
    //printingFredholmDerivativesCircle(); // OK

    // PLotting testing
    //plottingTestingRectangle();
    //plottingRectangle_Detail();
    //plottingCircle_Detail();
    //plottingTestingCircle();
    //plottingTestingCircle_lowE();
    //plottingTestingCircle_HighE();
    //plottingTestingQuarterCircle();
    //plottingTestingMushroomFull();
    //plottingTestingMushroomFull_Semicircle();
    //plotTestingEllipseFull();
    //plotTestingEllipseHalf();
    //plotTestingProsenFull();
    //plotTestingQuarterProsen();
    //plotTestingC3Full();
    //plotTestingC3Desymmetrized();
    //plotTestingC3Desymmetrized_HighE();
    //plotTestingSquareWithParabolicTop();
    //plotTestingRobnikFull_LowE();
    //plotTestingRobnikFull_MiddleE();
    //plotTestingRobnikFull_MiddleEMore();
    //plotTestingRobnik_MiddleEHusimi();
    //plotTestingRobnik_MiddleENormalDer();
    //plotTestingRobnik_MiddleEMomentumDensity();
    //plotTestingPolygonalBilliards();
    //plot_six_consecutive_mushroomFull_Dirichlet();
    //plot_six_consecutive_mushroomHalf_Dirichlet();
    //plot_six_consecutive_mushroomFull_Neumann();
    //plot_six_consecutive_mushroomHalf_Neumann();
    //plot_six_consecutive_quarterBunimovich();
    //plot_six_consecutiveFullBunimovich();
    //plot_Bowtie_Orbit_Bunimovich();
    //plotTestingQuarterBunimovich();
    //plot_eight_consecutive_rightTriangle();
    //plot_eight_consecutive_rightTriangleHigh();

    // Printing analytical eigenvalues for the rectangle and the circle
    //AnalysisTools::printRectangleAnalyticalEigenvalues(5.0, 101, 2, 1);
    //AnalysisTools::printCircleAnalyticalEigenvalues(0.5, 20, 1.0);

    // Testing the spectral rigidity and number variance
    //plot_Test_Number_Variance_Rectangle();
    //plot_Test_Spectral_Rigidity_Rectangle();

    // PROJECT HALF MUSHROOM
    //HalfMushroomProject::PROJECT_HALF_MUSHROOM_PLOTS();
    //HalfMushroomProject::PROJECT_HALF_MUSHROOM_SVD_RUN(1.0, 1.0, 2.0, 0.5, 40, 15, 400000, false);
    //const std::vector<double> ks{19.8243041215206084, 19.5561777808889055, 19.4536072680363432, 19.2087835439177219, 19.0656528282641418, 19.0006200031000176, 18.9411447057235307, 18.4276096380481924, 18.3654043270216363, 18.3467817339086707, 18.0533052665263334, 17.8092615463077344};
    //HalfMushroomProject::PROJECT_HALF_MUSHROOM_PLOT_WAVEFUNCTION_DENSITY("hm_plots.png", Boundary::Point{0.0, 0.0}, 1.0, 1.0, 2.0, ks, 4, 3, 15);
    //onst auto fig = figure(true);
    //fig->size(1200, 1000);
    //const auto ax_main = fig->add_subplot(2, 1, 0);
    //const auto ax_precise = fig->add_subplot(2, 1, 1);
    //HalfMushroomProject::PROJECT_HALF_MUSHROOM_SVD_SAVE(1.0, 1.0, 2.0, 0.5, 20, 15, 100000, "Half_m_test.csv");
    //HalfMushroomProject::PROJECT_HALF_MUSHROOM_SVD_PLOT("Half_m_test.csv", ax_main, 1, -0.1, 1.0);
    //HalfMushroomProject::PROJECT_HALF_MUSHROOM_SVD_PLOT("Half_m_test.csv", ax_main, 2, -0.1, 1.0);
    //HalfMushroomProject::PROJECT_HALF_MUSHROOM_SVD_PLOT("Half_m_test.csv", ax_precise, 1, -0.001, 0.005);
    //HalfMushroomProject::PROJECT_HALF_MUSHROOM_SVD_PLOT("Half_m_test.csv", ax_precise, 2, -0.001, 0.005);
    //BIM::KRangeSolver::calculateDegeneraciesFromCSV("Half_m_test.csv", 0.05, 1e-4, "Half_m_test_degeneracy.csv");
    //show(fig);
    //HalfMushroomProject::PROJECT_HALF_MUSHROOM_MAIN(0.5, 50, 500000, 15);
    //HalfMushroomProject::PROJECT_HALF_MUSHROOM_MAIN_OSCILATORY_PART_AND_WEYL_TOGETHER("HalfMushroomOscillatory.png", "HalfMushroomWeyl.png");
    //HalfMushroomProject::PROJECT_HALF_MUSHROOM_MAIN_NNLS_CUMUL();
    return 0;
}
