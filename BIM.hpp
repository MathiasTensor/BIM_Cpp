#ifndef TEMPIDEABIM_HPP
#define TEMPIDEABIM_HPP

/**
 * @file BIM.hpp
 * @brief Header for instances of the Boundary Integral class.
 * @author Matic Orel
 *
 * This file makes use of the following third-party libraries:
 *
 * - **GSL (GNU Scientific Library):** Licensed under the GNU General Public License (GPL).
 * - **Eigen:** Licensed under the Mozilla Public License 2.0 (MPL2).
 * - **Matplot++:** Licensed under the MIT License.
 *
 * Please ensure that you comply with the respective licenses of these libraries if you distribute or modify this code.
 *
 * Note: This project does not include a specific license for the source code. If you plan to use, distribute, or modify
 * this code, please contact the author for permissions or clarification on usage rights.
 */

#pragma once
#define EIGEN_USE_THREADS // Parallelization of the SVD in the Eigen library. Not noticeable for small matrices but for larger ones it is
#include "Boundary.hpp"
#include "KernelIntegrationStrategy.hpp"
#include "QuantumAnalysisTools.hpp"

#include <iostream>
#include <iomanip>
#include <Eigen/Core>
// ReSharper disable once CppUnusedIncludeDirective
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <utility>
#include <vector>
#include <cmath>
#include <functional>
#include <complex>
#include <future>
#include <ranges>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_spline.h>
#include <stdexcept>
#include <matplot/matplot.h>
#include <armadillo>

namespace BIM {
    class BoundaryIntegral;
    using namespace Boundary;
    using namespace KernelIntegrationStrategies;

    namespace Matrix_Eigen_Armadillo_Conversion {
        static arma::cx_mat EigenToArma(const Eigen::MatrixXcd& eigenMatrix, double lambda);
        static Eigen::MatrixXcd ArmaToEigen(const arma::cx_mat& armaMatrix);
    }

    namespace Debugging {
        enum class PRINT_TYPES;
        void writeComplexMatrixToCSV_Mathematica(const Eigen::MatrixXcd& matrix, const std::string& filename);
        void printFredholmMatrixAndDerivatives(const BoundaryIntegral& bi, bool printDiscretizationPointsAndNormals, PRINT_TYPES printOut);
        void writeComplexMatrixToMAT(const Eigen::MatrixXcd& matrix, const std::string& filename, const std::string& varName);
    }
/**
 * @class BoundaryIntegral
 * @brief Solves the boundary integral method using a specified kernel strategy.
 *
 * This class provides methods for setting up and solving the boundary integral method for a given boundary
 * using different kernel computation strategies. The kernel strategies are specified by the `IKernelStrategy` interface.
 */
class BoundaryIntegral {
private:
    std::vector<Point> points;
    std::vector<Point> normals;
    std::vector<double> t_values;  // To store parametric t values for each point. This is to help with curvature calculations since we will not have to use a root finding algorithm
    std::vector<double> s_values;
    std::vector<double> curvatures;
    Eigen::MatrixXcd matrix;
    Eigen::MatrixXcd matrixDerivative;
    Eigen::MatrixXcd matrixSecondDerivative;
    Eigen::MatrixXcd sortedEigenvectors;
    Eigen::VectorXd sortedSingularValues;
    double k;
    double totalArcLength;
    int scalingFactor;
    std::shared_ptr<AbstractBoundary> boundary;
    std::shared_ptr<IKernelStrategy> kernelStrategy;

    /**
     * @brief Computes the angular difference between the normals of consecutive boundary points.
     *
     * This method calculates the angular difference between the x and y components of the normals
     * for each pair of consecutive boundary points using the dot product.
     *
     * @return A vector of tuples where each tuple contains the parameterization value t, s value, and the angular difference value.
     */
    [[nodiscard]] std::vector<std::tuple<double, double, double>> computeAngularDifferenceOfNormals() const {
        const auto& points = getBoundaryPoints();  // Retrieve boundary points
        const auto& normals = getBoundaryNormals();  // Retrieve boundary normals
        const auto& t_values = getParametrizationValues();  // Retrieve parameterization values
        const auto& s_values = getArclengthValues();  // Retrieve arc length values
        const int num_points = points.size(); // NOLINT(*-narrowing-conversions)

        std::vector<std::tuple<double, double, double>> angular_diff_values(num_points);
        const int num_threads = std::thread::hardware_concurrency();  // Get the number of available threads NOLINT(*-narrowing-conversions)
        const int chunk_size = num_points / num_threads;  // Determine the size of each chunk

        auto compute_chunk = [&](const int start_idx, const int end_idx) {
            std::vector<std::tuple<double, double, double>> chunk_result;
            for (int i = start_idx; i < end_idx; ++i) {
                const int next_idx = (i + 1) % num_points;
                const auto& n1 = normals[i];
                const auto& n2 = normals[next_idx];
                const double dot_product = n1.x * n2.x + n1.y * n2.y;  // Dot product of normals
                const double magnitude1 = std::sqrt(n1.x * n1.x + n1.y * n1.y);  // Magnitude of first normal
                const double magnitude2 = std::sqrt(n2.x * n2.x + n2.y * n2.y);  // Magnitude of second normal
                // Cosine of the angle between the normals
                const double cos_theta = dot_product / (magnitude1 * magnitude2);
                const double theta = std::acos(std::clamp(cos_theta, -1.0, 1.0));  // Angle in radians
                chunk_result.emplace_back(t_values[i], s_values[i], theta);  // Store t, s, and angle difference
            }
            return chunk_result;
        };

        std::vector<std::future<std::vector<std::tuple<double, double, double>>>> futures;
        for (int i = 0; i < num_threads; ++i) {
            int start_idx = i * chunk_size;
            int end_idx = (i == num_threads - 1) ? num_points : start_idx + chunk_size;
            futures.push_back(std::async(std::launch::async, compute_chunk, start_idx, end_idx));
        }

        for (int i = 0; i < num_threads; ++i) {
            auto chunk_result = futures[i].get();
            std::ranges::move(chunk_result, angular_diff_values.begin() + i * chunk_size);
        }
        return angular_diff_values;
    }

    /**
     * @brief Redistributes boundary points based on sigmoid-smoothed angular difference of normals.
     *
     * This method redistributes the points along the boundary based on the sigmoid-smoothed angular difference
     * of normals. It uses sigmoid smoothing, spline smoothing, and numerical integration of the spline
     * to determine the new distribution of points.
     */
    void redistributePointsBasedOnSigmoidSmoothedAngularDifference() {
        // Step 1: Compute the angular difference of normals
        const auto angular_diff_values = computeAngularDifferenceOfNormals();

        std::vector<double> angular_diff, s_values_initial;
        for (const auto& angular_diff_value : angular_diff_values) {
            angular_diff.emplace_back(std::get<2>(angular_diff_value));  // Extract angular differences
            s_values_initial.emplace_back(std::get<1>(angular_diff_value));  // Extract corresponding s values
        }

        // Step 2: Apply sigmoid smoothing to the angular differences
        std::vector<double> sigmoid_smoothed_diff;
        sigmoid_smoothed_diff.reserve(angular_diff.size());
        for (const auto& diff : angular_diff) {
            constexpr double alpha = 1.0;
            double sigmoid_value = 0.5 + 1.0 / (1.0 + std::exp(-alpha * (diff - 0.5)));  // Sigmoid smoothing with added 0.5 to give better redistributing
            sigmoid_smoothed_diff.push_back(sigmoid_value);
        }

        // Step 3: Apply spline smoothing to the sigmoid-smoothed angular differences
        gsl_interp_accel* acc = gsl_interp_accel_alloc();
        gsl_spline* spline = gsl_spline_alloc(gsl_interp_cspline, s_values_initial.size());
        gsl_spline_init(spline, s_values_initial.data(), sigmoid_smoothed_diff.data(), s_values_initial.size());

        // Step 4: Compute the CDF using numerical integration of the spline
        std::vector<double> cdf(s_values_initial.size(), 0.0);

        auto integrand = [](const double x, void* params) -> double {
            // ReSharper disable once CppDeclarationHidesUncapturedLocal
            // ReSharper disable once CppLocalVariableMayBeConst
            auto* spline = static_cast<gsl_spline*>(params);
            return gsl_spline_eval(spline, x, gsl_interp_accel_alloc());  // Evaluate spline
        };

        gsl_function F;
        F.function = integrand;
        F.params = spline;

        int status = 0;
        size_t limit = 10000;

        // Use a while loop to handle memory allocation dynamically
        while (true) {
            // ReSharper disable once CppDeclarationHidesLocal
            gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(limit);  // Allocate workspace

            for (size_t i = 1; i < s_values_initial.size(); ++i) {
                double result = 0.0, error = 0.0;
                status = gsl_integration_qag(&F, s_values_initial[i-1], s_values_initial[i], 0, 1e-8, limit, GSL_INTEG_GAUSS15, workspace, &result, &error);  // Perform integration
                cdf[i] = cdf[i-1] + result;  // Accumulate the CDF values
            }

            gsl_integration_workspace_free(workspace);  // Free workspace

            if (status == GSL_ENOMEM) {
                limit *= 2;  // Double the workspace limit if out of memory
            } else {
                break;  // Exit loop if integration is successful
            }
        }

        // Normalize the CDF
        const double cdf_max = cdf.back();
        for (double& value : cdf) {
            value /= cdf_max;  // Normalize to [0, 1]
        }

        // Step 5: Distribute the points based on the CDF
        std::vector<double> distributed_t_values(s_values_initial.size());
        std::vector<double> target_cdf(s_values_initial.size());
        std::iota(target_cdf.begin(), target_cdf.end(), 0.0);
        for (auto& val : target_cdf) {
            val /= (target_cdf.size() - 1); // Normalize target_cdf to range [0, 1] NOLINT(*-narrowing-conversions)
        }

        for (size_t i = 0; i < target_cdf.size(); ++i) {
            if (auto it = std::ranges::lower_bound(cdf, target_cdf[i]); it != cdf.end()) {
                const size_t index = std::distance(cdf.begin(), it);
                distributed_t_values[i] = t_values[index];  // Map to new t values based on CDF
            } else {
                distributed_t_values[i] = t_values.back();  // Fallback to last value if necessary
            }
        }
        // Step 6: Update the boundary points, normals, and other values based on the new distribution
        updateBoundaryValues(distributed_t_values);
        // Free the GSL memory allocations
        gsl_spline_free(spline);
        gsl_interp_accel_free(acc);
    }

    /**
     * @brief Updates the boundary points, normals, t_values, and s_values based on new t_values.
     *
     * @param new_t_values The redistributed parametric values.
     */
    void updateBoundaryValues(const std::vector<double>& new_t_values) {
        points.clear();
        normals.clear();
        t_values = new_t_values;
        s_values.clear();
        curvatures.clear();

        // Recompute boundary values based on the new parameterization
        for (const auto& t : t_values) {
            points.push_back(boundary->curveParametrization(t));  // Update points
            normals.push_back(boundary->calculateNormal(t));  // Update normals
            s_values.push_back(boundary->calculateArcParameter(t));  // Update arc lengths
            curvatures.push_back(boundary->computeCurvature(t));  // Update curvatures
        }
    }

public:
    /**
     * @brief Constructor for BoundaryIntegral.
     * @param k The wavenumber.
     * @param b The scaling factor for point density.
     * @param boundary A shared pointer to an AbstractBoundary object.
     * @param kernelStrategy A shared pointer to an IKernelStrategy object.
     */
    BoundaryIntegral(const double k, const int b, const std::shared_ptr<AbstractBoundary>& boundary, std::shared_ptr<IKernelStrategy> kernelStrategy)
    : k(k), scalingFactor(b), boundary(boundary), kernelStrategy(std::move(kernelStrategy)) {
        totalArcLength = boundary->calculateArcLength();
        const int N = pointDensityPerWavelength(k);
        points.reserve(N);
        normals.reserve(N);
        t_values.reserve(N); // Prepare to store t values
        matrix.resize(N, N);
        initializePointsAndNormals_homogenous(N);
        redistributePointsBasedOnSigmoidSmoothedAngularDifference();
    }
    //TODO Add the gsl method for calcualting the moving averages
    /**
     * Get the k value for which we are doing BIM
     * @return k The wavenumber corresponding to this BIM
     */
    [[nodiscard]] double get_k() const {
        return k;
    }

    /**
     * Getter for the boundary/geometry of the problem that carries all geometrical information
     * @return The boundary of the problem
     */
    [[nodiscard]] std::shared_ptr<AbstractBoundary> getBoundary() const {
        return boundary;
    }
    /**
     * @brief Calculates the arc length of the boundary.
     * @return The total arc length.
     */
    [[nodiscard]] double calculateArcLength() const {
        return boundary->calculateArcLength();
    };
    /**
     * @brief Computes the curvature at a given parameter t.
     * @param t The parameter t.
     * @return The curvature.
     */
    [[nodiscard]] double computeCurvature(const double t) const {
        return boundary->computeCurvature(t);
    }

    /**
     * Computes the arclength at the parametrization t
     * @param t the parametrization parameter t
     * @return the arclength at the parametrization t
     */
    [[nodiscard]] double computeArcLength(const double t) const {
        return boundary->calculateArcParameter(t);
    }

    /**
     * @brief Determines the point density per wavelength.
     * @param k The wavenumber.
     * @return The point density.
     */
    [[nodiscard]] int pointDensityPerWavelength(const double k) const {
        return std::ceil(scalingFactor * totalArcLength * k / (2 * M_PI));
    }

    /**
     * @brief Initializes points and normals using a uniform distribution along the boundary.
     * @param N Number of points.
     */
    void initializePointsAndNormals_homogenous(const int N) {
        for (int i = 0; i < N; ++i) {
            double t = static_cast<double>(i) / N;  // Uniform parametric distribution
            points.push_back(boundary->curveParametrization(t));
            normals.push_back(boundary->calculateNormal(t));
            t_values.push_back(t);
            s_values.push_back(boundary->calculateArcParameter(t));
            curvatures.push_back(boundary->computeCurvature(t));
        }
    }

    /**
     * @brief Plots the angular differences between normals vs. arc length parameter s.
     *
     * @param ax The matplot::axes_handle for plotting.
     * @param aspect_ratio The aspect ratio for the plot.
     * @param plotSegmentLabels Whether to plot segment labels (default: false).
     * @param segmentLabels An optional vector of strings for labeling the segments (default: std::nullopt).
     */
    void plotAngularDifferencesBetweenNormals(
        const matplot::axes_handle &ax,
        const double aspect_ratio = 1.0,
        const bool plotSegmentLabels = false,
        const std::optional<std::vector<std::string>> &segmentLabels = std::nullopt) const
    {
        using namespace matplot;
        const auto angular_diff_values = computeAngularDifferenceOfNormals();
        const auto totalArcLength = calculateArcLength();

        std::vector<double> angular_diff, s_values;
        for (const auto &angular_diff_value : angular_diff_values) {
            angular_diff.emplace_back(std::get<2>(angular_diff_value));
            s_values.emplace_back(std::get<1>(angular_diff_value));
        }

        ax->scatter(s_values, angular_diff)->marker_size(0.5);
        ax->xlabel("Arc Length Parameter (s)");
        ax->ylabel("Angular Difference of Normals");

        // Plot the smoothed angular differences (using splines)
        gsl_interp_accel* acc = gsl_interp_accel_alloc();
        gsl_spline* spline = gsl_spline_alloc(gsl_interp_cspline, s_values.size());
        gsl_spline_init(spline, s_values.data(), angular_diff.data(), s_values.size());

        std::vector<double> smoothed_angle_diff(s_values.size());
        for (size_t i = 0; i < s_values.size(); ++i) {
            smoothed_angle_diff[i] = gsl_spline_eval(spline, s_values[i], acc);
        }

        ax->hold(matplot::on);  // Hold the plot to overlay the smoothed curve
        ax->plot(s_values, smoothed_angle_diff)->line_width(0.5).color("red");

        // Identify segment markers for CompositeBoundary and plot vertical lines
        if (plotSegmentLabels && std::dynamic_pointer_cast<CompositeBoundary>(getBoundary())) {
            std::vector<double> segmentMarkers;
            double accumulatedArcLength = 0.0;
            for (const auto& segment : std::dynamic_pointer_cast<CompositeBoundary>(getBoundary())->getSegments()) {
                accumulatedArcLength += segment->calculateArcLength();
                segmentMarkers.push_back(accumulatedArcLength);
            }

            // Plot vertical dashed lines at segment markers
            for (size_t i = 0; i < segmentMarkers.size(); ++i) {
                const double si = segmentMarkers[i];
                ax->plot({si, si}, {-0.99**std::ranges::max_element(angular_diff), 0.99**std::ranges::max_element(angular_diff)}, "--k");

                // Add segment labels if provided
                if (segmentLabels && i < segmentLabels->size()) {
                    const auto text_handle = ax->text(si - totalArcLength / 20, 0.9 * (*std::ranges::max_element(angular_diff)), segmentLabels->at(i));
                    text_handle->alignment(matplot::labels::alignment::center);
                    text_handle->font_size(10);
                }
            }
            ax->hold(matplot::off);
        }

        ax->xlim({0.0, totalArcLength});
        ax->ylim({0.0, 1.2**std::ranges::max_element(angular_diff)});
        ax->axes_aspect_ratio(aspect_ratio); // NOLINT(*-narrowing-conversions)

        gsl_spline_free(spline);  // Free the memory allocated for the spline
        gsl_interp_accel_free(acc);  // Free the memory allocated for the accelerator
    }

    /**
     * @brief Gets the parametric representation of the curve at parameter t.
     * @param t The parameter t.
     * @return The point on the curve.
     */
    [[nodiscard]] Point curveParametrization(const double t) const {
        return boundary->curveParametrization(t);
    }

    /**
     * @brief Gets the points on the boundary.
     * @return A vector of points on the boundary.
     */
    [[nodiscard]] std::vector<Point> getBoundaryPoints() const {
        return points;
    }

    /**
     * Simple getter for the parametriztion points of the boundary
     * @return A vector of t values for the point (they define the boundary points)
     */
    [[nodiscard]] std::vector<double> getParametrizationValues() const {
        return t_values;
    }

    /**
     * Gets the normals on the boundary
     * @return A vector of normals on the boundary
     */
    [[nodiscard]] std::vector<Point> getBoundaryNormals() const {
        return normals;
    }

    /**
     * A simple getter for the arclengths of the discretized boundary
     * @return the arclengths of the boundary
     */
    [[nodiscard]] std::vector<double> getArclengthValues() const {
        return s_values;
    }

    /**
     * A simple getter for the differences between consecutive arclengths of the discretized boundary,
     * including the difference between the first and last elements to account for periodic boundaries.
     * @return the differences between consecutive arclengths of the boundary
     */
    [[nodiscard]] std::vector<double> getArcDiffLengthValues() const {
        if (s_values.size() < 2) {
            throw std::runtime_error("The number of points is 2");
        }
        std::vector<double> arc_diff_lengths;
        arc_diff_lengths.reserve(s_values.size()); // Reserve space for all differences
        // Compute differences between consecutive elements
        std::ranges::transform(s_values | std::views::drop(1), s_values, std::back_inserter(arc_diff_lengths), [](const double next, const double current) {
            return next - current;
        });
        // Add the difference between the first and last element (closing the loop)
        arc_diff_lengths.push_back(std::abs(s_values.back() - s_values.front() - this->calculateArcLength()));
        return arc_diff_lengths;
    }

    /**
     * A simple getter for the curvatures of the discretized boundary
     * @return the curvatures of the boundary
     */
    [[nodiscard]] std::vector<double> getCurvatureValues() const {
        return curvatures;
    }

    /**
     * @brief Calculates the normal vector at parameter t.
     * @param t The parameter t.
     * @return The normal vector.
     */
    [[nodiscard]] Point calculateNormal(const double t) const {
        return boundary->calculateNormal(t);
    }
    /**
     * @brief Prints the discretization points and their corresponding normals.
     */
    void printDiscretizationPoints() const {
        for (size_t i = 0; i < points.size(); ++i) {
            std::cout << "Point number [" << i << "] on the boundary is: ["
                      << points[i].x << ", " << points[i].y << "], the normal is ["
                      << normals[i].x << ", " << normals[i].y << "], and the parametrization is: " << t_values[i] << std::endl;
        }
    }

    /**
     * @brief Constructs the Fredholm matrix for the boundary integral method.
     * @param beta The beta parameter for the modified Helmholtz kernel (default is 0.0 for standard kernel).
     * @return The constructed Fredholm matrix.
     */
    // ReSharper disable once CppMemberFunctionMayBeConst
    Eigen::MatrixXcd constructFredholmMatrix(const double beta = 0.0) {
        if (points.empty()) {
            std::cerr << "Point vector is empty. Matrix will not be computed." << std::endl;
            return {};
        }
        const auto delta_s = getArcDiffLengthValues();
        Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Identity(points.size(), points.size()); // NOLINT(*-narrowing-conversions)

        // Determine the number of threads to use
        const size_t num_threads = std::thread::hardware_concurrency();
        const size_t chunk_size = points.size() / num_threads;

        // Vector to store futures
        std::vector<std::future<void>> futures;

        // Function to compute a chunk of the matrix
        auto compute_chunk = [&](const size_t start, const size_t end) {
            for (size_t i = start; i < end; ++i) {
                for (size_t j = 0; j < points.size(); ++j) {
                    matrix(i, j) -= delta_s[i] * kernelStrategy->computeKernel(points[i], points[j], normals[i], k, t_values[i], beta); // NOLINT(*-narrowing-conversions)
                }
            }
        };

        // Launch asynchronous tasks
        for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            size_t start = thread_idx * chunk_size;
            size_t end = (thread_idx == num_threads - 1) ? points.size() : (thread_idx + 1) * chunk_size;
            futures.push_back(std::async(std::launch::async, compute_chunk, start, end));
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.get();
        }

        return matrix;
    }

    /**
     * @brief Constructs the first derivative of the Fredholm matrix for the boundary integral method with respect to the wavenumber k.
     *
     * This method calculates the first derivative of the Fredholm matrix elements using the derivative of the kernel function.
     * The resulting matrix represents the change in the Fredholm matrix with respect to small changes in the wavenumber k.
     *
     * @param beta The beta parameter for the modified Helmholtz kernel (default is 0.0 for the standard kernel).
     * @param useDefaultHelmholtz If we have the helmholtz equation set this to true. It uses the analyric epxpression for the derivative
     * @return The constructed Fredholm matrix derivative.
     * @exception std::runtime_error Thrown if the point vector is empty.
     */
    Eigen::MatrixXcd constructFredholmMatrixDerivative(const double beta = 0.0, const bool useDefaultHelmholtz = true) {
        if (points.empty()) {
            std::cerr << "Point vector is empty. Matrix will not be computed." << std::endl;
            return {};
        }
        const auto delta_s = getArcDiffLengthValues();
        matrixDerivative = Eigen::MatrixXcd::Zero(points.size(), points.size()); // NOLINT(*-narrowing-conversions)
        // Use hardware concurrency to determine the number of threads
        const auto num_threads = std::thread::hardware_concurrency();
        const auto chunk_size = points.size() / num_threads;
        std::vector<std::future<void>> futures;

        // Define the function to compute each chunk
        auto compute_chunk = [this, delta_s, beta, useDefaultHelmholtz](const size_t start, const size_t end) {
            for (size_t i = start; i < end; ++i) {
                for (size_t j = 0; j < points.size(); ++j) {
                    matrixDerivative(i, j) -= delta_s[i] * kernelStrategy->computeKernelDerivative(points[i], points[j], normals[i], this->k, t_values[i], beta, useDefaultHelmholtz); // NOLINT(*-narrowing-conversions)
                }
            }
        };

        // Launch asynchronous tasks
        for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            size_t start = thread_idx * chunk_size;
            size_t end = (thread_idx == num_threads - 1) ? points.size() : (thread_idx + 1) * chunk_size;
            futures.push_back(std::async(std::launch::async, compute_chunk, start, end));
        }

        // Wait for all tasks to complete
        for (auto &future : futures) {
            future.get();
        }

        return matrixDerivative;
    }

    /**
     * @brief Constructs the second derivative of the Fredholm matrix for the boundary integral method with respect to the wavenumber k.
     *
     * This method calculates the second derivative of the Fredholm matrix elements using the second derivative of the kernel function.
     * The resulting matrix represents the change in the first derivative of the Fredholm matrix with respect to small changes in the wavenumber k.
     *
     * @param beta The beta parameter for the modified Helmholtz kernel (default is 0.0 for the standard kernel).
     * @param useDefaultHelmholtz If we have the helmholtz equation set this to true. It uses the analyric epxpression for the derivative
     * @return The constructed Fredholm matrix second derivative.
     * @exception std::runtime_error Thrown if the point vector is empty.
     */
    Eigen::MatrixXcd constructFredholmMatrixSecondDerivative(const double beta = 0.0, const bool useDefaultHelmholtz = true) {
        if (points.empty()) {
            std::cerr << "Point vector is empty. Matrix will not be computed." << std::endl;
            return {};
        }
        const auto delta_s = getArcDiffLengthValues();
        matrixSecondDerivative = Eigen::MatrixXcd::Zero(points.size(), points.size()); // NOLINT(*-narrowing-conversions)
        // Use hardware concurrency to determine the number of threads
        const auto num_threads = std::thread::hardware_concurrency();
        const auto chunk_size = points.size() / num_threads;
        std::vector<std::future<void>> futures;

        // Define the function to compute each chunk
        auto compute_chunk = [this, delta_s, beta, useDefaultHelmholtz](const size_t start, const size_t end) {
            for (size_t i = start; i < end; ++i) {
                for (size_t j = 0; j < points.size(); ++j) {
                    matrixSecondDerivative(i, j) -= delta_s[i] * kernelStrategy->computeKernelSecondDerivative(points[i], points[j], normals[i], this->k, t_values[i], beta, useDefaultHelmholtz); // NOLINT(*-narrowing-conversions)
                }
            }
        };

        // Launch asynchronous tasks
        for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            size_t start = thread_idx * chunk_size;
            size_t end = (thread_idx == num_threads - 1) ? points.size() : (thread_idx + 1) * chunk_size;
            futures.push_back(std::async(std::launch::async, compute_chunk, start, end));
        }

        // Wait for all tasks to complete
        for (auto &future : futures) {
            future.get();
        }
        return matrixSecondDerivative;
    }

    /**
     * @brief Constructs a matrix of cos_phi values for the discretization points.
     *
     * This method constructs an Eigen matrix where each element (i, j) is the cos_phi value
     * computed between the i-th and j-th discretization points using the normal vector at the i-th point.
     *
     * @return Eigen::MatrixXcd The matrix of cos_phi values.
     */
    [[nodiscard]] Eigen::MatrixXcd constructCosPhiMatrix() const {
        const size_t n = points.size();
        Eigen::MatrixXcd cosPhiMatrix(n, n);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                cosPhiMatrix(i, j) = std::complex<double>(kernelStrategy->computeCosPhi(points[i], points[j], normals[i], t_values[i]), 0.0); // NOLINT(*-narrowing-conversions)
            }
        }

        return cosPhiMatrix;
    }

    /**
     * @brief Constructs a matrix of Hankel function values for the discretization points.
     *
     * This method constructs an Eigen matrix where each element (i, j) is the Hankel function
     * value computed between the i-th and j-th discretization points for the given wavenumber k.
     *
     * @return Eigen::MatrixXcd The matrix of Hankel function values as complex numbers.
     */
    [[nodiscard]] Eigen::MatrixXcd constructHankelMatrix() const {
        const size_t n = points.size();
        Eigen::MatrixXcd hankelMatrix(n, n);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                hankelMatrix(i, j) = kernelStrategy->computeHankel(points[i], points[j], k); // NOLINT(*-narrowing-conversions)
            }
        }
        return hankelMatrix;
    }

    /**
     * @brief Constructs the Fredholm matrix with detailed information for the boundary integral method.
     * @param beta The beta parameter for the modified Helmholtz kernel (default is 0.0 for standard kernel).
     * @return The constructed Fredholm matrix as a matrix of strings.
     */
    [[nodiscard]] Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> constructFredholmMatrixString(const double beta = 0.0) const {
        if (points.empty()) {
            std::cerr << "Point vector is empty. Matrix will not be computed." << std::endl;
            return {};
        }
        const auto delta_s = getArcDiffLengthValues();
        Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> fredholmMatrix(points.size(), points.size());

        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = 0; j < points.size(); ++j) {
                std::complex<double> value = (i == j ? 1.0 : 0.0) - delta_s[i] * kernelStrategy->computeKernel(points[i], points[j], normals[i], k, t_values[i], beta);

                std::stringstream ss;
                ss.precision(8);
                ss << std::fixed << "Value=" << value << ", p1={" << points[i].x << ", " << points[i].y << "}, "
                   << "p2={" << points[j].x << ", " << points[j].y << "}, "
                   << "n={" << normals[i].x << ", " << normals[i].y << "}";

                fredholmMatrix(i, j) = ss.str(); // NOLINT(*-narrowing-conversions)
            }
        }

        return fredholmMatrix;
    }

    /**
     * @brief Constructs the first derivative of the Fredholm matrix with detailed information for the boundary integral method.
     * @param beta The beta parameter for the modified Helmholtz kernel (default is 0.0 for standard kernel).
     * @return The constructed first derivative of the Fredholm matrix as a matrix of strings.
     */
    [[nodiscard]] Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> constructFredholmMatrixDerivativeString(const double beta = 0.0) const {
        if (points.empty()) {
            std::cerr << "Point vector is empty. Matrix will not be computed." << std::endl;
            return {};
        }
        const auto delta_s = getArcDiffLengthValues();
        Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> fredholmMatrixDerivative(points.size(), points.size());

        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = 0; j < points.size(); ++j) {
                std::complex<double> value = -delta_s[i] * kernelStrategy->computeKernelDerivative(points[i], points[j], normals[i], k, t_values[i], beta, true);

                std::stringstream ss;
                ss.precision(8);
                ss << std::fixed << "Value=" << value << ", p1={" << points[i].x << ", " << points[i].y << "}, "
                   << "p2={" << points[j].x << ", " << points[j].y << "}, "
                   << "n={" << normals[i].x << ", " << normals[i].y << "}";

                fredholmMatrixDerivative(i, j) = ss.str(); // NOLINT(*-narrowing-conversions)
            }
        }

        return fredholmMatrixDerivative;
    }

    /**
     * @brief Constructs the second derivative of the Fredholm matrix with detailed information for the boundary integral method.
     * @param beta The beta parameter for the modified Helmholtz kernel (default is 0.0 for standard kernel).
     * @return The constructed second derivative of the Fredholm matrix as a matrix of strings.
     */
    [[nodiscard]] Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> constructFredholmMatrixSecondDerivativeString(const double beta = 0.0) const {
        if (points.empty()) {
            std::cerr << "Point vector is empty. Matrix will not be computed." << std::endl;
            return {};
        }
        const auto delta_s = getArcDiffLengthValues();
        Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> fredholmMatrixSecondDerivative(points.size(), points.size());

        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = 0; j < points.size(); ++j) {
                std::complex<double> value = -delta_s[i] * kernelStrategy->computeKernelSecondDerivative(points[i], points[j], normals[i], k, t_values[i], beta, true);

                std::stringstream ss;
                ss.precision(8);
                ss << std::fixed << "Value=" << value << ", p1={" << points[i].x << ", " << points[i].y << "}, "
                   << "p2={" << points[j].x << ", " << points[j].y << "}, "
                   << "n={" << normals[i].x << ", " << normals[i].y << "}";

                fredholmMatrixSecondDerivative(i, j) = ss.str(); // NOLINT(*-narrowing-conversions)
            }
        }

        return fredholmMatrixSecondDerivative;
    }

    /**
     * @brief Constructs the combined first and second derivatives of the Fredholm matrix with detailed information for the boundary integral method.
     * @param beta The beta parameter for the modified Helmholtz kernel (default is 0.0 for standard kernel).
     * @return The constructed combined derivatives of the Fredholm matrix as a matrix of strings.
     */
    [[nodiscard]] Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> constructCombinedFredholmMatrixDerivativeString(const double beta = 0.0) const {
        if (points.empty()) {
            std::cerr << "Point vector is empty. Matrix will not be computed." << std::endl;
            return {};
        }
        const auto delta_s = getArcDiffLengthValues();
        Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> combinedFredholmMatrixDerivative(points.size(), points.size());

        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = 0; j < points.size(); ++j) {
                std::complex<double> value = (i == j ? 1.0 : 0.0) - delta_s[i] * kernelStrategy->computeKernel(points[i], points[j], normals[i], k, t_values[i], beta);
                std::complex<double> firstDerivativeValue = -delta_s[i] * kernelStrategy->computeKernelDerivative(points[i], points[j], normals[i], k, t_values[i], beta, true);
                std::complex<double> secondDerivativeValue = -delta_s[i] * kernelStrategy->computeKernelSecondDerivative(points[i], points[j], normals[i], k, t_values[i], beta, true);

                std::stringstream ss;
                ss.precision(8);
                ss << std::fixed << "Fredholm Matrix Value=" << value << ", First Derivative Value=" << firstDerivativeValue << ", Second Derivative Value=" << secondDerivativeValue
                   << ", p1={" << points[i].x << ", " << points[i].y << "}, "
                   << "p2={" << points[j].x << ", " << points[j].y << "}, "
                   << "n={" << normals[i].x << ", " << normals[i].y << "}";

                combinedFredholmMatrixDerivative(i, j) = ss.str(); // NOLINT(*-narrowing-conversions)
            }
        }

        return combinedFredholmMatrixDerivative;
    }

    /**
     * @brief Constructs the kernel matrix evaluated at the given wavenumber k.
     *
     * This method constructs a matrix where each element is a string containing the result of evaluating
     * the kernel strategy between pairs of boundary points and the coordinates of those points.
     *
     * @return The constructed kernel matrix as a matrix of strings.
     */
    [[nodiscard]] Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> constructKernelMatrixString(const double beta = 0.0) const {
        const size_t num_points = points.size();
        Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> kernelMatrix(num_points, num_points);

        for (size_t i = 0; i < num_points; ++i) {
            for (size_t j = 0; j < num_points; ++j) {
                std::complex<double> value = kernelStrategy->computeKernel(points[i], points[j], normals[i], k, t_values[i], beta);

                std::stringstream ss;
                ss << "Value=" << value << ", p1={" << points[i].x << ", " << points[i].y << "}, "
                   << "p2={" << points[j].x << ", " << points[j].y << "}, "
                   << "n={" << normals[i].x << ", " << normals[i].y << "}";

                kernelMatrix(i, j) = ss.str(); // NOLINT(*-narrowing-conversions)
            }
        }
        return kernelMatrix;
    }

    /**
     * @brief Constructs a matrix of strings containing cos_phi values and the coordinates of points and normals.
     *
     * This method constructs an Eigen matrix where each element (i, j) is a string
     * containing the cos_phi value, the coordinates of the i-th and j-th discretization points,
     * and the normal vector at the i-th point.
     *
     * @return Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> The matrix of strings.
     */
    [[nodiscard]] Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> constructCosPhiMatrixString() const {
        const size_t n = points.size();
        Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> cosPhiMatrix(n, n);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                const double cos_phi = kernelStrategy->computeCosPhi(points[i], points[j], normals[i], t_values[i]);

                std::stringstream ss;
                ss << "cos_phi=" << cos_phi << ", p1={" << points[i].x << ", " << points[i].y << "}, "
                   << "p2={" << points[j].x << ", " << points[j].y << "}, "
                   << "n={" << normals[i].x << ", " << normals[i].y << "}";

                cosPhiMatrix(i, j) = ss.str(); // NOLINT(*-narrowing-conversions)
            }
        }

        return cosPhiMatrix;
    }

    /**
     * @brief Constructs a matrix of Hankel function values for the discretization points.
     *
     * This method constructs an Eigen matrix where each element (i, j) is a string
     * containing the Hankel function value and the coordinates of the i-th and j-th discretization points.
     *
     * @return Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> The matrix of strings.
     */
    [[nodiscard]] Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> constructHankelMatrixString() const {
        const size_t n = points.size();
        Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> hankelMatrix(n, n);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                std::complex<double> hankelValue = kernelStrategy->computeHankel(points[i], points[j], k);

                std::stringstream ss;
                ss << "Hankel=" << hankelValue << ", p1={" << points[i].x << ", " << points[i].y << "}, "
                   << "p2={" << points[j].x << ", " << points[j].y << "}";

                hankelMatrix(i, j) = ss.str(); // NOLINT(*-narrowing-conversions)
            }
        }

        return hankelMatrix;
    }

    /**
     * @brief Constructs the Fredholm matrix using cos_phi and Hankel function matrices, returning a matrix of strings.
     * @return The constructed Fredholm matrix as a matrix of strings.
     */
    [[nodiscard]] Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> constructFredholmMatrixFromCosPhiMatrixAndHankelMatrix() const {
        if (points.empty()) {
            std::cerr << "Point vector is empty. Matrix will not be computed." << std::endl;
            return {};
        }

        const size_t n = points.size();
        const auto delta_s = getArcDiffLengthValues();

        // Construct cos_phi and Hankel matrices
        Eigen::MatrixXcd cosPhiMatrix = constructCosPhiMatrix();
        Eigen::MatrixXcd hankelMatrix = constructHankelMatrix();

        // Initialize the Fredholm matrix of strings
        Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic> fredholmMatrix(n, n); // NOLINT(*-narrowing-conversions)

        // Fill the Fredholm matrix using the cos_phi and Hankel function matrices
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                std::stringstream ss;

                std::complex<double> value;

                if (i == j) {
                    value = std::complex<double>(1.0, 0.0);
                } else {
                    // Compute the Fredholm matrix element
                    std::complex<double> term = -delta_s[i] * (std::complex<double>(0.0, k) / 2.0) * cosPhiMatrix(i, j) * hankelMatrix(i, j); // NOLINT(*-narrowing-conversions)
                    value = - term;
                }

                ss << "p1={" << points[i].x << ", " << points[i].y << "}, "
                   << "p2={" << points[j].x << ", " << points[j].y << "}, "
                   << "n={" << normals[i].x << ", " << normals[i].y << "}, "
                   << "cos_phi=" << cosPhiMatrix(i, j).real() << ", " // NOLINT(*-narrowing-conversions)
                   << "hankel=" <<(i == j ? std::complex<double>(0.0, 0.0) : hankelMatrix(i, j)) << ", " // NOLINT(*-narrowing-conversions)
                   << "Fredholm value=" << value; // NOLINT(*-narrowing-conversions)

                fredholmMatrix(i, j) = ss.str(); // NOLINT(*-narrowing-conversions)
            }
        }
        return fredholmMatrix;
    }

    /**
    * @brief Prints the discretization points and their normals in a simple list format.
    */
    void printDiscretizationPointsAndNormals() const {
        std::cout << "Discretization Points and Normals:\n";
        for (size_t i = 0; i < points.size(); ++i) {
            std::cout << "[" << points[i].x << ", " << points[i].y << "] ";
            std::cout << "Normal: [" << normals[i].x << ", " << normals[i].y << "]\n";
        }
    }

    /**
     * @brief Computes the singular value decomposition (SVD) of the Fredholm matrix.
     *
     * This method computes the SVD of the Fredholm matrix using BDCSVD. By default, it does not compute the unitary matrices `U` and `V`.
     * If the computation options are provided to compute `U` and `V`, it sorts the singular values and the corresponding eigenvectors.
     *
     * Detailed Steps:
     * 1. Constructs the Fredholm matrix using the `constructFredholmMatrix` method.
     * 2. Performs SVD on the Fredholm matrix using the `Eigen::BDCSVD` class.
     * 3. Extracts the singular values from the SVD.
     * 4. Pairs each singular value with its corresponding index.
     * 5. Sorts the singular values and their indices.
     * 6. If options indicate the computation of unitary matrices `U` and `V`:
     *    - Extracts the unitary matrix `U`.
     *    - Sorts the corresponding eigenvectors based on the sorted singular values.
     *    - Returns the sorted singular values and the corresponding sorted eigenvectors.
     * 7. If options do not indicate the computation of unitary matrices `U` and `V`, only the sorted singular values are returned.
     *
     * @param options Options for computing the SVD. Default is 0, meaning no unitary matrices `U` or `V` are computed.
     * @return A pair consisting of:
     *         - Eigen::VectorXd: Sorted singular values.
     *         - Eigen::MatrixXcd: Corresponding sorted eigenvectors if computed, otherwise an empty matrix.
     */
    std::pair<Eigen::VectorXd, Eigen::MatrixXcd> computeSVD(auto options = 0) {
        Eigen::setNbThreads(std::thread::hardware_concurrency()); // NOLINT(*-narrowing-conversions)
    try {
        // Step 1: Construct the Fredholm matrix
        auto F = constructFredholmMatrix();

        // Step 2: Perform SVD on the Fredholm matrix
        const Eigen::BDCSVD<Eigen::MatrixXcd> svd(F, options);

        // Step 3: Extract the singular values
        Eigen::VectorXd singularValues = svd.singularValues();

        // Step 4: Pair each singular value with its corresponding index
        std::vector<std::pair<double, int>> indexedSingularValues(singularValues.size());
        for (int i = 0; i < singularValues.size(); i++) {
            indexedSingularValues[i] = {singularValues[i], i};
        }

        // Step 5: Sort the singular values and corresponding eigenvectors
        std::sort(indexedSingularValues.begin(), indexedSingularValues.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first < b.first;
        });

        Eigen::VectorXd sortedSingularValues(singularValues.size());
        Eigen::MatrixXcd sortedEigenvectors;
        if (options != 0) {
            // Extract the unitary matrix U
            Eigen::MatrixXcd eigenvectors = svd.matrixU();
            sortedEigenvectors.resize(eigenvectors.rows(), eigenvectors.cols());

            // The second element of the pairs in indexedSingularValues remains unchanged and represents the original index of the singular value in the singularValues vector
            for (int i = 0; i < indexedSingularValues.size(); i++) {
                sortedSingularValues(i) = indexedSingularValues[i].first;
                sortedEigenvectors.col(i) = eigenvectors.col(indexedSingularValues[i].second);
            }
        } else {
            for (int i = 0; i < indexedSingularValues.size(); i++) {
                sortedSingularValues(i) = indexedSingularValues[i].first;
            }
        }

        // Step 7: Return the sorted singular values and corresponding eigenvectors if computed
        return {sortedSingularValues, sortedEigenvectors};

        } catch (const std::exception& e) {
            std::cerr << "Exception during matrix computation: " << e.what() << std::endl;
            return {};
        }
    }

    /**
     * @brief Computes the singular value decomposition (SVD) of the Fredholm matrix with a beta variation.
     *
     * This method computes the SVD of the Fredholm matrix with an additional beta parameter using BDCSVD. By default, it does not compute the unitary matrices `U` and `V`.
     * If the computation options are provided to compute `U` and `V`, it sorts the singular values and the corresponding eigenvectors.
     *
     * Detailed Steps:
     * 1. Constructs the Fredholm matrix using the `constructFredholmMatrix` method with the beta parameter.
     * 2. Performs SVD on the Fredholm matrix using the `Eigen::BDCSVD` class.
     * 3. Extracts the singular values from the SVD.
     * 4. Pairs each singular value with its corresponding index.
     * 5. Sorts the singular values and their indices.
     * 6. If options indicate the computation of unitary matrices `U` and `V`:
     *    - Extracts the unitary matrix `U`.
     *    - Sorts the corresponding eigenvectors based on the sorted singular values.
     *    - Returns the sorted singular values and the corresponding sorted eigenvectors.
     * 7. If options do not indicate the computation of unitary matrices `U` and `V`, only the sorted singular values are returned.
     *
     * @param options Options for computing the SVD. Default is 0, meaning no unitary matrices `U` or `V` are computed.
     * @param beta The beta parameter for the modified Helmholtz kernel.
     * @return A pair consisting of:
     *         - Eigen::VectorXd: Sorted singular values.
     *         - Eigen::MatrixXcd: Corresponding sorted eigenvectors if computed, otherwise an empty matrix.
     */
    std::pair<Eigen::VectorXd, Eigen::MatrixXcd> computeSVDWithBeta(auto options = 0, const double beta = 1.0) {
        Eigen::setNbThreads(std::thread::hardware_concurrency()); // NOLINT(*-narrowing-conversions)
        try {
            // Step 1: Construct the Fredholm matrix with the beta parameter
            auto F = constructFredholmMatrix(beta);

            // Step 2: Perform SVD on the Fredholm matrix
            const Eigen::BDCSVD<Eigen::MatrixXcd> svd(F, options);

            // Step 3: Extract the singular values
            Eigen::VectorXd singularValues = svd.singularValues();

            // Step 4: Pair each singular value with its corresponding index
            std::vector<std::pair<double, int>> indexedSingularValues(singularValues.size());
            for (int i = 0; i < singularValues.size(); i++) {
                indexedSingularValues[i] = {singularValues[i], i};
            }

            // Step 5: Sort the singular values and corresponding eigenvectors
            std::sort(indexedSingularValues.begin(), indexedSingularValues.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                return a.first < b.first;
            });

            Eigen::VectorXd sortedSingularValues(singularValues.size());
            Eigen::MatrixXcd sortedEigenvectors;
            if (options != 0) {
                // Extract the unitary matrix U
                Eigen::MatrixXcd eigenvectors = svd.matrixU();
                sortedEigenvectors.resize(eigenvectors.rows(), eigenvectors.cols());

                // The second element of the pairs in indexedSingularValues remains unchanged and represents the original index of the singular value in the singularValues vector
                for (int i = 0; i < indexedSingularValues.size(); i++) {
                    sortedSingularValues(i) = indexedSingularValues[i].first;
                    sortedEigenvectors.col(i) = eigenvectors.col(indexedSingularValues[i].second);
                }
            } else {
                for (int i = 0; i < indexedSingularValues.size(); i++) {
                    sortedSingularValues(i) = indexedSingularValues[i].first;
                }
            }

            // Step 7: Return the sorted singular values and corresponding eigenvectors if computed
            return {sortedSingularValues, sortedEigenvectors};
        } catch (const std::exception& e) {
            std::cerr << "Exception during matrix computation for k=" << k << " with beta=" << beta << ": " << e.what() << std::endl;
            return {};
        }
    }

    /**
     * @brief Computes the determinant of the matrix for the boundary integral method.
     *
     * This method constructs the matrix \(A_k\) for the boundary integral equation by evaluating
     * the kernel function at each pair of discretized boundary points. The matrix \(A_k\) represents
     * the interactions between boundary points as mediated by the Helmholtz kernel, and incorporates
     * the discretization length \(\Delta s\).
     *
     * The steps involved are:
     * 1. Compute the discretization length \(\Delta s\) as \( \Delta s = \frac{L}{N} \), where \(L\) is the
     *    total arc length of the boundary and \(N\) is the number of discretization points.
     * 2. Iterate over all pairs of discretized boundary points.
     * 3. For each pair \((p_i, p_j)\), compute the kernel value using the provided `IKernelStrategy`.
     * 4. Form the matrix \(A_k\) as \( A_{ij} = \delta_{ij} - \Delta s \cdot Q_k(s_i, s_j) \).
     * 5. Compute the determinant of the matrix \(A_k\).
     *
     * @return The determinant of the matrix \(A_k\).
     *
     * @details
     * The determinant of the matrix \(A_k\) is useful for finding the eigenvalues of the integral equation.
     * The real zeros of the determinant as a function of \(k\) provide approximations to the eigenvalues.
     */
    std::complex<double> computeMatrixDeterminant(const bool showCout) {
        try {
            const auto F = constructFredholmMatrix();
            const std::complex<double> determinant = F.determinant();
            if (showCout) {
                std::cout << "Computed determinant for k=" << k << ": " << determinant << std::endl;
            }
            return determinant;
        } catch (const std::exception& e) {
            std::cerr << "Exception during determinant computation for k=" << k << ": " << e.what() << std::endl;
            return 0.0;
        }
    }

    /**
     * @brief Computes the determinant of the matrix for the boundary integral method with a beta variation.
     *
     * This method constructs the matrix \(A_k\) for the boundary integral equation by evaluating
     * the kernel function at each pair of discretized boundary points, incorporating an additional
     * parameter \(\beta\). The matrix \(A_k\) represents the interactions between boundary points
     * as mediated by the modified Helmholtz kernel, and incorporates the discretization length \(\Delta s\).
     *
     * The steps involved are:
     * 1. Compute the discretization length \(\Delta s\) as \( \Delta s = \frac{L}{N} \), where \(L\) is the
     *    total arc length of the boundary and \(N\) is the number of discretization points.
     * 2. Iterate over all pairs of discretized boundary points.
     * 3. For each pair \((p_i, p_j)\), compute the kernel value using the provided `IKernelStrategy`,
     *    including the \(\beta\) term.
     * 4. Form the matrix \(A_k\) as \( A_{ij} = \delta_{ij} - \Delta s \cdot Q_k(s_i, s_j) \).
     * 5. Compute the determinant of the matrix \(A_k\).
     *
     * @param beta The beta parameter for the modified Helmholtz kernel.
     * @param showCout To show the cout of the determinants
     * @return The determinant of the matrix \(A_k\).
     *
     * @details
     * The determinant of the matrix \(A_k\) with the \(\beta\) parameter is useful for finding the eigenvalues
     * of the integral equation with more complex boundary interactions. The real zeros of the determinant as a function
     * of \(k\) provide approximations to the eigenvalues.
     */
     std::complex<double> computeMatrixDeterminantWithBeta(const double beta, const bool showCout) {
        try {
            const auto F = constructFredholmMatrix(beta);
            const std::complex<double> determinant = F.determinant();
            if (showCout) {
                std::cout << "Computed determinant for k=" << k << " with beta=" << beta << ": " << determinant << std::endl;
            }
            return determinant;
        } catch (const std::exception& e) {
            std::cerr << "Exception during determinant computation for k=" << k << " with beta=" << beta << ": " << e.what() << std::endl;
            return 0.0;
        }
    }

    [[nodiscard]] Eigen::VectorXd getSortedSingularValues() const {
        return sortedSingularValues;
    }

    [[nodiscard]] Eigen::MatrixXcd getSortedEigenvectors() const {
        return sortedEigenvectors;
    }

    /**
     * @brief Checks the validity of the interior Dirichlet problem by finding the maximum value of the first layer potential.
     *
     * This method calculates the maximum value of the first layer potential for a given eigenvector.
     * The first layer potential is computed by summing the contributions from the Green's function
     * evaluated at all pairs of boundary points, except for the diagonal elements where the Green's function
     * is singular. The logic handles the singularity by setting the contribution to zero at those points.
     *
     * @param eigenvectorToCheck A tuple containing the eigenvalue, wave number, and the corresponding eigenvector (complex values).
     * @return The maximum value of the first layer potential as a real number, representing the maximum "error" in the solution.
     */
     [[nodiscard]] double checkValidityOfInteriorProblem(const std::tuple<double, double, Eigen::VectorXcd>& eigenvectorToCheck) const {
        const auto eigenvec = std::get<2>(eigenvectorToCheck);

        // Collect all partial results synchronously
        std::vector<std::complex<double>> partialResults;
        partialResults.reserve(points.size());

        if (points.empty()) {
            std::cerr << "Error: points vector is empty." << std::endl;
            return 0.0;
        }

        for (int i = 0; i < points.size(); ++i) {
            std::complex<double> partial(0.0, 0.0);
            for (int j = 0; j < points.size(); ++j) {
                if (i != j) {
                    partial += eigenvec[j] * kernelStrategy->computeGreensFunction(points[i], points[j], k);
                } else {
                    partial += std::complex<double>(0, 0);
                }
            }
            partialResults.push_back(partial);
        }
        // Find the maximum partial result by magnitude
        std::complex<double> maxResult(0.0, 0.0);
        std::cout << "Partial results vector has size: " << partialResults.size() << std::endl;
        for (const auto& partial : partialResults) {
            if (std::abs(partial) > std::abs(maxResult)) {
                maxResult = partial;
            }
        }
        // Return the value of the maximum partial result
        return std::abs(maxResult);
    }
};

/**
 * @class KRangeSolver
 * @brief Computes the SVD and determinants of matrices (Fredholm) for a range of wavenumbers.
 *
 * This class computes the Singular Value Decomposition (SVD) for a range of wavenumbers (k values)
 * using the boundary integral method. It uses the provided kernel strategy for computation.
 */
class KRangeSolver {
    std::vector<double> k_values;
    std::shared_ptr<AbstractBoundary> boundary;
    std::shared_ptr<IKernelStrategy> kernelStrategy;
    std::vector<std::complex<double>> determinants;
    std::vector<std::vector<std::complex<double>>> determinantsWithBeta; // Each sub-vector corresponds to a beta value
    std::vector<Eigen::VectorXd> svd_results; // Only singular values
    std::vector<std::vector<Eigen::VectorXd>> svdResultsWithBeta; // Only singular values with beta variation
    std::vector<std::pair<double, Eigen::VectorXd>> refined_svd_results; // Only singular values
    const int SIZE_K;
    int scalingFactor;
public:
    /**
     * @brief Constructor for KRangeSolver.
     * @param k_min The minimum wavenumber.
     * @param k_max The maximum wavenumber.
     * @param SIZE_K The number of wavenumber values.
     * @param b The scaling factor for point density.
     * @param boundary A shared pointer to an AbstractBoundary object.
     * @param kernelStrategy A shared pointer to an IKernelStrategy object.
     */
    // ReSharper disable once CppPossiblyUninitializedMember
    // ReSharper disable once CppParameterMayBeConst
    KRangeSolver(const double k_min, const double k_max, const int SIZE_K, const int b,
                     const std::shared_ptr<AbstractBoundary>& boundary, std::shared_ptr<IKernelStrategy> kernelStrategy)
            : boundary(boundary), kernelStrategy(std::move(kernelStrategy)), SIZE_K(SIZE_K), scalingFactor(b) {
        const double dk = (k_max - k_min) / (SIZE_K - 1);
        k_values.resize(SIZE_K);
        for (int i = 0; i < SIZE_K; ++i) {
            k_values[i] = k_min + i * dk;
        }
        determinants.resize(SIZE_K);
        svd_results.resize(SIZE_K);
    }

    // *****************************************************************************//
    // FOR THIS CLASS BELOW ARE METHODS WHERE WE DO NOT SAVE ANY RESULTS TO .CSV FILES
    // *****************************************************************************//

    /**
     * @brief Computes the singular value decomposition (SVD) for a range of wavenumbers.
     *
     * This method computes the SVD of the Fredholm matrix for a range of wavenumbers using parallel computation.
     * By default, it does not compute the unitary matrices `U` and `V` for the SVD.
     *
     * Detailed Steps:
     * 1. Initializes futures for parallel computation using hardware concurrency.
     * 2. For each wavenumber, asynchronously computes the SVD of the Fredholm matrix.
     * 3. Extracts the singular values from the SVD.
     * 4. Collects the results from the futures and stores them in `svd_results`.
     */
    void computeSingularValueDecomposition(auto options = 0) {
        std::vector<std::future<std::vector<Eigen::VectorXd>>> futures;

        // Use hardware concurrency to determine the number of threads
        const auto num_threads = std::thread::hardware_concurrency();
        const auto chunk_size = k_values.size() / num_threads;

        for (size_t i = 0; i < k_values.size(); i += chunk_size) {
            futures.emplace_back(std::async(std::launch::async, [this, i, chunk_size, options]() {
                std::vector<Eigen::VectorXd> local_results;
                for (size_t j = i; j < i + chunk_size && j < k_values.size(); ++j) {
                    const double k_value = k_values[j];
                    auto bi = new BoundaryIntegral(k_value, scalingFactor, boundary, kernelStrategy);
                    auto result = bi->computeSVD(options).first; // Only store singular values
                    local_results.push_back(result);
                    delete bi;
                }
                std::cout << "Completed chunk from index " << i << " to " << i + chunk_size - 1 << std::endl;
                return local_results;
            }));
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Small delay to manage memory usage
        }

        // Collect the results from the futures
        svd_results.clear();
        for (auto& future : futures) {
            auto local_results = future.get();
            svd_results.insert(svd_results.end(), local_results.begin(), local_results.end());
        }
        std::cout << "Completed SVD computation for all k values." << std::endl;
    }

    /**
     * @brief Computes the singular value decomposition with beta variation for the range of wavenumbers.
     *
     * This method computes the SVD of the Fredholm matrix with an additional beta parameter for a range of wavenumbers
     * using parallel computation. By default, it does not compute the unitary matrices `U` and `V` for the SVD.
     *
     * Detailed Steps:
     * 1. Initializes futures for parallel computation using hardware concurrency.
     * 2. For each beta value and wavenumber, asynchronously computes the SVD of the Fredholm matrix.
     * 3. Extracts the singular values from the SVD.
     * 4. Collects the results from the futures and stores them in `svdResultsWithBeta`.
     *
     * @param beta_values A vector of beta values.
     * @param options Options for computing the SVD. Default is 0, meaning no unitary matrices `U` or `V` are computed.
     */
    void computeSingularValueDecompositionWithBetaVariation(const std::vector<double>& beta_values, auto options = 0) {
        svdResultsWithBeta.resize(beta_values.size(), std::vector<Eigen::VectorXd>(SIZE_K));

        // Use hardware concurrency to determine the number of threads
        const auto num_threads = std::thread::hardware_concurrency();
        const auto chunk_size = k_values.size() / num_threads;

        for (size_t beta_idx = 0; beta_idx < beta_values.size(); ++beta_idx) {
            std::vector<std::future<std::vector<Eigen::VectorXd>>> futures;
            for (size_t i = 0; i < k_values.size(); i += chunk_size) {
                futures.emplace_back(std::async(std::launch::async, [this, i, chunk_size, options, beta = beta_values[beta_idx]]() {
                    std::vector<Eigen::VectorXd> local_results;
                    for (size_t j = i; j < i + chunk_size && j < k_values.size(); ++j) {
                        const double k_value = k_values[j];
                        auto bi = new BoundaryIntegral(k_value, scalingFactor, boundary, kernelStrategy);
                        auto result = bi->computeSVDWithBeta(options, beta).first; // Only store singular values
                        local_results.push_back(result);
                        delete bi;
                    }
                    std::cout << "Completed chunk from index " << i << " to " << i + chunk_size - 1 << " for beta " << beta << std::endl;
                    return local_results;
                }));
                std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Small delay to manage memory usage
            }

            // Collect the results from the futures
            size_t index = 0;
            for (auto& future : futures) {
                for (auto local_results = future.get(); const auto & local_result : local_results) {
                    svdResultsWithBeta[beta_idx][index++] = local_result;
                }
            }
            std::cout << "Completed SVD computation for beta " << beta_values[beta_idx] << std::endl;
        }
        std::cout << "Completed SVD computation for all beta values." << std::endl;
    }

    /**
     * @brief Computes the determinants for the range of k values using the standard kernel.
     */
    void computeDeterminants(const bool showCout) {
        std::vector<std::future<std::complex<double>>> futures;
        futures.reserve(SIZE_K);
        for (int i = 0; i < SIZE_K; ++i) {
            futures.emplace_back(std::async(std::launch::async, [this, i, showCout]() {
                // Manually allocate memory for BoundaryIntegral
                const auto bi = new BoundaryIntegral(k_values[i], scalingFactor, boundary, kernelStrategy);
                const auto result = bi->computeMatrixDeterminant(showCout);
                // Manually free the allocated memory
                delete bi;
                return result;
            }));
        }

        for (int i = 0; i < SIZE_K; ++i) {
            determinants[i] = futures[i].get();
        }
    }

    /**
     * @brief Computes the determinants for the range of k values using the beta variation.
     * @param beta_values A vector of beta values.
     * @param showCout To show the logging of the determinat values in the cout
     */
    void computeDeterminantsWithBeta(const std::vector<double>& beta_values, const bool showCout) {
        determinantsWithBeta.resize(beta_values.size(), std::vector<std::complex<double>>(SIZE_K));

        for (size_t beta_idx = 0; beta_idx < beta_values.size(); ++beta_idx) {
            std::vector<std::future<std::complex<double>>> futures;
            for (int i = 0; i < SIZE_K; ++i) {
                futures.emplace_back(std::async(std::launch::async, [this, i, showCout, beta = beta_values[beta_idx]]() { // NOLINT(*-inefficient-vector-operation)
                    // Manually allocate memory for BoundaryIntegral
                    const auto bi = new BoundaryIntegral(k_values[i], scalingFactor, boundary, kernelStrategy);
                    const auto result = bi->computeMatrixDeterminantWithBeta(beta, showCout);
                    // Manually free the allocated memory
                    delete bi;
                    return result;
                }));
            }

            for (int i = 0; i < SIZE_K; ++i) {
                determinantsWithBeta[beta_idx][i] = futures[i].get();
            }
        }
    }

    /**
     * @brief Plots the real and imaginary parts of the determinants as functions of k and returns the roots.
     * @param ax The matplot axes to plot into
     * @param a, b The y limit to render: [a, b]
     * @param printCrossingLabels If the approximate roots will be rendered in the plot
     * @return A pair of vectors containing the roots of the real and imaginary parts of the determinants.
     */
    [[nodiscard]] std::pair<std::vector<double>, std::vector<double>> plotDeterminants(const matplot::axes_handle& ax, const double a, const double b, const bool printCrossingLabels) const {
        using namespace matplot;

        std::vector<double> real_parts(SIZE_K), imag_parts(SIZE_K);
        for (int i = 0; i < SIZE_K; ++i) {
            real_parts[i] = determinants[i].real();
            imag_parts[i] = determinants[i].imag();
        }

        ax->plot(k_values, real_parts, "g")->display_name("Real(det(A_k))"); // Plot real part in green
        hold(ax, on);
        ax->plot(k_values, imag_parts, "r")->display_name("Imag(det(A_k))"); // Plot imaginary part in red
        ax->xlabel("k");
        ax->ylabel("Value of det(A_k)");
        ax->ylim({a, b}); // Limit y-axis to interval [a, b]

        // Draw y=0 line
        ax->line(k_values.front(), 0, k_values.back(), 0)->color("black").line_style("--");

        constexpr double epsilon = std::numeric_limits<double>::epsilon();
        const double threshold = std::sqrt(epsilon);

        for (size_t i = 0; i < determinants.size(); ++i) {
            if (std::abs(determinants[i]) < threshold) {
                ax->plot({k_values[i]}, {real_parts[i]}, "rs");
                ax->plot({k_values[i]}, {imag_parts[i]}, "rs");
            }
        }
        std::vector<double> zero_k_values_real, zero_real_values, zero_k_values_imag, zero_imag_values;
        for (size_t i = 1; i < determinants.size(); ++i) {
            // Check for zero crossings in the real part
            if ((real_parts[i-1] > 0 && real_parts[i] < 0) || (real_parts[i-1] < 0 && real_parts[i] > 0)) {
                double zero_k_real = (k_values[i-1] + k_values[i]) / 2.0; // Linear interpolation
                zero_k_values_real.push_back(zero_k_real);
                zero_real_values.push_back(0.0);
                // Alternate label position above and below the intersection
                const double label_y = (i % 2 == 0) ? 2.0 : -2.0;

                if (printCrossingLabels) {
                    ax->text(zero_k_real, label_y, std::to_string(zero_k_real).substr(0, 5));
                    std::cout << "Zero crossing (Real): k = " << zero_k_real << std::endl;
                }
            }

            // Check for zero crossings in the imaginary part
            if ((imag_parts[i-1] > 0 && imag_parts[i] < 0) || (imag_parts[i-1] < 0 && imag_parts[i] > 0)) {
                double zero_k_imag = (k_values[i-1] + k_values[i]) / 2.0; // Linear interpolation
                zero_k_values_imag.push_back(zero_k_imag);
                zero_imag_values.push_back(0.0);
                // Alternate label position above and below the intersection
                const double label_y = (i % 2 == 0) ? 2.0 : -2.0;

                if (printCrossingLabels) {
                    ax->text(zero_k_imag, label_y, std::to_string(zero_k_imag).substr(0, 5));
                    std::cout << "Zero crossing (Imag): k = " << zero_k_imag << std::endl;
                }
            }
        }

        // Scatter plot zero crossings
        ax->scatter(zero_k_values_real, zero_real_values); // Green circles for real zero crossings
        ax->scatter(zero_k_values_imag, zero_imag_values); // Red circles for imaginary zero crossings


        // Refine x-axis ticks for more precise values
        std::vector<double> xticks;
        const double step = (k_values.back() - k_values.front()) / 25.0;
        for (double tick = k_values.front(); tick <= k_values.back(); tick += step) { // NOLINT(*-flp30-c)
            xticks.push_back(tick);
        }

        ax->xticks(xticks);
        ax->font_size(12);

        ax->legend({"Abscisa", "Approximate real root", "Approximate imaginary root"});
        return {zero_k_values_real, zero_k_values_imag};
    }

    /**
    * @brief Plots the real and imaginary parts of the determinants as functions of k for multiple beta values and returns the roots.
    * @param ax The matplot axes to plot into
    * @param a, b The y limit to render: [a, b]
    * @param beta_values A vector of real beta values.
    * @param printCrossingLabels If we wish to render the string that contains the root information on to the plot
    * @return A vector of pairs of vectors containing the roots of the real and imaginary parts of the determinants for each beta value.
    */
    [[nodiscard]] std::vector<std::pair<std::vector<double>, std::vector<double>>> plotDeterminantsWithBeta(const matplot::axes_handle& ax, const double a, const double b, const std::vector<double>& beta_values, const bool printCrossingLabels) const {
        using namespace matplot;
        std::vector<std::pair<std::vector<double>, std::vector<double>>> roots_with_beta;

        for (size_t beta_idx = 0; beta_idx < beta_values.size(); ++beta_idx) {
            std::vector<double> real_parts(SIZE_K), imag_parts(SIZE_K);
            for (int i = 0; i < SIZE_K; ++i) {
                real_parts[i] = determinantsWithBeta[beta_idx][i].real();
                imag_parts[i] = determinantsWithBeta[beta_idx][i].imag();
            }

            ax->plot(k_values, real_parts, "g")->display_name("Real(det(A_k)) with beta " + std::to_string(beta_values[beta_idx])); // Plot real part in green
            hold(ax, on);
            ax->plot(k_values, imag_parts, "r")->display_name("Imag(det(A_k)) with beta " + std::to_string(beta_values[beta_idx])); // Plot imaginary part in red
            ax->ylim({a, b}); // Limit y-axis to interval [a, b]

            std::vector<double> zero_k_values_real, zero_real_values, zero_k_values_imag, zero_imag_values;

            for (size_t i = 1; i < determinantsWithBeta[beta_idx].size(); ++i) {
                // Check for zero crossings in the real part
                if ((real_parts[i-1] > 0 && real_parts[i] < 0) || (real_parts[i-1] < 0 && real_parts[i] > 0)) {
                    double zero_k_real = (k_values[i-1] + k_values[i]) / 2.0; // Linear interpolation
                    zero_k_values_real.push_back(zero_k_real);
                    zero_real_values.push_back(0.0);
                    // Alternate label position above and below the intersection
                    const double label_y = (i % 2 == 0) ? 2.0 : -2.0;

                    if (printCrossingLabels) {
                        ax->text(zero_k_real, label_y, std::to_string(zero_k_real).substr(0, 5));
                        std::cout << "Zero crossing (Real) with beta " << beta_values[beta_idx] << ": k = " << zero_k_real << std::endl;
                    }

                }

                // Check for zero crossings in the imaginary part
                if ((imag_parts[i-1] > 0 && imag_parts[i] < 0) || (imag_parts[i-1] < 0 && imag_parts[i] > 0)) {
                    double zero_k_imag = (k_values[i-1] + k_values[i]) / 2.0; // Linear interpolation
                    zero_k_values_imag.push_back(zero_k_imag);
                    zero_imag_values.push_back(0.0);
                    // Alternate label position above and below the intersection
                    const double label_y = (i % 2 == 0) ? 2.0 : -2.0;

                    if (printCrossingLabels) {
                        ax->text(zero_k_imag, label_y, std::to_string(zero_k_imag).substr(0, 5));
                        std::cout << "Zero crossing (Imag) with beta " << beta_values[beta_idx] << ": k = " << zero_k_imag << std::endl;
                    }

                }
            }

            ax->scatter(zero_k_values_real, zero_real_values);
            ax->scatter(zero_k_values_imag, zero_imag_values);
            roots_with_beta.emplace_back(zero_k_values_real, zero_k_values_imag);
        }

        // Draw y=0 line using line method
        ax->line(k_values.front(), 0, k_values.back(), 0)->color("black").line_style("--");

        ax->xlabel("k");
        ax->ylabel("Value of det(A_k)");

        // Set x-axis ticks with larger font size for better readability
        std::vector<double> xticks;
        const double step = (k_values.back() - k_values.front()) / 25.0;  // Reduce the number of ticks
        for (double tick = k_values.front(); tick <= k_values.back(); tick += step) { // NOLINT(*-flp30-c)
            xticks.push_back(tick);
        }

        ax->xticks(xticks);
        ax->font_size(12);

        ax->legend({"Abscisa", "Approximate real root", "Approximate imaginary root"});
        return roots_with_beta;
    }

    /**
     * @brief Plots the magnitude of the determinants as functions of k.
     * @param ax The matplot axes to plot into
     * @param a, b The y limit to render: [a, b]
     * @param printCrossingLabels If the approximate roots will be rendered in the plot
     * @return A std::vector<std::pair<double, double>> containing the values of the determinant magnitude for a given k (k, mag) pair. This can then be used to determine the local minima same as the SVD calculations
     */
    [[nodiscard]] std::vector<std::pair<double, double>> plotMagnitudeDeterminants(const matplot::axes_handle& ax, const double a, const double b, const bool printCrossingLabels) const {
        using namespace matplot;

        std::vector<double> magnitude(SIZE_K);
        std::vector<std::pair<double, double>> k_magnitude_pairs;
        for (int i = 0; i < SIZE_K; ++i) {
            magnitude[i] = std::sqrt(std::pow(determinants[i].real(), 2) + std::pow(determinants[i].imag(), 2));
            k_magnitude_pairs.emplace_back(k_values[i], magnitude[i]);
        }

        ax->plot(k_values, magnitude, "b")->display_name("Magnitude of det(A_k)"); // Plot magnitude in blue
        hold(ax, on);
        ax->xlabel("k");
        ax->ylabel("Magnitude of det(A_k)");
        ax->ylim({a, b}); // Limit y-axis to interval [a, b]

        // Draw y=0 line
        ax->line(k_values.front(), 0, k_values.back(), 0)->color("black").line_style("--");

        constexpr double epsilon = std::numeric_limits<double>::epsilon();
        const double threshold = std::sqrt(epsilon);

        std::vector<double> zero_k_values;
        for (size_t i = 1; i < determinants.size(); ++i) {
            // Check for zero crossings in the magnitude
            if ((magnitude[i-1] > 0 && magnitude[i] < 0) || (magnitude[i-1] < 0 && magnitude[i] > 0)) {
                double zero_k = (k_values[i-1] + k_values[i]) / 2.0; // Linear interpolation
                zero_k_values.push_back(zero_k);
                // Alternate label position above and below the intersection
                const double label_y = (i % 2 == 0) ? 2.0 : -2.0;

                if (printCrossingLabels) {
                    ax->text(zero_k, label_y, std::to_string(zero_k).substr(0, 5));
                    std::cout << "Zero crossing (Magnitude): k = " << zero_k << std::endl;
                }
            }

            if (std::abs(determinants[i]) < threshold) {
                ax->plot({k_values[i]}, {magnitude[i]}, "bs");
            }
        }

        // Scatter plot zero crossings
        ax->scatter(zero_k_values, std::vector<double>(zero_k_values.size(), 0)); // Blue circles for zero crossings

        // Refine x-axis ticks for more precise values
        std::vector<double> xticks;
        const double step = (k_values.back() - k_values.front()) / 25.0;
        for (double tick = k_values.front(); tick <= k_values.back(); tick += step) { // NOLINT(*-flp30-c)
            xticks.push_back(tick);
        }

        ax->xticks(xticks);
        ax->font_size(12);

        ax->legend({"Magnitude of det(A_k)", "Approximate roots"});
        return k_magnitude_pairs;
    }

    /**
     * @brief Prints the refined results of k values and their smallest singular values. Must be called after the SVD calculations
     */
    void printRefinedResults() const {
        std::cout << "Refined k values and their smallest singular values:" << std::endl;
        for (const auto&[fst, snd] : refined_svd_results) {
            const double k_value = fst;
            const Eigen::VectorXd& singularValues = snd;

            if (const double smallest_sv = singularValues(0); smallest_sv < std::sqrt(std::numeric_limits<double>::epsilon())) {
                std::cout << "k: " << k_value << ", Smallest SV: " << smallest_sv << std::endl;
            }
        }
    }

    /**
     * @brief Plots the smallest singular values as functions of k using matplot.
     * @param ax The matplot axes to plot into
     * @param a, b The y limit to render: [a, b]
     */
    [[nodiscard]] matplot::line_handle plotSmallestSingularValues(const matplot::axes_handle& ax, const double a, const double b) const {
        using namespace matplot;

        std::vector<double> smallest_singular_values(SIZE_K);
        for (int i = 0; i < SIZE_K; ++i) {
            smallest_singular_values[i] = svd_results[i](0); // First singular value which is the smallest
        }

        const auto plt = ax->plot(k_values, smallest_singular_values, "b");
        plt->display_name("Smallest Singular Values");
        plt->line_width(0.5);
        hold(ax, on);
        ax->xlabel("k");
        ax->ylabel("Smallest Singular Value");
        ax->ylim({a, b}); // Limit y-axis to interval [a, b]

        // Refine x-axis ticks for more precise values
        std::vector<double> xticks;
        const double step = (k_values.back() - k_values.front()) / 25.0;
        for (double tick = k_values.front(); tick <= k_values.back(); tick += step) { // NOLINT(*-flp30-c)
            xticks.push_back(tick);
        }

        ax->xticks(xticks);
        ax->font_size(12);
        ax->legend({"Smallest Singular Values"});
        return plt;
    }

    /**
     * @brief Plots the singular values as functions of k using matplot.
     * @param ax The matplot axes to plot into
     * @param index The index of the singular value to plot (1 for smallest, 2 for second smallest, etc.)
     * @param a, b The y limit to render: [a, b]
     */
    [[nodiscard]] matplot::line_handle plotSingularValues(const matplot::axes_handle& ax, const int index, const double a, const double b) const {
        using namespace matplot;

        std::vector<double> singular_values(SIZE_K);
        for (int i = 0; i < SIZE_K; ++i) {
            singular_values[i] = svd_results[i](index - 1); // Index adjustment for 0-based indexing
        }

        const auto plt = ax->plot(k_values, singular_values, "b");
        plt->display_name("Singular Value #" + std::to_string(index));
        plt->line_width(0.5);
        if (index == 2) {
            plt->color("red");
        }
        if (index == 3) {
            plt->color("green");
        }
        if (index == 4) {
            plt->color("orange");
        }
        hold(ax, on);
        ax->xlabel("k");
        ax->ylabel("Singular Value");
        ax->ylim({a, b}); // Limit y-axis to interval [a, b]

        // Refine x-axis ticks for more precise values
        std::vector<double> xticks;
        const double step = (k_values.back() - k_values.front()) / 25.0;
        for (double tick = k_values.front(); tick <= k_values.back(); tick += step) { // NOLINT(*-flp30-c)
            xticks.push_back(tick);
        }

        ax->xticks(xticks);
        ax->font_size(12);
        ax->legend({"Singular Value " + std::to_string(index)});
        return plt;
    }

    /**
     * @brief Plots the smallest singular values as functions of k for multiple beta values using matplot.
     * @param ax The matplot axes to plot into
     * @param a, b The y limit to render: [a, b]
     * @param beta_values A vector of real beta values.
     */
    void plotSmallestSingularValuesWithBeta(const matplot::axes_handle& ax, const double a, const double b, const std::vector<double>& beta_values) const {
        using namespace matplot;
        hold(ax, on);
        for (size_t beta_idx = 0; beta_idx < beta_values.size(); ++beta_idx) {
            std::vector<double> smallest_singular_values(SIZE_K);
            for (int i = 0; i < SIZE_K; ++i) {
                smallest_singular_values[i] = svdResultsWithBeta[beta_idx][i](0); // First singular value
            }

            const auto l = ax->plot(k_values, smallest_singular_values, "b");
            l->display_name("Smallest Singular Values with beta " + std::to_string(beta_values[beta_idx]));
            if (beta_idx == 0) {
                l->color("red");
            }
            if (beta_idx == 1) {
                l->color("green");
            }
            if (beta_idx == 2) {
                l->color("black");
            }
            if (beta_idx == 3) {
                l->color("cyan");
            }

            ax->xlabel("k");
            ax->ylabel("Smallest Singular Value");
            ax->ylim({a, b}); // Limit y-axis to interval [a, b]
        }

        // Draw y=0 line using line method
        auto l = ax->line(k_values.front(), 0, k_values.back(), 0)->color("black").line_style("--");
        l.line_width(0.5);

        // Refine x-axis ticks for more precise values
        std::vector<double> xticks;
        const double step = (k_values.back() - k_values.front()) / 25.0;
        for (double tick = k_values.front(); tick <= k_values.back(); tick += step) { // NOLINT(*-flp30-c)
            xticks.push_back(tick);
        }

        ax->xticks(xticks);
        ax->font_size(12);
        ax->legend({"Smallest Singular Values"});
    }

    /**
     * @brief Plots the singular values as functions of k for multiple beta values using matplot.
     * @param ax The matplot axes to plot into
     * @param index The index of the singular value to plot (1 for smallest, 2 for second smallest, etc.)
     * @param a, b The y limit to render: [a, b]
     * @param beta_values A vector of real beta values.
     */
    void plotSingularValuesWithBetaVariation(const matplot::axes_handle& ax, const int index, const double a, const double b, const std::vector<double>& beta_values) const {
        using namespace matplot;
        hold(ax, on);
        for (size_t beta_idx = 0; beta_idx < beta_values.size(); ++beta_idx) {
            std::vector<double> singular_values(SIZE_K);
            for (int i = 0; i < SIZE_K; ++i) {
                singular_values[i] = svdResultsWithBeta[beta_idx][i](index - 1); // Index adjustment for 0-based indexing
            }

            const auto l = ax->plot(k_values, singular_values, "b");
            l->display_name("Singular Value " + std::to_string(index) + " with beta " + std::to_string(beta_values[beta_idx]));
            if (beta_idx == 0) {
                l->color("red");
            }
            if (beta_idx == 1) {
                l->color("green");
            }
            if (beta_idx == 2) {
                l->color("black");
            }
            if (beta_idx == 3) {
                l->color("cyan");
            }

            ax->xlabel("k");
            ax->ylabel("Singular Value " + std::to_string(index));
            ax->ylim({a, b}); // Limit y-axis to interval [a, b]
        }

        // Draw y=0 line using line method
        ax->line(k_values.front(), 0, k_values.back(), 0)->color("black").line_style("--");

        // Refine x-axis ticks for more precise values
        std::vector<double> xticks;
        const double step = (k_values.back() - k_values.front()) / 25.0;
        for (double tick = k_values.front(); tick <= k_values.back(); tick += step) { // NOLINT(*-flp30-c)
            xticks.push_back(tick);
        }

        ax->xticks(xticks);
        ax->font_size(12);
        ax->legend({"Smallest Singular Values"});
    }

    /**
     * @brief Finds the local minima of the smallest singular values from the SVD results that are below a specified threshold.
     *
     * This method iterates through the `svd_results` to identify local minima of the smallest singular values.
     * It returns a vector of tuples, each containing the `k` value, the smallest singular value, and the eigenvector
     * associated with that smallest singular value for each local minimum found that is below the specified threshold.
     *
     * Additionally, this method handles cases where the smallest singular values go to very small values (below sqrt(epsilon)) in sequences by
     * identifying the middle of such sequences as local minima.
     *
     * @param threshold The value below which a local minimum is considered.
     * @return A vector of tuples containing the `k` value, smallest singular value.
     */
    [[nodiscard]] std::vector<std::tuple<double, double>> findLocalMinima(const double threshold = 0.1) const {
        std::vector<std::tuple<double, double>> localMinima;

        // Ensure there are enough points to compare
        if (svd_results.size() < 3) {
            std::cerr << "Not enough points to determine local minima." << std::endl;
            return localMinima;
        }

        const double epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
        for (size_t i = 1; i < svd_results.size() - 1; ++i) {
            const double prevValue = svd_results[i - 1](0);
            const double currValue = svd_results[i](0);

            // Check if current value is a local minimum and below the threshold
            if (const double nextValue = svd_results[i + 1](0); currValue < prevValue && currValue < nextValue && currValue < threshold) {
                const double k = k_values[i];
                localMinima.emplace_back(k, currValue);
            }

            // Check for sequences of very small values, this happens with desymmetrizations
            if (currValue < epsilon) {
                const size_t zero_start = i;
                while (i < svd_results.size() && svd_results[i](0) < epsilon) { // Increment i unitl we have no more uniterrupted sequence of 0's
                    ++i;
                }
                if (const size_t zero_end = i - 1; zero_end > zero_start) {
                    const size_t zero_middle = (zero_start + zero_end) / 2;
                    const double k = k_values[zero_middle];
                    localMinima.emplace_back(k, svd_results[zero_middle](0));
                }
            }
        }
        return localMinima;
    }

    /**
     * @brief Finds the local minima of the smallest singular values from the SVD results that are below a specified threshold.
     *
     * This method iterates through the `svd_results` to identify local minima of the smallest singular values.
     * It returns a vector of tuples, each containing the `k` value, the smallest singular value, and the eigenvector
     * associated with that smallest singular value for each local minimum found that is below the specified threshold.
     *
     * Additionally, this method handles cases where the smallest singular values go to very small values (below sqrt(epsilon)) in sequences by
     * identifying the middle of such sequences as local minima.
     *
     * @param indexOfSingularValue Which singular values are we considering (first smallest, second smallest, third smallest, etc.)
     * @param threshold The value below which a local minimum is considered.
     * @return A vector of tuples containing the `k` value, smallest singular value for each local minimum.
     */
    [[nodiscard]] std::vector<std::tuple<double, double>> findLocalMinima(const int indexOfSingularValue, const double threshold = 0.1) const {
        std::vector<std::tuple<double, double>> localMinima;

        // Ensure there are enough points to compare
        if (svd_results.size() < 3) {
            std::cerr << "Not enough points to determine local minima." << std::endl;
            return localMinima;
        }

        const double epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
        for (size_t i = 1; i < svd_results.size() - 1; ++i) {
            const double prevValue = svd_results[i - 1](indexOfSingularValue - 1);
            const double currValue = svd_results[i](indexOfSingularValue - 1);

            // Check if current value is a local minimum and below the threshold
            if (const double nextValue = svd_results[i + 1](indexOfSingularValue - 1); currValue < prevValue && currValue < nextValue && currValue < threshold) {
                const double k = k_values[i];
                localMinima.emplace_back(k, currValue);
            }

            // Check for sequences of very small values. This happens when we have
            if (currValue < epsilon) {
                const size_t zero_start = i;
                while (i < svd_results.size() && svd_results[i](indexOfSingularValue - 1) < epsilon) {
                    ++i;
                }
                if (const size_t zero_end = i - 1; zero_end > zero_start) {
                    const size_t zero_middle = (zero_start + zero_end) / 2;
                    const double k = k_values[zero_middle];
                    localMinima.emplace_back(k, svd_results[zero_middle](indexOfSingularValue - 1));
                }
            }
        }
        return localMinima;
    }

    /**
     * @brief Checks the validity of the interior Dirichlet problem for given eigenvectors.
     *
     * This method iterates through a collection of eigenvectors, and for each eigenvector,
     * it constructs a BoundaryIntegral object using the corresponding wavenumber \(k\). It then
     * checks the validity of the interior Dirichlet problem using the `checkValidityOfInteriorProblem`
     * method of the BoundaryIntegral class. If the result is below the specified tolerance, it indicates
     * that the interior Dirichlet problem is valid; otherwise, it indicates that the problem should be discarded
     * as an exterior Neumann problem.
     *
     * @param eigenvectorsToCheck A vector of tuples where each tuple contains:
     *                            - A double representing the wavenumber \(k\).
     *                            - A double representing another parameter (unused in this method).
     *                            - An Eigen::VectorXcd object representing the eigenvector to be checked.
     * @param tolerance The tolerance value used to determine the validity of the interior Dirichlet problem.
     *
     * @details
     * - For each tuple in the eigenvectorsToCheck vector:
     *   - Extract the wavenumber \(k\) from the tuple.
     *   - Construct a BoundaryIntegral object using \(k\), and other necessary member variables (assumed to be `b`, `boundary`, and `kernelStrategy`).
     *   - Use the BoundaryIntegral object's `checkValidityOfInteriorProblem` method to check the validity of the interior Dirichlet problem.
     *   - If the result is less than the specified tolerance, print a message indicating the interior Dirichlet problem is OK.
     *   - If the result is greater than or equal to the specified tolerance, print a message indicating the problem should be discarded as an exterior Neumann problem.
     */
    void checkInteriorDirichletProblem(const std::vector<std::tuple<double, double, Eigen::VectorXcd>>& eigenvectorsToCheck, const double tolerance) const {
        // Save the original format flags
        const std::ios_base::fmtflags originalFlags = std::cout.flags();

        for (const auto & i : eigenvectorsToCheck) {
            const auto k = std::get<0>(i);
            BoundaryIntegral bi(k, scalingFactor, boundary, kernelStrategy);
            std::cout << "We are doing k: " << bi.get_k() << std::endl;

            // Additional debug info to ensure points are initialized
            bi.initializePointsAndNormals_homogenous(bi.pointDensityPerWavelength(k));
            const auto& boundaryPoints = bi.getBoundaryPoints();
            std::cout << "Points size: " << boundaryPoints.size() << std::endl;

            if (boundaryPoints.empty()) {
                std::cerr << "Points size is 0 for k=" << k << std::endl;
                continue;
            }

            for (size_t j = 0; j < boundaryPoints.size(); ++j) {
                std::cout << "Point " << j << ": (" << boundaryPoints[j].x << ", " << boundaryPoints[j].y << ")" << std::endl;
            }

            if (const auto res = bi.checkValidityOfInteriorProblem(i); res < tolerance) {
                std::cout << std::fixed << std::setprecision(16) << "For k=" << k << " we have an interior Dirichlet problem - OK. Result: " << res << std::endl;
            } else {
                std::cout << std::fixed << std::setprecision(16) << "For k=" << k << " we have an exterior Neumann problem - DISCARD. Result: " << res << std::endl;
            }
            std::cout << "k=" << k << std::endl;
            std::cout << "Eigenvector " << std::get<2>(i).transpose() << std::endl;
        }

        std::cout.flags(originalFlags);
    }

    /**
     * @brief Identifies and prints the local minima of the smallest singular values from the SVD results.
     *
     * This method iterates through the `svd_results` to identify local minima of the smallest singular values.
     * For each local minimum found that is below the specified threshold, it prints the corresponding `k` value,
     * the smallest singular value, and the eigenvector associated with that smallest singular value.
     *
     * @param threshold The value below which a local minimum is considered.
     */
    void printLocalMinimaOfSingularValues(const double threshold = 0.05) const {
        std::vector<std::tuple<double, double>> localMinima = findLocalMinima(threshold);
        // Print the results
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Local Minima of Smallest Singular Values (below " << threshold << "):" << std::endl;
        for (const auto& [k, singularValue] : localMinima) {
            std::cout << "k: " << k << ", Smallest Singular Value: " << singularValue << std::endl;
        }
    }

    /**
     * @brief Calculates the degeneracies for each k value found as a local minimum of the first singular value.
     *
     * This method finds local minima of the first singular value, then checks higher order singular values at those k values.
     * If the higher order singular values are below a given threshold relative to the first singular value, it counts them as degenerate.
     *
     * @param thresholdForLocalMinima The threshold below which a singular value is considered small enough to indicate a local minimum of at least a non-degenerate level.
     * @param thresholdBetweenDegeneracies The threshold that a singular value of a higher order degeneracy is a true indication of degeneracy
     * @return A vector of tuples containing (k, number_of_degeneracy, vector_of_eigenvectors_at_that_k), sorted by k. The final element is the maximum degeneracy for further analysis
     */
    std::tuple<std::vector<std::tuple<double, int>>, int> calculateDegeneracies(const double thresholdForLocalMinima, const double thresholdBetweenDegeneracies) {

        std::vector<std::tuple<double, int>> result;
        // Get local minima for the first singular value
        auto localMinima = findLocalMinima(1, thresholdForLocalMinima);

        // Extract k-values from localMinima for clustering
        std::vector<double> k_values_local_minima;
        k_values_local_minima.reserve(localMinima.size());
        for (const auto& [k, singularValue] : localMinima) {
            k_values_local_minima.push_back(k);
        }

        std::vector<std::future<std::tuple<double, int>>> futures;
        futures.reserve(localMinima.size());

        for (const auto& [k, singularValue] : localMinima) {
            futures.emplace_back(std::async(std::launch::async, [this, k, singularValue, thresholdBetweenDegeneracies]() {
                int degeneracyCount = 1; // Start with the first singular value being a local minimum

                // Find the index of k in k_values
                if (const auto it = std::ranges::find(this->k_values, k); it != this->k_values.end()) {
                    const int k_index = std::distance(this->k_values.begin(), it); // NOLINT(*-narrowing-conversions)

                    // Check higher order singular values dynamically
                    int currentIndex = 2; // Start checking from the second singular value
                    bool continueChecking = true;
                    while (continueChecking && currentIndex <= svd_results[k_index].size()) { // We have not reached the limitations of the Fredholm matrix size -> svd vector results size
                        if (const double currentValue = svd_results[k_index](currentIndex - 1); std::abs(currentValue - singularValue) < thresholdBetweenDegeneracies) {
                            ++degeneracyCount;
                            ++currentIndex;
                        } else {
                            continueChecking = false;
                        }
                    }
                }

                return std::make_tuple(k, degeneracyCount);
            }));
        }

        // Collect results from futures
        result.reserve(futures.size());
        for (auto& future : futures) {
            result.emplace_back(future.get());
        }

        // Sort results based on k-values
        std::ranges::sort(result, [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });

        // Find the maximum degeneracy
        int maxDegeneracy = 1;
        for (const auto& [k, degeneracyCount] : result) {
            if (degeneracyCount > maxDegeneracy) {
                maxDegeneracy = degeneracyCount;
            }
        }

        return std::make_tuple(result, maxDegeneracy);
    }

    /**
     * @brief Prints the calculated degeneracies.
     *
     * This method prints the k values, the number of degeneracies at each k, and the corresponding eigenvectors.
     *
     * @param degeneracies The vector of tuples containing (k, number_of_degeneracy, vector_of_eigenvectors_at_that_k).
     */
    static void printDegeneracies(const std::vector<std::tuple<double, int>>& degeneracies) {
        std::cout << "Degeneracies:\n";
        for (const auto& [k, count] : degeneracies) {
            std::cout << "k: " << std::fixed << std::setprecision(16) << k << ", Degeneracy Count: " << count << "\n";
            std::cout << "-----------------------------\n";
        }
    }

    /**
     * @brief Normalizes the eigenvectors for each k value. This procedure is based on the Backer paper where we use a boundary integral to calculate the normalization of the normal derivative of the wavefunction u and this implies the normalization of Psi. Altough this method is here for archaic reasons as the normalization method in the EigenfunctionAndPlotting class will automatically do this
     *
     * This method normalizes the eigenvectors based on the provided k values,
     * points, and normal vectors.
     *
     * @param eigenvaluesAndEigenvectors The vector of tuples containing (k, degeneracy, eigenvectors) from the calculateDegeneracies method.
     * @return A vector of tuples containing (k, degeneracy, normalized eigenvectors).
     */
    [[nodiscard]] std::vector<std::tuple<double, int, std::vector<Eigen::VectorXcd>>> normalizeEigenvectors(
        const std::vector<std::tuple<double, int, std::vector<Eigen::VectorXcd>>>& eigenvaluesAndEigenvectors) const {

        std::vector<std::tuple<double, int, std::vector<Eigen::VectorXcd>>> normalizedResults;
        // Iterate over each tuple containing (k, degeneracy, eigenvectors)
        for (const auto& [k, degeneracy, eigenvectors] : eigenvaluesAndEigenvectors) {
            // Initialize BoundaryIntegral for the current kr
            BoundaryIntegral bi(k, scalingFactor, boundary, kernelStrategy);
            // Get points and normals for the current k
            const auto& points = bi.getBoundaryPoints();
            const auto& normals = bi.getBoundaryNormals();
            // Vector to store the normalized eigenvectors for the current k
            std::vector<Eigen::VectorXcd> normalizedEigenvectors;
            // Iterate over each eigenvector to normalize
            for (const auto& tilde_u : eigenvectors) {
                // Calculate the sum for normalization
                double sum = 0.0;
                for (size_t i = 0; i < points.size(); ++i) {
                    // Sum up the terms using points, normals, and the eigenvector values
                    sum += std::norm(tilde_u[i]) * (normals[i].x * points[i].x + normals[i].y * points[i].y); // NOLINT(*-narrowing-conversions)
                }
                // Calculate the normalization factor based on the sum
                double normalizationFactor = std::sqrt(2 * k * k / sum);
                // Normalize the eigenvector using the normalization factor
                Eigen::VectorXcd u = tilde_u * normalizationFactor;
                // Store the normalized eigenvector
                normalizedEigenvectors.push_back(u);
            }
            // Store the results as a tuple of (k, degeneracy, normalized eigenvectors)
            normalizedResults.emplace_back(k, degeneracy, normalizedEigenvectors);
        }
        // Return the vector of tuples containing (k, degeneracy, normalized eigenvectors)
        return normalizedResults;
    }

    /**
     * Simple getter for the SVD results
     * @return Eigen::VectorXd of the SVD results for this instance of the KRangeSolver class which are the singular values
     */
    [[nodiscard]] std::vector<Eigen::VectorXd> getSVDResults() const {
        return svd_results;
    }

    // *****************************************************************//
    // FOR THIS CLASS BELOW ARE METHODS WHERE WE SAVE AND PLOT THE RESULTS
    // *****************************************************************//

    /**
     * @brief Writes the k values and a specified number of the lowest sorted SVD results into a .csv file.
     *
     * This method writes the k values and their corresponding specified number of lowest sorted SVD results
     * into a specified .csv file.
     *
     * @param filename The name of the .csv file to write to.
     * @param num_singular_values The number of lowest sorted singular values to write for each k value. If the number
     *                            is -1 or exceeds the available singular values, it writes all available singular values.
     */
    void writeSVDResultsToCSV(const std::string& filename, const int num_singular_values = -1) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }
        // Write the k values and their corresponding sorted singular values
        for (size_t i = 0; i < k_values.size(); ++i) {
            file << k_values[i] << ",";
            std::vector<double> singular_values(svd_results[i].data(), svd_results[i].data() + svd_results[i].size());
            std::ranges::sort(singular_values);
            const int limit = (num_singular_values == -1) ? singular_values.size() : std::min(num_singular_values, static_cast<int>(singular_values.size())); // NOLINT(*-narrowing-conversions)
            for (int j = 0; j < limit; ++j) {
                file << singular_values[j];
                if (j < limit - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }
        file.close();
        std::cout << "SVD results written to " << filename << std::endl;
    }

    /**
     * @brief Reads the k values and their corresponding SVD results from a .csv file.
     *
     * This method reads the k values and their corresponding SVD results from the specified .csv file and
     * returns them as a vector of pairs, where each pair consists of a k value and a vector of SVD values for that k.
     *
     * @param filename The name of the .csv file to read from.
     * @return A vector of pairs where each pair consists of a k value and a vector of SVD values for that k.
     */
    static std::vector<std::pair<double, std::vector<double>>> readSVDResultsFromCSV(const std::string& filename) {
        std::vector<std::pair<double, std::vector<double>>> svd_data;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return svd_data; // Return an empty vector
        }
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;
            std::vector<double> svd_values;
            // Read the k value
            std::getline(ss, value, ',');
            double k = std::stod(value);
            // Read the SVD values
            while (std::getline(ss, value, ',')) {
                svd_values.push_back(std::stod(value));
            }
            svd_data.emplace_back(k, svd_values);
        }
        file.close();
        return svd_data;
    }

    /**
     * @brief Plots the singular values as functions of k using matplot, based on data read from a CSV file.
     *
     * This method reads the SVD results from a CSV file and plots the specified index of the singular values.
     *
     * @param filename The name of the CSV file to read the SVD results from.
     * @param ax The matplot axes to plot into.
     * @param index The index of the singular value to plot (1 for smallest, 2 for second smallest, etc.).
     * @param a The lower limit of the y-axis.
     * @param b The upper limit of the y-axis.
     * @param from_k The lower limit of the k values to plot.
     * @param to_k The upper limit of the k values to plot.
     */
    static void plotSingularValuesFromCSV(const std::string& filename, const matplot::axes_handle& ax, const int index, const double a, const double b, const double from_k = -1, const double to_k = -1) {
        std::vector<std::pair<double, std::vector<double>>> svd_results = readSVDResultsFromCSV(filename);

        if (svd_results.empty()) {
            std::cerr << "No data to plot." << std::endl;
            return;
        }

        std::vector<double> k_values;
        std::vector<double> singular_values;

        for (const auto& [k, svd_values] : svd_results) {
            if ((from_k == -1 || k >= from_k) && (to_k == -1 || k <= to_k)) {
                k_values.push_back(k);
                if (index - 1 < svd_values.size()) {
                    singular_values.push_back(svd_values[index - 1]);
                } else {
                    singular_values.push_back(std::numeric_limits<double>::quiet_NaN());
                }
            }
        }

        using namespace matplot;
        const auto plt = ax->plot(k_values, singular_values, "b");
        plt->display_name("Singular Value #" + std::to_string(index));
        plt->line_width(0.5);

        if (index == 2) {
            plt->color("red");
        }
        if (index == 3) {
            plt->color("green");
        }
        if (index == 4) {
            plt->color("orange");
        }

        hold(ax, on);
        ax->xlabel("k");
        ax->ylabel("Singular Value");
        ax->ylim({a, b}); // Limit y-axis to interval [a, b]
        if (index == 1) {
            ax->line(k_values.front(), 0, k_values.back(), 0)->color("black").line_style("--").line_width(0.5).display_name("y=0"); // Plot y=0 axis
        }
        // Refine x-axis ticks for more precise values
        std::vector<double> xticks;
        const double step = (k_values.back() - k_values.front()) / 25.0;
        for (double tick = k_values.front(); tick <= k_values.back(); tick += step) { // NOLINT(*-flp30-c)
            xticks.push_back(tick);
        }

        ax->xticks(xticks);
        ax->font_size(12);
        ax->legend({"Singular Value " + std::to_string(index)});
    }

    /**
     * @brief Finds the local minima of the specified singular value index from the SVD results read from a CSV file.
     *
     * This method iterates through the SVD results to identify local minima of the specified singular value.
     * It returns a vector of tuples, each containing the k value and the singular value at that k for each local minimum found that is below the specified threshold.
     *
     * Additionally, this method handles cases where the specified singular value goes to very small values (below sqrt(epsilon)) in sequences by identifying the middle of such sequences as local minima.
     *
     * @param filename The name of the CSV file to read the SVD results from.
     * @param indexOfSingularValue The index of the singular value to check (1 for smallest, 2 for second smallest, etc.).
     * @param threshold The value below which a local minimum is considered.
     * @return A vector of tuples containing the k value and the singular value at that k for each local minimum.
     */
    [[nodiscard]] static std::vector<std::tuple<double, double>> findLocalMinimaFromCSV(const std::string& filename, const int indexOfSingularValue, const double threshold = 0.1) {
        std::vector<std::pair<double, std::vector<double>>> svd_results = readSVDResultsFromCSV(filename);
        std::vector<std::tuple<double, double>> localMinima;

        // Ensure there are enough points to compare
        if (svd_results.size() < 3) {
            std::cerr << "Not enough points to determine local minima." << std::endl;
            return localMinima;
        }
        const double epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
        for (size_t i = 1; i < svd_results.size() - 1; ++i) {
            const double prevValue = svd_results[i - 1].second[indexOfSingularValue - 1];
            const double currValue = svd_results[i].second[indexOfSingularValue - 1];

            // Check if current value is a local minimum and below the threshold
            if (const double nextValue = svd_results[i + 1].second[indexOfSingularValue - 1]; currValue < prevValue && currValue < nextValue && currValue < threshold) {
                const double k = svd_results[i].first;
                localMinima.emplace_back(k, currValue);
            }

            // Check for sequences of very small values. This happens when we have
            if (currValue < epsilon) {
                const size_t zero_start = i;
                while (i < svd_results.size() && svd_results[i].second[indexOfSingularValue - 1] < epsilon) {
                    ++i;
                }
                if (const size_t zero_end = i - 1; zero_end > zero_start) {
                    const size_t zero_middle = (zero_start + zero_end) / 2;
                    const double k = svd_results[zero_middle].first;
                    localMinima.emplace_back(k, svd_results[zero_middle].second[indexOfSingularValue - 1]);
                }
            }
        }
        return localMinima;
    }

    /**
     * @brief Calculates the degeneracies for each k value found as a local minimum of the first singular value.
     *
     * This method finds local minima of the first singular value, then checks higher order singular values at those k values.
     * If the higher order singular values are below a given threshold relative to the first singular value, it counts them as degenerate.
     *
     * @param filename The name of the CSV file to read the SVD results from.
     * @param thresholdForLocalMinima The threshold below which a singular value is considered small enough to indicate a local minimum of at least a non-degenerate level.
     * @param thresholdBetweenDegeneracies The threshold that a singular value of a higher order degeneracy is a true indication of degeneracy.
     * @param outputFilename The name of the CSV file to write the k values and degeneracy into.
     * @return A vector of tuples containing (k, number_of_degeneracy), sorted by k. The final element is the maximum degeneracy for further analysis if needed.
     */
    static std::tuple<std::vector<std::tuple<double, int>>, int> calculateDegeneraciesFromCSV(const std::string& filename, const double thresholdForLocalMinima, const double thresholdBetweenDegeneracies, const std::string& outputFilename) {
        // Read SVD results from CSV file
        std::vector<std::pair<double, std::vector<double>>> svd_results = readSVDResultsFromCSV(filename);
        std::vector<std::tuple<double, int>> result;

        // Get local minima for the first singular value
        auto localMinima = findLocalMinimaFromCSV(filename, 1, thresholdForLocalMinima);
        std::vector<std::future<std::tuple<double, int>>> futures;
        futures.reserve(localMinima.size());

        for (const auto& [k, singularValue] : localMinima) {
            futures.emplace_back(std::async(std::launch::async, [k, singularValue, &svd_results, thresholdBetweenDegeneracies]() {
                int degeneracyCount = 1; // Start with the first singular value being a local minimum

                // Find the index of k in svd_results

                if (const auto it = std::ranges::find_if(svd_results, [k](const auto& pair) {
                    return pair.first == k;
                }); it != svd_results.end()) {
                    const auto& svd_values = it->second;
                    // Check higher order singular values dynamically
                    for (size_t currentIndex = 1; currentIndex < svd_values.size(); ++currentIndex) {
                        if (const double currentValue = svd_values[currentIndex]; std::abs(currentValue - singularValue) < thresholdBetweenDegeneracies) {
                            ++degeneracyCount;
                        } else {
                            break;
                        }
                    }
                }
                return std::make_tuple(k, degeneracyCount);
            }));
        }

        // Collect results from futures
        result.reserve(futures.size());
        for (auto& future : futures) {
            result.emplace_back(future.get());
        }

        // Sort results based on k-values
        std::ranges::sort(result, [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });

        // Find the maximum degeneracy
        int maxDegeneracy = 1;
        for (const auto& [k, degeneracyCount] : result) {
            if (degeneracyCount > maxDegeneracy) {
                maxDegeneracy = degeneracyCount;
            }
        }

        // Write results to CSV file
        if (std::ofstream file(outputFilename); file.is_open()) {
            file << std::fixed << std::setprecision(8);
            for (const auto& [k, degeneracyCount] : result) {
                file << k << "," << degeneracyCount << "\n";
            }
            file.close();
            std::cout << "Degeneracies successfully written to " << outputFilename << std::endl;
        } else {
            std::cerr << "Unable to open file: " << outputFilename << std::endl;
        }
        return std::make_tuple(result, maxDegeneracy);
    }

    /**
     * @brief Counts the number of k values up to a given k value, including degeneracy if present.
     *
     * This method uses the calculateDegeneraciesFromCSV method to find the local minima of singular values,
     * calculates the degeneracy for each k value, and counts the total number of k values up to each k value,
     * including degeneracies.
     *
     * @param inputFilename The name of the CSV file to read the SVD results from.
     * @param thresholdForLocalMinima The threshold below which a singular value is considered small enough to indicate a local minimum of at least a non-degenerate level.
     * @param thresholdBetweenDegeneracies The threshold that a singular value of a higher order degeneracy is a true indication of degeneracy.
     * @param degeneracyOutputFilename The name of the CSV file to write the degeneracy results into.
     * @param countOutputFilename The name of the CSV file to write the cumulative k count results into.
     */
    static void countKValuesUpTo(const std::string& inputFilename, const double thresholdForLocalMinima, const double thresholdBetweenDegeneracies, const std::string& degeneracyOutputFilename, const std::string& countOutputFilename) {
        // Calculate degeneracies from the CSV file
        auto [degeneracies, maxDegeneracy] = calculateDegeneraciesFromCSV(inputFilename, thresholdForLocalMinima, thresholdBetweenDegeneracies, degeneracyOutputFilename);

        std::vector<std::tuple<double, int>> k_count;
        int cumulative_count = 0;

        for (const auto& [k, degeneracyCount] : degeneracies) {
            cumulative_count += degeneracyCount;
            k_count.emplace_back(k, cumulative_count);
        }

        // Write the results to the output CSV file
        if (std::ofstream file(countOutputFilename); file.is_open()) {
            file << std::fixed << std::setprecision(8);
            for (const auto& [k, count] : k_count) {
                file << k << "," << count << "\n";
            }
            file.close();
            std::cout << "K value counts successfully written to " << countOutputFilename << std::endl;
        } else {
            std::cerr << "Unable to open file: " << countOutputFilename << std::endl;
        }
    }

private:
    int b{};
};
    
namespace Matrix_Eigen_Armadillo_Conversion {
    /**
     * @brief Converts an Eigen::MatrixXcd to an arma::cx_mat and applies Tikhonov regularization.
     *
     * This function converts an Eigen complex matrix to an Armadillo complex matrix and adds
     * a regularization term to the diagonal elements to improve numerical stability (Tikhonov regularization improves numerical stability)
     *
     * @param eigenMatrix The input Eigen::MatrixXcd.
     * @param lambda The regularization parameter to be added to the diagonal.
     * @return The output arma::cx_mat with regularization applied.
     */
    // ReSharper disable once CppDFAConstantParameter
    inline arma::cx_mat EigenToArma(const Eigen::MatrixXcd& eigenMatrix, const double lambda = 0.0) {
        // Get the number of rows and columns
        const auto rows = eigenMatrix.rows();
        const auto cols = eigenMatrix.cols();

        // Initialize an Armadillo matrix with the same size
        arma::cx_mat armaMatrix(rows, cols);

        // Copy elements from the Eigen matrix to the Armadillo matrix
        for (Eigen::Index i = 0; i < rows; ++i) {
            for (Eigen::Index j = 0; j < cols; ++j) {
                armaMatrix(i, j) = eigenMatrix(i, j);
            }
        }

        // Apply Tikhonov regularization by adding lambda to the diagonal elements
        for (Eigen::Index i = 0; i < std::min(rows, cols); ++i) {
            armaMatrix(i, i) += lambda;
        }

        return armaMatrix;
    }

    /**
     * @brief Converts an Armadillo matrix to an Eigen matrix.
     *
     * This function converts a given Armadillo matrix of complex numbers to an Eigen matrix.
     *
     * @param armaMatrix The Armadillo matrix to convert.
     * @return Eigen::MatrixXcd The converted Eigen matrix.
     */
    static Eigen::MatrixXcd ArmaToEigen(const arma::cx_mat& armaMatrix) {
        Eigen::MatrixXcd eigenMatrix(armaMatrix.n_rows, armaMatrix.n_cols);
        std::memcpy(eigenMatrix.data(), armaMatrix.memptr(), armaMatrix.n_elem * sizeof(std::complex<double>));
        return eigenMatrix;
    }
}

/**
 * Namespace for debugging methods, mostly to check the correctnes of the matrices constructed
 */
namespace Debugging {
    enum class PRINT_TYPES {
        COS_PHI,
        HANKEL,
        KERNEL,
        FREDHOLM,
        FREDHOLM_DERIVATIVE,
        FREDHOLM_SECOND_DERIVATIVE,
        FREDHOLM_COMBINED_DERIVATIVE,
        NONE,
    };

    inline void printFredholmMatrixAndDerivatives(const BoundaryIntegral& bi, const bool printDiscretizationPointsAndNormals, const PRINT_TYPES printOut) {
        if (printDiscretizationPointsAndNormals) {
            std::cout << "Number of discretization points: " << bi.getBoundaryPoints().size() << std::endl;
            std::cout << "The length of the boundary is "<< bi.calculateArcLength() << std::endl;
            bi.printDiscretizationPointsAndNormals();
        }
        switch (printOut) {
            case (PRINT_TYPES::COS_PHI):
                std::cout << "CosPhi Matrix for the last k=" << bi.get_k() << ":\n" << bi.constructCosPhiMatrixString() << std::endl;
            break;
            case (PRINT_TYPES::HANKEL):
                std::cout << "Hankel Matrix for the last k=" << bi.get_k() << ":\n" << bi.constructHankelMatrixString() << std::endl;
            break;
            case (PRINT_TYPES::KERNEL):
                std::cout << "Integration kernel for k=" << bi.get_k() << ":\n" << bi.constructKernelMatrixString() << std::endl;
            break;
            case (PRINT_TYPES::FREDHOLM):
                std::cout << "Our manually constructed Fredholm matrix for k=" << bi.get_k() << "\n" << bi.constructFredholmMatrixFromCosPhiMatrixAndHankelMatrix() << std::endl;
                std::cout << "Fredholm Matrix for k=" << bi.get_k() << ":\n" << bi.constructFredholmMatrixString() << std::endl;
            break;
            case (PRINT_TYPES::FREDHOLM_DERIVATIVE):
                std::cout << "The first derivative of the Fredholm matrix for k=" << bi.get_k() << ":\n" << bi.constructFredholmMatrixDerivativeString() << std::endl;
                std::cout << "Constructed first derivative of Fredholm matrix of size (" << bi.constructFredholmMatrixDerivativeString().rows() << ", " << bi.constructFredholmMatrixDerivativeString().cols() << ") " << std::endl;
            break;
            case (PRINT_TYPES::FREDHOLM_SECOND_DERIVATIVE):
                std::cout << "The second derivative of the Fredholm matrix for k=" << bi.get_k() << ":\n" << bi.constructFredholmMatrixSecondDerivativeString() << std::endl;
                std::cout << "Constructed second derivative of Fredholm matrix of size: (" << bi.constructFredholmMatrixSecondDerivativeString().rows() << ", " << bi.constructFredholmMatrixSecondDerivativeString().cols() << ") " << std::endl;
            break;
            case (PRINT_TYPES::FREDHOLM_COMBINED_DERIVATIVE):
                std::cout << "The first and second derivative of the Fredholm matrix for k=" << bi.get_k() << ":\n" << bi.constructCombinedFredholmMatrixDerivativeString() << std::endl;
            std::cout << "Constructed zeroth, first and second derivative of Fredholm matrix of size: (" << bi.constructCombinedFredholmMatrixDerivativeString().rows() << ", " << bi.constructCombinedFredholmMatrixDerivativeString().cols() << ") " << std::endl;
            break;
            default:
                break;
        }
    }

    /**
     * @brief Print the Fredholm matrix and its derivatives for a specific analytical eigenvalue.
     *
     * This function finds the analytical eigenvalue specified by its indices (m, n) and prints
     * the Fredholm matrix and its derivatives for that eigenvalue.
     *
     * @param boundary The boundary of the problem.
     * @param kernelStrategy The kernel integration strategy.
     * @param analytical_eigenvalues A vector of tuples containing analytical eigenvalues and their indices.
     * @param m The index of the analytical eigenvalue (m).
     * @param n The index of the analytical eigenvalue (n).
     * @param b The scaling factor used in the BoundaryIntegral constructor.
     * @param printDiscretizationPointsAndNormals Flag to indicate whether to print the discretization points and normals.
     * @param printOut The type of matrix or derivative to print.
     */
    inline void printFredholmMatrixAndDerivativesForEigenvalue(
        const std::shared_ptr<AbstractBoundary>& boundary,
        const std::shared_ptr<IKernelStrategy>& kernelStrategy,
        const std::vector<std::tuple<double, int, int>>& analytical_eigenvalues,
        int m, int n,
        const double b,
        const bool printDiscretizationPointsAndNormals = true,
        const PRINT_TYPES printOut = PRINT_TYPES::FREDHOLM_COMBINED_DERIVATIVE)
    {
        // Find the analytical eigenvalue with indices (m, n)

        // Check if the selected eigenvalue was found
        if (const auto selected_eigenvalue = std::ranges::find_if(analytical_eigenvalues, [m, n](const std::tuple<double, int, int>& eigenvalue) { return std::get<1>(eigenvalue) == m && std::get<2>(eigenvalue) == n;});

        selected_eigenvalue != analytical_eigenvalues.end()) {
            // Get the k value for the selected eigenvalue
            const double k_value = std::get<0>(*selected_eigenvalue);
            std::cout << "Printing Fredholm matrix and derivatives for analytical eigenvalue k(" << m << "," << n << ") = " << k_value << std::endl;

            // Create a BoundaryIntegral object with the selected k value
            const BoundaryIntegral bi(k_value, b, boundary, kernelStrategy); // NOLINT(*-narrowing-conversions)

            // Print the Fredholm matrix and its derivatives
            printFredholmMatrixAndDerivatives(bi, printDiscretizationPointsAndNormals, printOut);
        } else {
            std::cerr << "Analytical eigenvalue with indices (" << m << "," << n << ") not found." << std::endl;
        }
    }

    /**
     * @brief Print the Fredholm matrix and its derivatives for a given k value.
     *
     * This function prints the Fredholm matrix and its derivatives for a specified k value.
     *
     * @param boundary The boundary of the problem.
     * @param kernelStrategy The kernel integration strategy.
     * @param k_value The k value for which the Fredholm matrix will be computed.
     * @param b The scaling factor used in the BoundaryIntegral constructor.
     * @param printDiscretizationPointsAndNormals Flag to indicate whether to print the discretization points and normals.
     * @param printOut The type of matrix or derivative to print.
     */
    inline void printFredholmMatrixAndDerivativesForKValue(
        const std::shared_ptr<AbstractBoundary>& boundary,
        const std::shared_ptr<IKernelStrategy>& kernelStrategy,
        const double k_value,
        double b,
        const bool printDiscretizationPointsAndNormals = true,
        const PRINT_TYPES printOut = PRINT_TYPES::FREDHOLM_COMBINED_DERIVATIVE)
    {
        std::cout << "Printing Fredholm matrix and derivatives for k value = " << k_value << std::endl;

        // Create a BoundaryIntegral object with the specified k value
        const BoundaryIntegral bi(k_value, b, boundary, kernelStrategy); // NOLINT(*-narrowing-conversions)

        // Print the Fredholm matrix and its derivatives
        printFredholmMatrixAndDerivatives(bi, printDiscretizationPointsAndNormals, printOut);
    }

    /**
     * @brief Writes a complex double Eigen matrix to a CSV file in a format readable by Mathematica.
     *
     * This function is specifically designed to export the Fredholm matrix and its derivatives
     * into a CSV file. The Fredholm matrix is an integral operator represented in matrix form,
     * used extensively in solving integral equations. The derivatives of the Fredholm matrix
     * are also crucial in various numerical methods and theoretical analyses. This function
     * ensures that the complex numbers are written in a format (a + bI) that Mathematica can
     * correctly interpret for further computations and visualizations.
     *
     * The CSV file will contain the complex numbers in the format `a + bI`, which is the standard
     * format used by Mathematica for complex numbers.
     *
     * Usage in Mathematica:
     * 1. Import the CSV file using the Import function:
     *    `matrix = Import["matrix.csv", "CSV"]`
     * 2. The imported matrix will be a list of lists where each sublist corresponds to a row of the matrix.
     * 3. To convert the imported data to a matrix format in Mathematica, use:
     *    `matrix = Map[ToExpression, matrix, {2}]`
     *    This will convert the string representations of the complex numbers to Mathematica's complex number format.
     *
     * Example:
     * ```
     * matrix = Import["matrix.csv", "CSV"];
     * matrix = Map[ToExpression, matrix, {2}];
     * ```
     *
     * @param matrix The complex double Eigen matrix to be written to the CSV file.
     * @param filename The name of the CSV file to write the matrix into.
     */
    inline void writeComplexMatrixToCSV_Mathematica(const Eigen::MatrixXcd& matrix, const std::string& filename) {
        if (std::ofstream file(filename); file.is_open()) {
            // Loop through the matrix elements
            for (int i = 0; i < matrix.rows(); ++i) {
                for (int j = 0; j < matrix.cols(); ++j) {
                    // Get the real and imaginary parts
                    const double real = matrix(i, j).real();
                    const double imag = matrix(i, j).imag();
                    // Write in Mathematica-friendly format a + bI
                    file << real << (imag >= 0 ? "+" : "") << imag << "I";
                    // Add a comma if not the last element
                    if (j < matrix.cols() - 1) {
                        file << ",";
                    }
                }
                file << "\n";  // Newline at the end of each row
            }
            file.close();
            std::cout << "Matrix successfully written to " << filename << std::endl;
        } else {
            std::cerr << "Unable to open file: " << filename << std::endl;
        }
    }

     /**
     * @brief Writes the boundary points to a CSV file in a format readable by Mathematica.
     *
     * This function exports the boundary points of a given BoundaryIntegral object to a CSV file.
     * The points are written in a format where each line contains the x and y coordinates separated by a space.
     *
     * @param boundaryIntegral The BoundaryIntegral object containing the boundary points.
     * @param filename The name of the file to write the boundary points to.
     * @throws std::runtime_error If the file cannot be opened for writing.
     */
     inline void writePointsToCSV_Mathematica(const BoundaryIntegral& boundaryIntegral, const std::string& filename) {
         const std::string outputFilename = filename + ".csv";
        std::ofstream outFile(outputFilename);
        if (!outFile.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + outputFilename);
        }
        // Set precision for output
        outFile << std::fixed << std::setprecision(8);
        // Write points to CSV
        for (const auto& point : boundaryIntegral.getBoundaryPoints()) {
            outFile << point.x << "," << point.y << "\n";
        }
        outFile.close();
        std::cout << "Points written to " << outputFilename << std::endl;
    }

    /**
     * @brief Writes the boundary normals to a CSV file in a format readable by Mathematica.
     *
     * This function exports the boundary normals of a given BoundaryIntegral object to a CSV file.
     * The normals are written in a format where each line contains the x and y components separated by a space.
     *
     * @param boundaryIntegral The BoundaryIntegral object containing the boundary normals.
     * @param filename The name of the file to write the boundary normals to.
     * @throws std::runtime_error If the file cannot be opened for writing.
     */
    inline void writeNormalsToCSV_Mathematica(const BoundaryIntegral& boundaryIntegral, const std::string& filename) {
        const std::string outputFilename = filename + ".csv";
        std::ofstream outFile(outputFilename);
        if (!outFile.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + outputFilename);
        }
        // Set precision for output
        outFile << std::fixed << std::setprecision(8);
        // Write normals to CSV
        for (const auto& normal : boundaryIntegral.getBoundaryNormals()) {
            outFile << normal.x << "," << normal.y << "\n";
        }
        outFile.close();
        std::cout << "Normals written to " << outputFilename << std::endl;
    }

    /**
     * @namespace Circle
     * @brief Namespace for debugging purposes, containing functions to compute wavefunctions and their gradients for a circular boundary.
     */
    namespace Circle {
        /**
         * @enum Parity
         * @brief Enumeration for specifying the parity of the wavefunction.
         */
        enum class Parity { EVEN, ODD };

        /**
         * @brief Computes the wavefunction in polar coordinates for a circular boundary.
         * @param m The angular quantum number.
         * @param n The radial quantum number.
         * @param parity The parity of the wavefunction (EVEN or ODD).
         * @return A std::function<double(double, double)> representing the wavefunction in polar coordinates (r, theta).
         */
        inline std::function<double(double, double)> computeWavefunctionPolar(int m, const int n, Parity parity) {
            double k_mn = gsl_sf_bessel_zero_Jnu(m, n); // m-th zero of the Bessel function of order m
            return [m, k_mn, parity](const double r, const double theta) {
                if (parity == Parity::EVEN) {
                    return gsl_sf_bessel_Jn(m, k_mn * r) * std::cos(m * theta);
                } else {
                    return gsl_sf_bessel_Jn(m, k_mn * r) * std::sin(m * theta);
                }
            };
        }

        /**
         * @brief Computes the wavefunction in Cartesian coordinates for a circular boundary.
         * @param m The angular quantum number.
         * @param n The radial quantum number.
         * @param parity The parity of the wavefunction (EVEN or ODD).
         * @return A std::function<double(double, double)> representing the wavefunction in Cartesian coordinates (x, y).
         */
        inline std::function<double(double, double)> computeWavefunctionCartesian(int m, const int n, Parity parity) {
            double k_mn = gsl_sf_bessel_zero_Jnu(m, n); // m-th zero of the Bessel function of order m
            return [m, k_mn, parity](const double x, const double y) {
                const double r = std::sqrt(x * x + y * y);
                const double theta = std::atan2(y, x);
                if (parity == Parity::EVEN) {
                    return gsl_sf_bessel_Jn(m, k_mn * r) * std::cos(m * theta);
                } else {
                    return gsl_sf_bessel_Jn(m, k_mn * r) * std::sin(m * theta);
                }
            };
        }

        /**
         * @brief Computes the gradient of the wavefunction in polar coordinates for a circular boundary.
         * @param m The angular quantum number.
         * @param n The radial quantum number.
         * @param parity The parity of the wavefunction (EVEN or ODD).
         * @return A std::function<std::pair<double, double>(double, double)> representing the gradient of the wavefunction in polar coordinates (r, theta).
         */
        inline std::function<std::pair<double, double>(double, double)> computeGradientPolar(int m, const int n, Parity parity) {
            double k_mn = gsl_sf_bessel_zero_Jnu(m, n); // m-th zero of the Bessel function of order m
            return [m, k_mn, parity](const double r, const double theta) {
                const double J_m = gsl_sf_bessel_Jn(m, k_mn * r);
                const double J_m_prime = 0.5 * (gsl_sf_bessel_Jn(m - 1, k_mn * r) - gsl_sf_bessel_Jn(m + 1, k_mn * r));
                double dPsi_dr, dPsi_dTheta;
                if (parity == Parity::EVEN) {
                    dPsi_dr = k_mn * J_m_prime * std::cos(m * theta);
                    dPsi_dTheta = -m * J_m * std::sin(m * theta);
                } else {
                    dPsi_dr = k_mn * J_m_prime * std::sin(m * theta);
                    dPsi_dTheta = m * J_m * std::cos(m * theta);
                }
                return std::make_pair(dPsi_dr, dPsi_dTheta);
            };
        }

        /**
         * @brief Computes the gradient of the wavefunction in Cartesian coordinates for a circular boundary.
         * @param m The angular quantum number.
         * @param n The radial quantum number.
         * @param parity The parity of the wavefunction (EVEN or ODD).
         * @return A std::function<std::pair<double, double>(double, double)> representing the gradient of the wavefunction in Cartesian coordinates (x, y).
         */
        inline std::function<std::pair<double, double>(double, double)> computeGradientCartesian(int m, const int n, Parity parity) {
            double k_mn = gsl_sf_bessel_zero_Jnu(m, n); // m-th zero of the Bessel function of order m
            return [m, k_mn, parity](const double x, const double y) {
                const double r = std::sqrt(x * x + y * y);
                const double theta = std::atan2(y, x);
                const double J_m = gsl_sf_bessel_Jn(m, k_mn * r);
                const double J_m_prime = 0.5 * (gsl_sf_bessel_Jn(m - 1, k_mn * r) - gsl_sf_bessel_Jn(m + 1, k_mn * r));
                double dPsi_dr, dPsi_dTheta;
                if (parity == Parity::EVEN) {
                    dPsi_dr = k_mn * J_m_prime * std::cos(m * theta);
                    dPsi_dTheta = -m * J_m * std::sin(m * theta);
                } else {
                    dPsi_dr = k_mn * J_m_prime * std::sin(m * theta);
                    dPsi_dTheta = m * J_m * std::cos(m * theta);
                }
                double dPsi_dx = dPsi_dr * std::cos(theta) - dPsi_dTheta * std::sin(theta) / r;
                double dPsi_dy = dPsi_dr * std::sin(theta) + dPsi_dTheta * std::cos(theta) / r;
                return std::make_pair(dPsi_dx, dPsi_dy);
            };
        }
    }

    /**
 * @namespace Rectangle
 * @brief Namespace for debugging purposes, containing functions to compute wavefunctions and their gradients for a rectangular boundary.
 */
    namespace Rectangle {

        /**
         * @brief Computes the wavefunction in Cartesian coordinates for a rectangular boundary.
         * @param n The quantum number for the x direction.
         * @param m The quantum number for the y direction.
         * @param Lx The length of the rectangle in the x direction.
         * @param Ly The length of the rectangle in the y direction.
         * @return A std::function<double(double, double)> representing the wavefunction in Cartesian coordinates (x, y).
         */
        inline std::function<double(double, double)> computeWavefunction(int n, int m, double Lx, double Ly) {
            return [n, m, Lx, Ly](const double x, const double y) {
                const double kx = n * M_PI / Lx;
                const double ky = m * M_PI / Ly;
                return std::sin(kx * x) * std::sin(ky * y);
            };
        }

        /**
         * @brief Computes the gradient of the wavefunction in Cartesian coordinates for a rectangular boundary.
         * @param n The quantum number for the x direction.
         * @param m The quantum number for the y direction.
         * @param Lx The length of the rectangle in the x direction.
         * @param Ly The length of the rectangle in the y direction.
         * @return A std::function<std::vector<double>(double, double)> representing the gradient of the wavefunction in Cartesian coordinates (x, y).
         */
        inline std::function<std::vector<double>(double, double)> computeGradient(int n, int m, double Lx, double Ly) {
            return [n, m, Lx, Ly](const double x, const double y) {
                const double kx = n * M_PI / Lx;
                const double ky = m * M_PI / Ly;
                const double dPsi_dx = kx * std::cos(kx * x) * std::sin(ky * y);
                const double dPsi_dy = ky * std::sin(kx * x) * std::cos(ky * y);
                return std::vector<double>{dPsi_dx, dPsi_dy};
            };
        }
    }

    namespace Plot2DFunctions {

        /**
         * @brief Plots a heatmap of a given 2D function using matplot++.
         *
         * This function plots a heatmap for a given 2D function using Cartesian coordinates.
         *
         * @param ax The matplot::axes_handle for plotting.
         * @param func The 2D function to plot.
         * @param x_min The minimum x value for the plot.
         * @param x_max The maximum x value for the plot.
         * @param y_min The minimum y value for the plot.
         * @param y_max The maximum y value for the plot.
         * @param x_points The number of points in the x direction.
         * @param y_points The number of points in the y direction.
         */
        inline void plotHeatmap(const matplot::axes_handle& ax, const std::function<double(double, double)>& func, const double x_min, const double x_max, const double y_min, const double y_max, const size_t x_points = 500, const size_t y_points = 500) {
            using namespace matplot;

            // Create vectors to store x and y values
            const std::vector<double> x_vals = linspace(x_min, x_max, x_points);
            const std::vector<double> y_vals = linspace(y_min, y_max, y_points);

            // Create a 2D grid to store the function values
            std::vector<std::vector<double>> z_vals(y_points, std::vector<double>(x_points));

            // Compute the function values on the grid
            for (size_t i = 0; i < y_points; ++i) {
                for (size_t j = 0; j < x_points; ++j) {
                    z_vals[i][j] = func(x_vals[j], y_vals[i]);
                }
            }

            // Plot the heatmap
            ax->heatmap(z_vals);
            colorbar(ax, false);
            ax->xlabel("X");
            ax->ylabel("Y");
            ax->title("Heatmap of 2D Function");
        }

    }

}

} // namespace BIM

#endif //TEMPIDEABIM_HPP
