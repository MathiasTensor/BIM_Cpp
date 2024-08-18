#ifndef PLOTTING_HPP
#define PLOTTING_HPP

#pragma once
#include "Boundary.hpp"
#include <Eigen/Dense>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_expint.h>
#include "BIM.hpp"

/**
 * @file Plotting.hpp
 * @brief Header for constructing the wavefunctions and it's associated plots.
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

/**
 * @class EigenfunctionsAndPlotting
 * @brief Computes and plots eigenfunctions and their intensity heatmaps for given refined SVD results and boundary points.
 *
 * This class provides methods for computing eigenfunctions based on refined SVD results and for plotting the intensity
 * heatmap of the wavefunction using matplot++ library. The wavefunction is represented as the square of its absolute value.
 */
class EigenfunctionsAndPlotting {
private:
    std::tuple<double, int, std::vector<Eigen::VectorXcd>> degeneracy_result;
    std::shared_ptr<Boundary::AbstractBoundary> boundary;
    BIM::BoundaryIntegral boundary_integral_;
public:
    /**
     * @brief Constructor for EigenfunctionsAndPlotting.
     *
     * This constructor initializes the class with refined degeneracy results and boundary points.
     *
     * @param bi The BoundaryIntegral object containing all necessary information.
     * @param max_degeneracy The maximum degeneracy to consider.
     */
    EigenfunctionsAndPlotting(BIM::BoundaryIntegral& bi, const int max_degeneracy) : boundary_integral_(bi) {
        // Perform SVD and compute eigenvectors up to max_degeneracy
        auto [_, eigenvectors] = bi.computeSVD(Eigen::ComputeFullU);
        const double k = bi.get_k();
        // Filter and group the eigenvectors by degeneracy
        for (size_t i = 0; i < max_degeneracy; ++i) {
            std::vector<Eigen::VectorXcd> degenerateEigenvectors;
            for (int j = 0; j < max_degeneracy && j < eigenvectors.cols(); ++j) {
                degenerateEigenvectors.emplace_back(eigenvectors.col(j));
            }
            std::get<0>(degeneracy_result) = k;
            std::get<1>(degeneracy_result) = max_degeneracy;
            std::get<2>(degeneracy_result) = degenerateEigenvectors;
        }
        boundary = bi.getBoundary();

        // Calculate normalization for each eigenvector
        std::vector<double> normalization_factors;
        normalization_factors.reserve(std::get<2>(degeneracy_result).size());
        for (int i = 0; i < std::get<2>(degeneracy_result).size(); ++i) {
            normalization_factors.push_back(calculateWavefunctionNormalization(500, 500, i));
        }

        // Normalize eigenvectors
        normalizeEigenvectors(normalization_factors);

    }

    /**
     * @brief Plots the wavefunction intensity heatmap for all degenerate wavefunctions.
     *
     * This method plots a heatmap of the wavefunction intensities for the specified index using matplot++.
     * The intensity is calculated as the square of the absolute value of the wavefunction. If there are multiple
     * degenerate wavefunctions, they will be plotted in a grid layout.
     *
     * @param ax The matplot::axis_handle that will contain all the subplots.
     * @param gridX The size of the grid for plotting the heatmap in the x-direction.
     * @param gridY The size of the grid for plotting the heatmap in the y-direction.
     * @param aspect_ratio For better looking plots that do not have the same x and y scale
     * @param index The index of the degeneracy. If it is not degenerate this is 0. For 1 degeneracy we have 1 etc.
     * @param plotCoordinateAxes If we want to plot the coordinate axes, default true
     * @param plotOutsideBoundary To observe the behaviour outside the boundary
     */
    void plotWavefunctionDensityHeatmap(const matplot::axes_handle &ax, const int gridX = 500, const int gridY = 500, const float aspect_ratio = 1.0, const int index = 0, const bool plotCoordinateAxes = true, const bool plotOutsideBoundary = false) const {
        const auto squareWavefunctions = getSquareWavefunction();

        if (const int numWavefunctions = squareWavefunctions.size(); index < 0 || index >= numWavefunctions) { // NOLINT(*-narrowing-conversions)
            throw std::out_of_range("Index out of range for available wavefunctions");
        }

        double minX, maxX, minY, maxY;
        boundary->getBoundingBox(minX, maxX, minY, maxY);

        if (plotOutsideBoundary) {
            const double centerX = (minX + maxX) / 2.0;
            const double centerY = (minY + maxY) / 2.0;
            const double width = (maxX - minX) * 2.0;
            const double height = (maxY - minY) * 2.0;
            minX = centerX - width / 2.0;
            maxX = centerX + width / 2.0;
            minY = centerY - height / 2.0;
            maxY = centerY + height / 2.0;
        }

        const double dx = (maxX - minX) / (gridX - 1);
        const double dy = (maxY - minY) / (gridY - 1);

        std::vector<std::vector<double>> intensity(gridY, std::vector<double>(gridX, 0.0));
        std::vector<double> x_coords(gridX), y_coords(gridY);

        for (int i = 0; i < gridX; ++i) {
            x_coords[i] = minX + i * dx;
        }

        for (int i = 0; i < gridY; ++i) {
            y_coords[i] = minY + i * dy;
        }

        // Generate boundary points using curveParametrization
        std::vector<Boundary::Point> points;
        constexpr int numBoundaryPoints = 10000;
        for (int i = 0; i <= numBoundaryPoints; ++i) {
            const double t = static_cast<double>(i) / numBoundaryPoints;
            points.push_back(boundary->curveParametrization(t));
        }

        // Check if the first and last points coincide
        const double distance = std::hypot(points.front().x - points.back().x, points.front().y - points.back().y);
        const double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());

        if (const bool isClosed = distance <= tolerance; !isClosed) {
            // Add points from the last point to the origin (0,0)
            for (int i = 0; i <= numBoundaryPoints; ++i) {
                const double t = static_cast<double>(i) / numBoundaryPoints;
                points.emplace_back(
                    points.back().x * (1 - t),
                    points.back().y * (1 - t)
                );
            }
            // Add points from the origin (0,0) to the first point
            for (int i = 0; i <= numBoundaryPoints; ++i) {
                const double t = static_cast<double>(i) / numBoundaryPoints;
                points.emplace_back(
                    points.front().x * t,
                    points.front().y * t
                );
            }
        }

        // Function to compute a chunk of the intensity grid
        auto compute_chunk = [&](const int start, const int end) {
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < gridY; ++j) {
                    const double x = x_coords[i];
                    const double y = y_coords[j];
                    if (!plotOutsideBoundary) {
                        if (boundary->supportsIsInside()) {
                            if (isPointInside(points, x, y)) {
                                if (squareWavefunctions[index]) {
                                    intensity[gridY - j - 1][i] = (*squareWavefunctions[index])(x, y);
                                } else {
                                    throw std::runtime_error("Wavefunction pointer is null");
                                }
                            }
                        } else {
                            if (squareWavefunctions[index]) {
                                intensity[gridY - j - 1][i] = (*squareWavefunctions[index])(x, y);
                            } else {
                                throw std::runtime_error("Wavefunction pointer is null");
                            }
                        }
                    } else {
                        if (squareWavefunctions[index]) {
                            intensity[gridY - j - 1][i] = (*squareWavefunctions[index])(x, y);
                        } else {
                            throw std::runtime_error("Wavefunction pointer is null");
                        }
                    }
                }
            }
        };

        // Launch tasks to compute chunks of the grid in parallel
        const int num_threads = std::thread::hardware_concurrency(); // NOLINT(*-narrowing-conversions)
        const int chunk_size = gridX / num_threads;
        std::vector<std::future<void>> futures;

        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = (t == num_threads - 1) ? gridX : start + chunk_size;
            futures.emplace_back(std::async(std::launch::async, compute_chunk, start, end));
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.get();
        }

        // Use curveParametrization to generate a more precise boundary
        std::vector<double> boundary_x, boundary_y;
        constexpr int numPoints = 1000;
        for (int i = 0; i <= numPoints; ++i) {
            const double t = static_cast<double>(i) / numPoints;
            const auto point = boundary->curveParametrization(t);
            double scaled_x = (point.x - minX) / (maxX - minX) * (gridX - 1);
            const double scaled_y = (point.y - minY) / (maxY - minY) * (gridY - 1);
            boundary_x.push_back(scaled_x);
            boundary_y.push_back(gridY - scaled_y - 1);
        }

        // Determine the center of the grid for the coordinate axes
        const double center_x = (0.0 - minX) / (maxX - minX) * (gridX - 1);
        const double center_y = gridY - (0.0 - minY) / (maxY - minY) * (gridY - 1);

        using namespace matplot;
        auto heatmap_handle = ax->heatmap(intensity);
        colorbar(ax, false);
        ax->hold(true);
        ax->plot(boundary_x, boundary_y, "")->color("red");

        if (plotCoordinateAxes) {
            // Plot dashed lines for x and y coordinate axes
            ax->plot(std::vector<double>{0, static_cast<double>(gridX - 1)}, std::vector<double>{center_y, center_y}, "k--");
            ax->plot(std::vector<double>{center_x, center_x}, std::vector<double>{0, static_cast<double>(gridY - 1)}, "k--");
        }

        // Render the value of k in the top right corner
        const double k_value = boundary_integral_.get_k(); // Replace this with the actual k value you want to display
        std::ostringstream k_stream;
        k_stream << "k = " << std::fixed << std::setprecision(8) << k_value;
        const auto k_text = ax->text(gridX - gridX / 4, gridY / 20, k_stream.str()); // Adjust position as needed NOLINT(*-integer-division)

        ax->hold(false);
        ax->xticks({});
        ax->yticks({});
        ax->axes_aspect_ratio(aspect_ratio);
    }

    /**
     * @brief Plots the Poincare-Husimi function.
     *
     * This method plots the Poincare-Husimi function for a given boundary.
     *
     * @param ax The matplot::axes_handle that will contain the plot.
     * @param index The index of the wavefunction
     * @param grid_size The size of the grid for plotting the Husimi function.
     * @param aspect_ratio The aspect ratio for the plot.
     * @param segmentLabels Provide segment labels if wanted
     */
    void plotPoincareHusimiHeatmap(const matplot::axes_handle& ax, const int grid_size = 500, const double aspect_ratio = 1.0, const std::optional<std::vector<std::string>>& segmentLabels = std::nullopt, const int index = 0) const {
        using namespace matplot;

        // Identify segment markers for CompositeBoundary (if possible)
        std::vector<double> segmentMarkers;
        if (const auto compositeBoundary = std::dynamic_pointer_cast<Boundary::CompositeBoundary>(boundary)) {
            double totalArcLength = 0.0;
            for (const auto& segment : compositeBoundary->getSegments()) {
                totalArcLength += segment->calculateArcLength();
                segmentMarkers.push_back(totalArcLength);
            }
        }

        // Calculate Poincare-Husimi function
        const auto husimi = calculatePoincareHusimi(boundary_integral_.getBoundaryPoints(), grid_size, index);

        // Plot the heatmap
        auto heatmap_handle = ax->heatmap(husimi);
        colorbar(ax, false);
        ax->xlabel("s");
        ax->ylabel("p");
        ax->xticks({});
        ax->yticks({});
        ax->axes_aspect_ratio(aspect_ratio); // NOLINT(*-narrowing-conversions)

        // Convert segment markers to grid coordinates and plot vertical dashed lines
        const double L = boundary->calculateArcLength();
        const double dx = L / grid_size; // Grid spacing in s direction

        ax->hold(matplot::on); // Ensure further plots overlay on top of the heatmap
        for (size_t i = 0; i < segmentMarkers.size(); ++i) {
            const double marker_grid_x = segmentMarkers[i] / dx;
            ax->plot(std::vector<double>{marker_grid_x, marker_grid_x}, std::vector<double>{0.0, static_cast<double>(grid_size)}, "--k");

            // Add segment labels if provided
            if (segmentLabels && i < segmentLabels->size()) {
                const auto text_handle = ax->text(marker_grid_x, 0.95*grid_size, segmentLabels->at(i));
                text_handle->alignment(matplot::labels::alignment::right);
                text_handle->font_size(10);
            }
        }
        ax->hold(matplot::off);
    }

    /**
     * @brief Plots the radially integrated momentum density for the specified eigenvector.
     *
     * This method plots the radially integrated momentum density \( I_n(\phi) \) for the specified eigenvector
     * using the computeRadiallyIntegratedDensity function. The plot is generated using the matplot++ library.
     *
     * @param ax The matplot::axis_handle for plotting.
     * @param index The index of the eigenvector (in case of degeneracy).
     * @param grid_size The size of the grid for plotting the density.
     * @param aspect_ratio The aspect ratio for the plot.
     * @return The matplot::line_handle containing the plot.
     */
    [[nodiscard]] matplot::line_handle plotRadiallyIntegratedMomentumDensity(const matplot::axes_handle &ax, const int index = 0, const int grid_size = 500, const float aspect_ratio = 1.0) const {
        using namespace matplot;
        const auto densities = computeRadiallyIntegratedDensity(); // Compute the radially integrated densities
        if (index < 0 || index >= densities.size()) {
            throw std::out_of_range("Index out of range for available densities");
        }
        // Generate angles from 0 to 2*pi
        std::vector<double> angles(grid_size);
        for (int i = 0; i < grid_size; ++i) {
            angles[i] = i * 2 * M_PI / grid_size;
        }
        std::vector<double> values(grid_size);
        for (int i = 0; i < grid_size; ++i) {
            values[i] = (*densities[index])(angles[i]); // Evaluate the density at each angle
        }
        // Determine the maximum y value for setting the plot limits
        const double max_y = *std::ranges::max_element(values);
        // Scale values by the maximum y value
        for (auto &value : values) {
            value /= max_y;
        }
        // Create the plot
        const auto plt = ax->plot(angles, values);
        ax->xlabel("φ");
        ax->ylabel("Radially Integrated Momentum Density (Relative)");
        ax->xlim({0, 2*M_PI});
        ax->ylim({0, 1.0});
        ax->axes_aspect_ratio(aspect_ratio);

        // Set custom x-ticks
        const std::vector<double> x_ticks = {0, M_PI / 4, M_PI / 2, 3 * M_PI / 4, M_PI, 5 * M_PI / 4, 3 * M_PI / 2, 7 * M_PI / 4, 2 * M_PI};
        const std::vector<std::string> x_tick_labels = {"0", "π/4", "π/2", "3π/4", "π", "5π/4", "3π/2", "7π/4", "2π"};

        ax->xticks(x_ticks);
        ax->xticklabels(x_tick_labels);

        return plt;
    }

    /**
     * @brief Plots the angular integrated momentum density for the specified eigenvector.
     *
     * This method plots the angular integrated momentum density \( R_n(r) \) for the specified eigenvector
     * using the computeAngularIntegratedMomentumDensity function. The plot is generated using the matplot++ library.
     *
     * @param ax The matplot::axis_handle for plotting.
     * @param index The index of the eigenvector (in case of degeneracy).
     * @param grid_size The size of the grid for plotting the density.
     * @param aspect_ratio The aspect ratio for the plot.
     * @return The std::pair containing the matplot::axis_handle and the line_handle of the vertical line at r = k.
     */
    [[nodiscard]] std::pair<matplot::line_handle, matplot::line_handle> plotAngularIntegratedMomentumDensity(const matplot::axes_handle &ax, const int index = 0, const int grid_size = 500, const float aspect_ratio = 1.0) const {
        using namespace matplot;
        const auto densities = computeAngularIntegratedMomentumDensity(); // Compute the angular integrated densities
        if (index < 0 || index >= densities.size()) {
            throw std::out_of_range("Index out of range for available densities");
        }
        // Get k value from degeneracy result
        const auto [k, degeneracy, eigenvectors] = degeneracy_result;
        // Generate r values from 0 to 2*k
        std::vector<double> r_values(grid_size);
        for (int i = 0; i < grid_size; ++i) {
            r_values[i] = i * 2 * k / (grid_size - 1);
        }
        std::vector<double> values(grid_size);
        for (int i = 0; i < grid_size; ++i) {
            values[i] = (*densities[index])(r_values[i]); // Evaluate the density at each r value
        }
        // Determine the maximum y value for setting the plot limits
        const double max_y = *std::ranges::max_element(values);
        // Scale values by the maximum y value
        for (auto &value : values) {
            value /= max_y;
        }
        // Create the plot
        const auto line_handle_plt = ax->plot(r_values, values);
        ax->xlabel("r");
        ax->ylabel("Angular Integrated Momentum Density (Relative)");
        ax->ylim({0, 1.0});
        ax->axes_aspect_ratio(aspect_ratio);
        // Add vertical red line at r = k
        ax->hold(matplot::on);
        const auto line_handle_vert = ax->plot(std::vector<double>{k, k}, std::vector<double>{0, 1}, "r--");
        ax->hold(matplot::off);
        return {line_handle_plt, line_handle_vert};
    }

    /**
     * @brief Plots the heatmap of the momentum wavefunction intensity in polar coordinates.
     *
     * This method generates a heatmap plot of the square of the momentum wavefunction's absolute value
     * (intensity) as a function of polar coordinates (r, phi). The method uses the eigenfunctions computed
     * by the `computeMomentumEigenfunctionPolar()` method and transforms them to Cartesian coordinates for
     * plotting using matplot++.
     *
     * @param ax The matplot::axes_handle that will contain the heatmap plot.
     * @param gridX The number of grid points in the x-direction (phi).
     * @param gridY The number of grid points in the y-direction (r).
     * @param aspect_ratio The aspect ratio for the plot.
     * @param index The index of the momentum wavefunction (in case of degeneracy).
     */
    void plotMomentumDensityHeatmapPolar(const matplot::axes_handle &ax, const int gridX = 500, const int gridY = 500,
                                         const double aspect_ratio = 1.0, const int index = 0) const {
        const auto squareMomentumWavefunctions = getSquareMomentumWavefunctionPolar();

        if (index < 0 || index >= squareMomentumWavefunctions.size()) {
            throw std::out_of_range("Index out of range for available wavefunctions");
        }

        const auto [k, degeneracy, eigenvectors] = degeneracy_result;
        const double k_magnitude = k;
        const double r_max = 1.0 * k_magnitude;
        constexpr double r_min = 0;
        constexpr double phi_max = 2 * M_PI;
        constexpr double phi_min = 0;

        const std::vector<double> r_values = matplot::linspace(r_min, r_max, gridY);
        const std::vector<double> phi_values = matplot::linspace(phi_min, phi_max, gridX);
        std::vector<std::vector<double>> intensity(gridY, std::vector<double>(gridX, 0.0));

        for (int i = 0; i < gridY; ++i) {
            for (int j = 0; j < gridX; ++j) {
                intensity[i][j] = (*squareMomentumWavefunctions[index])(r_values[i], phi_values[j]);
            }
        }

        using namespace matplot;
        // Plot the heatmap of intensity
        auto hm = ax->heatmap(intensity);
        hold(ax, on);
        // Set up the axis labels and ticks
        ax->xlabel("φ");
        ax->ylabel("r");

        // Setting xticks and yticks to align correctly with the grid
        const std::vector<double> x_ticks = {0.0, gridX / 8.0, gridX / 4.0, 3.0 * gridX / 8.0, gridX / 2.0, 5.0 * gridX / 8.0, 3.0 * gridX / 4.0, 7.0 * gridX / 8.0, static_cast<double>(gridX)};
        const std::vector<std::string> x_tick_labels = {"0", "π/4", "π/2", "3π/4", "π", "5π/4", "3π/2", "7π/4", "2π"};

        // Add custom vertical and horizontal lines
        for (const auto &x_tick : x_ticks) {
            ax->plot(std::vector<double>{x_tick, x_tick}, std::vector<double>{0.0, static_cast<double>(gridY)}, "k--");
        }

        // Render custom x-axis tick labels
        for (size_t i = 0; i < x_ticks.size(); ++i) {
            const auto txt = ax->text(x_ticks[i], gridY/20, x_tick_labels[i]); // NOLINT(*-integer-division)
            txt->font_size(10);
            txt->color("black");
            txt->alignment(matplot::labels::alignment::center);
        }

        ax->xticks({});
        ax->yticks({});
        ax->yticklabels({});
        ax->axes_aspect_ratio(aspect_ratio); // NOLINT(*-narrowing-conversions)
        colorbar(ax, false);
        hold(ax, off);
    }

    /**
     * @brief Plots the normal derivative of the wavefunction as a function of the arc length parameter s.
     *
     * This method extracts the normal derivatives of the wavefunction, represented as the eigenvector u[s],
     * and plots them against the arc length parameter s. It supports handling composite boundaries by
     * adding vertical dashed lines at segment boundaries.
     *
     * @param ax The matplot::axes_handle for plotting.
     * @param index The index of the eigenvector to use (for higher degeneracies).
     * @param aspect_ratio The aspect ratio for the plot.
     * @param plotSegmentMarkers Whether to plot vertical lines for segment boundaries (default: true).
     * @param segmentLabels If we have a composite boundary we can add the labels for the parts that constitute the boundary
     * @return The matplot::axes_handle containing the plot.
     */
    [[nodiscard]] matplot::axes_handle plotNormalDerivativeOfWavefunction(const matplot::axes_handle& ax, const int index = 0, const double aspect_ratio = 1.0, const bool plotSegmentMarkers = true, const std::optional<std::vector<std::string>>& segmentLabels = std::nullopt) const {
        using namespace matplot;
        try {
            // Extract the normal derivatives of the wavefunction (u[s])
            const auto& [k, degeneracy, eigenvectors] = degeneracy_result;

            // Ensure the index is valid
            if (index < 0 || index >= eigenvectors.size()) {
                throw std::runtime_error("Eigenfunction index out of range");
            }

            // Use the specified eigenvector for the plot
            const Eigen::VectorXcd& u_s = eigenvectors[index];

            // Get boundary points, parametrization values, and arc lengths
            const auto& boundary_points = boundary_integral_.getBoundaryPoints();
            const auto& s_values = boundary_integral_.getArclengthValues();
            const double totalArcLength = boundary_integral_.calculateArcLength();

            std::vector<double> u_real(boundary_points.size());
            for (size_t i = 0; i < boundary_points.size(); ++i) {
                u_real[i] = u_s[i].real(); // NOLINT(*-narrowing-conversions)
            }

            // Plot the normal derivatives
            ax->plot(s_values, u_real);
            ax->xlabel("s");
            ax->ylabel("Normal Derivative of Wavefunction u[s]");

            // Set x-axis limits
            ax->xlim({0.0, totalArcLength});
            ax->ylim({-*std::ranges::max_element(u_real), *std::ranges::max_element(u_real)});

            // Identify segment markers for CompositeBoundary and plot vertical lines
            if (plotSegmentMarkers && std::dynamic_pointer_cast<Boundary::CompositeBoundary>(boundary)) {
                std::vector<double> segmentMarkers;
                double accumulatedArcLength = 0.0;
                for (const auto& segment : std::dynamic_pointer_cast<Boundary::CompositeBoundary>(boundary)->getSegments()) {
                    accumulatedArcLength += segment->calculateArcLength();
                    segmentMarkers.push_back(accumulatedArcLength);
                }

                ax->hold(matplot::on);
                // Plot vertical dashed lines at segment markers
                for (size_t i = 0; i < segmentMarkers.size(); ++i) {
                    const double si = segmentMarkers[i];
                    ax->plot({si, si}, {-0.99**std::ranges::max_element(u_real), 0.99**std::ranges::max_element(u_real)}, "--k");

                    // Add segment labels if provided
                    if (segmentLabels && i < segmentLabels->size()) {
                        const auto text_handle = ax->text(si - totalArcLength/20, 0.9*(*std::ranges::max_element(u_real)), segmentLabels->at(i));
                        text_handle->alignment(matplot::labels::alignment::center);
                        text_handle->font_size(10);
                    }
                }
                ax->hold(matplot::off);
            }

            ax->axes_aspect_ratio(aspect_ratio); // NOLINT(*-narrowing-conversions)
        } catch (const std::exception &e) {
            std::cerr << "Error plotting normal derivative of wavefunction: " << e.what() << std::endl;
        }

        return ax;
    }

    /**
     * @brief Plots the value of the wavefunction on the boundary as a function of the arc length parameter s. For this one should choose a high point dendity per wavelength (b parameter) to get small values on the boundary
     *
     * This method extracts the wavefunction values from the square of the wavefunction, and plots them against the arc length parameter s.
     *
     * @param ax The matplot::axes_handle for plotting.
     * @param index The index of the wavefunction.
     * @param aspect_ratio The aspect ratio for the plot.
     * @param plotSegmentMarkers Whether to plot vertical lines for segment boundaries (default: true).
     * @param segmentLabels If we have a composite boundary we can add the labels for the parts that constitute the boundary.
     * @return The matplot::axes_handle containing the plot.
     */
    [[nodiscard]] matplot::axes_handle plotWavefunctionOnBoundary(const matplot::axes_handle& ax, const int index = 0, const double aspect_ratio = 1.0, const bool plotSegmentMarkers = true, const std::optional<std::vector<std::string>>& segmentLabels = std::nullopt) const {
        using namespace matplot;
        try {
            // Ensure the index is valid
            const auto Wavefunctions = computeEigenfunction();
            if (index < 0 || index >= Wavefunctions.size()) {
                throw std::runtime_error("Wavefunction index out of range");
            }

            // Get boundary points, parametrization values, and arc lengths
            const auto& boundary_points = boundary_integral_.getBoundaryPoints();
            const auto& s_values = boundary_integral_.getArclengthValues();
            const double totalArcLength = boundary_integral_.calculateArcLength();

            std::vector<double> wavefunction_values(boundary_points.size());
            for (size_t i = 0; i < boundary_points.size(); ++i) {
                const double x = boundary_points[i].x;
                const double y = boundary_points[i].y;
                wavefunction_values[i] = std::abs((*Wavefunctions[index])(x, y));
            }

            // Plot the wavefunction values
            ax->scatter(s_values, wavefunction_values);
            ax->xlabel("s");
            ax->ylabel("Wavefunction Value on Boundary |ψ(s)|");

            // Set x-axis limits
            ax->xlim({0.0, totalArcLength});
            ax->ylim({0.0, *std::ranges::max_element(wavefunction_values)});

            // Identify segment markers for CompositeBoundary and plot vertical lines
            if (plotSegmentMarkers && std::dynamic_pointer_cast<Boundary::CompositeBoundary>(boundary)) {
                std::vector<double> segmentMarkers;
                double accumulatedArcLength = 0.0;
                for (const auto& segment : std::dynamic_pointer_cast<Boundary::CompositeBoundary>(boundary)->getSegments()) {
                    accumulatedArcLength += segment->calculateArcLength();
                    segmentMarkers.push_back(accumulatedArcLength);
                }

                ax->hold(matplot::on);
                // Plot vertical dashed lines at segment markers
                for (size_t i = 0; i < segmentMarkers.size(); ++i) {
                    const double si = segmentMarkers[i];
                    ax->plot({si, si}, {0.0, 0.99**std::ranges::max_element(wavefunction_values)}, "--k");

                    // Add segment labels if provided
                    if (segmentLabels && i < segmentLabels->size()) {
                        const auto text_handle = ax->text(si - totalArcLength/20, 0.9*(*std::ranges::max_element(wavefunction_values)), segmentLabels->at(i));
                        text_handle->alignment(matplot::labels::alignment::center);
                        text_handle->font_size(10);
                    }
                }
                ax->hold(matplot::off);
            }

            ax->axes_aspect_ratio(aspect_ratio); // NOLINT(*-narrowing-conversions)
        } catch (const std::exception &e) {
            std::cerr << "Error plotting wavefunction on boundary: " << e.what() << std::endl;
        }

        return ax;
    }

    /**
     * @brief Calculates the largest value of the wavefunction on the grid.
     *
     * This method computes the largest value of the wavefunction intensity on a specified grid.
     * It divides the computation into chunks and uses multiple threads to parallelize the process.
     * The method checks if a point is inside the boundary before computing the wavefunction value.
     *
     * @param gridX The number of grid points in the x-direction.
     * @param gridY The number of grid points in the y-direction.
     * @param index The index of the wavefunction.
     * @return The largest value of the wavefunction intensity on the grid.
     * @throws std::runtime_error if the boundary does not support the isInside method.
     */
    [[nodiscard]] double calculateLargestWavefunctionValue(const int gridX = 500, const int gridY = 500, const int index = 0) const {
        const auto squareWavefunctions = getSquareWavefunction();

        if (index < 0 || index >= squareWavefunctions.size()) {
            throw std::out_of_range("Index out of range for available wavefunctions");
        }

        double minX, maxX, minY, maxY;
        boundary->getBoundingBox(minX, maxX, minY, maxY);
        const double dx = (maxX - minX) / (gridX - 1);
        const double dy = (maxY - minY) / (gridY - 1);

        std::vector<double> x_coords(gridX), y_coords(gridY);

        for (int i = 0; i < gridX; ++i) {
            x_coords[i] = minX + i * dx;
        }

        for (int i = 0; i < gridY; ++i) {
            y_coords[i] = minY + i * dy;
        }

        std::vector<Boundary::Point> points;
        constexpr int numBoundaryPoints = 1000;
        for (int i = 0; i <= numBoundaryPoints; ++i) {
            const double t = static_cast<double>(i) / numBoundaryPoints;
            points.push_back(boundary->curveParametrization(t));
        }

        const int num_threads = std::thread::hardware_concurrency(); // NOLINT(*-narrowing-conversions)
        const int chunk_size = gridX / num_threads;
        std::vector<std::future<double>> futures;

        auto find_max_in_chunk = [&](const int start, const int end) {
            double max_value = 0.0;
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < gridY; ++j) {
                    const double x = x_coords[i];
                    const double y = y_coords[j];
                    if (boundary->supportsIsInside()) {
                        if (isPointInside(points, x, y)) {
                            max_value = std::max(max_value, (*squareWavefunctions[index])(x, y));
                        }
                    } else {
                        throw std::runtime_error("Boundary does not support isInside. Implementation required.");
                    }
                }
            }
            return max_value;
        };

        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = (t == num_threads - 1) ? gridX : start + chunk_size;
            futures.emplace_back(std::async(std::launch::async, find_max_in_chunk, start, end));
        }

        double max_value = 0.0;
        for (auto& future : futures) {
            max_value = std::max(max_value, future.get());
        }

        return max_value;
    }

    /**
     * @brief Calculates the area integral of the wavefunction density on the grid.
     *
     * This method computes the area integral of the wavefunction intensity on a specified grid.
     * It divides the computation into chunks and uses multiple threads to parallelize the process.
     * The method checks if a point is inside the boundary before computing the wavefunction value.
     *
     * @param gridX The number of grid points in the x-direction.
     * @param gridY The number of grid points in the y-direction.
     * @param index The index of the wavefunction.
     * @return The area integral of the wavefunction intensity on the grid.
     * @throws std::runtime_error if the boundary does not support the isInside method.
     */
    [[nodiscard]] double calculateWavefunctionNormalization(const int gridX = 500, const int gridY = 500, const int index = 0) const {
        const auto squareWavefunctions = getSquareWavefunction();

        if (index < 0 || index >= squareWavefunctions.size()) {
            throw std::out_of_range("Index out of range for available wavefunctions");
        }

        double minX, maxX, minY, maxY;
        boundary->getBoundingBox(minX, maxX, minY, maxY);
        const double dx = (maxX - minX) / (gridX - 1);
        const double dy = (maxY - minY) / (gridY - 1);

        std::vector<double> x_coords(gridX), y_coords(gridY);

        for (int i = 0; i < gridX; ++i) {
            x_coords[i] = minX + i * dx;
        }

        for (int i = 0; i < gridY; ++i) {
            y_coords[i] = minY + i * dy;
        }

        std::vector<Boundary::Point> points;
        constexpr int numBoundaryPoints = 1000;
        for (int i = 0; i <= numBoundaryPoints; ++i) {
            const double t = static_cast<double>(i) / numBoundaryPoints;
            points.push_back(boundary->curveParametrization(t));
        }

        const int num_threads = std::thread::hardware_concurrency(); // NOLINT(*-narrowing-conversions)
        const int chunk_size = gridX / num_threads;
        std::vector<std::future<double>> futures;

        auto compute_integral_in_chunk = [&](const int start, const int end) {
            double local_integral = 0.0;
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < gridY; ++j) {
                    const double x = x_coords[i];
                    const double y = y_coords[j];
                    if (boundary->supportsIsInside()) {
                        if (isPointInside(points, x, y)) {
                            local_integral += (*squareWavefunctions[index])(x, y) * dx * dy;
                        }
                    } else {
                        throw std::runtime_error("Boundary does not support isInside. Implementation required.");
                    }
                }
            }
            return local_integral;
        };

        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = (t == num_threads - 1) ? gridX : start + chunk_size;
            futures.emplace_back(std::async(std::launch::async, compute_integral_in_chunk, start, end));
        }

        double integral = 0.0;
        for (auto& future : futures) {
            integral += future.get();
        }

        return integral;
    }

    /**
     * @brief Plots the curvature as a function of the arc length parameter. Can have problems plotting if all the y-values are 0 like in polygonal billiards
     *
     * This method plots the curvature of the boundary as a function of the arc length parameter (s).
     * It also supports handling composite boundaries by adding vertical dashed lines at segment boundaries.
     * Optionally, labels can be added to these segment boundaries.
     *
     * @param ax The matplot::axes_handle for plotting.
     * @param aspect_ratio The aspect ratio for the plot.
     * @param plotSegmentMarkers If the segment markers for the composite boundary will be plotted. If not a composite boundary it does nothing
     * @param segmentLabels An optional vector of strings for labeling the segments (default: std::nullopt).
     *
     * @note This method uses the curvature values and arc length parameters provided by the BoundaryIntegral class.
     *       It assumes that the curvature values and arc length parameters are already computed and available.
     *
     * @throws std::exception If an error occurs during plotting.
     */
    void plotCurvature(const matplot::axes_handle& ax, const double aspect_ratio = 1.0, const bool plotSegmentMarkers = false, const std::optional<std::vector<std::string>>& segmentLabels = std::nullopt) const {
        using namespace matplot;
        try {
            // Get arc length parameters and curvature values
            const auto& arc_parameters = boundary_integral_.getArclengthValues();
            const auto& curvature_values = boundary_integral_.getCurvatureValues();
            const auto totalArcLength = boundary->calculateArcLength();
            // Plot the curvature values
            ax->plot(arc_parameters, curvature_values);
            ax->xlabel("Arc Length Parameter (s)");
            ax->ylabel("Curvature");

            // Set x-axis limits
            ax->xlim({0.0, totalArcLength});
            ax->ylim({-1.2**std::ranges::max_element(curvature_values), 1.2**std::ranges::max_element(curvature_values)});

            // Identify segment markers for CompositeBoundary and plot vertical lines
            if (plotSegmentMarkers && std::dynamic_pointer_cast<Boundary::CompositeBoundary>(boundary)) {
                std::vector<double> segmentMarkers;
                double accumulatedArcLength = 0.0;
                for (const auto& segment : std::dynamic_pointer_cast<Boundary::CompositeBoundary>(boundary)->getSegments()) {
                    accumulatedArcLength += segment->calculateArcLength();
                    segmentMarkers.push_back(accumulatedArcLength);
                }

                ax->hold(matplot::on);
                // Plot vertical dashed lines at segment markers
                for (size_t i = 0; i < segmentMarkers.size(); ++i) {
                    const double si = segmentMarkers[i];
                    ax->plot({si, si}, {-0.99**std::ranges::max_element(curvature_values), 0.99**std::ranges::max_element(curvature_values)}, "--k");

                    // Add segment labels if provided
                    if (segmentLabels && i < segmentLabels->size()) {
                        const auto text_handle = ax->text(si - totalArcLength / 20, 0.9 * (*std::ranges::max_element(curvature_values)), segmentLabels->at(i));
                        text_handle->alignment(matplot::labels::alignment::center);
                        text_handle->font_size(10);
                    }
                }
                ax->hold(matplot::off);
            }

            ax->axes_aspect_ratio(aspect_ratio); // NOLINT(*-narrowing-conversions)
        } catch (const std::exception &e) {
            std::cerr << "Error plotting curvature: " << e.what() << std::endl;
        }
    }

    /**
     * @brief Computes the angular difference between the normals of consecutive boundary points.
     *
     * This method calculates the angular difference between the x and y components of the normals
     * for each pair of consecutive boundary points using the dot product.
     *
     * @return A vector of pairs where each pair contains the parameterization value t , s value and the angular difference value.
     */
    [[nodiscard]] std::vector<std::tuple<double, double, double>> computeAngularDifferenceOfNormals() const {
        const auto& points = boundary_integral_.getBoundaryPoints();
        const auto& normals = boundary_integral_.getBoundaryNormals();
        const auto& t_values = boundary_integral_.getParametrizationValues();
        const auto& s_values = boundary_integral_.getArclengthValues();
        const int num_points = points.size(); // NOLINT(*-narrowing-conversions)

        std::vector<std::tuple<double, double, double>> angular_diff_values;

        for (int i = 0; i < num_points; ++i) {
            const int next_idx = (i + 1) % num_points;
            const auto& n1 = normals[i];
            const auto& n2 = normals[next_idx];

            const double dot_product = n1.x * n2.x + n1.y * n2.y;
            const double magnitude1 = std::sqrt(n1.x * n1.x + n1.y * n1.y);
            const double magnitude2 = std::sqrt(n2.x * n2.x + n2.y * n2.y);

            // Cosine of the angle between the normals
            const double cos_theta = dot_product / (magnitude1 * magnitude2);
            const double theta = std::acos(std::clamp(cos_theta, -1.0, 1.0)); // Angle in radians
            angular_diff_values.emplace_back(t_values[i], s_values[i], theta);
        }

        return angular_diff_values;
    }

    /**
     * @brief Plots the angular difference between the normals vs. the arc parameter s.
     *
     * This method computes the angular difference between the normals for consecutive points
     * and plots these values against the arc length parameter s.
     *
     * @param ax The matplot::axes_handle for plotting.
     * @param aspect_ratio The aspect ratio for the plot.
     * @param plotSegmentLabels Whether to plot segment labels (default: false).
     * @param segmentLabels An optional vector of strings for labeling the segments (default: std::nullopt).
     * @return A tuple of vectors that contain the parametrization (t) values, the arc parameter (s) values and the smoothed angular difference values (sigmoid scaled)
     * @note This method assumes that the boundary points and normals are already computed and available.
     */
    [[nodiscard]] std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> plotAngularDifferenceOfNormalsVsArcLength(
        const matplot::axes_handle& ax,
        const double aspect_ratio = 1.0,
        const bool plotSegmentLabels = false,
        const std::optional<std::vector<std::string>>& segmentLabels = std::nullopt
    ) const {
        using namespace matplot;

        const auto angular_diff_values = computeAngularDifferenceOfNormals();
        const auto totalArcLength = boundary->calculateArcLength();

        std::vector<double> angular_diff, angular_diff_sigmoid, s_values, t_values;
        for (const auto & angular_diff_value : angular_diff_values) {
            constexpr double scaling_factor = 1.0;
            angular_diff_sigmoid.emplace_back(scaling_factor * 1 / (1 + std::exp(-std::get<2>(angular_diff_value))));
            angular_diff.emplace_back(std::get<2>(angular_diff_value));
            s_values.emplace_back(std::get<1>(angular_diff_value));
            t_values.emplace_back(std::get<0>(angular_diff_value));
        }

        auto sc = ax->scatter(s_values, angular_diff)->marker_size(2.0);
        sc.color("blue");
        ax->xlabel("Arc Length Parameter (s)");
        ax->ylabel("Angular Difference of Normals");

        // Fit a spline to the angular differences (ignoring the start for now)
        gsl_interp_accel* acc_sigm = gsl_interp_accel_alloc();
        gsl_interp_accel* acc_reg = gsl_interp_accel_alloc();
        gsl_spline* spline_sigm = gsl_spline_alloc(gsl_interp_cspline, s_values.size());
        gsl_spline* spline_reg = gsl_spline_alloc(gsl_interp_cspline, s_values.size());
        gsl_spline_init(spline_sigm, s_values.data(), angular_diff_sigmoid.data(), s_values.size());
        gsl_spline_init(spline_reg, s_values.data(), angular_diff.data(), s_values.size());


        // Evaluate the spline at the original s_values to get the smoothed angle differences
        std::vector<double> smoothed_angle_diff_sigmoid(s_values.size());
        std::vector<double> smoothed_angle_diff_reg(s_values.size());
        for (size_t i = 0; i < s_values.size(); ++i) {
            smoothed_angle_diff_sigmoid[i] = gsl_spline_eval(spline_sigm, s_values[i], acc_sigm);
            smoothed_angle_diff_reg[i] = gsl_spline_eval(spline_reg, s_values[i], acc_reg);
        }

        // Assign the last value to the first position b/c of periodicity
        smoothed_angle_diff_sigmoid[0] = smoothed_angle_diff_sigmoid.back();
        smoothed_angle_diff_reg[0] = smoothed_angle_diff_reg.back();

        // Plot the smoothed angular differences
        ax->hold(matplot::on);  // Hold the plot to overlay the smoothed curve
        auto plt_sigmoid = ax->plot(s_values, smoothed_angle_diff_sigmoid)->line_width(1.5).color("red");
        plt_sigmoid.display_name("Sigmoid Angle diff Cubic spline");
        auto plt_reg = ax->plot(s_values, smoothed_angle_diff_reg)->line_width(0.5).color("green");
        plt_reg.display_name("Regular Angle diff Cubic spline");

        // Identify segment markers for CompositeBoundary and plot vertical lines
        if (plotSegmentLabels && std::dynamic_pointer_cast<Boundary::CompositeBoundary>(boundary)) {
            std::vector<double> segmentMarkers;
            double accumulatedArcLength = 0.0;
            for (const auto& segment : std::dynamic_pointer_cast<Boundary::CompositeBoundary>(boundary)->getSegments()) {
                accumulatedArcLength += segment->calculateArcLength();
                segmentMarkers.push_back(accumulatedArcLength);
            }

            // Plot vertical dashed lines at segment markers
            for (size_t i = 0; i < segmentMarkers.size(); ++i) {
                const double si = segmentMarkers[i];
                ax->plot({si, si}, {-0.99**std::ranges::max_element(angular_diff), 1.15**std::ranges::max_element(angular_diff)}, "--k");

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
        const auto l = matplot::legend(ax, {"Curvature", "Sigmoid spline", "Regular Spline"});
        l->location(legend::general_alignment::topleft);
        l->visible(false);

        gsl_spline_free(spline_reg);  // Free the memory allocated for the spline
        gsl_interp_accel_free(acc_reg);  // Free the memory allocated for the accelerator
        gsl_spline_free(spline_sigm);
        gsl_interp_accel_free(acc_sigm);
        return std::make_tuple(t_values, s_values, smoothed_angle_diff_sigmoid);
    }

    /**
     * @brief Calculates the overlap between the Poincare Husimi grid values and the classical phase space.
     *
     * This method computes the overlap between the Poincare Husimi function values (H) and the classical phase
     * space grid (A). The classical phase space grid is provided as input and contains counts of a classical
     * trajectory visiting each cell. The method processes this grid to create a binary grid (A) where cells
     * visited by the chaotic component of the trajectory are marked as +1, and cells not visited are marked as -1.
     * It then calculates the Poincare Husimi function values on the same grid and computes the matrix product of
     * H and the transpose of A, taking the trace to obtain the final overlap value (M).
     *
     * @param classicalPhaseSpaceChaoticCount A grid containing counts of a classical trajectory visiting each cell.
     * @return The overlap value (M) as a double.
     * @throws std::invalid_argument If the classical phase space chaotic count grid is empty.
     * @throws std::runtime_error If the classical phase space chaotic count grid has inconsistent column sizes,
     * or if the sizes of H and A do not match.
     */
    [[nodiscard]] double calculatePoincareHusimiOverlapWithClassicalPhaseSpace(const std::vector<std::vector<int>>& classicalPhaseSpaceChaoticCount) const {
        if (classicalPhaseSpaceChaoticCount.empty() || classicalPhaseSpaceChaoticCount[0].empty()) {
            throw std::invalid_argument("Classical phase space chaotic count grid is empty.");
        }

        const size_t rows = classicalPhaseSpaceChaoticCount.size();
        const size_t cols = classicalPhaseSpaceChaoticCount[0].size();

        // Runtime check to ensure all rows have the same number of columns
        for (const auto& row : classicalPhaseSpaceChaoticCount) {
            if (row.size() != cols) {
                throw std::runtime_error("Inconsistent number of columns in classicalPhaseSpaceChaoticCount.");
            }
        }

        // Step 1: Convert classicalPhaseSpaceChaoticCount to grid A
        std::vector<std::vector<int>> A(rows, std::vector<int>(cols, 0));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                A[i][j] = (classicalPhaseSpaceChaoticCount[i][j] > 0) ? 1 : -1; // If we have any (p,s) trajectory point in the cell this is chaotic
            }
        }

        // Step 2: Calculate Poincare Husimi function values H
        const auto H = calculatePoincareHusimi(boundary_integral_.getBoundaryPoints(), rows); // NOLINT(*-narrowing-conversions)

        if (H.size() != rows || H[0].size() != cols) {
            throw std::runtime_error("Size mismatch between H and A.");
        }

        // Step 3: Calculate the overlap using multithreading
        const int num_threads = std::thread::hardware_concurrency(); // NOLINT(*-narrowing-conversions)
        const size_t chunk_size = rows / num_threads;
        std::vector<std::future<double>> futures;

        auto compute_overlap_chunk = [&](const size_t start_row, const size_t end_row) {
            double chunk_overlap = 0.0;
            for (size_t i = start_row; i < end_row; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    chunk_overlap += H[i][j] * static_cast<double>(A[i][j]);
                }
            }
            return chunk_overlap;
        };

        for (int t = 0; t < num_threads; ++t) {
            size_t start_row = t * chunk_size;
            size_t end_row = (t == num_threads - 1) ? rows : start_row + chunk_size;
            futures.emplace_back(std::async(std::launch::async, compute_overlap_chunk, start_row, end_row));
        }

        double M = 0.0;
        for (auto& future : futures) {
            M += future.get();
        }

        return M;
    }

    /**
     * @brief Arranges the axes in a tightly ordered layout within the given figure.
     *
     * This method takes a figure handle, a vector of axes handles (in plot_id ascending order),
     * and arranges the axes in a tightly ordered layout based on the specified number of rows and columns.
     *
     * @param fig The figure handle.
     * @param axes_handles The vector of axes handles in plot_id ascending order.
     * @param nrows The number of rows in the layout.
     * @param ncols The number of columns in the layout.
     * @param height The height of the figure.
     * @param margin The margin between plots (default: 0.005).
     * @param edge_margin
     */
    static void arrangeAxesInFigure(const matplot::figure_handle& fig, const std::vector<matplot::axes_handle>& axes_handles, const int nrows, const int ncols, const double height, const double margin = 0.005, double edge_margin = 0.05) {
        const double width = static_cast<double>(ncols) / nrows * height;
        fig->size(width, height); // NOLINT(*-narrowing-conversions)

        const double subplot_width = (1.0 - 2 * edge_margin - margin * (ncols - 1)) / ncols;
        const double subplot_height = (1.0 - 2 * edge_margin - margin * (nrows - 1)) / nrows;

        for (int row = 0; row < nrows; ++row) {
            for (int col = 0; col < ncols; ++col) {
                const int plot_id = row * ncols + col;
                if (plot_id >= axes_handles.size()) {
                    return; // End the function if we've placed all axes
                }

                const auto& ax = axes_handles[plot_id];
                const double x_pos = edge_margin + col * (subplot_width + margin);
                const double y_pos = 1.0 - edge_margin - (row + 1) * subplot_height - row * margin;

                try {
                    ax->position({static_cast<float>(x_pos), static_cast<float>(y_pos), static_cast<float>(subplot_width), static_cast<float>(subplot_height)});

                    // Check if the axis data is empty or constant
                    if (ax->ylim()[1] == ax->ylim()[0]) {
                        std::cout << "Warning: Axis " << plot_id << " has constant or empty y data." << std::endl;
                        ax->ylim({ax->ylim()[0] - 1, ax->ylim()[1] + 1}); // Adjust the y-limits manually
                    }

                    // Special handling for the last axis
                    if (plot_id == axes_handles.size() - 1) {
                        if (ax->ylim()[1] == ax->ylim()[0]) {
                            std::cout << "Warning: Last axis " << plot_id << " has constant or empty y data." << std::endl;
                            ax->ylim({ax->ylim()[0] - 1, ax->ylim()[1] + 1}); // Adjust the y-limits manually
                        }
                        if (ax->xlim()[1] == ax->xlim()[0]) {
                            ax->xlim({ax->xlim()[0] - 1, ax->xlim()[1] + 1}); // Adjust the x-limits manually
                        }
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error positioning axis " << plot_id << ": " << e.what() << std::endl;
                }
            }
        }
    }

    /**
     * @brief Computes the eigenfunctions using the refined degeneracy results.
     *
     * This method constructs a vector of eigenfunctions based on the refined degeneracy results.
     * Each eigenfunction is represented as a complex function of x and y coordinates. The eigenfunctions
     * are computed using the boundary points and the Bessel function of the second kind.
     *
     * The method assumes that the eigenvectors are entirely real and calculates the wavefunction
     * intensity as the sum of the real part of the eigenvector components multiplied by the Bessel
     * function value at the given point's distance from the boundary points.
     *
     * @return A vector of shared pointers to functions representing the eigenfunctions. Each function
     * takes two double arguments (x and y coordinates) and returns a complex double representing
     * the wavefunction at that point.
     *
     * @note The Bessel function of the second kind (Y0) is used to calculate the wavefunction intensity.
     * The wavefunction intensity is scaled by a factor of 1/4.
     *
     * @throws std::out_of_range If the index is out of the range of the degeneracy results.
     */
    [[nodiscard]] std::vector<std::shared_ptr<std::function<std::complex<double>(double, double)>>> computeEigenfunction() const {
        const auto [k, degeneracy, eigenvectors] = degeneracy_result;
        std::vector<std::shared_ptr<std::function<std::complex<double>(double, double)>>> results;

        const auto points = boundary_integral_.getBoundaryPoints();
        for (const auto &u_s : eigenvectors) {
            results.emplace_back(std::make_shared<std::function<std::complex<double>(double, double)>>(
                [k, u_s, points](const double x, const double y) {
                    double psi = 0.0;
                    for (size_t idx = 0; idx < points.size(); ++idx) {
                        const auto &point = points[idx];
                        double distance = std::hypot(x - point.x, y - point.y);
                        if (distance < sqrt(std::numeric_limits<double>::epsilon())) {
                            distance = std::sqrt(std::numeric_limits<double>::epsilon());
                        }
                        const double Y0 = gsl_sf_bessel_Y0(k * distance);
                        psi += u_s[idx].real() * Y0; // NOLINT(*-narrowing-conversions) // The u(s) should be entirely real or have a small Im part!
                    }
                    return (1.0 / 4.0) * psi;
                }));
        }
        return results;
    }

private:

    /**
     * @brief Computes the momentum wavefunction for all eigenvectors.
     *
     * This method constructs a vector of momentum wavefunctions based on the normalized eigenvectors.
     * Each momentum wavefunction is represented as a complex function of p_x and p_y coordinates.
     * The momentum wavefunctions are computed using the boundary points and the Fourier transform of
     * the spatial wavefunction.
     *
     * The method assumes that the eigenvectors are entirely real and calculates the momentum wavefunction
     * as the sum of the normal derivative of the wavefunction components multiplied by an exponential term
     * representing the Fourier transform.
     *
     * @return A vector of shared pointers to functions representing the momentum wavefunctions. Each function
     * takes two double arguments (p_x and p_y coordinates) and returns a complex double representing
     * the momentum wavefunction at that point.
     *
     * @throws std::runtime_error If the wavefunction pointer is null.
     */
    [[nodiscard]] std::vector<std::shared_ptr<std::function<std::complex<double>(double, double)>>> computeMomentumEigenfunctionCartesian() const {
        const auto [k, deg, eigvecs] = degeneracy_result;
        const auto points = boundary_integral_.getBoundaryPoints();
        std::vector<std::shared_ptr<std::function<std::complex<double>(double, double)>>> results;
        constexpr std::complex<double> i(0.0, 1.0); // Imaginary unit

        for (const auto& u : eigvecs) {
            results.emplace_back(std::make_shared<std::function<std::complex<double>(double, double)>>(
                [k, u, points, i](const double px, const double py) {
                    const double k_squared = k * k;
                    std::complex<double> psi_p = 0.0;
                    for (size_t idx = 0; idx < points.size(); ++idx) {
                        const auto& p = points[idx];
                        const double pq_dot = px * p.x + py * p.y;
                        const std::complex<double> exp_term = std::exp(-i * pq_dot);
                        psi_p += exp_term * pq_dot * u[idx].real(); // NOLINT(*-narrowing-conversions)
                    }
                    const std::complex<double> prefactor = -i / (4 * M_PI * k_squared);
                    return prefactor * psi_p;
                }));
        }
        return results;
    }

    /**
     * @brief Computes the momentum wavefunction for all eigenvectors in polar coordinates.
     *
     * This method constructs a vector of momentum wavefunctions based on the normalized eigenvectors.
     * Each momentum wavefunction is represented as a complex function of r and phi coordinates.
     * The momentum wavefunctions are computed using the boundary points and the Fourier transform of
     * the spatial wavefunction.
     *
     * The method assumes that the eigenvectors are entirely real and calculates the momentum wavefunction
     * as the sum of the normal derivative of the wavefunction components multiplied by an exponential term
     * representing the Fourier transform.
     *
     * @return A vector of shared pointers to functions representing the momentum wavefunctions. Each function
     * takes two double arguments (r and phi coordinates) and returns a complex double representing
     * the momentum wavefunction at that point.
     *
     * @throws std::runtime_error If the wavefunction pointer is null.
     */
    [[nodiscard]] std::vector<std::shared_ptr<std::function<std::complex<double>(double, double)>>> computeMomentumEigenfunctionPolar() const {
        const auto momentum_wavefunctions = computeMomentumEigenfunctionCartesian();
        std::vector<std::shared_ptr<std::function<std::complex<double>(double, double)>>> results;
        results.reserve(momentum_wavefunctions.size());

        for (const auto& psi_p_cartesian : momentum_wavefunctions) {
            results.emplace_back(std::make_shared<std::function<std::complex<double>(double, double)>>(
                [psi_p_cartesian](const double r, const double phi) {
                    const double px = r * std::cos(phi);
                    const double py = r * std::sin(phi);
                    return (*psi_p_cartesian)(px, py);
                }));
        }
        return results;
    }

    /**
     * @brief Computes the radially integrated momentum density for all eigenvectors.
     *
     * This method computes the radially integrated momentum density for all eigenvectors in the degeneracy.
     * The radially integrated density is calculated using the boundary points and the normalized eigenvectors.
     * The calculation uses the function \( f(x) = \sin(x) \cdot \text{Ci}(x) - \cos(x) \cdot \text{Si}(x) \), where
     * Si and Ci are the sine and cosine integrals, respectively.
     *
     * @return A vector of shared pointers to functions representing the radially integrated momentum density for each eigenvector.
     *
     * @throws std::runtime_error If the wavefunction pointer is null.
     */
    [[nodiscard]] std::vector<std::shared_ptr<std::function<double(double)>>> computeRadiallyIntegratedDensity() const {
        const auto [k, deg, eigvecs] = degeneracy_result;  // Extract wave number, degeneracy, and eigenvectors from the degeneracy result
        const auto points = boundary_integral_.getBoundaryPoints();  // Get boundary points
        std::vector<std::shared_ptr<std::function<double(double)>>> results;  // Vector to store the radially integrated densities
        const int num_points = points.size(); // NOLINT(*-narrowing-conversions)
        const int num_threads = std::thread::hardware_concurrency();  // Number of available hardware threads NOLINT(*-narrowing-conversions)
        const int chunk_size = num_points / num_threads;  // Size of each chunk

        for (const auto& u : eigvecs) {  // Iterate over each eigenvector
            results.emplace_back(std::make_shared<std::function<double(double)>>(
                [k, u, points, num_threads, chunk_size, num_points](const double phi) {
                    double I_phi = 0.0;
                    const double p = k;  // Set p to the wave number k
                    std::vector<std::future<double>> futures;

                    auto compute_chunk = [&](const int start, const int end) {
                        double chunk_I_phi = 0.0;
                        for (int i = start; i < end; ++i) {
                            for (int j = 0; j < num_points; ++j) {
                                const double alpha = std::abs(std::cos(phi) * (points[i].x - points[j].x) + std::sin(phi) * (points[i].y - points[j].y));
                                double x = alpha * p;
                                if (std::abs(x) < std::sqrt(std::numeric_limits<double>::epsilon())) { // Added to handle numerical instability for Ci(x)
                                    x = std::sqrt(std::numeric_limits<double>::epsilon());
                                }
                                const double Si_x = gsl_sf_Si(x);  // Sine integral from GSL under the exponential integrals header
                                const double Ci_x = gsl_sf_Ci(x);  // Cosine integral from GSL under the exponential integrals header
                                const double f_x = std::sin(x) * Ci_x - std::cos(x) * Si_x;  // Compute f(x)
                                chunk_I_phi += f_x * (u[i].real() * u[j].real());  // Accumulate the result NOLINT(*-narrowing-conversions)
                            }
                        }
                        return chunk_I_phi;
                    };
                    for (int t = 0; t < num_threads; ++t) {
                        int start = t * chunk_size;
                        int end = (t == num_threads - 1) ? num_points : start + chunk_size;
                        futures.emplace_back(std::async(std::launch::async, compute_chunk, start, end));
                    }
                    for (auto& future : futures) {
                        I_phi += future.get();
                    }
                    return (1.0 / (8 * M_PI * M_PI)) * I_phi;  // Normalize and return the result
                }));
        }
        return results;
    }

    /**
     * @brief Computes the angular integrated momentum density for all normalized eigenvectors.
     *
     * This method calculates the angular integrated momentum density \( R_n(r) \) for all
     * normalized eigenvectors. The calculation is based on the equation:
     *
     * \[
     * R_n(r) = \frac{r}{(r^2 - k^2)^2} \frac{1}{2\pi} \iint_{\partial \Omega \times \partial \Omega}
     * u_n(s) u_n(s') J_0(|q(s) - q(s')| r) \, ds \, ds'
     * \]
     *
     * Where \( J_0 \) is the Bessel function of the first kind, \( u_n \) are the normalized eigenvectors,
     * \( q(s) \) and \( q(s') \) are boundary points, and \( k^2 \) is the square of the wave number.
     *
     * The method handles the potential singularity at \( r = \sqrt{k^2} \) by computing the value
     * just before and after the critical point and then averaging these values.
     *
     * @return A vector of shared pointers to functions representing the angular integrated momentum
     * densities. Each function takes a single double argument \( r \) and returns a double representing
     * the density at that point.
     *
     * @note The Bessel function of the first kind (J0) is used in the calculation.
     * The method assumes the eigenvectors are entirely real.
     *
     * @throws std::out_of_range If the index is out of the range of the degeneracy results.
     */
    [[nodiscard]] std::vector<std::shared_ptr<std::function<double(double)>>> computeAngularIntegratedMomentumDensity() const {
        const auto [k, degeneracy, eigenvectors] = degeneracy_result; // Retrieve the degeneracy result containing k, degeneracy, and eigenvectors
        const double k_squared = k * k; // Compute k squared
        std::vector<std::shared_ptr<std::function<double(double)>>> results;
        // Define a small epsilon for handling singularities
        const double epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
        const auto points = boundary_integral_.getBoundaryPoints(); // Retrieve boundary points
        const int num_threads = std::thread::hardware_concurrency(); // Get the number of available hardware threads NOLINT(*-narrowing-conversions)
        const size_t chunk_size = points.size() / num_threads; // Determine chunk size for parallelization

        for (const auto& u_s : eigenvectors) {
            results.emplace_back(std::make_shared<std::function<double(double)>>(
                [k_squared, u_s, points, epsilon, chunk_size, num_threads](const double r) -> double {
                    double R_r = 0.0; // Initialize the result for R(r)

                    // Function to compute a chunk of the integral
                    auto compute_chunk = [&](const size_t start, const size_t end) {
                        double chunk_sum = 0.0;
                        for (size_t i = start; i < end; ++i) {
                            for (size_t j = 0; j < points.size(); ++j) {
                                const auto& point = points[i];
                                const auto& point_prime = points[j];
                                const double distance = std::hypot(point.x - point_prime.x, point.y - point_prime.y);
                                chunk_sum += u_s[i].real() * u_s[j].real() * gsl_sf_bessel_J0(distance * r); // Add the contribution from each pair of boundary points NOLINT(*-narrowing-conversions)
                            }
                        }
                        return chunk_sum;
                    };

                    // Launch tasks to compute chunks of the grid in parallel
                    std::vector<std::future<double>> futures;
                    for (int t = 0; t < num_threads; ++t) {
                        size_t start = t * chunk_size;
                        size_t end = (t == num_threads - 1) ? points.size() : start + chunk_size;
                        futures.emplace_back(std::async(std::launch::async, compute_chunk, start, end));
                    }

                    // Sum the results from all tasks
                    for (auto &future : futures) {
                        R_r += future.get();
                    }

                    // Handle the potential singularity at r = sqrt(k_squared)
                    if (std::abs(r - std::sqrt(k_squared)) < epsilon) {
                        const double r_left = r - epsilon;
                        const double r_right = r + epsilon;
                        const double R_left = (r_left / std::pow(r_left * r_left - k_squared, 2)) * R_r;
                        const double R_right = (r_right / std::pow(r_right * r_right - k_squared, 2)) * R_r;
                        return (R_left + R_right) / 2.0;
                    }
                    return (r / std::pow(r * r - k_squared, 2)) * R_r; // Compute the final value for R(r)
                }
            ));
        }
        return results; // Return the computed angular integrated momentum densities
    }

    /**
     * @brief Normalizes the eigenvectors for each k value.
     *
     * This method normalizes the eigenvectors stored in degeneracy_result.
     * @param normalization_factors The normalization factors for each eigenvector.
     */
    void normalizeEigenvectors(const std::vector<double>& normalization_factors) {
        auto& [k, degeneracy, eigenvectors] = degeneracy_result;

        // Normalize each eigenvector using the corresponding normalization factor
        for (int i = 0; i < eigenvectors.size(); ++i) {
            eigenvectors[i] /= std::sqrt(normalization_factors[i]);
        }
    }

    /**
    * @brief Computes the square of the wavefunction's absolute value.
    *
    * This method constructs a vector of functions representing the square of the wavefunction's absolute value,
    * which corresponds to the intensity of the wavefunction at each point (x, y). The method utilizes the
    * eigenfunctions computed by the computeEigenfunction() method.
    *
    * Each function in the resulting vector takes two double arguments (x and y coordinates) and returns a
    * double representing the wavefunction intensity (the square of the absolute value of the wavefunction)
    * at that point.
    *
    * @return A vector of shared pointers to functions representing the square of the wavefunctions (with possible degeneracies).
    *
    * @note The wavefunction intensity is calculated using the std::norm function, which computes the square
    * of the absolute value of a complex number.
    */
    [[nodiscard]] std::vector<std::shared_ptr<std::function<double(double, double)>>> getSquareWavefunction() const {
        const auto wavefunctions = computeEigenfunction();
        std::vector<std::shared_ptr<std::function<double(double, double)>>> results;
        results.reserve(wavefunctions.size());
        for (const auto& psi : wavefunctions) {
            results.emplace_back(std::make_shared<std::function<double(double, double)>>(
                [psi](const double x, const double y) -> double {
                    const std::complex<double> wave_f_xy = (*psi)(x, y);
                    return std::norm(wave_f_xy); // Square of the absolute value of the wavefunction
                }));
        }
        return results;
    }

    /**
     * @brief Computes the square of the wavefunction's absolute value in momentum space.
     *
     * This method constructs a vector of functions representing the square of the wavefunction's absolute value
     * in momentum space, which corresponds to the intensity of the wavefunction at each momentum point (px, py).
     * The method utilizes the eigenfunctions computed by the computeMomentumEigenfunction() method.
     *
     * Each function in the resulting vector takes two double arguments (px and py coordinates) and returns a
     * double representing the wavefunction intensity (the square of the absolute value of the wavefunction)
     * at that point.
     *
     * @return A vector of shared pointers to functions representing the square of the wavefunctions in momentum space.
     *
     * @note The wavefunction intensity is calculated using the std::norm function, which computes the square
     * of the absolute value of a complex number.
     */
    [[nodiscard]] std::vector<std::shared_ptr<std::function<double(double, double)>>> getSquareMomentumWavefunctionCartesian() const {
        const auto momentum_wavefunctions = computeMomentumEigenfunctionCartesian();
        std::vector<std::shared_ptr<std::function<double(double, double)>>> results;
        results.reserve(momentum_wavefunctions.size());
        for (const auto& psi_p : momentum_wavefunctions) {
            results.emplace_back(std::make_shared<std::function<double(double, double)>>(
                [psi_p](const double px, const double py) -> double {
                    const std::complex<double> wave_f_pxy = (*psi_p)(px, py);
                    return std::norm(wave_f_pxy); // Square of the absolute value of the wavefunction in momentum space
                }));
        }
        return results;
    }

    /**
     * @brief Computes the square of the momentum wavefunction's absolute value in polar coordinates.
     *
     * This method constructs a vector of functions representing the square of the momentum wavefunction's absolute value,
     * which corresponds to the intensity of the momentum wavefunction at each point (r, phi). The method utilizes the
     * momentum wavefunctions computed by the computeMomentumEigenfunctionPolar() method.
     *
     * Each function in the resulting vector takes two double arguments (r and phi coordinates) and returns a
     * double representing the wavefunction intensity (the square of the absolute value of the wavefunction)
     * at that point.
     *
     * @return A vector of shared pointers to functions representing the square of the momentum wavefunctions in polar coordinates.
     *
     * @note The wavefunction intensity is calculated using the std::norm function, which computes the square
     * of the absolute value of a complex number.
     */
    [[nodiscard]] std::vector<std::shared_ptr<std::function<double(double, double)>>> getSquareMomentumWavefunctionPolar() const {
        const auto momentum_wavefunctions_polar = computeMomentumEigenfunctionPolar();
        std::vector<std::shared_ptr<std::function<double(double, double)>>> results;
        results.reserve(momentum_wavefunctions_polar.size());
        for (const auto& psi_p_polar : momentum_wavefunctions_polar) {
            results.emplace_back(std::make_shared<std::function<double(double, double)>>(
                [psi_p_polar](const double r, const double phi) -> double {
                    const std::complex<double> wave_f_polar = (*psi_p_polar)(r, phi);
                    return std::norm(wave_f_polar); // Square of the absolute value of the wavefunction in momentum space
                }));
        }
        return results;
    }

    /**
     * @brief Checks if a point is inside the boundary.
     *
     * This method determines whether a given point (px, py) is inside the boundary defined by the provided points.
     *
     * @param points The points defining the boundary.
     * @param px The x-coordinate of the point.
     * @param py The y-coordinate of the point.
     * @return True if the point is inside the boundary, false otherwise.
     */
    static bool isPointInside(const std::vector<Boundary::Point>& points, const double px, const double py) {
        int windingNumber = 0;
        const size_t numPoints = points.size();
        for (size_t i = 0; i < numPoints; ++i) {
            const auto& p1 = points[i];
            const auto& p2 = points[(i + 1) % numPoints];
            if (p1.y <= py) {
                if (p2.y > py && isLeft(p1, p2, {px, py}) > 0) {
                    ++windingNumber;
                }
            } else {
                if (p2.y <= py && isLeft(p1, p2, {px, py}) < 0) {
                    --windingNumber;
                }
            }
        }
        return windingNumber != 0;
    }

    /**
     * @brief Helper function to determine if a point is on a symmetry line segment for plotting.
     *
     * This method checks if a point lies on a line segment defined by two points.
     *
     * @param p1 The first point of the line segment.
     * @param p2 The second point of the line segment.
     * @param p The point to check.
     * @return True if the point is on the line segment, false otherwise.
     */
    static bool isPointOnSymmetryLine(const Boundary::Point& p1, const Boundary::Point& p2, const Boundary::Point& p) {
        if (const double cross = (p.y - p1.y) * (p2.x - p1.x) - (p.x - p1.x) * (p2.y - p1.y); std::abs(cross) > std::numeric_limits<double>::epsilon()) {
            return false;
        }

        if (const double dot = (p.x - p1.x) * (p2.x - p1.x) + (p.y - p1.y) * (p2.y - p1.y); dot < 0 || dot > (p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y)) {
            return false;
        }

        return true;
    }

    /**
     * @brief Helper function to determine if a point is to the left of a line segment.
     *
     * This method calculates the relative position of a point with respect to a line segment.
     *
     * @param p1 The first point of the line segment.
     * @param p2 The second point of the line segment.
     * @param p The point to check.
     * @return A positive value if the point is to the left of the line segment, negative if to the right, and 0 if on the line.
     */
    static double isLeft(const Boundary::Point& p1, const Boundary::Point& p2, const Boundary::Point& p) {
        return (p2.x - p1.x) * (p.y - p1.y) - (p.x - p1.x) * (p2.y - p1.y);
    }

    /**
     * @brief Calculates the Poincare-Husimi function.
     *
     * This method computes the Poincare-Husimi function on a given grid size for the boundary points.
     *
     * @param boundary_points The boundary points of the domain.
     * @param grid_size The size of the grid for the Husimi function.
     * @param index The index of the wavefunction
     * @return A 2D vector containing the Husimi function values.
     */
    [[nodiscard]] std::vector<std::vector<double>> calculatePoincareHusimi(const std::vector<Boundary::Point>& boundary_points, const int grid_size, const int index = 0) const {
        // Initialize the Husimi function grid
        std::vector<std::vector<double>> husimi(grid_size, std::vector<double>(grid_size, 0.0));

        // Total perimeter length
        const double L = boundary->calculateArcLength();
        // Grid spacing in q and p directions
        const double dq = L / grid_size;
        const double dp = 2.0 / grid_size;

        // Extract the wave number, degeneracy, and eigenvectors from the degeneracy result (normalized since in the constructor they are normalized)
        const auto [k, degeneracy, eigenvectors] = degeneracy_result;

        // Create futures to parallelize the computation
        std::vector<std::future<void>> futures;
        const int num_threads = std::thread::hardware_concurrency(); // NOLINT(*-narrowing-conversions)
        const int chunk_size = grid_size / num_threads;

        // Generate arc length parameters for each boundary point
        const std::vector<double> arc_parameters = boundary_integral_.getArclengthValues();

        auto compute_chunk = [&](const int start, const int end) {
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < grid_size; ++j) {
                    const double q = i * dq;
                    const double p = -1.0 + j * dp;
                    std::complex<double> projection = 0.0;
                    // Loop over eigenvectors and boundary points to compute the projection
                    for (size_t idx = 0; idx < boundary_points.size(); ++idx) {
                        projection += std::conj(coherentState(q, p, arc_parameters[idx], k, L)) * eigenvectors[index][idx].real(); // NOLINT(*-narrowing-conversions)
                    }
                    // Store the squared magnitude of the projection std::abs(.)^2
                    husimi[j][i] = std::norm(projection); // Just like in the wavefunction the indexing is changed as y,x AND NOT x,y due to filling of the vector of vectors
                }
            }
        };

        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = (t == num_threads - 1) ? grid_size : (t + 1) * chunk_size;
            futures.emplace_back(std::async(std::launch::async, compute_chunk, start, end));
        }

        for (auto& future : futures) {
            future.get();
        }

        // Parallelize the normalization step
        double sum = 0.0;
        std::vector<std::future<double>> sum_futures;
        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = (t == num_threads - 1) ? grid_size : (t + 1) * chunk_size;
            sum_futures.emplace_back(std::async(std::launch::async, [&, start, end]() {
                double local_sum = 0.0;
                for (int i = start; i < end; ++i) {
                    local_sum = std::accumulate(husimi[i].begin(), husimi[i].end(), local_sum);
                }
                return local_sum;
            }));
        }

        for (auto& future : sum_futures) { // Add all the paralelliaztion steps to summation here
            sum += future.get();
        }

        auto normalize_chunk = [&](const int start, const int end) { // Ech cell is normalized with the sum
            for (int i = start; i < end; ++i) {
                for (auto& value : husimi[i]) {
                    value /= sum;
                }
            }
        };

        futures.clear();
        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = (t == num_threads - 1) ? grid_size : (t + 1) * chunk_size;
            futures.emplace_back(std::async(std::launch::async, normalize_chunk, start, end));
        }

        for (auto& future : futures) {
            future.get();
        }
        return husimi;
    }

    /**
     * @brief Constructs a coherent state.
     *
     * This method constructs a coherent state centered at (q, p) for a given boundary parameter s.
     *
     * @param q The position in the q-direction.
     * @param p The position in the p-direction.
     * @param s The boundary parameter.
     * @param k The wave number.
     * @param L The perimeter length of the boundary.
     * @return The coherent state as a complex number.
     */
    [[nodiscard]] static std::complex<double> coherentState(const double q, const double p, const double s, const double k, const double L) {
        constexpr std::complex<double> i(0, 1);
        const double phase = k * p * (s - q);
        const double gaussian = std::exp(-k/2 * (s - q) * (s - q));
        return std::exp(i * phase) * gaussian;
    }
};

#endif //PLOTTING_HPP
