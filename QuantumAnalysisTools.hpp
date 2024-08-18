#ifndef QUANTUMANALYSISTOOLS_HPP
#define QUANTUMANALYSISTOOLS_HPP

#pragma once
#include <tuple>
#include <numbers>
#include <Eigen/Dense>
#include <gsl/gsl_sf_bessel.h>
#include <matplot/freestanding/axes_functions.h>
#include "Boundary.hpp"
#include <set>

/**
 * @file QuantumAnalysisTools.hpp
 * @brief Header for analyzing results.
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

namespace AnalysisTools {
    using namespace Boundary;

        namespace BerryRobnik {

        /**
         * @brief Computes the complementary error function.
         *
         * This function calculates the complementary error function, which is defined as 1 minus the error function.
         * The error function is a mathematical function that is used in probability, statistics, and partial differential equations.
         *
         * @param x The input value for which the complementary error function is computed.
         * @return The value of the complementary error function at x.
         */
        inline double erfc(const double x) {
            return 1.0 - std::erf(x);
        }

        /**
         * @brief Computes the Berry-Robnik distribution function.
         *
         * The Berry-Robnik distribution describes the spacing distribution of energy levels in quantum chaotic systems
         * where there is a mixture of regular and chaotic dynamics. This function calculates the Berry-Robnik distribution
         * for a given spacing s and mixing parameter rho.
         *
         * @param s The spacing value.
         * @param rho The mixing parameter (0 < rho < 1). For rho = 1 we have Poisson and for rho = 0 we have Wigner (GOE) distribution
         * @return The value of the Berry-Robnik distribution at spacing s for the given rho.
         */
        inline double probabilityBerryRobnik(const double s, const double rho) {
            constexpr double pi = M_PI;
            const double rho1 = rho;
            const double rho2 = 1.0 - rho;
            return (std::pow(rho1, 2) * erfc(std::sqrt(pi / 2.0) * rho2 * s) +
                    (2.0 * rho1 * rho2 + (pi / 2.0) * std::pow(rho2, 3) * s) *
                    std::exp(-(pi / 4.0) * std::pow(rho2, 2) * std::pow(s, 2))) *
                   std::exp(-rho1 * s);
        }

        /**
         * @brief Computes the cumulative Berry-Robnik distribution function.
         *
         * The cumulative Berry-Robnik distribution function describes the cumulative probability distribution of the spacing
         * of energy levels in quantum chaotic systems where there is a mixture of regular and chaotic dynamics. This function
         * calculates the cumulative Berry-Robnik distribution for a given spacing s and mixing parameter rho.
         *
         * @param s The spacing value.
         * @param rho The mixing parameter (0 < rho < 1). For rho = 1 we have Poisson and for rho = 0 we have Wigner (GOE) distribution
         * @return The value of the cumulative Berry-Robnik distribution at spacing s for the given rho.
         */
        inline double cumulativeBRDistribution(const double s, const double rho) {
            constexpr double pi = M_PI;
            return 1.0 - std::exp(-rho * s - 0.25 * pi * std::pow((1.0 - rho), 2) * std::pow(s, 2))
                        - std::exp(-rho * s) * rho
                        + std::exp(-rho * s - 0.25 * pi * std::pow((1.0 - rho), 2) * std::pow(s, 2)) * rho
                        + std::exp(std::pow(rho, 2) / (pi * std::pow((1.0 - rho), 2))) * rho * std::erf(rho / (std::sqrt(pi) * (1.0 - rho)))
                        - std::exp(std::pow(rho, 2) / (2.0 * pi * std::pow((1.0 - rho), 2))) * rho * std::erf(rho / (std::sqrt(2.0 * pi) * (1.0 - rho)))
                        + std::exp(-rho * s) * rho * std::erf(std::sqrt(pi / 2.0) * (1.0 - rho) * s)
                        + std::exp(std::pow(rho, 2) / (2.0 * pi * std::pow((1.0 - rho), 2))) * rho * std::erf((rho + pi * s - 2.0 * pi * rho * s + pi * std::pow(rho, 2) * s) / (std::sqrt(2.0 * pi) * (1.0 - rho)))
                        - std::exp(std::pow(rho, 2) / (pi * std::pow((1.0 - rho), 2))) * rho * std::erf((2.0 * rho + pi * s - 2.0 * pi * rho * s + pi * std::pow(rho, 2) * s) / (2.0 * std::sqrt(pi) * (1.0 - rho)));
        }
    }// Namespace Berry Robnik

    namespace MushroomTools {
        /**
         * @brief Calculates the total 3D phase-space volume of a mushroom billiard.
         *
         * This function calculates the total 3D phase-space volume of a mushroom billiard
         * characterized by a stem of given height and width, and a cap of given radius.
         * The volume is computed using the formula:
         * V_3D_tot = 2π * (h * w + (π/2) * r^2)
         *
         * @param h The height of the stem.
         * @param w The width of the stem.
         * @param r The radius of the cap.
         * @return double The total 3D phase-space volume.
         */
        inline double calculateTotalPhaseSpaceVolume(const double h, const double w, const double r) {
            return 2 * M_PI * (h * w + (M_PI / 2) * std::pow(r, 2));
        }

        /**
         * @brief Calculates the regular 3D phase-space volume of a mushroom billiard.
         *
         * This function calculates the regular (integrable) component of the 3D phase-space volume
         * of a mushroom billiard. The volume is calculated using the formula:
         * V_3D_reg = 2π * r^2 * (acos(w / (2 * r)) -
         *                        (w / (2 * r)) * sqrt(1 - (w^2 / (4 * r^2))))
         *
         * @param h The height of the stem.
         * @param w The width of the stem.
         * @param r The radius of the cap.
         * @return double The regular 3D phase-space volume.
         */
        inline double calculateRegularPhaseSpaceVolume(double h, const double w, const double r) {
            return 2 * M_PI * std::pow(r, 2) * (std::acos(w / (2 * r)) -
                                                (w / (2 * r)) * std::sqrt(1 - std::pow(w, 2) / (4 * std::pow(r, 2))));
        }

        /**
         * @brief Calculates the chaotic 3D phase-space volume of a mushroom billiard.
         *
         * This function calculates the chaotic (non-integrable) component of the 3D phase-space volume
         * of a mushroom billiard. It is computed as the difference between the total phase-space volume
         * and the regular phase-space volume:
         * V_3D_cha = V_3D_tot - V_3D_reg
         *
         * @param h The height of the stem.
         * @param w The width of the stem.
         * @param r The radius of the cap.
         * @return double The chaotic 3D phase-space volume.
         */
        inline double calculateChaoticPhaseSpaceVolume(const double h, const double w, const double r) {
            return calculateTotalPhaseSpaceVolume(h, w, r) - calculateRegularPhaseSpaceVolume(h, w, r);
        }

        /**
         * @brief Calculates the regular phase-space portion of the full 3D phase-space of a mushroom billiard.
         *
         * This function calculates the portion of the full 3D phase-space that is regular (integrable)
         * for a mushroom billiard. It is computed as the ratio of the regular phase-space volume to the total
         * phase-space volume:
         * g_r_3D = V_3D_reg / V_3D_tot
         *
         * @param h The height of the stem.
         * @param w The width of the stem.
         * @param r The radius of the cap.
         * @return double The regular phase-space portion.
         */
        inline double calculateRegularPhaseSpacePortion(const double h, const double w, const double r) {
            return calculateRegularPhaseSpaceVolume(h, w, r) / calculateTotalPhaseSpaceVolume(h, w, r);
        }

        /**
         * @brief Calculates the chaotic phase-space portion of the full 3D phase-space of a mushroom billiard.
         *
         * This function calculates the portion of the full 3D phase-space that is chaotic (non-integrable)
         * for a mushroom billiard. It is computed as one minus the regular phase-space portion:
         * g_c_3D = 1 - g_r_3D
         *
         * @param h The height of the stem.
         * @param w The width of the stem.
         * @param r The radius of the cap.
         * @return double The chaotic phase-space portion.
         */
        inline double calculateChaoticPhaseSpacePortion(const double h, const double w, const double r) {
            return 1.0 - calculateRegularPhaseSpacePortion(h, w, r);
        }
    } // namespace MushroomTools

    /**
     * @brief Prints the differences between numerical and analytical k values for local minima.
     *
     * @param minima The local minima to compare.
     * @param analytical_eigenvalues The analytical eigenvalues for comparison. They need to be 2 indexed with int numbers
     * @param label A label for the output.
     * @param suppressPrint If true, suppresses printing.
     * @param printLastPercent If true, prints only the last percentage of the data.
     * @param lastPercent The percentage of the data to print if printLastPercent is true.
     */
    void printDifferences(const auto& minima, const std::vector<std::tuple<double, int, int>>& analytical_eigenvalues, const std::string& label, const bool suppressPrint = false, const bool printLastPercent = false, const double lastPercent = 0.05) {
        std::cout << "\nDifferences between numerical and analytical k values for " << label << ":\n";
        if (minima.empty()) {
            std::cout << "Minima are empty. Skipping comparison.\n";
            return;
        }
        const size_t startIndex = printLastPercent ? static_cast<size_t>(minima.size() * (1.0 - lastPercent)) : 0;

        for (size_t i = startIndex; i < minima.size(); ++i) {
            const auto& [numerical_k, smallest_singular_value] = minima[i];
            // Find the closest analytical eigenvalue
            auto closest_analytical = *std::ranges::min_element(analytical_eigenvalues,
                                                                [numerical_k](const auto& a, const auto& b) {
                                                                    return std::abs(std::get<0>(a) - numerical_k) < std::abs(std::get<0>(b) - numerical_k);
                                                                });

            double closest_analytical_k = std::get<0>(closest_analytical);
            int m = std::get<1>(closest_analytical);
            int n = std::get<2>(closest_analytical);

            double absolute_difference = std::abs(numerical_k - closest_analytical_k);
            double relative_difference = absolute_difference / closest_analytical_k;

            if (!suppressPrint) {
                std::cout << "Numerical k: " << numerical_k
                          << ", Analytical k: " << closest_analytical_k
                          << " (m: " << m << ", n: " << n << ")"
                          << ", Smallest Singular Value: " << smallest_singular_value
                          << ", Absolute Difference: " << absolute_difference
                          << ", Relative Difference: " << relative_difference << std::endl;
            }
        }
    }

    /**
     * @brief Compares numerical k values between two sets of local minima within a tolerance.
     *
     * @param minima1 The first set of local minima.
     * @param minima2 The second set of local minima.
     * @param label1 The label for the first set of local minima.
     * @param label2 The label for the second set of local minima.
     * @param suppressPrint If true, suppresses printing.
     * @param printLastPercent If true, prints only the last percentage of the data.
     * @param lastPercent The percentage of the data to print if printLastPercent is true.
     */
    void compareMinima(const auto& minima1, const auto& minima2, const std::string& label1, const std::string& label2, const bool suppressPrint = false, const bool printLastPercent = false, const double lastPercent = 0.05) {
        if (minima1.empty() || minima2.empty()) {
            std::cout << "One or both sets of minima are empty. Skipping comparison.\n";
            return;
        }

        std::cout << "\nDifferences between " << label1 << " and " << label2 << " (min element approach):\n";
        const size_t startIndex1 = printLastPercent ? static_cast<size_t>(minima1.size() * (1.0 - lastPercent)) : 0;
        const size_t startIndex2 = printLastPercent ? static_cast<size_t>(minima2.size() * (1.0 - lastPercent)) : 0;

        for (size_t i = startIndex1; i < minima1.size(); ++i) {
            const auto& [numerical_k1, smallest_singular_value1] = minima1[i];
            // Find the closest numerical k in minima2
            auto closest_minima2 = *std::ranges::min_element(minima2.begin() + startIndex2, minima2.end(),
                                                             [numerical_k1](const auto& a, const auto& b) {
                                                                 return std::abs(std::get<0>(a) - numerical_k1) < std::abs(std::get<0>(b) - numerical_k1);
                                                             });

            double numerical_k2 = std::get<0>(closest_minima2);
            double smallest_singular_value2 = std::get<1>(closest_minima2);

            double absolute_difference = std::abs(numerical_k1 - numerical_k2);
            double relative_difference = absolute_difference / numerical_k1;

            if (!suppressPrint) {
                std::cout << "Numerical k (" << label1 << "): " << numerical_k1
                          << ", Numerical k (" << label2 << "): " << numerical_k2
                          << ", Absolute Difference: " << absolute_difference
                          << ", Relative Difference: " << relative_difference
                          << ", Smallest Singular Value (" << label1 << "): " << smallest_singular_value1
                          << ", Smallest Singular Value (" << label2 << "): " << smallest_singular_value2 << std::endl;
            }
        }
    }

    /**
     * @brief Prints the differences between numerical and analytical energies for local minima.
     *
     * @param minima The local minima to compare.The first one of the must have k values, the other is arbitrary
     * @param analytical_eigenvalues The analytical eigenvalues for comparison.
     * @param label A label for the output.
     * @param area The area of the boundary
     * @param perimeter The modified perimeter of the boundary wrt to Neumann/Dirichlet bc
     * @param suppressPrint If true, suppresses printing.
     * @param printLastPercent If true, prints only the last percentage of the data.
     * @param lastPercent The percentage of the data to print if printLastPercent is true.
     */
    void printDifferencesEnergies(const auto& minima, const std::vector<std::tuple<double, int, int>>& analytical_eigenvalues, const std::string& label, const double area, const double perimeter, const bool suppressPrint = false, const bool printLastPercent = false, const double lastPercent = 0.05) {
        std::cout << "\nDifferences between numerical and analytical energies for " << label << ":\n";
        if (minima.empty()) {
            std::cout << "Minima are empty. Skipping comparison.\n";
            return;
        }
        const size_t startIndex = printLastPercent ? static_cast<size_t>(minima.size() * (1.0 - lastPercent)) : 0;

        double total_normalized_difference = 0.0;
        size_t count = 0;

        for (size_t i = startIndex; i < minima.size(); ++i) {
            const auto& [numerical_k, _] = minima[i];
            double numerical_energy = numerical_k * numerical_k;

            // Find the closest analytical eigenvalue
            auto closest_analytical = *std::ranges::min_element(analytical_eigenvalues,
                                                                [numerical_energy](const auto& a, const auto& b) {
                                                                    const double analytical_energy_a = std::get<0>(a) * std::get<0>(a);
                                                                    const double analytical_energy_b = std::get<0>(b) * std::get<0>(b);
                                                                    return std::abs(analytical_energy_a - numerical_energy) < std::abs(analytical_energy_b - numerical_energy);
                                                                });

            const double closest_analytical_k = std::get<0>(closest_analytical);
            const double closest_analytical_energy = closest_analytical_k * closest_analytical_k;
            const int m = std::get<1>(closest_analytical);
            const int n = std::get<2>(closest_analytical);

            const double absolute_difference = std::abs(numerical_energy - closest_analytical_energy);
            const double relative_difference = absolute_difference / closest_analytical_energy;
            const double mean_level_spacing = 1.0 / ((area / (4 * M_PI)) - (perimeter / (8 * M_PI * std::sqrt((numerical_energy + closest_analytical_energy) / 2.0))));
            const double normalized_difference = absolute_difference / mean_level_spacing;

            total_normalized_difference += normalized_difference;
            count++;

            if (!suppressPrint) {
                std::cout << "Numerical energy: " << numerical_energy
                          << ", Analytical energy: " << closest_analytical_energy
                          << " (m: " << m << ", n: " << n << ")"
                          << ", Absolute Difference: " << absolute_difference
                          << ", Relative Difference: " << relative_difference
                          << ", Normalized Difference: " << normalized_difference << std::endl;
            }
        }

        if (count > 0) {
            const double average_normalized_difference = total_normalized_difference / count; // NOLINT(*-narrowing-conversions)
            std::cout << "Average Normalized Difference: " << average_normalized_difference << std::endl;
        }
    }

    /**
     * @brief Compares numerical energies between two sets of local minima within a tolerance.
     *
     * @param minima1 The first set of local minima. The first one must have k values as argument
     * @param minima2 The second set of local minima. The first one must have k values as argument
     * @param label1 The label for the first set of local minima.
     * @param label2 The label for the second set of local minima.
     * @param area The area of the boundary
     * @param perimeter The modified perimeter of the boundary wrt to Neumann/Dirichlet bc
     * @param suppressPrint If true, suppresses printing.
     * @param printLastPercent If true, prints only the last percentage of the data.
     * @param lastPercent The percentage of the data to print if printLastPercent is true.
     */
    void compareMinimaEnergies(const auto& minima1, const auto& minima2, const std::string& label1, const std::string& label2, const double area, const double perimeter, const bool suppressPrint = false, const bool printLastPercent = false, const double lastPercent = 0.05) {
        if (minima1.empty() || minima2.empty()) {
            std::cout << "One or both sets of minima are empty. Skipping comparison.\n";
            return;
        }

        std::cout << "\nDifferences between " << label1 << " and " << label2 << " (min element approach):\n";
        const size_t startIndex1 = printLastPercent ? static_cast<size_t>(minima1.size() * (1.0 - lastPercent)) : 0;
        const size_t startIndex2 = printLastPercent ? static_cast<size_t>(minima2.size() * (1.0 - lastPercent)) : 0;

        double total_normalized_difference = 0.0;
        size_t count = 0;

        for (size_t i = startIndex1; i < minima1.size(); ++i) {
            const auto& [numerical_k1, _] = minima1[i];
            double numerical_energy1 = numerical_k1 * numerical_k1;

            // Find the closest numerical energy in minima2
            auto closest_minima2 = *std::ranges::min_element(minima2.begin() + startIndex2, minima2.end(),
                                                             [numerical_energy1](const auto& a, const auto& b) {
                                                                 const double energy_a = std::get<0>(a) * std::get<0>(a);
                                                                 const double energy_b = std::get<0>(b) * std::get<0>(b);
                                                                 return std::abs(energy_a - numerical_energy1) < std::abs(energy_b - numerical_energy1);
                                                             });

            const double numerical_k2 = std::get<0>(closest_minima2);
            const double numerical_energy2 = numerical_k2 * numerical_k2;

            const double absolute_difference = std::abs(numerical_energy1 - numerical_energy2);
            const double relative_difference = absolute_difference / numerical_energy1;
            const double mean_level_spacing = 1.0 / ((area / (4 * M_PI)) - (perimeter / (8 * M_PI * std::sqrt((numerical_energy1 + numerical_energy2) / 2.0))));
            const double normalized_difference = absolute_difference / mean_level_spacing;

            total_normalized_difference += normalized_difference;
            count++;

            if (!suppressPrint) {
                std::cout << "Numerical energy (" << label1 << "): " << numerical_energy1
                          << ", Numerical energy (" << label2 << "): " << numerical_energy2
                          << ", Absolute Difference: " << absolute_difference
                          << ", Relative Difference: " << relative_difference
                          << ", Normalized Difference: " << normalized_difference << std::endl;
            }
        }

        if (count > 0) {
            const double average_normalized_difference = total_normalized_difference / count; // NOLINT(*-narrowing-conversions)
            std::cout << "Average Normalized Difference: " << average_normalized_difference << std::endl;
        }
    }

    /**
     * @brief Extracts the k values and degeneracy counts from a vector of tuples containing k values, degeneracy counts,
     *        and eigenvectors.
     *
     * This function takes a vector of tuples, where each tuple contains:
     * - A double representing the k value.
     * - An int representing the degeneracy count.
     * - A vector of Eigen::VectorXcd representing the eigenvectors.
     *
     * It returns a new vector of tuples containing only the k values and degeneracy counts.
     *
     * @param original_vector The original vector of tuples containing k values, degeneracy counts, and eigenvectors.
     * @return A vector of tuples containing only the k values and degeneracy counts.
     */
    inline std::vector<std::tuple<double, int>> extractDegeneracyInfo(
        const std::vector<std::tuple<double, int, std::vector<Eigen::VectorXcd>>>& original_vector) {
        std::vector<std::tuple<double, int>> result;
        result.reserve(original_vector.size());

        for (const auto& entry : original_vector) {
            result.emplace_back(std::get<0>(entry), std::get<1>(entry));
        }

        return result;
    }

    /**
     * @brief Extracts the k values and smallest singular values from a vector of tuples containing local minima information.
     *
     * This function takes a vector of tuples where each tuple contains a k value, the smallest singular value,
     * and the corresponding eigenvector. It returns a vector of tuples containing only the k values and smallest singular values.
     *
     * @param localMinima A vector of tuples containing (k, smallest_singular_value, eigenvector).
     * @return A vector of tuples containing (k, smallest_singular_value).
     */
    inline std::vector<std::tuple<double, double>> extractLocalMinimaWithoutEigenvectors(
        const std::vector<std::tuple<double, double, Eigen::VectorXcd>>& localMinima) {
        std::vector<std::tuple<double, double>> result;
        result.reserve(localMinima.size());

        for (const auto& [k, singularValue, eigenvector] : localMinima) {
            result.emplace_back(k, singularValue);
        }
        return result;
    }

    /**
     * @brief Plots the SVD results including the smallest singular values and their higher order counterparts.
     *
     * This method plots the smallest singular values and their higher order counterparts on the provided axes.
     * It also overlays the analytical solutions as purple diamonds, and adds text annotations for the (m,n) pairs
     * corresponding to the analytical eigenvalues.
     *
     * @param ax The matplot axes handle for the main plot.
     * @param ax2 The matplot axes handle for the zoomed-in plot.
     * @param solver The KRangeSolver instance containing the SVD results.
     * @param analytical_eigenvalues The analytical eigenvalues for comparison.
     * @param a The lower bound for the y-axis in the main plot.
     * @param b The upper bound for the y-axis in the main plot.
     * @param a2 The lower bound for the y-axis in the zoomed-in plot.
     * @param b2 The upper bound for the y-axis in the zoomed-in plot.
     */
    void plotSVDResults(const matplot::axes_handle& ax, const matplot::axes_handle& ax2, auto& solver,
                               const std::vector<std::tuple<double, int, int>>& analytical_eigenvalues,
                               double a, double b, double a2, double b2) {

        // Plotting the smallest singular values
        solver.plotSmallestSingularValues(ax, a, b);
        solver.plotSmallestSingularValues(ax2, a2, b2);

        // Plotting the second smallest singular values
        solver.plotSingularValues(ax, 2, a, b);
        solver.plotSingularValues(ax2, 2, a2, b2);

        // Plotting the third smallest singular values
        solver.plotSingularValues(ax, 3, a, b);
        solver.plotSingularValues(ax2, 3, a2, b2);

        // Plotting the fourth smallest singular values
        solver.plotSingularValues(ax, 4, a, b);
        solver.plotSingularValues(ax2, 4, a2, b2);

        // Plot analytical solutions as purple diamonds
        std::vector<double> analytical_k_values;
        for (const auto& [k_mn, m, n] : analytical_eigenvalues) {
            analytical_k_values.push_back(k_mn); // NOLINT(*-inefficient-vector-operation)
        }
        const std::vector<double> analytical_x(analytical_k_values.size(), 0);
        const auto l = ax->scatter(analytical_k_values, analytical_x);
        l->display_name("Analytical");
        l->marker_style(matplot::line_spec::marker_style::diamond);
        l->marker_size(15);
        l->marker_color("purple");

        const auto l2 = ax2->scatter(analytical_k_values, analytical_x);
        l2->display_name("Analytical");
        l2->marker_style(matplot::line_spec::marker_style::diamond);
        l2->marker_size(15);
        l2->marker_color("purple");

        // Add labels for (m,n) pairs without adding to the legend
        std::map<double, int> label_count;
        for (const auto& [k_mn, m, n] : analytical_eigenvalues) {
            constexpr double offset = 0.001;
            constexpr double offset2 = 0.1;
            const int count = label_count[k_mn];
            const double offset_y = -0.001 - count * offset; // Adjust text position to avoid overlap
            const double offset_y2 = -0.1 - count * offset2; // Adjust text position to avoid overlap

            // Adjust text position for bottom plot (ax2)
            const auto txt2 = ax2->text(k_mn, offset_y, "(" + std::to_string(m) + "," + std::to_string(n) + ")");
            txt2->font_size(10);
            txt2->color("black");

            // Adjust text position for bottom plot (ax2)
            const auto txt = ax->text(k_mn, offset_y2, "(" + std::to_string(m) + "," + std::to_string(n) + ")");
            txt->font_size(10);
            txt->color("black");

            // Update label count for next iteration
            label_count[k_mn]++;
        }

        matplot::legend(ax, false);
        matplot::legend(ax2, true);
    }

    /**
     * @brief Calculates the degeneracies for each k value found as a local minimum of the first singular value.
     *
     * This method finds local minima of the first singular value, then checks higher order singular values at those k values.
     * If the higher order singular values are below a given threshold relative to the first singular value, it counts them as degenerate.
     *
     * @param merged_results A pair containing the k values and the SVD results.
     * @param thresholdForLocalMinima The threshold below which a singular value is considered small enough to indicate a local minimum of at least a non-degenerate level.
     * @param thresholdBetweenDegeneracies The threshold that a singular value of a higher order degeneracy is a true indication of degeneracy
     * @return A tuple containing a vector of tuples (k, number_of_degeneracy), sorted by k, and the maximum degeneracy.
     */
    inline std::tuple<std::vector<std::tuple<double, int>>, int> calculateDegeneraciesMerged(
        const std::pair<std::vector<double>, std::vector<Eigen::VectorXd>>& merged_results,
        const double thresholdForLocalMinima = 0.01,
        const double thresholdBetweenDegeneracies = 0.003) {

        const auto& k_values = merged_results.first;
        const auto& svd_results = merged_results.second;

        // Step 1: Identify local minima of the first singular value
        std::vector<std::tuple<double, double>> localMinima;
        const double epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
        for (size_t i = 1; i < svd_results.size() - 1; ++i) {
            const double prevValue = svd_results[i - 1][0];
            const double currValue = svd_results[i][0];
            // ReSharper disable once CppTooWideScopeInitStatement
            const double nextValue = svd_results[i + 1][0];

            if (currValue < prevValue && currValue < nextValue && currValue < thresholdForLocalMinima) {
                localMinima.emplace_back(k_values[i], currValue);
            }

            // Check for sequences of very small values -> this can happen when we have small k and a desymmetrized boundary when there are jumps in curvature. Higher k values and b values resolve this...
            if (currValue < epsilon) {
                const size_t zero_start = i;
                while (i < svd_results.size() && svd_results[i][0] < epsilon) {
                    ++i;
                }
                if (const size_t zero_end = i - 1; zero_end > zero_start) {
                    const size_t zero_middle = (zero_start + zero_end) / 2;
                    localMinima.emplace_back(k_values[zero_middle], svd_results[zero_middle][0]);
                }
            }
        }

        // Step 2: Initialize result vector and futures
        std::vector<std::tuple<double, int>> result;
        std::vector<std::future<std::tuple<double, int>>> futures;
        futures.reserve(localMinima.size());

        // Step 3: Process each local minimum in parallel
        for (const auto& [k, singularValue] : localMinima) {
            futures.emplace_back(std::async(std::launch::async, [&k_values, &svd_results, k, singularValue, thresholdBetweenDegeneracies]() {
                int degeneracyCount = 1;  // Start with the first singular value being a local minimum

                // Find the index of k in k_values
                if (const auto it = std::ranges::find(k_values, k); it != k_values.end()) {
                    const int k_index = std::distance(k_values.begin(), it);  // NOLINT(*-narrowing-conversions)

                    // Check higher order singular values dynamically
                    int currentIndex = 2;  // Start checking from the second singular value
                    bool continueChecking = true;
                    while (continueChecking && currentIndex <= svd_results[k_index].size()) {
                        if (std::abs(svd_results[k_index][currentIndex - 1] - singularValue) < thresholdBetweenDegeneracies) {
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

        // Step 4: Collect results from futures
        result.reserve(futures.size());
        for (auto& future : futures) {
            result.emplace_back(future.get());
        }

        // Step 5: Sort results based on k-values
        std::ranges::sort(result, [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });

        // Step 6: Find the maximum degeneracy
        int maxDegeneracy = 1;
        for (const auto& [k, degeneracyCount] : result) {
            if (degeneracyCount > maxDegeneracy) {
                maxDegeneracy = degeneracyCount;
            }
        }
        return std::make_tuple(result, maxDegeneracy);
    }

    /**
     * @brief Finds local minima of the specified order singular values.
     *
     * This method identifies local minima of the specified order singular value (1 for smallest, 2 for second smallest, etc.).
     *
     * @param svd_results The merged results containing k-values and SVD results.
     * @param indexOfSingularValue The order of the singular value to check for local minima.
     * @param threshold The threshold for identifying local minima.
     * @return A vector of tuples containing (k, singular_value) for each identified local minimum.
     */
    inline std::vector<std::tuple<double, double>> findLocalMinima(const std::pair<std::vector<double>, std::vector<Eigen::VectorXd>>& svd_results, const int indexOfSingularValue, const double threshold = 0.1) {
        std::vector<std::tuple<double, double>> localMinima;
        const auto& [k_values, svd_values] = svd_results;

        if (svd_values.size() < 3) {
            std::cerr << "Not enough points to determine local minima." << std::endl;
            return localMinima;
        }

        const double epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
        for (size_t i = 1; i < svd_values.size() - 1; ++i) {
            const double prevValue = svd_values[i - 1](indexOfSingularValue - 1);
            const double currValue = svd_values[i](indexOfSingularValue - 1);

            if (const double nextValue = svd_values[i + 1](indexOfSingularValue - 1); currValue < prevValue && currValue < nextValue && currValue < threshold) {
                const double k = k_values[i];
                localMinima.emplace_back(k, currValue);
            }

            // Check for sequences of very small values -> can happen with small k where we have corner singularity with desymmetrized boundary
            if (currValue < epsilon) {
                const size_t zero_start = i;
                while (i < svd_values.size() && svd_values[i](indexOfSingularValue - 1) < epsilon) {
                    ++i;
                }
                if (const size_t zero_end = i - 1; zero_end > zero_start) {
                    const size_t zero_middle = (zero_start + zero_end) / 2;
                    const double k = k_values[zero_middle];
                    localMinima.emplace_back(k, svd_values[zero_middle](indexOfSingularValue - 1));
                }
            }
        }
        return localMinima;
    }

    /**
     * @brief Calculates and plots the normalized energy differences using Weyl's law.
     *
     * This function takes the numerical and analytical wavenumbers, computes their corresponding
     * energies \( E = k^2 \), and then calculates the normalized energy differences
     * \((E_{\text{numerical}} - E_{\text{analytical}}) / \rho(E)\), where \(\rho(E)\)
     * is the density of states derived from Weyl's law. The results are plotted using the
     * provided matplot++ axis handle. Optionally, the results can also be printed to the standard output.
     *
     * @param ax The matplot::axes_handle where the normalized energy differences will be plotted.
     * @param numerical_k A vector of numerical wavenumbers.
     * @param analytical_k A vector of analytical wavenumbers.
     * @param modifiedCircumference The length of the domain, modified to take into account the potential Neumann boundary conditions.
     *                              If all we have is Dirichlet, then this equals the total circumference of the boundary.
     * @param boundary The boundary we are doing the simulation on. This is so we can get the area the boundary encloses.
     * @param printToStd A boolean flag indicating whether to print the results to the standard output. Default is true.
     *
     * @throws std::invalid_argument If the size of numerical and analytical vectors are not the same.
     *
     * @details
     * The function computes the energy values from the wavenumbers as \( E = k^2 \). Using Weyl's law,
     * the density of states is calculated as \( \rho(E) = \frac{\text{area}}{4 * \pi} - \frac{\text{length of boundary}}{8 * \pi \sqrt{E}} \).
     * The normalized energy differences are then calculated and plotted. If `printToStd` is true,
     * the method outputs the index, numerical energy, analytical energy, and normalized difference in a tabular format.
     */
    inline std::tuple<std::shared_ptr<class matplot::line>, std::vector<double>> calculateAndPlotEnergyDifferencesUsingWeyl(
        const matplot::axes_handle& ax,
        const std::vector<double>& numerical_k,
        const std::vector<double>& analytical_k,
        const double modifiedCircumference,
        const std::shared_ptr<AbstractBoundary>& boundary,
        const bool printToStd = true)
    {
        if (numerical_k.size() != analytical_k.size()) {
            std::cout << "Size of the numerical_k" << numerical_k.size() << std::endl;
            std::cout << "Size of the analytical k" << analytical_k.size() << std::endl;
            throw std::invalid_argument("The size of numerical and analytical vectors must be the same.");
        }

        std::vector<double> numerical_E(numerical_k.size());
        std::vector<double> analytical_E(analytical_k.size());

        // Compute the energy values E = k^2
        for (size_t i = 0; i < numerical_k.size(); ++i) {
            numerical_E[i] = std::pow(numerical_k[i], 2);
            analytical_E[i] = std::pow(analytical_k[i], 2);
        }

        // Compute the normalized differences
        std::vector<double> normalizedDifferences(numerical_E.size());
        const double area = boundary->getArea();

        for (size_t i = 0; i < numerical_E.size(); ++i) {
            const double deltaE = std::abs(numerical_E[i] - analytical_E[i]);
            const double densityOfStates = (area / (4 * M_PI)) - (modifiedCircumference / (8 * M_PI * std::sqrt(analytical_E[i])));
            normalizedDifferences[i] = deltaE / (1.0 / densityOfStates);
        }

        // Plot the normalized differences using matplot++
        matplot::hold(ax, matplot::on);
        auto sct = ax->scatter(analytical_E, normalizedDifferences);
        sct->marker(matplot::line_spec::marker_style::asterisk);
        ax->xlabel("Analytical E");
        ax->ylabel("(E_n - E_a) / Density of States");
        ax->title("Energy Differences (num vs. anal) in comparison to the density of states");

        // Output the results
        if (printToStd) {
            std::cout << "Index\tk_numerical\tk_analytical\tE_numerical\tE_analytical\tDifference comparison to the density of states\n";
            for (size_t i = 0; i < numerical_E.size(); ++i) {
                std::cout << i << "\t" << numerical_k[i] << "\t" << analytical_k[i] << "\t"
                          << numerical_E[i] << "\t" << analytical_E[i] << "\t"
                          << normalizedDifferences[i] << "\n";
            }
        }
        return std::make_tuple(sct, normalizedDifferences);
    }

    /**
     * @brief Calculates and plots the log-log plot of average errors with a linear fit.
     *
     * This function takes vectors of error values and corresponding b values, computes the average error
     * for each unique b, transforms both b values and average errors to their logarithms, and plots them using matplot++.
     *
     * @param ax The matplot::axes_handle where the results will be plotted.
     * @param errors A vector of vectors representing the error values for each b value.
     * @param b_values A vector of integers representing the b values.
     */
    inline void plotLogLogAverageErrorsWithLinearFit(const matplot::axes_handle& ax, const std::vector<std::vector<double>>& errors, const std::vector<int>& b_values) {
        using namespace matplot;

        // Check if the sizes of errors and b_values vectors are the same
        if (errors.size() != b_values.size()) {
            throw std::invalid_argument("The size of errors and b_values vectors must be the same.");
        }

        // Calculate average errors for each unique b
        std::map<int, std::vector<double>> error_map;
        for (size_t i = 0; i < errors.size(); ++i) {
            error_map[b_values[i]].insert(error_map[b_values[i]].end(), errors[i].begin(), errors[i].end());
        }

        std::map<int, double> avg_errors;
        for (const auto&[fst, snd] : error_map) {
            const double sum = std::accumulate(snd.begin(), snd.end(), 0.0);
            avg_errors[fst] = sum / snd.size(); // NOLINT(*-narrowing-conversions)
        }

        // Extract the data for plotting
        std::vector<double> b_unique, avg_error_values;
        for (const auto&[fst, snd] : avg_errors) {
            b_unique.push_back(fst);
            avg_error_values.push_back(snd);
        }

        // Transform b values and average errors to their logarithms
        std::vector<double> log_b_values(b_unique.size()), log_avg_errors(avg_error_values.size());
        for (size_t i = 0; i < b_unique.size(); ++i) {
            log_b_values[i] = std::log(b_unique[i]);
            log_avg_errors[i] = std::log(avg_error_values[i]);
        }

        // Plot the log-log transformed average errors
        hold(ax, true);
        const auto scatter_plot = scatter(ax, log_b_values, log_avg_errors);
        scatter_plot->display_name("Log-Log Average Errors");

        // Fit a linear model to the log-log data
        Eigen::VectorXd x(log_b_values.size()), y(log_avg_errors.size());
        for (size_t i = 0; i < log_b_values.size(); ++i) {
            x(i) = log_b_values[i]; // NOLINT(*-narrowing-conversions)
            y(i) = log_avg_errors[i]; // NOLINT(*-narrowing-conversions)
        }

        Eigen::VectorXd ones = Eigen::VectorXd::Ones(x.size());
        Eigen::MatrixXd A(x.size(), 2);
        A << x, ones;

        Eigen::VectorXd solution = A.householderQr().solve(y);
        double slope = solution(0);
        double intercept = solution(1);

        // Calculate R^2 value
        Eigen::VectorXd y_fit = A * solution;
        const double ss_res = (y - y_fit).squaredNorm();
        const double ss_tot = (y.array() - y.mean()).matrix().squaredNorm();
        const double r_squared = 1 - (ss_res / ss_tot);

        // Generate fitted values
        std::vector<double> fitted_values;
        fitted_values.reserve(log_b_values.size());
        for (const double log_b_value : log_b_values) {
            fitted_values.push_back(slope * log_b_value + intercept);
        }

        // Plot the fitted line
        const auto plot_fit = plot(ax, log_b_values, fitted_values);
        plot_fit->line_width(2).color("r").display_name("Fit: y = " + std::to_string(slope) + "x + " + std::to_string(intercept) + ", R^2 = " + std::to_string(r_squared));

        // Set plot labels and title
        ax->xlabel("log(b values)");
        ax->ylabel("log(Average (E_n - E_a) / Density of States)");
        ax->title("Log-Log Plot of Average Error vs b with Linear Fit");

        matplot::legend(ax, on);
    }

    /**
     * @brief Calculates and plots the normalized k differences using Weyl's law.
     *
     * This function takes the numerical and analytical wavenumbers and then calculates the normalized k differences
     * \((k_{\text{numerical}} - k_{\text{analytical}}) / \rho(k)\), where \(\rho(k)\)
     * is the density of states derived from Weyl's law. The results are plotted using the
     * provided matplot++ axis handle. Optionally, the results can also be printed to the standard output.
     *
     * @param ax The matplot::axes_handle where the normalized energy differences will be plotted.
     * @param numerical_k A vector of numerical wavenumbers.
     * @param analytical_k A vector of analytical wavenumbers.
     * @param modifiedCircumference The length of the domain, modified to take into account the potential Neumann boundary conditions.
     *                              If all we have is Dirichlet, then this equals the total circumference of the boundary.
     * @param boundary The boundary we are doing the simulation on. This is so we can get the area the boundary encloses.
     * @param printToStd A boolean flag indicating whether to print the results to the standard output. Default is true.
     *
     * @throws std::invalid_argument If the size of numerical and analytical vectors are not the same.
     */
    inline std::tuple<std::shared_ptr<class matplot::line>, std::vector<double>> calculateAndPlotKDifferencesUsingWeyl(
        const matplot::axes_handle& ax,
        const std::vector<double>& numerical_k,
        const std::vector<double>& analytical_k,
        const double modifiedCircumference,
        const std::shared_ptr<AbstractBoundary>& boundary,
        const bool printToStd = true)
    {
        if (numerical_k.size() != analytical_k.size()) {
            throw std::invalid_argument("The size of numerical and analytical vectors must be the same.");
        }

        // Compute the normalized differences
        std::vector<double> normalizedDifferences(numerical_k.size());
        const double area = boundary->getArea();

        for (size_t i = 0; i < numerical_k.size(); ++i) {
            const double deltaE = std::abs(numerical_k[i] - analytical_k[i]);
            const double densityOfStates = (area / (2 * M_PI)) * analytical_k[i] - (modifiedCircumference / (4 * M_PI));
            normalizedDifferences[i] = deltaE / (1.0 / densityOfStates);
        }

        // Plot the normalized differences using matplot++
        matplot::hold(ax, matplot::on);
        auto sct = ax->scatter(analytical_k, normalizedDifferences);
        sct->marker(matplot::line_spec::marker_style::asterisk);
        ax->xlabel("Analytical k");
        ax->ylabel("(k_n - k_a) / Density of States");
        ax->title("wavenumber differences (num vs. anal) in comparison to the density of states");

        // Output the results
        if (printToStd) {
            std::cout << "Index\tk_numerical\tk_analytical\tDifference comparison to the density of states\n";
            for (size_t i = 0; i < numerical_k.size(); ++i) {
                std::cout << i << "\t" << numerical_k[i] << "\t" << analytical_k[i] << "\t"
                          << numerical_k[i] << "\t" << analytical_k[i] << "\t"
                          << normalizedDifferences[i] << "\n";
            }
        }
        return std::make_tuple(sct, normalizedDifferences);
    }

    /**
    * @brief Computes the analytical eigenvalues for a circle within the given range [k_min, k_max].
    *
    * This function computes the eigenvalues for a circle using the zeros of the Bessel function of the first kind.
    * Each eigenvalue is represented as a tuple (k, m, n), where k is the eigenvalue, m is the order of the Bessel function,
    * and n is the index of the zero of the Bessel function.
    *
    * @param k_min The minimum k value.
    * @param k_max The maximum k value.
    * @param radius The radius of the circle.
    * @return A vector of tuples containing the analytical eigenvalues in the form (k, m, n).
    */
    inline std::vector<std::tuple<double, int, int>> computeCircleAnalyticalEigenvalues(const double k_min, const double k_max, const double radius) {
        std::vector<std::tuple<double, int, int>> analytical_eigenvalues;
        for (int nu = 0; ; ++nu) {
            bool stop_outer_loop = false;
            for (int zero_index = 1; ; ++zero_index) {
                const double zero = gsl_sf_bessel_zero_Jnu(nu, zero_index);
                double k_mn = zero / radius;
                if (k_mn > k_max) {
                    if (zero_index == 1) stop_outer_loop = true;
                    break;
                }
                if (k_mn >= k_min) {
                    analytical_eigenvalues.emplace_back(k_mn, nu, zero_index);
                    if (nu > 0) { // Add the negative nu case for doublets
                        analytical_eigenvalues.emplace_back(k_mn, -nu, zero_index);
                    }
                }
            }
            if (stop_outer_loop) break;
        }
        return analytical_eigenvalues;
    }

    /**
    * @brief Finds the closest analytical eigenvalue to the given k for a circle.
    *
    * This function finds the closest eigenvalue to the given k using the precomputed analytical eigenvalues.
    *
    * @param k The target k value.
    * @param radius The radius of the circle.
    * @return A tuple containing the closest eigenvalue k, m, and n.
    */
    inline std::tuple<double, int, int> findClosestEigenvalueCircle(const double k, const double radius) {
        const double k_min = k - 0.1*k;
        const double k_max = k + 0.1*k;
        auto eigenvalues = computeCircleAnalyticalEigenvalues(k_min, k_max, radius);

        const auto closest = std::ranges::min_element(eigenvalues, [k](const auto& a, const auto& b) {
            return std::abs(std::get<0>(a) - k) < std::abs(std::get<0>(b) - k);
        });
        return *closest;
    }

    /**
     * @brief Computes the analytical eigenvalues and their degeneracies for a circle within the given range [k_min, k_max].
     *
     * This function computes the eigenvalues for a circle using the zeros of the Bessel function of the first kind.
     * Each eigenvalue is represented as a tuple (k, degeneracy).
     *
     * @param k_min The minimum k value.
     * @param k_max The maximum k value.
     * @param radius The radius of the circle.
     * @return A vector of tuples containing the eigenvalue k and its degeneracy, sorted by k in increasing order.
     */
    inline std::vector<std::tuple<double, int>> computeCircleEigenvaluesWithoutIndices(const double k_min, const double k_max, const double radius) {
        std::vector<std::tuple<double, int>> eigenvaluesWithDegeneracies;
        for (int nu = 0; ; ++nu) {
            bool stop_outer_loop = false;
            for (int zero_index = 1; ; ++zero_index) {
                const double zero = gsl_sf_bessel_zero_Jnu(nu, zero_index);
                double k_mn = zero / radius;
                if (k_mn > k_max) {
                    if (zero_index == 1) stop_outer_loop = true;
                    break;
                }
                if (k_mn >= k_min) {
                    int degeneracy = (nu == 0) ? 1 : 2; // Doublet for non-zero nu
                    eigenvaluesWithDegeneracies.emplace_back(k_mn, degeneracy);
                }
            }
            if (stop_outer_loop) break;
        }

        // Sort the eigenvalues by k in increasing order
        std::ranges::sort(eigenvaluesWithDegeneracies, [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });

        return eigenvaluesWithDegeneracies;
    }

    /**
     * @brief Computes and prints the analytical eigenvalues for a circle within the given range [k_min, k_max].
     *
     * This function computes the eigenvalues for a circle using the zeros of the Bessel function of the first kind.
     * Each eigenvalue is represented as a tuple (k, m, n), where k is the eigenvalue, m is the order of the Bessel function,
     * and n is the index of the zero of the Bessel function. The results are printed to the standard output.
     *
     * @param k_min The minimum k value.
     * @param k_max The maximum k value.
     * @param radius The radius of the circle.
     */
    inline void printCircleAnalyticalEigenvalues(const double k_min, const double k_max, const double radius) {
        std::vector<std::tuple<double, int, int>> analytical_eigenvalues;
        for (int nu = 0; ; ++nu) {
            bool stop_outer_loop = false;
            for (int zero_index = 1; ; ++zero_index) {
                const double zero = gsl_sf_bessel_zero_Jnu(nu, zero_index);
                double k_mn = zero / radius;
                if (k_mn > k_max) {
                    if (zero_index == 1) stop_outer_loop = true;
                    break;
                }
                if (k_mn >= k_min) {
                    analytical_eigenvalues.emplace_back(k_mn, nu, zero_index);
                    if (nu > 0) { // Add the negative nu case for doublets
                        analytical_eigenvalues.emplace_back(k_mn, -nu, zero_index);
                    }
                }
            }
            if (stop_outer_loop) break;
        }

        // Sort the eigenvalues by k
        std::ranges::sort(analytical_eigenvalues,
                          [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });


        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Analytical eigenvalues for the circle:\n";
        for (const auto& [k, m, n] : analytical_eigenvalues) {
            std::cout << "k: " << k << ", m: " << m << ", n: " << n << "\n";
        }
    }

    /**
     * @brief Computes the analytical eigenvalues for a rectangle within the given range [k_min, k_max].
     *
     * This function computes the eigenvalues for a rectangle using the formula for the eigenvalues of a rectangular domain.
     * Each eigenvalue is represented as a tuple (k, m, n), where k is the eigenvalue, m and n are the integer indices.
     *
     * @param k_min The minimum k value.
     * @param k_max The maximum k value.
     * @param width The width of the rectangle.
     * @param height The height of the rectangle.
     * @return A vector of tuples containing the analytical eigenvalues in the form (k, m, n).
     */
    inline std::vector<std::tuple<double, int, int>> computeRectangleAnalyticalEigenvalues(const double k_min, const double k_max, const double width, const double height) {
        std::vector<std::tuple<double, int, int>> analytical_eigenvalues;
        for (int m = 1; ; ++m) {
            bool stop_outer_loop = false;
            for (int n = 1; ; ++n) {
                const double k_mn_squared = std::pow(m * M_PI / width, 2) + std::pow(n * M_PI / height, 2);
                if (k_mn_squared > k_max * k_max) {
                    if (n == 1) stop_outer_loop = true;
                    break;
                }
                if (k_min * k_min <= k_mn_squared) {
                    double k_mn = sqrt(k_mn_squared);
                    analytical_eigenvalues.emplace_back(k_mn, m, n);
                }
            }
            if (stop_outer_loop) break;
        }
        return analytical_eigenvalues;
    }

    /**
    * @brief Finds the closest analytical eigenvalue to the given k for a rectangle.
    *
    * This function finds the closest eigenvalue to the given k using the precomputed analytical eigenvalues.
    *
    * @param k The target k value.
    * @param width The width of the rectangle.
    * @param height The height of the rectangle.
    * @return A tuple containing the closest eigenvalue k, m, and n.
    */
    inline std::tuple<double, int, int> findClosestEigenvalue(const double k, const double width, const double height) {
        const double k_min = k - 0.1*k;
        const double k_max = k + 0.1*k;
        auto eigenvalues = computeRectangleAnalyticalEigenvalues(k_min, k_max, width, height);
        const auto closest = std::ranges::min_element(eigenvalues, [k](const auto& a, const auto& b) {
            return std::abs(std::get<0>(a) - k) < std::abs(std::get<0>(b) - k);
        });
        return *closest;
    }

    /**
     * @brief Computes the analytical eigenvalues and their degeneracies for a rectangle within the given range [k_min, k_max].
     *
     * This function computes the eigenvalues for a rectangle using the formula for the eigenvalues of a rectangular domain.
     * Each eigenvalue is represented as a tuple (k, degeneracy).
     *
     * @param k_min The minimum k value.
     * @param k_max The maximum k value.
     * @param width The width of the rectangle.
     * @param height The height of the rectangle.
     * @return A vector of tuples containing the eigenvalue k and its degeneracy, sorted by k in increasing order.
     */
    inline std::vector<std::tuple<double, int>> computeRectangleEigenvaluesWithoutIndices(const double k_min, const double k_max, const double width, const double height) {
        std::vector<std::tuple<double, int>> eigenvaluesWithDegeneracies;
        for (int m = 1; ; ++m) {
            bool stop_outer_loop = false;
            for (int n = 1; ; ++n) {
                const double k_mn_squared = std::pow(m * M_PI / width, 2) + std::pow(n * M_PI / height, 2);
                if (k_mn_squared > k_max * k_max) {
                    if (n == 1) stop_outer_loop = true;
                    break;
                }
                if (k_min * k_min <= k_mn_squared) {
                    double k_mn = sqrt(k_mn_squared);
                    eigenvaluesWithDegeneracies.emplace_back(k_mn, 1); // Degeneracy for rectangular eigenvalues
                }
            }
            if (stop_outer_loop) break;
        }

        // Sort the eigenvalues by k in increasing order
        std::ranges::sort(eigenvaluesWithDegeneracies, [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });

        return eigenvaluesWithDegeneracies;
    }

    /**
     * @brief Computes and prints the analytical eigenvalues for a rectangle within the given range [k_min, k_max].
     *
     * This function computes the eigenvalues for a rectangle using the formula for the eigenvalues of a rectangular domain.
     * Each eigenvalue is represented as a tuple (k, m, n), where k is the eigenvalue, m and n are the integer indices.
     * The results are printed to the standard output.
     *
     * @param k_min The minimum k value.
     * @param k_max The maximum k value.
     * @param width The width of the rectangle.
     * @param height The height of the rectangle.
     */
    inline void printRectangleAnalyticalEigenvalues(const double k_min, const double k_max, const double width, const double height) {
        std::vector<std::tuple<double, int, int>> analytical_eigenvalues;
        for (int m = 1; ; ++m) {
            bool stop_outer_loop = false;
            for (int n = 1; ; ++n) {
                const double k_mn_squared = std::pow(m * M_PI / width, 2) + std::pow(n * M_PI / height, 2);
                if (k_mn_squared > k_max * k_max) {
                    if (n == 1) stop_outer_loop = true;
                    break;
                }
                if (k_min * k_min <= k_mn_squared) {
                    double k_mn = sqrt(k_mn_squared);
                    analytical_eigenvalues.emplace_back(k_mn, m, n);
                }
            }
            if (stop_outer_loop) break;
        }

        // Sort the eigenvalues by k
        std::ranges::sort(analytical_eigenvalues,
                          [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Analytical eigenvalues for the rectangle:\n";
        for (const auto& [k, m, n] : analytical_eigenvalues) {
            std::cout << "k: " << k << ", m: " << m << ", n: " << n << "\n";
        }
    }

    /**
     * @brief Counts the analytical eigenvalues for a circle within a given k range.
     *
     * This method calculates the analytical eigenvalues for a circle and counts them,
     * considering their degeneracies.
     *
     * @param radius The radius of the circle.
     * @param k_min The minimum k value.
     * @param k_max The maximum k value.
     * @return The total count of eigenvalues considering their degeneracies.
     */
    inline int countCircleAnalyticalEigenvalues(const double radius, const double k_min, const double k_max) {
        const auto analytical_eigenvalues = computeCircleAnalyticalEigenvalues(radius, k_min, k_max);
        return analytical_eigenvalues.size(); // Each analytical eigenvalue in the circle case is a singlet NOLINT(*-narrowing-conversions)
    }

    /**
     * @brief Counts the analytical eigenvalues for a rectangle within a given k range.
     *
     * This method calculates the analytical eigenvalues for a rectangle and counts them,
     * considering their degeneracies.
     *
     * @param width The width of the rectangle.
     * @param height The height of the rectangle.
     * @param k_min The minimum k value.
     * @param k_max The maximum k value.
     * @return The total count of eigenvalues considering their degeneracies.
     */
    inline int countRectangleAnalyticalEigenvalues(const double width, const double height, const double k_min, const double k_max) {
        const auto analytical_eigenvalues = computeRectangleAnalyticalEigenvalues(width, height, k_min, k_max);
        return analytical_eigenvalues.size(); // Each analytical eigenvalue in the rectangle case is a singlet NOLINT(*-narrowing-conversions)
    }

    /**
     * @brief Counts the eigenvalues within a given range, taking degeneracy into account.
     *
     * This method iterates over the eigenvalues and adds 1 for each one. If the degeneracy
     * of a particular eigenvalue is greater than 1, it adds the degeneracy number instead.
     * This allows for accurate counting of the number of eigenvalues, considering their degeneracies.
     *
     * @param eigenvaluesWithDegeneracies A vector of tuples containing (k, degeneracy, vector_of_eigenvectors_at_that_k).
     * @param limitTo To what k we want to count the eigenvalues
     * @return The total count of eigenvalues considering their degeneracies.
     */
    inline int countEigenvaluesWithDegeneracy(const std::vector<std::tuple<double, int>>& eigenvaluesWithDegeneracies, const double limitTo) {
        int eigenvalueCount = 0;

        // Iterate over the eigenvalues and their degeneracies
        for (const auto& [k, degeneracy] : eigenvaluesWithDegeneracies) {
            if (k > limitTo) {
                break;
            }
            // Add the degeneracy number to the count
            eigenvalueCount += degeneracy;
        }
        return eigenvalueCount;
    }

    /**
     * @brief Plots the theoretical eigenvalues count versus k.
     *
     * This method iterates over the analytical eigenvalues, counts the number of eigenvalues up to each k value, and plots the count of eigenvalues against k.
     *
     * @param ax The axis handle from matplot++ for plotting.
     * @param analytical_eigenvalues A vector of tuples containing (k, m, n) for the analytical eigenvalues.
     * @param k_min The minimum k value for plotting.
     * @param k_max The maximum k value for plotting.
     */
    inline void plotTheoreticalEigenvalues(const matplot::axes_handle &ax, const std::vector<std::tuple<double, int, int>>& analytical_eigenvalues, const double k_min, const double k_max) {
        std::vector<double> k_values;
        std::vector<double> eigenvalueCounts;
        double count = 0;

        // Sort the analytical eigenvalues by k
        auto sorted_analytical_eigenvalues = analytical_eigenvalues;
        std::ranges::sort(sorted_analytical_eigenvalues,
                          [](const auto& a, const auto& b) {
                              return std::get<0>(a) < std::get<0>(b);
                          });

        double previous_k = -1.0;
        for (const auto& [k, m, n] : sorted_analytical_eigenvalues) {
            if (k >= k_min && k <= k_max) {
                if (k != previous_k) {
                    k_values.push_back(k);
                    count = count + 1.0;
                    eigenvalueCounts.push_back(count);
                    previous_k = k;
                } else {
                    count = count + 1.0;
                    eigenvalueCounts.back() = count;
                }
            }
        }

        hold(ax, matplot::on);
        const auto plt = ax->scatter(k_values, eigenvalueCounts);
        plt->display_name("Theoretical");
        plt->marker(matplot::line_spec::marker_style::diamond).marker_size(10.0);
    }

    /**
     * @brief Plots the numerical eigenvalues count versus k.
     *
     * This method iterates over the numerical eigenvalues, counts the number of eigenvalues (including degeneracies) up to each k value using countEigenvaluesWithDegeneracy, and plots the count of eigenvalues against k.
     *
     * @param ax The axis handle from matplot++ for plotting.
     * @param numerical_eigenvalues A vector of tuples containing (k, degeneracy, eigenvectors) for the numerical eigenvalues.
     * @param k_min The minimum k value for plotting.
     * @param k_max The maximum k value for plotting.
     */
    inline void plotNumericalEigenvalues(const matplot::axes_handle &ax, const std::vector<std::tuple<double, int>>& numerical_eigenvalues, const double k_min, double const k_max) {
        std::vector<double> k_values;
        std::vector<double> eigenvalueCounts;
        for (const auto& [k, degeneracy] : numerical_eigenvalues) {
            if (k >= k_min && k <= k_max) {
                k_values.push_back(k);
                double count = countEigenvaluesWithDegeneracy(numerical_eigenvalues, k);
                eigenvalueCounts.push_back(count);
            }
        }
        hold(ax, matplot::on);
        const auto plt = ax->scatter(k_values, eigenvalueCounts);
        plt->display_name("Numerical");
        plt->marker(matplot::line_spec::marker_style::circle).marker_size(5.0);
    }

    /**
     * @brief Plots the comparison between the numerical and theoretical eigenvalue counts.
     *
     * This method counts the number of numerical eigenvalues (including their degeneracies) up to each \( k \) value,
     * and compares it to the theoretical prediction given by Weyl's law. The counts are then plotted using matplot++.
     *
     * @param ax The axis handle from matplot++ for plotting.
     * @param numerical_eigenvalues A vector of tuples containing (k, degeneracy, eigenvectors) for the numerical eigenvalues.
     * @param boundary The boundary of the domain, used to get the area.
     * @param boundaryLength The length of the boundary.
     * @param C The curvature and corner corrections to Weyl's law
     */
    inline void compareEigenvalueCountsWithWeylDefault(
        const matplot::axes_handle &ax,
        const std::vector<std::tuple<double, int>>& numerical_eigenvalues,
        const std::shared_ptr<AbstractBoundary>& boundary,
        const double boundaryLength,
        const double C)
    {
        using namespace matplot;
        // Calculate the area of the boundary
        const double area = boundary->getArea();

        // Prepare vectors for k values and counts
        std::vector<double> k_values;
        std::vector<double> numerical_counts;
        std::vector<double> theoretical_counts;

        // Sort numerical eigenvalues by k
        auto sorted_numerical_eigenvalues = numerical_eigenvalues;
        std::ranges::sort(sorted_numerical_eigenvalues,
                          [](const auto& a, const auto& b) {
                              return std::get<0>(a) < std::get<0>(b);
                          });

        // Count numerical eigenvalues and calculate theoretical counts
        for (const auto& [k, degeneracy] : sorted_numerical_eigenvalues) {
            k_values.push_back(k);

            // Count numerical eigenvalues up to k
            numerical_counts.push_back(countEigenvaluesWithDegeneracy(sorted_numerical_eigenvalues, k));

            // Calculate theoretical count using Weyl's law with the previously calculated corner correction
            double theoretical_count = (area * k * k) / (4 * M_PI) - (boundaryLength * k) / (4 * M_PI) + C;
            theoretical_counts.push_back(theoretical_count);
        }

        // Plot numerical eigenvalue counts
        hold(ax, true);
        const auto num_plot = ax->scatter(k_values, numerical_counts);
        num_plot->display_name("Numerical Eigenvalues");
        num_plot->marker(line_spec::marker_style::circle).marker_size(3.0);

        // Plot theoretical eigenvalue counts
        const auto theo_plot = ax->scatter(k_values, theoretical_counts);
        theo_plot->display_name("Theoretical (Weyl's Law)");
        theo_plot->marker(line_spec::marker_style::diamond).marker_size(3.0);

        // Set plot labels and title
        ax->xlabel("k");
        ax->ylabel("Cumulative Count of Eigenvalues");
        ax->title("Comparison of Numerical Eigenvalue Counts with Weyl's Law");

        std::cout << "The value of C is: " << std::to_string(C) << std::endl;
        matplot::legend(ax, true);
    }

    /**
     * @brief Plots the theoretical and numerical eigenvalues with comparison to Weyl's law.
     *
     * This method compares the theoretical eigenvalues to the numerical results, and plots the eigenvalue counts
     * against Weyl's law.
     *
     * @param degeneraciesResult The result from calculateDegeneracies method.
     * @param analytical_eigenvalues The analytical eigenvalues for theoretical plotting
     * @param k_min The minimum k value.
     * @param k_max The maximum k value.
     * @param boundary The boundary of the geometry
     * @param modifiedLength If we need to take the compounded Dirichlet/Neumann bc into account
     * @param C The curvature and corner corrections to Weyl's law
     */
    inline void plotEigenvaluesComparisonWithWeyl(
        const std::tuple<std::vector<std::tuple<double, int, std::vector<Eigen::VectorXcd>>>, int>& degeneraciesResult, const std::vector<std::tuple<double, int, int>> &analytical_eigenvalues,
        const double k_min, const double k_max,
        const std::shared_ptr<AbstractBoundary>& boundary, const double modifiedLength, const double C) {
        using namespace matplot;
        const auto numerical_results = extractDegeneracyInfo(std::get<0>(degeneraciesResult));

        const auto figComparison = figure(true);
        figComparison->size(800, 800);
        const auto axComparison = figComparison->add_axes();
        axComparison->xlabel("k");
        axComparison->ylabel("Number of Eigenvalues");
        axComparison->title("Comparing theoretical vs. numerical eigenvalue counting functions");
        matplot::legend(axComparison, true);
        plotTheoreticalEigenvalues(axComparison, analytical_eigenvalues, k_min, k_max);
        plotNumericalEigenvalues(axComparison, numerical_results, k_min, k_max);

        const auto figWeyl2 = figure(true);
        figWeyl2->size(800, 800);
        const auto axWeyl2 = figWeyl2->add_axes();
        compareEigenvalueCountsWithWeylDefault(axWeyl2, numerical_results, boundary, modifiedLength, C);
    }

    /**
     * @brief Calculates and returns the mean level spacing of unfolded energy eigenvalues.
     *
     * This method computes the mean level spacing of a given set of unfolded energy eigenvalues.
     * If the unfolding is correct, the mean level spacing should be approximately 1.
     *
     * @param unfolded_energies A vector of unfolded energy eigenvalues.
     * @return The mean level spacing of the unfolded energy eigenvalues.
     */
    inline double calculateMeanLevelSpacing(const std::vector<double>& unfolded_energies) {
        double sum = 0.0;
        for (size_t i = 1; i < unfolded_energies.size(); ++i) {
            sum += (unfolded_energies[i] - unfolded_energies[i - 1]);
        }
        return sum / static_cast<double>(unfolded_energies.size() - 1);
    }

    /**
     * @brief Calculates and returns the mean level spacing of unfolded energy eigenvalues.
     *
     * This method computes the mean level spacing of a given set of unfolded energy eigenvalues.
     * If the unfolding is correct, the mean level spacing should be approximately 1.
     *
     * @param unfolded_energies_with_degeneracy A vector of tuples containing unfolded energy eigenvalues and their degeneracies.
     * @return The mean level spacing of the unfolded energy eigenvalues.
     */
    inline double calculateMeanLevelSpacingWithDegeneracy(const std::vector<std::tuple<double, int>>& unfolded_energies_with_degeneracy) {
        double sum = 0.0;
        int total_degeneracy = 0;

        for (size_t i = 1; i < unfolded_energies_with_degeneracy.size(); ++i) {
            const double delta_energy = std::get<0>(unfolded_energies_with_degeneracy[i]) - std::get<0>(unfolded_energies_with_degeneracy[i - 1]);
            sum += delta_energy * std::get<1>(unfolded_energies_with_degeneracy[i - 1]);
            total_degeneracy += std::get<1>(unfolded_energies_with_degeneracy[i - 1]);
        }

        return sum / static_cast<double>(total_degeneracy);
    }

    /**
     * @brief Unfolds the energy eigenvalues using Weyl's law.
     *
     * This method transforms numerical wavenumbers into unfolded energy eigenvalues by using the integrated density of states
     * derived from Weyl's law. The modifiedPerimeter of the boundary can be specified to account for potential Neumann boundary conditions.
     *
     * @param numerical_k_wavenumbers A vector of tuples containing (k, degeneracy) for the numerical wavenumbers.
     * @param boundary The boundary of the billiard
     * @param modifiedPerimeter The perimeter of the boundary modified by taking into account the combination of Dirichlet or Neumann boundary conditions
     * @param C The curvature and corner corrections to Weyl's law
     * @return A vector of unfolded energy eigenvalues.
     *
     * @note The input to this method should be numerical wavenumbers, not numerical energies.
     */
    inline std::vector<double> unfoldEnergyEigenvalues(
        const std::vector<std::tuple<double, int>>& numerical_k_wavenumbers,
        const std::shared_ptr<AbstractBoundary>& boundary,
        const double modifiedPerimeter, const double C)
    {
        // Step 1: Calculate the energy eigenvalues from the numerical wavenumbers
        std::vector<double> energies;
        energies.reserve(numerical_k_wavenumbers.size());
        for (const auto& [k, degeneracy] : numerical_k_wavenumbers) {
            energies.push_back(k * k);
        }

        // Area of the boundary
        const double area = boundary->getArea();

        // Step 2: Sort the energy eigenvalues
        std::ranges::sort(energies);

        // Step 3: Unfold the energy eigenvalues
        std::vector<double> unfolded_energies;
        unfolded_energies.reserve(energies.size());
        for (const double E : energies) {
            double N_E = (area * E) / (4 * M_PI) - (modifiedPerimeter * std::sqrt(E)) / (4 * M_PI) + C;
            unfolded_energies.push_back(N_E);
        }
        return unfolded_energies;
    }

    /**
     * @brief Unfolds the energy eigenvalues using Weyl's law and preserves degeneracy.
     *
     * This method transforms numerical wavenumbers into unfolded energy eigenvalues by using the integrated density of states
     * derived from Weyl's law. The modifiedPerimeter of the boundary can be specified to account for potential Neumann boundary conditions.
     * It also preserves the degeneracy of the eigenvalues.
     *
     * @param numerical_k_wavenumbers A vector of tuples containing (k, degeneracy) for the numerical wavenumbers.
     * @param boundary The boundary of the billiard.
     * @param modifiedPerimeter The perimeter of the boundary modified by taking into account the combination of Dirichlet or Neumann boundary conditions.
     * @param C The curvature and corner corrections to Weyl's law.
     * @return A vector of tuples containing unfolded energy eigenvalues and their degeneracies.
     *
     * @note The input to this method should be numerical wavenumbers, not numerical energies.
     */
    inline std::vector<std::tuple<double, int>> unfoldEnergyEigenvaluesWithDegeneracy(
        const std::vector<std::tuple<double, int>>& numerical_k_wavenumbers,
        const std::shared_ptr<AbstractBoundary>& boundary,
        const double modifiedPerimeter, const double C)
    {
        // Step 1: Calculate the energy eigenvalues from the numerical wavenumbers
        std::vector<std::tuple<double, int>> energies_with_degeneracy;
        energies_with_degeneracy.reserve(numerical_k_wavenumbers.size());
        for (const auto& [k, degeneracy] : numerical_k_wavenumbers) {
            double energy = k * k;
            energies_with_degeneracy.emplace_back(energy, degeneracy);
        }

        // Area of the boundary
        const double area = boundary->getArea();

        // Step 2: Sort the energy eigenvalues
        std::ranges::sort(energies_with_degeneracy, [](const auto& lhs, const auto& rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });

        // Step 3: Unfold the energy eigenvalues
        std::vector<std::tuple<double, int>> unfolded_energies_with_degeneracy;
        unfolded_energies_with_degeneracy.reserve(energies_with_degeneracy.size());
        for (const auto& [E, degeneracy] : energies_with_degeneracy) {
            double N_E = (area * E) / (4 * M_PI) - (modifiedPerimeter * std::sqrt(E)) / (4 * M_PI) + C;
            unfolded_energies_with_degeneracy.emplace_back(N_E, degeneracy);
        }
        return unfolded_energies_with_degeneracy;
    }

    /**
     * @brief Prints the k values and their corresponding unfolded energy values.
     *
     * This method uses the unfoldEnergyEigenvalues function to calculate the unfolded energy values
     * and then prints each k value along with its corresponding unfolded energy value.
     *
     * @param numerical_k_wavenumbers A vector of tuples containing (k, degeneracy) for the numerical wavenumbers.
     * @param boundary The boundary of the billiard
     * @param modifiedPerimeter The perimeter of the boundary modified by taking into account the combination of Dirichlet or Neumann boundary conditions
     * @param C The curvature and corner corrections to Weyl's law
     */
    inline void printUnfoldedEnergies(
        const std::vector<std::tuple<double, int>>& numerical_k_wavenumbers,
        const std::shared_ptr<AbstractBoundary>& boundary,
        const double modifiedPerimeter, const double C)
    {
        // Calculate unfolded energies
        const std::vector<double> unfolded_energies = unfoldEnergyEigenvalues(numerical_k_wavenumbers, boundary, modifiedPerimeter, C);

        // Print the k values and their corresponding unfolded energy values
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "k values and their corresponding unfolded energies:" << std::endl;
        for (size_t i = 0; i < numerical_k_wavenumbers.size(); ++i) {
            const double k = std::get<0>(numerical_k_wavenumbers[i]);
            const double unfolded_energy = unfolded_energies[i];
            std::cout << "k: " << k << ", Unfolded Energy: " << unfolded_energy << std::endl;
        }
    }

    /**
     * @brief Plots the spacing distribution of unfolded eigenvalues, including the empirical histogram and theoretical distributions (Poisson, GOE, GUE).
     *
     * This method computes the nearest neighbor spacings from the given unfolded eigenvalues, creates a histogram of these spacings,
     * and plots the empirical histogram alongside the theoretical distributions for Poisson, GOE, and GUE on the provided matplot++ axes handles.
     *
     * Detailed Explanation:
     *
     * 1. Helper Functions:
     * • computeNearestNeighborSpacings: Computes the nearest neighbor spacings from the given unfolded eigenvalues.
     * • calculateIQR: Calculates the interquartile range (IQR) which is used in the Freedman-Diaconis rule.
     * • createHistogram: Creates and normalizes the histogram to represent the probability density function (PDF) of the spacings.
     * • poissonDistribution, goeDistribution, gueDistribution: Define the theoretical distributions for Poisson, GOE, and GUE.
     *
     * 2. Main Function Steps:
     * • Step 1: Compute the nearest neighbor spacings from the unfolded eigenvalues.
     * • Step 2: Calculate the bin width using the Freedman-Diaconis rule to ensure a dynamic bin width based on the data.
     * • Step 3: Create and normalize the histogram using the computed bin width.
     * • Step 4: Plot the histogram of the nearest neighbor spacings on the upper axis.
     * • Step 5: Prepare and plot the theoretical distributions (Poisson, GOE, GUE) on the lower axis.
     * • Step 6: Plot the empirical p(s) from the histogram on the lower axis.
     *
     * @param ax_upper The upper axis handle from matplot++ for plotting the histogram of nearest neighbor spacings.
     * @param ax_lower The lower axis handle from matplot++ for plotting the theoretical distributions and empirical p(s).
     * @param unfolded_energy_eigenvalues A vector of unfolded eigenvalues.
     * @param bin_scaling The realtive scaling of the default Freedman-Diaconis rule for calculating the bin_width for the histograms
     * @param plotBerryRobnik Whether to plot the Berry-Robnik distribution (default: false).
     * @param rho The mixing parameter for the Berry-Robnik distribution (default: 0.5).
     * @return A tuple containing the histogram plot handle and the four line handles (our p(s), Poisson, GOE, GUE).
     */
    inline std::tuple<matplot::histogram_handle, matplot::line_handle, matplot::line_handle, matplot::line_handle, matplot::line_handle> plotSpacingDistribution(
        const matplot::axes_handle &ax_upper,
        const matplot::axes_handle &ax_lower,
        const std::vector<double> &unfolded_energy_eigenvalues, const double bin_scaling = 1.0, bool plotBerryRobnik = false, double rho = 0.5) {

        // Helper function to compute nearest neighbor spacings
        auto computeNearestNeighborSpacings = [](const std::vector<double> &eigenvalues) {
            std::vector<double> spacings;
            for (size_t i = 1; i < eigenvalues.size(); ++i) {
                spacings.push_back(eigenvalues[i] - eigenvalues[i - 1]);
            }
            return spacings;
        };

        // Helper function to calculate the interquartile range (IQR)
        auto calculateIQR = [](std::vector<double> data) {
            std::ranges::sort(data);
            const size_t q1_idx = data.size() / 4;
            const size_t q3_idx = 3 * data.size() / 4;
            return data[q3_idx] - data[q1_idx];
        };

        // Helper function to create and normalize histogram
        auto createHistogram = [](const std::vector<double> &spacings, const double bin_width) {
            // Determine the number of bins
            const size_t num_bins = static_cast<size_t>(std::ceil(*std::ranges::max_element(spacings) / bin_width));

            std::vector<double> histogram(num_bins, 0.0);
            std::vector<double> errors(num_bins, 0.0);
            std::vector<int> counts(num_bins, 0); // Store counts for annotation

            // Fill the histogram
            for (const auto& spacing : spacings) {
                if (const auto bin = static_cast<size_t>(std::floor(spacing / bin_width)); bin < num_bins) {
                    histogram[bin] += 1.0;
                    counts[bin] += 1; // Increment count
                }
            }

            // Calculate the errors (standard error of the mean for each bin)
            const auto total_count = static_cast<double>(spacings.size());
            for (size_t i = 0; i < histogram.size(); ++i) {
                if (histogram[i] > 0) {
                    errors[i] = std::sqrt(histogram[i]) / (total_count * bin_width);
                    histogram[i] /= (total_count * bin_width);
                }
            }

            return std::make_tuple(histogram, errors, counts);
        };

        // Helper functions for theoretical distributions
        auto poissonDistribution = [](const double s) {
            return std::exp(-s);
        };

        auto goeDistribution = [](const double s) {
            return (M_PI / 2) * s * std::exp(-M_PI * s * s / 4);
        };

        auto gueDistribution = [](const double s) {
            return (32 / (M_PI * M_PI)) * s * s * std::exp(-4 * s * s / M_PI);
        };

        // Step 1: Compute nearest neighbor spacings
        const auto spacings = computeNearestNeighborSpacings(unfolded_energy_eigenvalues);

        // Step 2: Calculate bin width using the Freedman-Diaconis rule
        const double iqr = calculateIQR(spacings);
        const double bin_width = 2.0 * iqr / std::pow(spacings.size(), 1.0 / 3.0) * bin_scaling;

        // Step 3: Create and normalize histogram
        const auto [histogram, errors, counts] = createHistogram(spacings, bin_width);

        // Prepare x values for the histogram

        std::vector<double> histogram_x(histogram.size());
        for (size_t i = 0; i < histogram.size(); ++i) {
            histogram_x[i] = (i) * bin_width; // Center x-values for error bars NOLINT(*-narrowing-conversions)
        }

        // Prepare x values for the error bars
        std::vector<double> errors_x(histogram.size());
        for (size_t i = 0; i < histogram.size(); ++i) {
            errors_x[i] = (i) * bin_width + 0.5 * bin_width; // Center x-values for error bars NOLINT(*-narrowing-conversions)
        }

        // Step 4: Plot histogram
        matplot::hold(ax_upper, true);
        auto hist_plot = ax_upper->hist(spacings, histogram_x);
        hist_plot->normalization(matplot::histogram::normalization::pdf);
        hist_plot->display_name("Empirical Histogram");
        ax_upper->xlabel("Spacing (s)");
        ax_upper->ylabel("Frequency");
        ax_upper->title("Histogram of Nearest Neighbor Spacings");
        ax_upper->xlim({0.0, histogram_x.back() + bin_width / 2.0});
        ax_upper->ylim({0.0, 1.5 * *std::ranges::max_element(histogram)});

        // Add error bars to the histogram
        const auto error_bar_plot = ax_upper->errorbar(errors_x, histogram, errors, "k");
        error_bar_plot->display_name("Error Bars");

        // Add annotations for counts
        for (size_t i = 0; i < counts.size(); ++i) {
            if (counts[i] > 0) {
                const auto annotation = ax_upper->text(errors_x[i], histogram[i] + 1.1 * errors[i], "N = " + std::to_string(counts[i]));
            }
        }

        // Add legend
        const auto l = matplot::legend(ax_upper, true);
        l->strings({"Num. of spacings"});

        // Step 5: Prepare and plot theoretical distributions
        constexpr size_t num_points = 1000; // Updated to 1000 points
        std::vector<double> s_values(num_points);
        std::vector<double> poisson_pdf(num_points);
        std::vector<double> goe_pdf(num_points);
        std::vector<double> gue_pdf(num_points);
        std::vector<double> berry_robnik_pdf(num_points);

        const double max_s = histogram_x.back() + bin_width;

        for (size_t i = 0; i < num_points; ++i) {
            const double s = max_s * (static_cast<double>(i) / (num_points - 1));
            s_values[i] = s;
            poisson_pdf[i] = poissonDistribution(s);
            goe_pdf[i] = goeDistribution(s);
            gue_pdf[i] = gueDistribution(s);
            if (plotBerryRobnik) {
                berry_robnik_pdf[i] = BerryRobnik::probabilityBerryRobnik(s, rho);
            }
        }

        matplot::hold(ax_lower, true);
        auto poisson_plot = ax_lower->plot(s_values, poisson_pdf);
        poisson_plot->line_width(1.0);
        poisson_plot->display_name("Poisson");
        auto goe_plot = ax_lower->plot(s_values, goe_pdf);
        goe_plot->line_width(1.0);
        goe_plot->display_name("GOE");
        auto gue_plot = ax_lower->plot(s_values, gue_pdf);
        gue_plot->line_width(1.0);
        gue_plot->display_name("GUE");

        if (plotBerryRobnik) {
            auto br_plot = ax_lower->plot(s_values, berry_robnik_pdf);
            br_plot->line_width(1.0);
            std::string berryRobnikDisplayName = "Berry-Robnik, rho=" + std::to_string(rho);
            br_plot->display_name(berryRobnikDisplayName);
        }
        // Step 6: Plot empirical p(s) from histogram
        const std::vector<double>& empirical_ps = histogram;
        auto empirical_plot = ax_lower->scatter(histogram_x, empirical_ps);
        empirical_plot->display_name("Empirical p(s)");

        // Add error bars to the empirical p(s) plot
        const auto error_bar_plot_ps = ax_lower->errorbar(histogram_x, empirical_ps, errors, "k");
        error_bar_plot_ps->display_name("Error Bars for Empirical p(s)");

        ax_lower->xlabel("Spacing (s)");
        ax_lower->ylabel("Probability Density");
        ax_lower->title("Probability Density of Nearest Neighbor Spacings");
        matplot::legend(ax_lower, true);

        // Return handles for the histogram and the four plots
        return std::make_tuple(hist_plot, empirical_plot, poisson_plot, goe_plot, gue_plot);
    }

    /**
     * @brief Plots the oscillatory part of the eigenvalue counts using Weyl's law.
     *
     * This method computes the cumulative count of numerical energy eigenvalues up to each \( E \) value,
     * calculates the smooth part of Weyl's law, and then plots the difference between the two.
     * This helps visualize the oscillatory behavior of the number of states.
     *
     * @param ax The axis handle from matplot++ for plotting.
     * @param numerical_eigenvalues A vector of tuples containing (k, degeneracy) for the numerical eigenvalues.
     * @param boundary The boundary of the domain, used to get the area.
     * @param boundaryLength The length of the boundary (to account for potential Neumann boundary conditions).
     * @param C The C is the corner and curvature correction for the billiard.
     * @param interval The interval over which to average oscillations.
     * @return The line handle of the plot.
     */
    inline matplot::line_handle plotOscillatoryPartWeyl(const matplot::axes_handle &ax, const std::vector<std::tuple<double, int>>& numerical_eigenvalues, const std::shared_ptr<AbstractBoundary>& boundary, const double boundaryLength, const double C, const double interval = 100) {
        using namespace matplot;

        // Calculate the area of the boundary
        const double area = boundary->getArea();

        // Prepare vectors for energy values and counts
        std::vector<double> E_values;
        std::vector<double> numerical_counts;
        std::vector<double> oscillatory_part;

        // Convert wavenumber k to energy E = k^2
        std::vector<std::tuple<double, int>> numerical_eigenvalues_with_energy;
        for (const auto& [k, degeneracy] : numerical_eigenvalues) {
            double E = k * k;
            numerical_eigenvalues_with_energy.emplace_back(E, degeneracy);
        }

        // Sort numerical eigenvalues by E
        std::ranges::sort(numerical_eigenvalues_with_energy,
                          [](const auto& a, const auto& b) {
                              return std::get<0>(a) < std::get<0>(b);
                          });

        // Count numerical eigenvalues and calculate oscillatory part
        for (const auto& [E, degeneracy] : numerical_eigenvalues_with_energy) {
            E_values.push_back(E);

            // Count numerical eigenvalues up to E
            numerical_counts.push_back(countEigenvaluesWithDegeneracy(numerical_eigenvalues_with_energy, E));

            // Calculate smooth part using Weyl's law with the previously calculated corner correction
            const double smooth_part = (area * E) / (4 * M_PI) - (boundaryLength * std::sqrt(E)) / (4 * M_PI) + C;
            oscillatory_part.push_back(numerical_counts.back() - smooth_part);
        }

        // Plot the oscillatory part as a scatter plot with semi-transparency
        hold(ax, true);
        const auto osc_plot = ax->scatter(E_values, oscillatory_part);
        osc_plot->display_name("Oscillatory Part of Weyl's Law");
        osc_plot->marker_style(matplot::line_spec::marker_style::circle);
        osc_plot->marker_face_alpha(0.3);  // Semi-transparent

        // Add a red line at y = 0 for reference
        const std::vector<double> zero_line_x = {E_values.front(), E_values.back()};
        const std::vector<double> zero_line_y = {0, 0};
        const auto zero_line = ax->plot(zero_line_x, zero_line_y);
        zero_line->color("red");
        zero_line->line_width(1.5);
        zero_line->display_name("y = 0");

        // Calculate the average oscillation over the specified interval and plot as a step function
        std::vector<double> step_x;
        std::vector<double> step_y;

        for (double bin_start = 0; bin_start < E_values.back(); bin_start += interval) { // NOLINT(*-flp30-c)
            const double bin_end = bin_start + interval;
            double bin_sum = 0.0;
            int bin_count = 0;

            for (size_t i = 0; i < E_values.size(); ++i) {
                if (E_values[i] >= bin_start && E_values[i] < bin_end) {
                    bin_sum += oscillatory_part[i];
                    ++bin_count;
                }
            }

            if (bin_count > 0) {
                double bin_average = bin_sum / bin_count;
                step_x.push_back(bin_start);
                step_y.push_back(bin_average);
                step_x.push_back(bin_end);
                step_y.push_back(bin_average);
            }
        }

        // Plot the step function
        const auto step_plot = ax->plot(step_x, step_y);
        step_plot->color("black");
        step_plot->line_width(1.5);
        step_plot->display_name("Average Oscillation over Interval");

        // Set plot labels and title
        ax->xlabel("E");
        ax->ylabel("N(num) - N(smooth - Weyl's Law)");
        ax->title("Oscillatory Part of Eigenvalue Counts (Weyl's Law)");

        matplot::legend(ax, false);
        return osc_plot;
    }

    /**
     * @brief Plots the spectral rigidity Δ3(L) for a given set of numerical eigenvalues.
     *
     * This method calculates the spectral rigidity Δ3(L) for a range of values L, comparing the numerical results
     * with the theoretical predictions for Poisson, GOE, and GUE ensembles. The method uses the unfolded energy eigenvalues
     * and accounts for eigenvalue degeneracies.
     *
     * @param ax The axis handle from matplot++ for plotting.
     * @param numerical_k_eigenvalues A vector of tuple containing (k, degeneracy) for the numerical eigenvalues.
     * @param boundary The boundary of the billiard.
     * @param modifiedPerimeter The modified perimeter to account for different boundary conditions.
     * @param C The curvature and corner correction to Weyl's formula.
     * @param L_min The minimum value of L for the spectral rigidity calculation.
     * @param L_max The maximum value of L for the spectral rigidity calculation.
     * @param num_L The number of points for L in the spectral rigidity calculation.
     * @return A line_handle for the plotted spectral rigidity.
     */
    inline matplot::line_handle plotSpectralRigidityDelta(const matplot::axes_handle &ax, const std::vector<std::tuple<double, int>>& numerical_k_eigenvalues, const std::shared_ptr<AbstractBoundary>& boundary, const double modifiedPerimeter, const double C, const double L_min = 0.1, const double L_max = 5, const int num_L = 100)
    {
        // Unfold the numerical eigenvalues using the previously defined method
        const std::vector<double> unfolded_energies = unfoldEnergyEigenvalues(numerical_k_eigenvalues, boundary, modifiedPerimeter, C);

        // Helper function to count eigenvalues with degeneracy
        auto countEigenvaluesWithDegeneracy = [](const std::vector<std::tuple<double, int>>& eigenvaluesWithDegeneracies, const double limitTo) {
            int eigenvalueCount = 0;
            for (const auto& [k, degeneracy] : eigenvaluesWithDegeneracies) {
                if (k * k > limitTo) {
                    break;
                }
                eigenvalueCount += degeneracy;
            }
            return eigenvalueCount;
        };

        // Helper function to compute deviation from linear fit using Eigen
        auto computeDeviation = [](const std::vector<double>& x, const std::vector<double>& N) {
            const size_t n = x.size();
            Eigen::MatrixXd A(n, 2); // Matrix A with n rows and 2 columns
            Eigen::VectorXd b(n);    // Vector b with n rows

            // Fill matrix A and vector b
            for (size_t i = 0; i < n; ++i) {
                A(i, 0) = x[i];   // First column: x_i values NOLINT(*-narrowing-conversions)
                A(i, 1) = 1.0;    // Second column: constant term 1 NOLINT(*-narrowing-conversions)
                b(i) = N[i];      // Vector b: N_i values NOLINT(*-narrowing-conversions)
            }

            // Solve for coefficients using pivot alogorithm instead of SVD
            Eigen::VectorXd coeff = A.colPivHouseholderQr().solve(b);
            const double a = coeff[0]; // Slope of the line
            const double b0 = coeff[1]; // Intercept of the line

            // Compute the sum of squared deviations
            double sum_sq_dev = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double fit = a * x[i] + b0; // Linear fit value
                const double dev = N[i] - fit;    // Deviation from the fit
                sum_sq_dev += dev * dev;    // Squared deviation
            }

            return sum_sq_dev; // Return the total sum of squared deviations
        };

        // Helper functions for theoretical spectral rigidities
        auto poissonDeltaL = [](const double L) {
            return L / 15.0;
        };

        auto goeDeltaL = [](const double L) {
            return (1 / (M_PI * M_PI)) * (std::log(2 * M_PI * L) + std::numbers::egamma - 5.0 / 4.0 - (M_PI * M_PI / 8));
        };

        auto gueDeltaL = [](const double L) {
            return (1 / (2 * M_PI * M_PI)) * (std::log(2 * M_PI * L) + std::numbers::egamma - 5.0 / 4.0);
        };

        // Calculate the spectral rigidity Δ3(L)
        std::vector<double> L_values;
        std::vector<double> delta_L;

        for (double L = L_min; L <= L_max; L += (L_max - L_min) / num_L) { // NOLINT(*-flp30-c)
            double sum_deviation = 0;
            int count = 0;

            // Iterate over all starting points
            for (const double alpha : unfolded_energies) {
                if (alpha + L > unfolded_energies.back()) break;

                std::vector<double> x;
                std::vector<double> N;

                // Collect points in the interval [alpha, alpha + L]
                for (double xi = alpha; xi <= alpha + L; ++xi) { // NOLINT(*-flp30-c)
                    x.push_back(xi);
                    N.push_back(countEigenvaluesWithDegeneracy(numerical_k_eigenvalues, xi));
                }

                // Compute the deviation from the linear fit
                const double deviation = computeDeviation(x, N);
                sum_deviation += deviation * deviation;
                count++;
            }

            // Average over all intervals
            const double delta = sum_deviation / count;
            L_values.push_back(L);
            delta_L.push_back(delta / L);
        }

        // Plot the spectral rigidity Δ3(L) as a scatter plot
        matplot::hold(ax, true);
        auto delta_plot = ax->scatter(L_values, delta_L);
        delta_plot->display_name("Δ3(L)");
        delta_plot->marker_face_color("blue");

        // Plot the theoretical predictions
        std::vector<double> theoretical_L(num_L);
        std::vector<double> poisson_delta_L(num_L);
        std::vector<double> goe_delta_L(num_L);
        std::vector<double> gue_delta_L(num_L);
        for (int i = 0; i < num_L; ++i) {
            const double L = L_min + i * (L_max - L_min) / num_L;
            theoretical_L[i] = L;
            poisson_delta_L[i] = poissonDeltaL(L);
            goe_delta_L[i] = goeDeltaL(L);
            gue_delta_L[i] = gueDeltaL(L);
        }

        // Plot Poisson Δ3(L) = L/15
        const auto poisson_plot = ax->plot(theoretical_L, poisson_delta_L, "--");
        poisson_plot->display_name("Theoretical Δ3(L) = L/15");
        poisson_plot->line_width(2.0);
        poisson_plot->color("r");

        // Plot GOE Δ3(L)
        const auto goe_plot = ax->plot(theoretical_L, goe_delta_L, "--");
        goe_plot->display_name("GOE Δ3(L)");
        goe_plot->line_width(2.0);
        goe_plot->color("g");

        // Plot GUE Δ3(L)
        const auto gue_plot = ax->plot(theoretical_L, gue_delta_L, "--");
        gue_plot->display_name("GUE Δ3(L)");
        gue_plot->line_width(2.0);
        gue_plot->color("b");

        // Set plot labels and title
        ax->xlabel("L");
        ax->ylabel("Δ3(L)");
        ax->title("Spectral Rigidity Δ3(L)");

        // Set y-axis limit
        ax->ylim({0, 1.25 * *std::ranges::max_element(delta_L)});

        matplot::legend(ax, true);
        return delta_plot;
    }

    /**
     * @brief Plots the empirical cumulative distribution function (CDF) of the nearest neighbor spacings
     *        from the unfolded energy eigenvalues along with the theoretical CDFs for Poisson, GOE, and GUE.
     *
     * @param ax The axis handle from matplot++ for plotting.
     * @param unfolded_energy_eigenvalues A vector of unfolded energy eigenvalues.
     * @param plotBerryRobnik Whether to plot the Berry-Robnik cumulative distribution (default: false).
     * @param rho The mixing parameter for the Berry-Robnik distribution (default: 0.5).
     * @return A tuple containing the line handles for the empirical CDF and the theoretical CDFs (Poisson, GOE, GUE).
     */
    inline std::tuple<matplot::line_handle, matplot::line_handle, matplot::line_handle, matplot::line_handle> plotCumulativeSpacingDistribution(
        const matplot::axes_handle &ax,
        const std::vector<double> &unfolded_energy_eigenvalues, const bool plotBerryRobnik = false, const double rho = 0.5)
    {
        using namespace matplot;

        // Helper function to compute nearest neighbor spacings
        auto computeNearestNeighborSpacings = [](const std::vector<double> &eigenvalues) {
            std::vector<double> spacings;
            for (size_t i = 1; i < eigenvalues.size(); ++i) {
                spacings.push_back(eigenvalues[i] - eigenvalues[i - 1]);
            }
            return spacings;
        };

        // Step 1: Compute nearest neighbor spacings
        std::vector<double> spacings = computeNearestNeighborSpacings(unfolded_energy_eigenvalues);

        // Step 2: Sort the spacings
        std::ranges::sort(spacings);

        // Step 3: Compute the empirical CDF
        std::vector<double> empirical_cdf(spacings.size());
        for (size_t i = 0; i < spacings.size(); ++i) {
            empirical_cdf[i] = static_cast<double>(i + 1) / spacings.size(); // NOLINT(*-narrowing-conversions)
        }

        // Helper functions for theoretical CDFs
        auto poissonCDF = [](const double s) {
            return 1 - std::exp(-s);
        };

        auto goeCDF = [](const double s) {
            return 1 - std::exp(-M_PI * s * s / 4);
        };

        auto gueCDF = [](const double s) {
            return 1 - std::exp(-4 * s * s / M_PI) * (1 + 4 * s * s / M_PI);
        };

        // Step 4: Compute the theoretical CDFs
        constexpr size_t num_points = 1000;
        std::vector<double> s_values(num_points);
        std::vector<double> poisson_cdf(num_points);
        std::vector<double> goe_cdf(num_points);
        std::vector<double> gue_cdf(num_points);
        std::vector<double> berry_robnik_cdf;

        const double max_s = *std::ranges::max_element(spacings);

        for (size_t i = 0; i < num_points; ++i) {
            const double s = max_s * static_cast<double>(i) / (num_points - 1);
            s_values[i] = s;
            poisson_cdf[i] = poissonCDF(s);
            goe_cdf[i] = goeCDF(s);
            gue_cdf[i] = gueCDF(s);
            if (plotBerryRobnik) {
                berry_robnik_cdf.push_back(BerryRobnik::cumulativeBRDistribution(s, rho));
            }
        }

        // Step 5: Plot the empirical CDF
        hold(ax, true);
        auto empirical_plot = ax->scatter(spacings, empirical_cdf);
        empirical_plot->display_name("Empirical CDF");
        empirical_plot->line_width(1.0);

        // Step 6: Plot the theoretical CDFs
        auto poisson_plot = ax->plot(s_values, poisson_cdf);
        poisson_plot->display_name("Poisson CDF");
        poisson_plot->line_width(1.0);

        auto goe_plot = ax->plot(s_values, goe_cdf);
        goe_plot->display_name("GOE CDF");
        goe_plot->line_width(1.0);

        auto gue_plot = ax->plot(s_values, gue_cdf);
        gue_plot->display_name("GUE CDF");
        gue_plot->line_width(1.0);

        if (plotBerryRobnik) {
            const auto berry_robnik_plot = ax->plot(s_values, berry_robnik_cdf);
            berry_robnik_plot->display_name("Berry-Robnik CDF");
            berry_robnik_plot->line_width(1.0);
        }

        // Set plot labels and title
        ax->xlabel("Spacing (s)");
        ax->ylabel("Cumulative Probability");
        ax->title("Cumulative Distribution of Nearest Neighbor Spacings");
        const auto leg = matplot::legend(ax, true);
        leg->location(matplot::legend::general_alignment::bottomright);

        // Return handles for the plots
        return std::make_tuple(empirical_plot, poisson_plot, goe_plot, gue_plot);
    }

    /**
     * @brief Plots the number variance Σ²(L) for a given set of numerical eigenvalues.
     *
     * This method calculates the number variance Σ²(L) for a range of values L, comparing the numerical results
     * with the theoretical predictions for Poisson, GOE, and GUE. The method uses the unfolded energy eigenvalues
     * and accounts for eigenvalue degeneracies.
     *
     * @param ax The axis handle from matplot++ for plotting.
     * @param numerical_k_eigenvalues A vector of tuples containing (k, degeneracy) for the numerical eigenvalues.
     * @param boundary The boundary of the billiard.
     * @param modifiedPerimeter The modified perimeter to account for different boundary conditions.
     * @param C The curvature and corner correction to Weyl's formula.
     * @param L_min The minimum value of L for the number variance calculation.
     * @param L_max The maximum value of L for the number variance calculation.
     * @param num_L The number of points for L in the number variance calculation.
     * @return A tuple containing the plot handles for the number variance and theoretical predictions.
     */
    inline std::tuple<matplot::line_handle, matplot::line_handle, matplot::line_handle, matplot::line_handle> plotNumberVarianceSigma(
        const matplot::axes_handle &ax,
        const std::vector<std::tuple<double, int>>& numerical_k_eigenvalues,
        const std::shared_ptr<AbstractBoundary>& boundary,
        const double modifiedPerimeter,
        const double C,
        const double L_min = 0.1,
        const double L_max = 5,
        const int num_L = 50)
        {
    // Unfold the numerical eigenvalues using the previously defined method
    const auto unfolded_energies_with_degeneracy = unfoldEnergyEigenvaluesWithDegeneracy(
        numerical_k_eigenvalues, boundary, modifiedPerimeter, C);

    // Extract the unfolded energies without degeneracy
    std::vector<double> unfolded_energies;
    std::set<double> unique_energies;
    for (const auto& [energy, degeneracy] : unfolded_energies_with_degeneracy) {
        if (!unique_energies.contains(energy)) {
            unfolded_energies.push_back(energy);
            unique_energies.insert(energy);
        }
    }

    // Sort the energies (just in case)
    std::ranges::sort(unfolded_energies);

    // Helper functions for theoretical number variances
    auto poissonSigmaL = [](const double L) {
        return L;
    };

    auto goeSigmaL = [](const double L) {
        return (2.0 / (M_PI * M_PI)) * (std::log(2.0 * M_PI * L) + std::numbers::egamma + 1.0 - (M_PI * M_PI / 8.0));
    };

    auto gueSigmaL = [](const double L) {
        return (1.0 / (M_PI * M_PI)) * (std::log(2.0 * M_PI * L) + std::numbers::egamma + 1.0);
    };

    // Initialize vectors to store L values and corresponding number variances
    std::vector<double> L_values(num_L);
    std::vector<double> sigma_L(num_L);

    // Calculate number variance for each L using multiple threads
    std::vector<std::future<void>> futures;
    futures.reserve(num_L);
    for (int i = 0; i < num_L; ++i) {
        futures.push_back(std::async(std::launch::async, [&unfolded_energies, &L_values, &sigma_L, i, L_min, L_max, num_L] {
            const double L = L_min + i * (L_max - L_min) / (num_L - 1);  // Ensure proper L spacing
            L_values[i] = L;
            double sum = 0.0;
            int count = 0;

            auto start_it = unfolded_energies.begin();
            auto end_it = unfolded_energies.begin();

            // Iterate over alpha in unfolded_energies
            for (const double alpha : unfolded_energies) {
                while (end_it != unfolded_energies.end() && *end_it <= alpha + L) {
                    ++end_it;
                }
                while (start_it != unfolded_energies.end() && *start_it < alpha) {
                    ++start_it;
                }

                const double n_L = std::distance(start_it, end_it); // NOLINT(*-narrowing-conversions)
                const double deviation = n_L - L;
                sum += deviation * deviation;
                ++count;

                if (end_it == unfolded_energies.end()) break;  // Handle edge case properly
            }

            // Average over all intervals
            sigma_L[i] = count > 0 ? sum / count : 0.0;
        }));
    }

    // Wait for all futures to complete
    for (auto& future : futures) {
        future.get();
    }

    // Plot the number variance Σ²(L)
    matplot::hold(ax, matplot::on);
    auto sigma_plot = ax->scatter(L_values, sigma_L);
    sigma_plot->display_name("Σ²(L)");
    sigma_plot->line_width(0.5);

    // Plot the theoretical predictions
    const std::vector<double> theoretical_L(num_L);
    std::vector<double> poisson_sigma_L(num_L);
    std::vector<double> goe_sigma_L(num_L);
    std::vector<double> gue_sigma_L(num_L);

    for (int i = 0; i < num_L; ++i) {
        const double L = L_min + i * (L_max - L_min) / (num_L - 1);  // Ensure consistent spacing        theoretical_L[i] = L;
        poisson_sigma_L[i] = poissonSigmaL(L);
        goe_sigma_L[i] = goeSigmaL(L);
        gue_sigma_L[i] = gueSigmaL(L);
    }

    // Plot Poisson Σ²(L)
    const auto poisson_plot = ax->plot(L_values, poisson_sigma_L, "--");
    poisson_plot->display_name("Poisson Σ²(L) = L");
    poisson_plot->line_width(1.0);
    poisson_plot->color("r");

    // Plot GOE Σ²(L)
    const auto goe_plot = ax->plot(L_values, goe_sigma_L, "--");
    goe_plot->display_name("GOE Σ²(L)");
    goe_plot->line_width(1.0);
    goe_plot->color("g");

    // Plot GUE Σ²(L)
    const auto gue_plot = ax->plot(L_values, gue_sigma_L, "--");
    gue_plot->display_name("GUE Σ²(L)");
    gue_plot->line_width(1.0);
    gue_plot->color("b");

    // Set plot labels and title
    ax->xlabel("L");
    ax->ylabel("Σ²(L)");
    ax->title("Number Variance Σ²(L)");

    // Set y-axis limit
    ax->ylim({0.0, 1.25 * *std::ranges::max_element(sigma_L)});
    matplot::legend(ax, true);

    // Return handles to the plotted number variance and theoretical predictions
    return std::make_tuple(sigma_plot, poisson_plot, goe_plot, gue_plot);
    }

    /**
     * @brief Unfolds the energy eigenvalues and plots various analyses including level spacing distribution,
     *        oscillatory part of the number of states, cumulative distribution function, spectral rigidity, and number variance.
     *
    * @param numerical_results The result from calculateDegeneracies method.
     * @param boundary The boundary of the geometry.
     * @param ax_hist The upper axis handle for the level spacing distribution histogram.
     * @param ax_pdf The lower axis handle for the level spacing probability density.
     * @param ax_oscillatory The axis handle for the oscillatory part of the number of states.
     * @param ax_cdf The axis handle for the cumulative distribution function.
     * @param ax_rigidity The axis handle for the spectral rigidity plot.
     * @param ax_number_variance The axis handle for the number variance plot.
     * @param C The modified length of the boundary to take into accont the Neumann/Dirichlet bc
     * @param L_min_spectral_rigidity Minimum L of spectral rigidity
     * @param L_max_spectral_rigidity Maximum L of spectral rigidity
     * @param L_min_number_variance Minimum L of number variance
     * @param L_max_number_variance Maximum L of number variance
     *
     * @return A tuple containing:
     * - A tuple with handles for the histogram of the level spacing distribution and the corresponding empirical and theoretical PDFs.
     * - A handle for the plot of the oscillatory part of the number of states.
     * - A handle for the cumulative spacing distribution plot.
     * - A handle for the spectral rigidity plot.
     * - A handle for the number variance plot.
     */
    inline std::tuple<std::tuple<std::shared_ptr<matplot::histogram>, std::shared_ptr<class matplot::line>,
    std::shared_ptr<class matplot::line>,
    std::shared_ptr<class matplot::line>,
    std::shared_ptr<class matplot::line>>,
    std::shared_ptr<class matplot::line>,
    std::tuple<std::shared_ptr<class matplot::line>,
    std::shared_ptr<class matplot::line>,
    std::shared_ptr<class matplot::line>,
    std::shared_ptr<class matplot::line>>,
    std::shared_ptr<class matplot::line>,
    std::tuple<std::shared_ptr<class matplot::line>,
    std::shared_ptr<class matplot::line>,
    std::shared_ptr<class matplot::line>,
    std::shared_ptr<class matplot::line>>> plotUnfoldedEnergiesAndAnalyses(
        const std::vector<std::tuple<double, int>> &
        numerical_results,
        const std::shared_ptr<AbstractBoundary> &boundary,
        const matplot::axes_handle &ax_hist,
        const matplot::axes_handle &ax_pdf,
        const matplot::axes_handle &ax_oscillatory,
        const matplot::axes_handle &ax_cdf,
        const matplot::axes_handle &ax_rigidity,
        const matplot::axes_handle &ax_number_variance, const double C, const double L_min_spectral_rigidity = 0.5,
        const double L_max_spectral_rigidity = 300.0, const double L_min_number_variance = 0.1,
        const double L_max_number_variance = 5.0) {

        using namespace matplot;
        // The level spacing distribution
        const auto unfolded_energies = AnalysisTools::unfoldEnergyEigenvalues(numerical_results, boundary, boundary->calculateArcLength(), C);
        std::cout << "The mean level spacing of unfolded energy eigenvalues is: " << calculateMeanLevelSpacing(unfolded_energies) << std::endl;
        auto pdf_tuple = plotSpacingDistribution(ax_hist, ax_pdf, unfolded_energies, 1.0);

        // The oscillating part of the number of energy eigenvalues
        auto u = plotOscillatoryPartWeyl(ax_oscillatory, numerical_results, boundary, boundary->calculateArcLength(), C);
        auto u2 = plotCumulativeSpacingDistribution(ax_cdf, unfolded_energies);

        auto u3 = plotSpectralRigidityDelta(ax_rigidity, numerical_results, boundary, boundary->calculateArcLength(), C, L_min_spectral_rigidity, L_max_spectral_rigidity, 50);
        auto u4 = plotNumberVarianceSigma(ax_number_variance, numerical_results, boundary, boundary->calculateArcLength(), C, L_min_number_variance, L_max_number_variance);
        return std::make_tuple(pdf_tuple, u, u2, u3, u4);
    }

    /**
     * @brief Plots the smallest singular values as a function of k. This is the more rudamentary method and is disfavoured with respect to the plotSVDResultsMerged method and that one should be used
     *
     * This function plots the smallest singular values from the SVD results against their corresponding k values.
     *
     * @param ax The matplot axes handle to plot into.
     * @param mergedResults The pair containing k values and corresponding SVD results.
     * @param a The lower bound of the y-axis.
     * @param b The upper bound of the y-axis.
     */
    inline void plotSmallestSingularValues(const matplot::axes_handle& ax, const std::pair<std::vector<double>, std::vector<Eigen::VectorXd>>& mergedResults, const double a, const double b) {
        using namespace matplot;

        const auto& [k_values, svd_values] = mergedResults;
        std::vector<double> singular_values;
        singular_values.reserve(svd_values.size());
        for (const auto& svd : svd_values) {
            singular_values.push_back(svd(0)); // First singular value
        }

        const auto plt = ax->plot(k_values, singular_values, "b");
        plt->display_name("Smallest Singular Value");
        plt->line_width(1.5);
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
        matplot::legend(ax, on);
    }

    /**
     * @brief Plots specified singular values as functions of k. This is the more rudamentary method and is disfavoured with respect to the plotSVDResultsMerged method and that one should be used
     *
     * This function plots the singular values from the SVD results against their corresponding k values.
     * The index specifies which singular value to plot (1 for smallest, 2 for second smallest, etc.).
     *
     * @param ax The matplot axes handle to plot into.
     * @param mergedResults The pair containing k values and corresponding SVD results.
     * @param index The index of the singular value to plot (1 for smallest, 2 for second smallest, etc.).
     * @param a The lower bound of the y-axis.
     * @param b The upper bound of the y-axis.
     */
    inline void plotSingularValues(const matplot::axes_handle& ax, const std::pair<std::vector<double>, std::vector<Eigen::VectorXd>>& mergedResults, const int index, const double a, const double b) {
        using namespace matplot;

        const auto& [k_values, svd_values] = mergedResults;
        std::vector<double> singular_values;
        singular_values.reserve(svd_values.size());
        for (const auto& svd : svd_values) {
            if (index - 1 < svd.size()) {
                singular_values.push_back(svd(index - 1));
            }
        }

        const auto plt = ax->plot(k_values, singular_values);
        plt->display_name("Singular Value #" + std::to_string(index));
        plt->line_width(1.5);
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
        matplot::legend(ax, on);
    }

    /**
     * @brief Plots the SVD results including the smallest singular values and their higher order counterparts. This is the preffered method for plotting SVD results of merged data for up to 4th order degeneracy
     *
     * This method plots the smallest singular values and their higher order counterparts on the provided axes.
     * It also overlays the analytical solutions as purple diamonds, and adds text annotations for the (m,n) pairs
     * corresponding to the analytical eigenvalues.
     *
     * @param ax The matplot axes handle for the main plot.
     * @param ax2 The matplot axes handle for the zoomed-in plot.
     * @param mergedResults The merged SVD results from SequentialKRangeSolver.
     * @param analytical_eigenvalues The analytical eigenvalues for comparison.
     * @param a The lower bound for the y-axis in the main plot.
     * @param b The upper bound for the y-axis in the main plot.
     * @param a2 The lower bound for the y-axis in the zoomed-in plot.
     * @param b2 The upper bound for the y-axis in the zoomed-in plot.
     */
    inline void plotSVDResultsMerge(const matplot::axes_handle& ax, const matplot::axes_handle& ax2,
                               const std::pair<std::vector<double>, std::vector<Eigen::VectorXd>>& mergedResults,
                               const double a, const double b, const double a2, const double b2,
                               const std::optional<std::vector<std::tuple<double, int, int>>>& analytical_eigenvalues = std::nullopt) {

        // Plotting the smallest singular values
        AnalysisTools::plotSmallestSingularValues(ax, mergedResults, a, b);
        AnalysisTools::plotSmallestSingularValues(ax2, mergedResults, a2, b2);

        // Plotting the second smallest singular values
        AnalysisTools::plotSingularValues(ax, mergedResults, 2, a, b);
        AnalysisTools::plotSingularValues(ax2, mergedResults, 2, a2, b2);

        // Plotting the third smallest singular values
        AnalysisTools::plotSingularValues(ax, mergedResults, 3, a, b);
        AnalysisTools::plotSingularValues(ax2, mergedResults, 3, a2, b2);

        // Plotting the fourth smallest singular values
        AnalysisTools::plotSingularValues(ax, mergedResults, 4, a, b);
        AnalysisTools::plotSingularValues(ax2, mergedResults, 4, a2, b2);

        if (analytical_eigenvalues) {
            // Plot analytical solutions as purple diamonds
            std::vector<double> analytical_k_values;
            for (const auto& [k_mn, m, n] : *analytical_eigenvalues) {
                analytical_k_values.push_back(k_mn); // NOLINT(*-inefficient-vector-operation)
            }
            const std::vector<double> analytical_x(analytical_k_values.size(), 0);
            const auto l = ax->scatter(analytical_k_values, analytical_x);
            l->display_name("Analytical");
            l->marker_style(matplot::line_spec::marker_style::diamond);
            l->marker_size(15);
            l->marker_color("purple");

            const auto l2 = ax2->scatter(analytical_k_values, analytical_x);
            l2->display_name("Analytical");
            l2->marker_style(matplot::line_spec::marker_style::diamond);
            l2->marker_size(15);
            l2->marker_color("purple");

            // Add labels for (m,n) pairs without adding to the legend
            std::map<double, int> label_count;
            for (const auto& [k_mn, m, n] : *analytical_eigenvalues) {
                constexpr double offset = 0.001;
                constexpr double offset2 = 0.1;
                const int count = label_count[k_mn];
                const double offset_y = -0.001 - count * offset; // Adjust text position to avoid overlap
                const double offset_y2 = -0.1 - count * offset2; // Adjust text position to avoid overlap

                // Adjust text position for bottom plot (ax2)
                const auto txt2 = ax2->text(k_mn, offset_y, "(" + std::to_string(m) + "," + std::to_string(n) + ")");
                txt2->font_size(10);
                txt2->color("black");

                // Adjust text position for bottom plot (ax2)
                const auto txt = ax->text(k_mn, offset_y2, "(" + std::to_string(m) + "," + std::to_string(n) + ")");
                txt->font_size(10);
                txt->color("black");

                // Update label count for next iteration
                label_count[k_mn]++;
            }
        }

        matplot::legend(ax, false);
        matplot::legend(ax2, false);
    }
}

#endif //QUANTUMANALYSISTOOLS_HPP
