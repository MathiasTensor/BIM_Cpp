#ifndef TEMPIDEABIMWITHSYMMETRIES_HPP
#define TEMPIDEABIMWITHSYMMETRIES_HPP

#pragma once
#include <cmath>
#include <complex>
#include <gsl/gsl_sf_bessel.h>
#include "Boundary.hpp"

/**
 * @file KernelIntegrationStrategy.hpp
 * @brief Header for constructing the integration kernels of the Fredholm matrix of the billiard with included desymmetrization options.
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
 * @namespace KernelIntegrationStrategies
 * @brief Provides strategies for computing Green's functions with various boundary conditions and symmetries.
 *
 * This namespace includes classes for handling different types of boundary conditions (Dirichlet, Neumann)
 * and symmetries (X, Y, XY reflections, and C3 rotational symmetry) for the boundary integral method.
 *
 * Classes:
 * - IKernelStrategy: Abstract base class for kernel computation strategies.
 * - DefaultKernelIntegrationStrategy: Default integration strategy without symmetry considerations
 * - XReflectionSymmetryDStandard: Dirichlet boundary condition with X reflection symmetry (standard).
 * - XReflectionSymmetryDBeta: Dirichlet boundary condition with X reflection symmetry (beta).
 * - XReflectionSymmetryNStandard: Neumann boundary condition with X reflection symmetry (standard).
 * - XReflectionSymmetryNBeta: Neumann boundary condition with X reflection symmetry (beta).
 * - YReflectionSymmetryDStandard: Dirichlet boundary condition with Y reflection symmetry (standard).
 * - YReflectionSymmetryDBeta: Dirichlet boundary condition with Y reflection symmetry (beta).
 * - YReflectionSymmetryNStandard: Neumann boundary condition with Y reflection symmetry (standard).
 * - YReflectionSymmetryNBeta: Neumann boundary condition with Y reflection symmetry (beta).
 * - XYReflectionSymmetryNNStandard: Neumann-Neumann boundary conditions with XY reflection symmetry (standard).
 * - XYReflectionSymmetryNNBeta: Neumann-Neumann boundary conditions with XY reflection symmetry (beta).
 * - XYReflectionSymmetryStandardDD: Dirichlet-Dirichlet boundary conditions with XY reflection symmetry (standard).
 * - XYReflectionSymmetryBetaDD: Dirichlet-Dirichlet boundary conditions with XY reflection symmetry (beta).
 * - XYReflectionSymmetryStandardND: Neumann-Dirichlet boundary conditions with XY reflection symmetry (standard).
 * - XYReflectionSymmetryBetaND: Neumann-Dirichlet boundary conditions with XY reflection symmetry (beta).
 * - XYReflectionSymmetryStandardDN: Dirichlet-Neumann boundary conditions with XY reflection symmetry (standard).
 * - XYReflectionSymmetryBetaDN: Dirichlet-Neumann boundary conditions with XY reflection symmetry (beta).
 * - C3SymmetryDStandard: Dirichlet boundary condition with C3 rotational symmetry (standard).
 * - C3SymmetryDBeta: Dirichlet boundary condition with C3 rotational symmetry (beta).
 * - C3SymmetryNStandard: Neumann boundary condition with C3 rotational symmetry (standard).
 * - C3SymmetryNBeta: Neumann boundary condition with C3 rotational symmetry (beta).
 *
 * To extend the IKernelStrategy class and create your own strategy, inherit from IKernelStrategy and implement the computeKernel method.
 */
namespace KernelIntegrationStrategies {
    using namespace Boundary;
    /**
     * @class IKernelStrategy
     * @brief Abstract base class for kernel computation strategies.
     *
     * This class provides a framework for computing Green's functions with specific boundary conditions and symmetries.
     */
    class IKernelStrategy {
public:
        /**
         * @brief Constructor for IKernelStrategy.
         * @param boundary A shared pointer to an AbstractBoundary object.
         */
        std::shared_ptr<AbstractBoundary> boundary;
        explicit IKernelStrategy(std::shared_ptr<AbstractBoundary> boundary) : boundary(std::move(boundary)) {};
    virtual ~IKernelStrategy() = default;
        /**
         * @brief Computes the kernel for the given points and boundary conditions.
         * @param p1 The first point.
         * @param p2 The second point.
         * @param normal The normal vector at the first point.
         * @param k The wavenumber.
         * @param t The parametrization of point 1
         * @param beta An optional parameter for the beta Helmholtz kernel.
         * @return The computed kernel as a complex number.
         */
    [[nodiscard]] virtual std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, double k, double t, double beta) const = 0;
    [[nodiscard]] virtual std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, double k, double t, double beta, bool useDefaultHelmholtz) const = 0;
    [[nodiscard]] virtual std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, double k, double t, double beta, bool useDefaultHelmholtz) const = 0;
    [[nodiscard]] virtual std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, double k) const = 0;

    /**
     * @brief Computes the cos_phi term for the normal Helmholtz kernel.
     *
     * This method calculates the cosine of the angle between the normal vector at `p1` and the vector from `p1` to `p2`.
     *
     * @param p1 The first point.
     * @param p2 The second point.
     * @param normal The normal vector at the first point.
     * @param t The parametrization of point 1
     * @return The cos_phi term as a double.
     */
    [[nodiscard]] double computeCosPhi(const Point& p1, const Point& p2, const Point& normal, const double t) const {
        const double dx = p1.x - p2.x;
        const double dy = p1.y - p2.y;
        if (const double distance = std::hypot(dx, dy); distance < std::numeric_limits<double>::epsilon()) {
            // Handle singularity when points coincide
            const double curvature = boundary->computeCurvature(t);
            return 1.0 / (2.0 * M_PI) * curvature;
        } else {
            return (normal.x * dx + normal.y * dy) / distance;
        }
    }

    /**
     * @brief Computes the Hankel function of the first kind.
     *
     * This method calculates the Hankel function of the first kind \( H_1^{(1)} \) for a given distance between points `p1` and `p2` with a given wavenumber `k`.
     * The Hankel function is defined as \( H_1^{(1)}(z) = J_1(z) + i Y_1(z) \), where \( J_1(z) \) is the Bessel
     * function of the first kind and \( Y_1(z) \) is the Bessel function of the second kind.
     *
     * @param p1 The first point.
     * @param p2 The second point.
     * @param k The wavenumber.
     * @return The computed Hankel function \( H_1^{(1)}(z) \) as a complex number.
     */
    static std::complex<double> computeHankel(const Point& p1, const Point& p2, const double k) {
        constexpr std::complex<double> i(0, 1);
        const double dx = p1.x - p2.x;
        const double dy = p1.y - p2.y;
        const double distance = std::hypot(dx, dy);
        if (distance == 0.0) {
            return {std::numeric_limits<double>::epsilon(), std::numeric_limits<double>::epsilon()}; // Handle the singularity, needs to be infinite but this will never ariese because it is supresed by the cos_phi term
        }
        const double J1 = gsl_sf_bessel_J1(k * distance);
        const double Y1 = gsl_sf_bessel_Y1(k * distance);
        return J1 + i * Y1;
    }

protected:
        /**
         * @brief Computes the normal Helmholtz kernel.
         *
         * This method calculates the normal Helmholtz kernel between two points `p1` and `p2` with a given wavenumber `k`.
         * It takes into account the normal vector at `p1`. If the distance between `p1` and `p2` is below a certain
         * threshold (machine epsilon), the method handles the singularity by using the curvature of the boundary at `p1`.
         *
         * @param p1 The first point.
         * @param p2 The second point.
         * @param normal The normal vector at the first point.
         * @param k The wavenumber.
         * @param t The parametrization of point 1
         * @return The computed Helmholtz kernel as a complex number.
         */
        [[nodiscard]] std::complex<double> normalHelmholtzKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t) const {
        constexpr std::complex<double> i(0, 1);
            return -i * k / 2.0 * computeCosPhi(p1, p2, normal, t) * computeHankel(p1, p2, k);
        }

        /**
         * @brief Computes the beta Helmholtz kernel.
         *
         * This method calculates the beta Helmholtz kernel between two points `p1` and `p2` with a given wavenumber `k` and parameter `beta`.
         * It takes into account the normal vector at `p1`. If the distance between `p1` and `p2` is below a certain
         * threshold (machine epsilon), the method handles the singularity by using the curvature of the boundary at `p1`.
         * Additionally, the method incorporates an extra term involving the Bessel function of the first kind.
         *
         * @param p1 The first point.
         * @param p2 The second point.
         * @param normal The normal vector at the first point.
         * @param k The wavenumber.
         * @param beta The beta parameter.
         * @return The computed Helmholtz kernel as a complex number.
         */
        // ReSharper disable once CppMemberFunctionMayBeStatic
        [[nodiscard]] std::complex<double> betaHelmholtzKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double beta) const { // NOLINT(*-convert-member-functions-to-static)
            return {0,0}; // NOT IMPLEMENTED; NEEDS REVISION
        }
    };

    /**
     * @brief Default kernel integration strategy without any symmetries.
     */
    class DefaultKernelIntegrationStrategy : public IKernelStrategy {
public:
    using IKernelStrategy::IKernelStrategy;

    /**
     * @brief Compute the kernel function for given points and normal using the default strategy.
     * @param p1 First point on the boundary.
     * @param p2 Second point on the boundary.
     * @param normal Normal vector at the first point.
     * @param k Wave number.
     * @param t The parametrization of point 1
     * @param beta Additional parameter for modified kernels (default is 0).
     * @return Complex value representing the kernel.
     */
    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        if (beta == 0) {
            return normalHelmholtzKernel(p1, p2, normal, k, t);
        } else {
            return betaHelmholtzKernel(p1, p2, normal, k, beta);
        }
    }
    /**
     * @brief Computes the first derivative of the kernel with respect to the wavenumber k.
     *
     * This method calculates the first derivative of the kernel function with respect to the wavenumber k
     * using the analytic expression derived from the Hankel function.
     * If the distance between points is smaller than or equal to std::numeric_limits<double>::epsilon(),
     * the derivative is considered to be 0, because the curvature is independent of k.
     *
     * @param p1 The first point on the boundary.
     * @param p2 The second point on the boundary.
     * @param normal The normal vector at the first point.
     * @param k The wavenumber at which the derivative is computed.
     * @param t The parametrization of point 1
     * @param beta An optional parameter for the beta Helmholtz kernel.
     * @param useDefaultHelmholtz A flag to determine if the default Helmholtz kernel should be used.
     * @return The first derivative of the kernel as a complex number.
     */
    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        if (!useDefaultHelmholtz) {
            return Derivative::compute([this, &p1, &p2, &normal, t, beta](const double kVal) {
                return this->computeKernel(p1, p2, normal, kVal, t, beta).real(); // Differentiate the real part
            }, k) +
            std::complex<double>(0, 1) * Derivative::compute([this, &p1, &p2, &normal, t, beta](const double kVal) {
                return this->computeKernel(p1, p2, normal, kVal, t, beta).imag(); // Differentiate the imaginary part
            }, k);
        } else {
            const double dx = p1.x - p2.x;
            const double dy = p1.y - p2.y;
            const double distance = std::hypot(dx, dy);
            if (distance <= std::numeric_limits<double>::epsilon()) {
                return {0.0, 0.0};
            }
            constexpr std::complex<double> i(0, 1);
            const double cos_phi = computeCosPhi(p1, p2, normal, t);
            const std::complex<double> H1_1 = gsl_sf_bessel_J1(k * distance) + i * gsl_sf_bessel_Y1(k * distance);
            const std::complex<double> H1_2 = gsl_sf_bessel_Jn(2, k * distance) + i * gsl_sf_bessel_Yn(2, k * distance);

            return -i / 2.0 * cos_phi * (2.0 * H1_1 - k * distance * H1_2);
        }
    }
    /**
     * @brief Computes the second derivative of the kernel with respect to the wavenumber k.
     *
     * This method calculates the second derivative of the kernel function with respect to the wavenumber k
     * using the analytic expression derived from the Hankel function.
     * If the distance between points is smaller than or equal to std::numeric_limits<double>::epsilon(),
     * the second derivative is considered to be 0, because the curvature is independent of k.
     *
     * @param p1 The first point on the boundary.
     * @param p2 The second point on the boundary.
     * @param normal The normal vector at the first point.
     * @param t The parametrization of point 1
     * @param k The wavenumber at which the second derivative is computed.
     * @param beta An optional parameter for the beta Helmholtz kernel.
     * @param useDefaultHelmholtz A flag to determine if the default Helmholtz kernel should be used.
     * @return The second derivative of the kernel as a complex number.
     */
    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        if (!useDefaultHelmholtz) {
            return Derivative::compute([this, &p1, &p2, &normal, t,  beta, useDefaultHelmholtz](const double kVal) {
                return this->computeKernelDerivative(p1, p2, normal, kVal, t, beta, useDefaultHelmholtz).real(); // Differentiate the real part
            }, k) +
            std::complex<double>(0, 1) * Derivative::compute([this, &p1, &p2, &normal, t, beta, useDefaultHelmholtz](const double kVal) {
                return this->computeKernelDerivative(p1, p2, normal, kVal, t, beta, useDefaultHelmholtz).imag(); // Differentiate the imaginary part
            }, k);
        } else {
            const double dx = p1.x - p2.x;
            const double dy = p1.y - p2.y;
            const double distance = std::hypot(dx, dy);
            if (distance <= std::numeric_limits<double>::epsilon()) {
                return {0.0, 0.0};
            }
            constexpr std::complex<double> i(0, 1);
            const double cos_phi = computeCosPhi(p1, p2, normal, t);
            const std::complex<double> H1_1 = gsl_sf_bessel_J1(k * distance) + i * gsl_sf_bessel_Y1(k * distance);
            const std::complex<double> H1_2 = gsl_sf_bessel_Jn(2, k * distance) + i * gsl_sf_bessel_Yn(2, k * distance);
            const std::complex<double> H1_3 = gsl_sf_bessel_Jn(3, k * distance) + i * gsl_sf_bessel_Yn(3, k * distance);

            const std::complex<double> term1 = 2.0 * H1_1 / k;
            const std::complex<double> term2 = -5.0 * distance * H1_2;
            const std::complex<double> term3 = k * distance * distance * H1_3;

            return -i / 2.0 * cos_phi * (term1 + term2 + term3);
        }
    }

    /**
     * Computes the Green's function for the free particle in 2D
     * @param p1 First point (x1,y1)
     * @param p2 Second point (x2, y2)
     * @param k The wavenumber in question
     * @return
     */
    [[nodiscard]] std::complex<double> computeGreensFunction(const Point &p1, const Point &p2, const double k) const override {
        constexpr std::complex<double> i(0.0, 1.0);
        const double dx = p1.x - p2.x;
        const double dy = p1.y - p2.y;
        const double distance = std::hypot(dx, dy);
        return (-1.0) * i / 4.0 * (gsl_sf_bessel_J0(k * distance) + i * gsl_sf_bessel_Y0(k * distance));
    };
};

/**
* @class XReflectionSymmetryDStandard
* @brief Dirichlet boundary condition with X reflection symmetry (standard).
*/
class XReflectionSymmetryDStandard final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }
    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }
    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2, k);
        return result;
    }
};
/**
* @class XReflectionSymmetryDBeta
* @brief Dirichlet boundary condition with X reflection symmetry (beta).
*/
class XReflectionSymmetryDBeta final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }
    // For XReflectionSymmetryDBeta
    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2, k);
        return result;
    }
};
/**
* @class XReflectionSymmetryNStandard
* @brief Neumann boundary condition with X reflection symmetry (standard).
*/
class XReflectionSymmetryNStandard final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2, k);
        return result;
    }
};
/**
* @class XReflectionSymmetryNBeta
* @brief Neumann boundary condition with X reflection symmetry (beta).
*/
class XReflectionSymmetryNBeta final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2 = p2.reflectX();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2, k);
        return result;
    }
};
/**
* @class YReflectionSymmetryDStandard
* @brief Dirichlet boundary condition with Y reflection symmetry (standard).
*/
class YReflectionSymmetryDStandard final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2, k);
        return result;
    }
};
/**
* @class YReflectionSymmetryDBeta
* @brief Dirichlet boundary condition with Y reflection symmetry (beta).
*/
class YReflectionSymmetryDBeta final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2, k);
        return result;
    }
};
/**
* @class YReflectionSymmetryNStandard
* @brief Neumann boundary condition with Y reflection symmetry (standard).
*/
class YReflectionSymmetryNStandard final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2, k);
        return result;
    }
};

/**
* @class YReflectionSymmetryNBeta
* @brief Neumann boundary condition with Y reflection symmetry (beta).
*/
class YReflectionSymmetryNBeta final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2 = p2.reflectY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2, k);
        return result;
    }
};

/**
* @class XYReflectionSymmetryNNStandard
* @brief Neumann-Neumann boundary conditions with XY reflection symmetry (standard).
*/
class XYReflectionSymmetryNNStandard final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k,t,  beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2X, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2Y, normal, k, t,  beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2XY, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t,  beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2XY, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2XY, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2X, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2Y, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2XY, k);
        return result;
    }
};
/**
* @class XYReflectionSymmetryNNBeta
* @brief Neumann-Neumann boundary conditions with XY reflection symmetry (beta).
*/
class XYReflectionSymmetryNNBeta final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2X, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2Y, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2XY, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2XY, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }
    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2XY, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2X, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2Y, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2XY, k);
        return result;
    }
};
/**
* @class XYReflectionSymmetryStandardDD
* @brief Dirichlet-Dirichlet boundary conditions with XY reflection symmetry (standard).
*/
class XYReflectionSymmetryStandardDD final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2X, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2Y, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2XY, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2XY, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2XY, normal, k, t,beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2X, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2Y, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2XY, k);
        return result;
    }
};
/**
* @class XYReflectionSymmetryBetaDD
* @brief Dirichlet-Dirichlet boundary conditions with XY reflection symmetry (beta).
*/
class XYReflectionSymmetryBetaDD final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2X, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2Y, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2XY, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2XY, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2XY, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2X, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2Y, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2XY, k);
        return result;
    }
};
/**
* @class XYReflectionSymmetryStandardND
* @brief Neumann-Dirichlet boundary conditions with XY reflection symmetry (standard).
*/
class XYReflectionSymmetryStandardND final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2X, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2Y, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2XY, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2XY, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }
    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2XY, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2X, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2Y, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2XY, k);
        return result;
    }
};
/**
* @class XYReflectionSymmetryBetaND
* @brief Neumann-Dirichlet boundary conditions with XY reflection symmetry (beta).
*/
class XYReflectionSymmetryBetaND final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2X, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2Y, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2XY, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2XY, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2XY, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2X, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2Y, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2XY, k);
        return result;
    }
};
/**
* @class XYReflectionSymmetryStandardDN
* @brief Dirichlet-Neumann boundary conditions with XY reflection symmetry (standard).
*/
class XYReflectionSymmetryStandardDN final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2X, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2Y, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2XY, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t,const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2XY, normal, k, t,  beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2XY, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2X, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2Y, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2XY, k);
        return result;
    }
};
/**
* @class XYReflectionSymmetryBetaDN
* @brief Dirichlet-Neumann boundary conditions with XY reflection symmetry (beta).
*/
class XYReflectionSymmetryBetaDN final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2X, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2Y, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, reflectedP2XY, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, reflectedP2XY, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2X, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2Y, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, reflectedP2XY, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const Point reflectedP2X = p2.reflectX();
        const Point reflectedP2Y = p2.reflectY();
        const Point reflectedP2XY = p2.reflectXY();
        std::complex result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2X, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2Y, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, reflectedP2XY, k);
        return result;
    }
};
/**
* @class C3SymmetryDStandard
* @brief Dirichlet boundary condition with C3 rotational symmetry (standard).
*/
class C3SymmetryDStandard final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    explicit C3SymmetryDStandard(const std::shared_ptr<AbstractBoundary>& boundary)
        : DefaultKernelIntegrationStrategy(boundary) {}

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;  // 120 degrees
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, rotatedP2a, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, rotatedP2b, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, rotatedP2a, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, rotatedP2b, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, rotatedP2a, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, rotatedP2b, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;  // 120 degrees
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, rotatedP2a, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, rotatedP2b, k);
        return result;
    }
};
/**
* @class C3SymmetryDBeta
* @brief Dirichlet boundary condition with C3 rotational symmetry (beta).
*/
class C3SymmetryDBeta final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;  // 120 degrees
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, rotatedP2a, normal, k, t, beta);
        result -= DefaultKernelIntegrationStrategy::computeKernel(p1, rotatedP2b, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, rotatedP2a, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, rotatedP2b, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, rotatedP2a, normal, k, t, beta, useDefaultHelmholtz);
        result -= DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, rotatedP2b, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;  // 120 degrees
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, rotatedP2a, k);
        result -= DefaultKernelIntegrationStrategy::computeGreensFunction(p1, rotatedP2b, k);
        return result;
    }
};
/**
* @class C3SymmetryNStandard
* @brief Neumann boundary condition with C3 rotational symmetry (standard).
*/
class C3SymmetryNStandard final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;  // 120 degrees
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, rotatedP2a, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, rotatedP2b, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, rotatedP2a, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, rotatedP2b, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, rotatedP2a, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, rotatedP2b, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;  // 120 degrees
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, rotatedP2a, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, rotatedP2b, k);
        return result;
    }
};
/**
* @class C3SymmetryNBeta
* @brief Neumann boundary condition with C3 rotational symmetry (beta).
*/
class C3SymmetryNBeta final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;  // 120 degrees
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, rotatedP2a, normal, k, t, beta);
        result += DefaultKernelIntegrationStrategy::computeKernel(p1, rotatedP2b, normal, k, t, beta);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, rotatedP2a, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, rotatedP2b, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex<double> result = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, rotatedP2a, normal, k, t, beta, useDefaultHelmholtz);
        result += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, rotatedP2b, normal, k, t, beta, useDefaultHelmholtz);
        return result;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        constexpr double rotationAngle = 2 * M_PI / 3;  // 120 degrees
        const Point rotatedP2a = p2.rotateByAngle(rotationAngle);
        const Point rotatedP2b = p2.rotateByAngle(-rotationAngle);
        std::complex result = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, rotatedP2a, k);
        result += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, rotatedP2b, k);
        return result;
    }
};
/**
 * @class CXSymmetryNStandard
 * @brief Standard kernel computation with CX rotational symmetry.
 *
 * This class computes the Green's function kernel for CX rotational symmetry with Dirichlet or Neumann boundary conditions.
 * The contributions from the image points are summed to respect the symmetry.
 */
class CXSymmetryNStandard final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    explicit CXSymmetryNStandard(const std::shared_ptr<AbstractBoundary>& boundary, const int C)
        : DefaultKernelIntegrationStrategy(boundary), C(C) {}

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const double rotationAngle = 2 * M_PI / C;
        std::complex<double> sum = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        for (int n = 1; n < C; ++n) {
            const Point rotatedP2 = p2.rotateByAngle(n * rotationAngle);
            sum += DefaultKernelIntegrationStrategy::computeKernel(p1, rotatedP2, normal, k, t, beta);
        }
        return sum;
    }

    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const double rotationAngle = 2 * M_PI / C;
        std::complex<double> sum = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        for (int n = 1; n < C; ++n) {
            const Point rotatedP2 = p2.rotateByAngle(n * rotationAngle);
            sum += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, rotatedP2, normal, k, t, beta, useDefaultHelmholtz);
        }
        return sum;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const double rotationAngle = 2 * M_PI / C;
        std::complex<double> sum = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        for (int n = 1; n < C; ++n) {
            const Point rotatedP2 = p2.rotateByAngle(n * rotationAngle);
            sum += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, rotatedP2, normal, k, t, beta, useDefaultHelmholtz);
        }
        return sum;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const double rotationAngle = 2 * M_PI / C;
        std::complex sum = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        for (int n = 1; n < C; ++n) {
            const Point rotatedP2 = p2.rotateByAngle(n * rotationAngle);
            sum += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, rotatedP2, k);
        }
        return sum;
    }

private:
    int C;  // The rotational symmetry order
};

/**
 * @class CXSymmetryNBeta
 * @brief Beta kernel computation with CX rotational symmetry.
 *
 * This class computes the Green's function kernel for CX rotational symmetry with an additional beta parameter.
 * The contributions from the image points are summed to respect the symmetry.
 */
class CXSymmetryNBeta final : public DefaultKernelIntegrationStrategy {
public:
    using DefaultKernelIntegrationStrategy::DefaultKernelIntegrationStrategy;

    explicit CXSymmetryNBeta(const std::shared_ptr<AbstractBoundary>& boundary, const int C)
        : DefaultKernelIntegrationStrategy(boundary), C(C) {}

    [[nodiscard]] std::complex<double> computeKernel(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta) const override {
        const double rotationAngle = 2 * M_PI / C;
        std::complex<double> sum = DefaultKernelIntegrationStrategy::computeKernel(p1, p2, normal, k, t, beta);
        for (int n = 1; n < C; ++n) {
            const Point rotatedP2 = p2.rotateByAngle(n * rotationAngle);
            sum += DefaultKernelIntegrationStrategy::computeKernel(p1, rotatedP2, normal, k, t, beta);
        }
        return sum;
    }
    [[nodiscard]] std::complex<double> computeKernelDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const double rotationAngle = 2 * M_PI / C;
        std::complex<double> sum = DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        for (int n = 1; n < C; ++n) {
            const Point rotatedP2 = p2.rotateByAngle(n * rotationAngle);
            sum += DefaultKernelIntegrationStrategy::computeKernelDerivative(p1, rotatedP2, normal, k, t, beta, useDefaultHelmholtz);
        }
        return sum;
    }

    [[nodiscard]] std::complex<double> computeKernelSecondDerivative(const Point& p1, const Point& p2, const Point& normal, const double k, const double t, const double beta, const bool useDefaultHelmholtz) const override {
        const double rotationAngle = 2 * M_PI / C;
        std::complex<double> sum = DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, p2, normal, k, t, beta, useDefaultHelmholtz);
        for (int n = 1; n < C; ++n) {
            const Point rotatedP2 = p2.rotateByAngle(n * rotationAngle);
            sum += DefaultKernelIntegrationStrategy::computeKernelSecondDerivative(p1, rotatedP2, normal, k, t, beta, useDefaultHelmholtz);
        }
        return sum;
    }

    [[nodiscard]] std::complex<double> computeGreensFunction(const Point& p1, const Point& p2, const double k) const override {
        const double rotationAngle = 2 * M_PI / C;
        std::complex sum = DefaultKernelIntegrationStrategy::computeGreensFunction(p1, p2, k);
        for (int n = 1; n < C; ++n) {
            const Point rotatedP2 = p2.rotateByAngle(n * rotationAngle);
            sum += DefaultKernelIntegrationStrategy::computeGreensFunction(p1, rotatedP2, k);
        }
        return sum;
    }
private:
    int C;  // The rotational symmetry order
};

}

#endif //TEMPIDEABIMWITHSYMMETRIES_HPP