#ifndef BOUNDARY_HPP
#define BOUNDARY_HPP

#pragma once
#include <vector>
#include <cmath>
#include <functional>
#include <complex>
#include <iostream>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_deriv.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_mode.h>
#include <gsl/gsl_sf_ellint.h>
#include <matplot/matplot.h>
#include <__algorithm/ranges_reverse.h>
#include <__algorithm/ranges_sort.h>
#include <future>

/**
 * @file Boundary.hpp
 * @brief Header for constructing the boundaries of billiards.
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
 * @namespace Boundary
 * @brief Documentation for constructing various boundary shapes using the provided classes.
 *
 * This namespace includes classes and methods designed for creating complex boundary shapes for simulations
 * that require geometric configurations like billiards, optical tables, or any system needing precise spatial boundaries.
 *
 * ## Base Classes:
 * - `AbstractBoundary`: Serves as the foundational class for all boundary shapes. It offers virtual methods that
 *   must be overridden by derived classes to describe specific boundary shapes. These methods include:
 *   - `curveParametrization(double t)`: Defines the curve's parametric form, returning a point on the boundary for a given parameter `t`.
 *   - `calculateNormal(double t)`: Computes the outward normal at a specific point on the boundary, crucial for physical simulations involving reflections or boundary interactions.
 *   - `calculateArcLength()`: Calculates the total arc length of the boundary, useful for parameterizing the curve by arc length.
 *   - `computeCurvature(double t)`: Determines the curvature of the boundary at a given point, which is essential for certain physics simulations and geometric analyses.
 *   - `plot(sciplot::Plot2D& plot, int numPoints)`: Plots the boundary using a specified number of points to visually represent the shape.
 *   - `getBoundingBox(double& minX, double& maxX, double& minY, double& maxY)`: Computes the axis-aligned bounding box of the boundary, which is useful for spatial indexing and quick rejection tests.
 *   - `isInside(const Point& point)`: Determines if a given point lies inside the closed boundary, supporting complex geometric queries and collision detections.
 *   - `speedAt(const double t)`: Computes the speed at which a point moves along the boundary with respect to the parameter `t`, useful for animations or simulations requiring consistent motion along the boundary.
 *   - `calculateArcParameter(const double t)`: Computes the arc length from the start of the boundary to a point described by parameter `t`, allowing for conversion between linear and parametric forms.
 *
 * - `CompositeBoundary`: Allows for the combination of multiple `AbstractBoundary` instances to form complex boundaries.
 *   Segments should be added in a specific order (clockwise or counterclockwise) to maintain the integrity of boundary traversal and inside checks.
 *
 * ## Predefined Shapes:
 * - `LineSegment`
 * - `SemiCircle`
 * - `SemiEllipse`
 * - `QuarterCircle`
 * - `ParabolicBoundary`
 *
 * ## Examples:
 * ### Example 1: Rectangle
 * Constructs a rectangle using four line segments. This example demonstrates how to use `CompositeBoundary` to construct a simple closed shape.
 * ```
 * class Rectangle final : public CompositeBoundary {
 * public:
 *     Rectangle(const Point& topLeft, double width, double height) {
 *         Point topRight(topLeft.x + width, topLeft.y);
 *         Point bottomRight(topLeft.x + width, topLeft.y + height);
 *         Point bottomLeft(topLeft.x, topLeft.y + height);
 *
 *         addSegment(std::make_shared<LineSegment>(topLeft, topRight, true));
 *         addSegment(std::make_shared<LineSegment>(topRight, bottomRight, true));
 *         addSegment(std::make_shared<LineSegment>(bottomRight, bottomLeft, true));
 *         addSegment(std::make_shared<LineSegment>(bottomLeft, topLeft, true));
 *     }
 * };
 * ```
 *
 * ### Example 2: ParabolicBoundary
 * Defines a boundary segment shaped as a parabola, defined by start and end points and a coefficient for the parabola's curvature.
 * ```
 * class ParabolicBoundary final : public ParametricBoundary {
 * public:
 *     ParabolicBoundary(const Point& start, const Point& end, double coefficient)
 *     : ParametricBoundary(
 *         [start, end, coefficient](double t) { return start.x + t * (end.x - start.x); },
 *         [start, end, coefficient](double t) { return start.y + coefficient * (start.x + t * (end.x - start.x) - start.x) * (start.x + t * (end.x - start.x) - start.x); }
 *       ) {}
 * };
 * ```
 *
 * ### Example 3: Square with Parabolic Top
 * Constructs a square where the top edge is replaced by a downward-facing parabolic boundary.
 * ```
 * class SquareWithParabolicTop final : public CompositeBoundary {
 * public:
 *     SquareWithParabolicTop(const Point& bottomLeft, double length, double coefficient) {
 *         Point bottomRight(bottomLeft.x + length, bottomLeft.y);
 *         Point topRight(bottomLeft.x + length, bottomLeft.y + length);
 *         Point topLeft(bottomLeft.x, bottomLeft.y + length);
 *
 *         addSegment(std::make_shared<ParabolicBoundary>(topLeft, topRight, coefficient));
 *         addSegment(std::make_shared<LineSegment>(bottomLeft, topLeft, true));
 *         addSegment(std::make_shared<LineSegment>(bottomRight, bottomLeft, true));
 *         addSegment(std::make_shared<LineSegment>(topRight, bottomRight, true));
 *     }
 * };
 * ```
 *
 * These examples illustrate how to construct boundaries using basic geometric primitives and composite boundaries. There are more examples in the code like the Cardioid, the HalfMushroom, the QuarterStadium etc.
 * Closed boundaries should ideally use `CompositeBoundary`, `ParametricBoundary`, or a custom implementation derived from `AbstractBoundary`.
 */
namespace Boundary {
 /**
 * @brief Represents a two-dimensional point or vector.
 *
 * This class is used to handle two-dimensional coordinates, providing basic arithmetic operations and vector normalization.
 */
struct Point {
    double x, y;
    /**
     * @brief Default constructor initializing point to the origin (0,0).
     */
    Point() : x(0), y(0) {}
    /**
     * @brief Default constructor initializing point to the origin (0,0).
     */
    Point(const double x, const double y) : x(x), y(y) {}
    /**
     * @brief Normalizes the vector represented by the point to unit length.
     * @return Normalized vector.
     * @exception std::runtime_error Thrown if the vector is zero-length.
     */
    [[nodiscard]] Point normalized() const {
        const double norm = std::sqrt(x * x + y * y);
        if (norm == 0) throw std::runtime_error("Attempt to normalize a zero-length vector.");
        return {x / norm, y / norm};
    }
    /**
     * @brief Adds this point to another point.
     * @param rhs The right-hand side point to add.
     * @return Resulting point after addition.
     */
    Point operator+(const Point& rhs) const {
        return {x + rhs.x, y + rhs.y};
    }
    /**
     * @brief Subtracts another point from this point.
     * @param rhs The right-hand side point to subtract.
     * @return Resulting point after subtraction.
     */
    Point operator-(const Point& rhs) const {
        return {x - rhs.x, y - rhs.y};
    }
    /**
     * @brief Multiplies this point by a scalar.
     * @param scalar The scalar to multiply by.
     * @return Scaled point.
     */
    Point operator*(const double scalar) const {
        return {x * scalar, y * scalar};
    }
    /**
     * @brief Divides this point by a scalar.
     * @param scalar The scalar to divide by.
     * @return Scaled point.
     * @exception std::runtime_error Thrown if scalar is zero.
     */
    Point operator/(const double scalar) const {
        if (scalar == 0) throw std::runtime_error("Attempt to divide by zero.");
        return {x / scalar, y / scalar};
    }
    /**
     * Reflects the point across the x-axis.
     * @return A new Point instance that is the reflection of the original point across the x-axis.
     */
    [[nodiscard]] Point reflectX() const {
        return {x, -y};
    }
    /**
     * Reflects the point across the y-axis.
     * @return A new Point instance that is the reflection of the original point across the y-axis.
     */
    [[nodiscard]] Point reflectY() const {
        return {-x, y};
    }
    /**
     * Reflects the point across both the x-axis and y-axis (reflection through the origin).
     * @return A new Point instance that is the reflection of the original point across both the x-axis and y-axis.
     */
    [[nodiscard]] Point reflectXY() const {
        return {-x, -y};
    }
    /**
     * Rotates the point around the origin by a specified angle.
     * @param angle The angle by which to rotate the point, in radians.
     * @return A new Point instance that is the rotation of the original point by the specified angle.
     */
    [[nodiscard]] Point rotateByAngle(const double angle) const {
        const double cos_theta = std::cos(angle);
        const double sin_theta = std::sin(angle);
        return {x * cos_theta - y * sin_theta, x * sin_theta + y * cos_theta};
    }
    /**
     * @brief Calculates the Euclidean distance from this point to another point.
     *
     * This method computes the straight-line distance between this point and a specified point
     * using the Euclidean distance formula.
     *
     * @param other The other point to which the distance is calculated.
     * @return The Euclidean distance between this point and the specified point.
     */
    [[nodiscard]] double distanceTo(const Point& other) const {
        return std::sqrt((x - other.x) * (x - other.x) + (y - other.y) * (y - other.y));
    }
};

/**
 * @brief Provides methods for computing numerical derivatives.
 */
class Derivative {
public:
    /**
     * @brief Computes the derivative of a function at a given point.
     * @param f The function to differentiate.
     * @param x The point at which to compute the derivative.
     * @param h Step size for the finite difference approximation, defaults to the square root of machine epsilon.
     * @param isForward Whether to use forward differencing (true) or backward differencing (false).
     * @return The computed derivative at x.
     * @exception std::runtime_error Thrown if the result is not a finite number.
     */
    template<typename Func>
    static double compute(Func f, const double x, const double h = sqrt(std::numeric_limits<double>::epsilon()), const bool isForward = true) {
        double result;
        double abserror;
        gsl_function F;
        F.function = [](double t, void* params) -> double {
            return (*(static_cast<Func*>(params)))(t);
        };
        F.params = &f;

        if (isForward) {
            gsl_deriv_forward(&F, x, h, &result, &abserror);
        } else {
            gsl_deriv_backward(&F, x, h, &result, &abserror);
        }

        if (std::isnan(result) || std::isinf(result)) {
            throw std::runtime_error("Failed to compute derivative at x=" + std::to_string(x));
        }
        return result;
    }
};

/**
 * @brief Provides methods for numerical integration.
 */
class Integral {
public:
    /**
     * @brief Computes the numerical derivative of a function using GSL's central differencing.
     * @param func The function to differentiate.
     * @param x The point at which to compute the derivative.
     * @param h Step size for the finite difference, defaults to the square root of machine epsilon.
     * @return The derivative of the function at point x.
     */
    static double computeDerivative(const std::function<double(double)>& func, const double x, const double h = std::sqrt(std::numeric_limits<double>::epsilon())) {
        gsl_function F;
        F.function = [](const double t, void* params) -> double {
            const auto function = static_cast<std::function<double(double)>*>(params);
            return (*function)(t);
        };

        // Create a copy of func to ensure it remains valid in the scope of GSL function
        auto* func_copy = new std::function<double(double)>(func);
        F.params = func_copy;

        double result, abserror;
        gsl_deriv_central(&F, x, h, &result, &abserror);

        // Clean up the dynamically allocated copy after use
        delete func_copy;
        return result;
    }
    /**
     * @brief Computes the arc length of a parametric curve using numerical integration.
     * @param xFunc Function representing the x-component of the curve.
     * @param yFunc Function representing the y-component of the curve.
     * @param tMin Minimum value of the parameter t.
     * @param tMax Maximum value of the parameter t.
     * @return The arc length of the curve from tMin to tMax.
     * @exception std::runtime_error Thrown if the integration fails to produce a finite result.
     */
    static double calculateArcLength(const std::function<double(double)>& xFunc, const std::function<double(double)>& yFunc, const double tMin, const double tMax) {
        gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(5000);
        gsl_function F;
        std::pair funcs = {xFunc, yFunc};
        F.function = &Integral::speedFunc;
        F.params = &funcs;
        double result, error;

        gsl_integration_qags(&F, tMin, tMax, 0, std::sqrt(std::numeric_limits<double>::epsilon()), 5000, workspace, &result, &error);
        gsl_integration_workspace_free(workspace);
        if (std::isnan(result) || std::isinf(result)) {
            throw std::runtime_error("Integration failed from tMin=" + std::to_string(tMin) + " to tMax=" + std::to_string(tMax));
        }
        return result;
    }
    /**
     * @brief General-purpose method for numerical integration of a single-variable function over an interval [a, b].
     * @param func The function to integrate.
     * @param a Lower limit of integration.
     * @param b Upper limit of integration.
     * @return The integral of the function from a to b.
     * @exception std::runtime_error Thrown if the integration fails to produce a finite result.
     */
    static double computeIntegral(const std::function<double(double)>& func, const double a, const double b) {
        gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(5000);
        gsl_function F;
        // ReSharper disable once CppParameterMayBeConstPtrOrRef
        F.function = [](const double t, void* params) -> double {
            const auto* function = static_cast<const std::function<double(double)>*>(params);
            return (*function)(t);
        };

        const std::function<double(double)>* func_ptr = &func;
        F.params = const_cast<std::function<double(double)>*>(func_ptr);

        double result, error;
        gsl_integration_qags(&F, a, b, 0, std::sqrt(std::numeric_limits<double>::epsilon()), 5000, workspace, &result, &error);
        gsl_integration_workspace_free(workspace);

        if (std::isnan(result) || std::isinf(result)) {
            throw std::runtime_error("Integration failed from a=" + std::to_string(a) + " to b=" + std::to_string(b));
        }
        return result;
    }
private:
    // Compute the speed function for the arc length of a parametric curve
    static double speedFunc(const double t, void* params) {
        const auto* p = static_cast<std::pair<std::function<double(double)>, std::function<double(double)>>*>(params);
        const double dx = computeDerivative(p->first, t);
        const double dy = computeDerivative(p->second, t);
        return std::sqrt(dx * dx + dy * dy);
    }
};

/**
 * @brief Abstract base class for defining geometric boundaries.
 *
 * This class provides the interface for parametric representations of boundaries, which are essential in simulations and geometric calculations.
 */
class AbstractBoundary {
public:
    virtual ~AbstractBoundary() = default;
    /**
     * @brief Parametrically defines the position on the boundary corresponding to parameter t.
     * @param t Parametric coordinate, typically normalized between 0 and 1.
     * @return The point on the boundary corresponding to the parameter t.
     */
    [[nodiscard]] virtual Point curveParametrization(double t) const = 0;
    /**
     * @brief Calculates the outward normal vector at a point on the boundary specified by parameter t.
     * @param t Parametric coordinate where the normal is to be calculated.
     * @return The normal vector at the parameter t, pointing outward.
     */
    [[nodiscard]] virtual Point calculateNormal(double t) const = 0;
    /**
     * @brief Computes the total arc length of the boundary.
     * @return The total length of the boundary.
     */
    [[nodiscard]] virtual double calculateArcLength() const = 0;
    /**
     * @brief Computes the curvature of the boundary at a given parametric point t.
     * @param t Parametric coordinate where the curvature is to be calculated.
     * @return The curvature at the parameter t.
     */
    [[nodiscard]] virtual double computeCurvature(double t) const = 0;
    /**
     * @brief Plots the boundary and its normals using matplot++.
     *
     * This method plots the boundary as a continuous line and overlays normal vectors at specified intervals.
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the boundary.
     * @param plotNormals If the normals are to be plotted
     * @param plotCoordinateAxes If the coordinate axes be plotted
     */
    virtual void plot(const matplot::axes_handle &ax, int numPoints, bool plotNormals, bool plotCoordinateAxes) const = 0;
    /**
     * @brief Computes the bounding box that completely contains the boundary.
     * @param minX Minimum x-coordinate of the bounding box.
     * @param maxX Maximum x-coordinate of the bounding box.
     * @param minY Minimum y-coordinate of the bounding box.
     * @param maxY Maximum y-coordinate of the bounding box.
     */
    virtual void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const = 0;
    /**
     * @brief Checks if the boundary supports point-inclusion testing.
     * @return True if the boundary supports testing if a point is inside it, otherwise false.
     */
    [[nodiscard]] virtual bool supportsIsInside() const { return false; } // Default implementation returns false
    /**
     * @brief Determines if a given point is inside the area enclosed by the boundary.
     * @param point The point to check.
     * @return True if the point is inside the boundary, false otherwise.
     *
     * @details
     * This method throws an exception if isInside is not supported for the boundary type, as indicated by supportsIsInside().
     */
    [[nodiscard]] virtual bool isInside(const Point& point) const {
        throw std::runtime_error("isInside not supported for this boundary type");
    }
    /**
     * Calculates the geometric area of the shape (unsigned). Is implemented in each subclass uniquely
     */
    [[nodiscard]] virtual double getArea() const {
        throw std::runtime_error("Must be implemented uniquely in the child classes");
    }
    /**
     * @brief Calculates the speed of the parametric curve at a specific parameter value t.
     *
     * The speed is defined as the magnitude of the first derivative of the curve parametrization,
     * which corresponds to the rate of change of the curve with respect to the parameter t.
     *
     * @param t Parametric coordinate, typically normalized between 0 and 1.
     * @return The speed (magnitude of the derivative) at the parameter t.
     *
     * @details
     * This method calculates the derivative of the curve parametrization using numerical differentiation.
     * The derivative is computed using:
     * - Forward differencing at the start of the boundary (t=0).
     * - Backward differencing at the end of the boundary (t=1).
     * - Central differencing elsewhere.
     *
     * This is essential for accurately calculating arc lengths and for integrations that depend on the curve's speed.
     */
    [[nodiscard]] virtual double speedAt(const double t) const {
        // Calculate derivatives conditionally based on the position of t
        // ReSharper disable once CppDFAUnusedValue
        double dx = 0.0;
        // ReSharper disable once CppDFAUnusedValue
        double dy = 0.0;
        const double h = std::sqrt(std::numeric_limits<double>::epsilon()); // For ordering of inputs

        if (t == 0.0) {
            // Use forward derivative at the start
            dx = Derivative::compute([this](const double u) { return this->curveParametrization(u).x; }, t, h, true);
            dy = Derivative::compute([this](const double u) { return this->curveParametrization(u).y; }, t, h, true);
        } else if (t == 1.0) {
            // Use backward derivative at the end
            dx = Derivative::compute([this](const double u) { return this->curveParametrization(u).x; }, t, h, false);
            dy = Derivative::compute([this](const double u) { return this->curveParametrization(u).y; }, t, h, false);
        } else {
            // Use forward derivative otherwise
            dx = Derivative::compute([this](const double u) { return this->curveParametrization(u).x; }, t, h);
            dy = Derivative::compute([this](const double u) { return this->curveParametrization(u).y; }, t, h);
        }

        return std::sqrt(dx * dx + dy * dy);
    }
    /**
     * @brief Calculates the arc length from the start of the boundary to a given parameter value t.
     *
     * This method uses numerical integration to compute the arc length from t=0 to a specified t,
     * leveraging the speed function defined by the speedAt method.
     *
     * @param t Parametric coordinate, typically normalized between 0 and 1, up to which the arc length is calculated.
     * @return The arc length of the boundary from the start to the parameter t.
     *
     * @details
     * The arc length is computed by integrating the speed function from 0 to t.
     * This method is crucial for applications requiring the exact length of the boundary covered up to a certain point,
     * such as in reparametrization of curves for uniform parameter distribution or motion along the curve.
     */
    [[nodiscard]] virtual double calculateArcParameter(double t) const = 0;
    /**
     * Finds the parameter t for a given point on the boundary using the secant method.
     *
     * The method iteratively approximates the parameter t such that the point on the
     * curve corresponding to this parameter is close to the given point p. The secant
     * method is used to find the root of the distance function between the given point
     * and the curve parametrization. The iteration stops when the distance is within
     * a specified tolerance or when a maximum number of iterations is reached.
     *
     * @param p The point for which to find the corresponding parameter t on the boundary.
     * @return The parameter t such that the point on the curve corresponding to this
     *         parameter is close to the given point p.
     */
    [[nodiscard]] virtual double findT(const Point& p) const {
        // Use the secant method to find the parameter t for the given point p
        double t0 = 0.0;
        double t1 = 1.0;
        constexpr double tolerance = std::numeric_limits<double>::epsilon();
        constexpr int max_iterations = 100;
        int iterations = 0;
        while (iterations < max_iterations) {
            const Point pt0 = curveParametrization(t0);
            const Point pt1 = curveParametrization(t1);
            const double f0 = (pt0.x - p.x) * (pt0.x - p.x) + (pt0.y - p.y) * (pt0.y - p.y);
            const double f1 = (pt1.x - p.x) * (pt1.x - p.x) + (pt1.y - p.y) * (pt1.y - p.y);
            if (std::abs(f1 - f0) < tolerance) {
                break;
            }
            const double t2 = t1 - f1 * (t1 - t0) / (f1 - f0);
            t0 = t1;
            t1 = t2;
            if (std::abs(f1) < tolerance) {
                break;
            }
            iterations++;
        }
        return t1;
    }
};

/**
 * @brief Represents a boundary defined by parametric equations.
 *
 * This class serves as a base class for boundaries described by parametric functions x(t) and y(t).
 */
class ParametricBoundary : public AbstractBoundary {
public:
    std::function<double(double)> x_func;
    std::function<double(double)> y_func;
    mutable double cachedArcLength;
    mutable bool arcLengthCached;
    /**
     * @brief Constructs a ParametricBoundary using parametric functions for the coordinates.
     * @param x Function defining the x-coordinate of the curve.
     * @param y Function defining the y-coordinate of the curve.
     */
    ParametricBoundary(std::function<double(double)> x, std::function<double(double)> y)
    : x_func(std::move(x)), y_func(std::move(y)), cachedArcLength(0.0), arcLengthCached(false) {}
    /**
     * @brief Returns the point on the curve at parameter t.
     * @param t A parameter ranging from 0 to 1.
     * @return Point on the curve at parameter t.
     *
     * @details
     * This method computes the position on the curve by evaluating the parametric functions x(t) and y(t).
     */
    [[nodiscard]] Point curveParametrization(const double t) const override {
        return {x_func(t), y_func(t)};
    }
    /**
     * @brief Calculates the normal vector at a given parameter on the curve.
     * @param t A parameter (0 to 1).
     * @return Normal vector at the given parameter.
     *
     * @details
     * - The normal vector is calculated by computing the derivatives of x(t) and y(t) to obtain the tangent vector.
     * - The normal is then obtained by rotating this tangent vector by 90 degrees, ensuring it points outward.
     */
    [[nodiscard]] Point calculateNormal(const double t) const override {
        const double dx_dt = Derivative::compute(x_func, t);
        const double dy_dt = Derivative::compute(y_func, t);
        return {dy_dt, -dx_dt}; // Rotate tangent to get normal, it needs to point outwards
    }
    /**
     * @brief Computes the curvature of the curve at a specific parameter.
     * @param t A parameter (0 to 1).
     * @return Curvature at the given parameter.
     *
     * @details
     * - Curvature is calculated using the formula for the curvature of a parametric curve, involving first and
     *   second derivatives of the parametric equations.
     */
    [[nodiscard]] double computeCurvature(const double t) const override {
        const double dx = Derivative::compute(x_func, t);
        const double dy = Derivative::compute(y_func, t);
        const double ddx = Derivative::compute([=](const double u) { return Derivative::compute(x_func, u); }, t);
        const double ddy = Derivative::compute([=](const double u) { return Derivative::compute(y_func, u); }, t);
        const double denominator = std::pow(dx * dx + dy * dy, 1.5);
        return denominator != 0 ? std::abs(dx * ddy - dy * ddx) / denominator : 0.0;
    }
    /**
     * @brief Calculates the total arc length of the parametric curve.
     * @return Total arc length of the curve.
     *
     * @details
     * - This method uses numerical integration to compute the arc length of the curve over the interval [0, 1].
     * - The length is cached upon first computation to optimize performance for subsequent calls.
     */
    [[nodiscard]] double calculateArcLength() const override {
        if (!arcLengthCached) { // Caching improves performance
            cachedArcLength = Integral::calculateArcLength(x_func, y_func, 0.0, 2 * M_PI);
            arcLengthCached = true;
        }
        return cachedArcLength;
    }
    /**
     * @brief Prints detailed information about the boundary at a set number of points along the curve.
     * @param numPoints Number of points at which to evaluate and print details.
     *
     * @details
     * - This method prints the parameter t, the corresponding point on the curve, the normal at that point,
     *   and the curvature, providing a comprehensive description of the curve's properties at regular intervals.
     */
    void printBoundaryInfo(const int numPoints) const {
        const double step = 1.0 / numPoints;
        std::cout << "Boundary Info (" << numPoints << " points):" << std::endl;
        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);
            const double curvature = computeCurvature(t);
            std::cout << "t: " << t
                      << ", Point: [" << p.x << ", " << p.y << "]"
                      << ", Normal: [" << n.x << ", " << n.y << "]"
                      << ", Curvature: " << curvature
                      << std::endl;
        }
    }
    /**
     * @brief Plots the boundary and its normals using matplot++.
     *
     * This method plots the boundary as a continuous line and overlays normal vectors at specified intervals.
     *
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the boundary.
     * @param plotNormals If the normals are to be plotted
     * @param plotCoordinateAxes If we want to plot the coordinate axes
     */
    void plot(const matplot::axes_handle &ax, const int numPoints, const bool plotNormals, const bool plotCoordinateAxes) const override {
        using namespace matplot;
        std::vector<double> x(numPoints + 1), y(numPoints + 1);
        std::vector<double> u(numPoints + 1), v(numPoints + 1);
        const double step = 1.0 / numPoints;
        ax->hold(matplot::on);

        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);

            x[i] = p.x;
            y[i] = p.y;
            if (plotNormals) {
                ax->arrow(p.x, p.y, p.x + 0.1 * n.x, p.y + 0.1 * n.y)->color("red");
            }
        }

        // Plot the boundary
        ax->plot(x, y, "-k");

        if (plotCoordinateAxes) {
            // Plot the coordinate system (x and y axes)
            double minX = *std::ranges::min_element(x);
            double maxX = *std::ranges::max_element(x);
            double minY = *std::ranges::min_element(y);
            double maxY = *std::ranges::max_element(y);

            // Extend axes limits a bit for better visualization
            const double xPadding = (maxX - minX) * 0.05;
            const double yPadding = (maxY - minY) * 0.05;

            minX -= xPadding;
            maxX += xPadding;
            minY -= yPadding;
            maxY += yPadding;

            // Plot x-axis
            ax->plot({minX, maxX}, std::vector<double>{0, 0}, "--k");

            // Plot y-axis
            ax->plot({0, 0}, {minY, maxY}, "--k");
        }

        ax->hold(matplot::off);
        ax->xlabel("x");
        ax->ylabel("y");
    }
    bool supportsIsInside() const override {
        return true;
    }
    /**
     * @brief Calculates the bounding box that completely encloses the curve.
     * @param minX Reference to store the minimum x-coordinate.
     * @param maxX Reference to store the maximum x-coordinate.
     * @param minY Reference to store the minimum y-coordinate.
     * @param maxY Reference to store the maximum y-coordinate.
     *
     * @details
     * - This method calculates the bounding box by evaluating the curve at a dense set of points along the curve,
     *   considering both the curve's nature and variations in curvature.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        // Initialize bounding box values to extreme opposites to ensure correct calculation
        minX = minY = std::numeric_limits<double>::infinity();
        maxX = maxY = -std::numeric_limits<double>::infinity();

        double t = 0.0;  // Start from the beginning of the normalized parametric interval
        while (t <= 1.0) {
            const double curvature = computeCurvature(t);  // Compute curvature at parameter t
            const double adaptiveStep = std::max(0.01, 1.0 / (1.0 + curvature));  // Smaller steps for higher curvature

            Point point = curveParametrization(t);
            minX = std::min(minX, point.x);
            maxX = std::max(maxX, point.x);
            minY = std::min(minY, point.y);
            maxY = std::max(maxY, point.y);

            t += adaptiveStep;  // Increment t based on the adaptive step size
        }

        // Check the final point if loop doesn't exactly reach t=1
        if (t != 1.0) {
            const Point endPoint = curveParametrization(1.0);
            minX = std::min(minX, endPoint.x);
            maxX = std::max(maxX, endPoint.x);
            minY = std::min(minY, endPoint.y);
            maxY = std::max(maxY, endPoint.y);
        }
    }
    /**
     * @brief Determines if a given point is inside the area enclosed by the boundary.
     * @param point The point to check.
     * @return True if the point is inside the boundary, false otherwise.
     *
     * @details
     * - The implementation for checking if a point is inside uses a ray-casting algorithm along the horizontal axis
     *   and counts the number of intersections with the boundary.
     */
    bool isInside(const Point& point) const override {
        int count = 0;
        gsl_root_fsolver* solver = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);
        gsl_function F;
        F.function = [](const double t, void* params) -> double {
            auto&[fst, snd] = *static_cast<std::pair<std::function<double(double)>, double>*>(params);
            return fst(t) - snd;
        };

        std::pair<std::function<double(double)>, double> params = {y_func, point.y};
        F.params = &params;

        // ReSharper disable once CppDFAUnusedValue
        double t_min = 0.0, t_max = 1.0, root = 0;
        gsl_root_fsolver_set(solver, &F, t_min, t_max);

        int status = GSL_CONTINUE;
        // Loop will continue as long as status indicates that iteration is necessary
        while (status == GSL_CONTINUE) {
            status = gsl_root_fsolver_iterate(solver);
            if (status != GSL_SUCCESS) {
                std::cout << "Root solver foe parametric boundary failed" << std::endl;
                break;
            }

            root = gsl_root_fsolver_root(solver);
            t_min = gsl_root_fsolver_x_lower(solver);
            t_max = gsl_root_fsolver_x_upper(solver);

            // Check for convergence
            if (fabs(t_max - t_min) < 0.001) { // Convergence criteria
                if (const double x_at_root = x_func(root); x_at_root > point.x) {
                    count++;
                }
                break; // Assume convergence
            }
        }

        gsl_root_fsolver_free(solver);
        return (count % 2) == 1;
    }
};

/**
 * @brief Represents a segment of a parametric curve, defining a boundary from a specified start to end parameter. THis is like choosing t_start=0.1 and t_end=0.4. THis will construct just a part of the boundary between the two parametrizations. This is useful when desymmetrizing the boundary as with certain choices for start t and end t we can correctly desymmetrize the billiard
 *
 * This class extends ParametricBoundary by allowing the specification of a sub-section of a parametric curve.
 */
class PartialParametricBoundary final : public ParametricBoundary {
    double t_start;
    double t_end;

public:
    /**
     * @brief Constructs a PartialParametricBoundary.
     * @param x Function defining the x-coordinate of the parametric curve.
     * @param y Function defining the y-coordinate of the parametric curve.
     * @param start The starting parametric value (inclusive).
     * @param end The ending parametric value (inclusive).
     */
    PartialParametricBoundary(
        std::function<double(double)> x,
        std::function<double(double)> y,
        const double start = 0.0,
        const double end = 1.0)
    : ParametricBoundary(std::move(x), std::move(y)), t_start(start), t_end(end) {}
    /**
     * @brief Returns the point on the parametric curve at a specific parameter within the segment.
     * @param t A parameter ranging from 0 to 1, remapped to the sub-interval [t_start, t_end].
     * @return Point on the curve at the remapped parameter.
     *
     * @details
     * This function remaps the parametric value t to the interval [t_start, t_end] and then evaluates the curve,
     * effectively focusing on a segment of the full parametric curve.
     */
    [[nodiscard]] Point curveParametrization(const double t) const override {
        // Adjust `t` to map to the partial segment
        const double mapped_t = t_start + t * (t_end - t_start);
        return ParametricBoundary::curveParametrization(mapped_t);
    }
    /**
     * @brief Calculates the normal vector at a given parameter within the segment.
     * @param t A parameter (0 to 1), remapped to [t_start, t_end] for the segment.
     * @return Normal vector at the remapped parameter on the curve.
     *
     * @details
     * - This method recalculates the parametric value to correspond to the segment and then calculates the normal.
     * - The normal direction is determined by the original curve's characteristics and the specified segment.
     */
    [[nodiscard]] Point calculateNormal(const double t) const override {
        const double mapped_t = t_start + t * (t_end - t_start);
        return ParametricBoundary::calculateNormal(mapped_t);
    }
    /**
     * @brief Computes the curvature of the curve at a specific parameter within the segment.
     * @param t A parameter, remapped to [t_start, t_end].
     * @return Curvature at the remapped parameter.
     */
    [[nodiscard]] double computeCurvature(const double t) const override {
        const double mapped_t = t_start + t * (t_end - t_start);
        return ParametricBoundary::computeCurvature(mapped_t);
    }
    /**
     * @brief Calculates the length of the curve segment between t_start and t_end.
     * @return Length of the curve segment.
     */
    [[nodiscard]] double calculateArcLength() const override {
        // Recalculate arc length only for the segment
        return Integral::calculateArcLength(x_func, y_func, t_start, t_end);
    }
    /**
     * @brief Plots the boundary and its normals using matplot++.
     *
     * This method plots the boundary as a continuous line and overlays normal vectors at specified intervals.
     *
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the boundary.
     * @param plotNormals If the normals are to be plotted
     * @param plotCoordinateAxes If we want to plot the coordinate axes
     */
    void plot(const matplot::axes_handle &ax, const int numPoints, const bool plotNormals, const bool plotCoordinateAxes) const override {
        using namespace matplot;
        std::vector<double> x(numPoints + 1), y(numPoints + 1);
        std::vector<double> u(numPoints + 1), v(numPoints + 1);
        const double step = 1.0 / numPoints;
        ax->hold(matplot::on);

        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);

            x[i] = p.x;
            y[i] = p.y;
            if (plotNormals) {
                ax->arrow(p.x, p.y, p.x + 0.1 * n.x, p.y + 0.1 * n.y)->color("red");
            }
        }

        // Plot the boundary
        ax->plot(x, y, "-k");

        if (plotCoordinateAxes) {
            // Plot the coordinate system (x and y axes)
            double minX = *std::ranges::min_element(x);
            double maxX = *std::ranges::max_element(x);
            double minY = *std::ranges::min_element(y);
            double maxY = *std::ranges::max_element(y);

            // Extend axes limits a bit for better visualization
            const double xPadding = (maxX - minX) * 0.05;
            const double yPadding = (maxY - minY) * 0.05;

            minX -= xPadding;
            maxX += xPadding;
            minY -= yPadding;
            maxY += yPadding;

            // Plot x-axis
            ax->plot({minX, maxX}, std::vector<double>{0, 0}, "--k");

            // Plot y-axis
            ax->plot({0, 0}, {minY, maxY}, "--k");
        }

        ax->hold(matplot::off);
        ax->xlabel("x");
        ax->ylabel("y");
    }
    bool supportsIsInside() const override {
        return true;
    }
    /**
     * @brief Calculates the bounding box that completely encloses the segment.
     * @param minX Reference to store the minimum x-coordinate.
     * @param maxX Reference to store the maximum x-coordinate.
     * @param minY Reference to store the minimum y-coordinate.
     * @param maxY Reference to store the maximum y-coordinate.
     *
     * @details
     * - This method calculates the bounding box by evaluating the curve at various points along the segment,
     *   taking curvature into account to adaptively sample densely in high-curvature areas.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        minX = minY = std::numeric_limits<double>::infinity();
        maxX = maxY = -std::numeric_limits<double>::infinity();

        double t = 0.0;  // Start from the beginning of the normalized parametric interval
        while (t <= 1.0) {
            const double mapped_t = t_start + t * (t_end - t_start);
            const double curvature = computeCurvature(mapped_t);  // Use the mapped t value
            const double adaptiveStep = std::max(0.01, 1.0 / (1.0 + curvature));  // Smaller steps for higher curvature

            Point point = curveParametrization(t);
            minX = std::min(minX, point.x);
            maxX = std::max(maxX, point.x);
            minY = std::min(minY, point.y);
            maxY = std::max(maxY, point.y);

            t += adaptiveStep;  // Increment t based on the adaptive step size
        }

        // Check the final point if loop doesn't exactly reach t_end
        if (t != 1.0) {
            const Point endPoint = curveParametrization(1.0);
            minX = std::min(minX, endPoint.x);
            maxX = std::max(maxX, endPoint.x);
            minY = std::min(minY, endPoint.y);
            maxY = std::max(maxY, endPoint.y);
        }
    }

    [[nodiscard]] double calculateArcParameter(double t) const override;
};

/**
 * @brief Represents a straight line segment defined by two endpoints and provides methods to calculate its properties.
 *
 * This class encapsulates a line segment, allowing for the computation of geometric properties such as the length, normal, and methods for plotting.
 */
class LineSegment final : public AbstractBoundary {
    Point start, end;
    bool clockwise;  // Indicates if the segment is part of a clockwise-ordered boundary

public:
    /**
     * @brief Constructor for the LineSegment class.
     * @param start Starting point of the line segment.
     * @param end Ending point of the line segment.
     * @param clockwise Indicates if the normal should be computed as if the segment is part of a clockwise traversal of a boundary.
     */
    LineSegment(const Point &start, const Point &end, const bool clockwise = true)
        : start(start), end(end), clockwise(clockwise) {}
    /**
     * @brief Computes the point on the line segment corresponding to a given parametric value t.
     * @param t A parameter ranging from 0 to 1, where 0 corresponds to the start point and 1 to the end point.
     * @return Point at the parametric position t on the line segment.
     *
     * @details
     * - The parametric equation of the line segment is:
     *   x(t) = start.x + (end.x - start.x) * t
     *   y(t) = start.y + (end.y - start.y) * t
     * - This linear interpolation ensures the line segment is traversed linearly as t progresses from 0 to 1.
     */
    [[nodiscard]] Point curveParametrization(const double t) const override {
        return {start.x + (end.x - start.x) * t, start.y + (end.y - start.y) * t};
    }
    /**
     * @brief Computes the length of the line segment.
     * @return The Euclidean distance between the start and end points.
     */
    [[nodiscard]] double calculateArcLength() const override {
        return std::hypot(end.x - start.x, end.y - start.y);
    }
    /**
     * @brief Calculates the normal vector at any point on the line segment.
     * @param t A parameter (not used here since the normal is constant along the line segment).
     * @return The outward normal vector of the line segment.
     *
     * @details
     * - The normal vector for a line segment is constant and can be computed from the vector (end - start).
     * - For a clockwise segment, the right-hand normal is used:
     *   If clockwise: normal = { (end.y - start.y) / length, -(end.x - start.x) / length }
     *   If not clockwise: normal = { -(end.y - start.y) / length, (end.x - start.x) / length }
     * - The normalization ensures the length of the normal vector is 1.
     */
    [[nodiscard]] Point calculateNormal(double t) const override {
        double dx = end.x - start.x;
        double dy = end.y - start.y;

        // Address precision issues by setting very small values to zero
        if (std::abs(dx) < std::numeric_limits<double>::epsilon()) dx = 0.0;
        if (std::abs(dy) < std::numeric_limits<double>::epsilon()) dy = 0.0;

        const double length = std::hypot(dx, dy);
        if (clockwise) {
            return {dy / length, -dx / length};  // Right-hand normal for clockwise
        } else {
            return {-dy / length, dx / length};  // Left-hand normal for counterclockwise
        }
    }
    /**
     * @brief Computes the curvature of the line segment.
     * @param t Parametric value (unused as curvature of a straight line is always zero).
     * @return Always returns 0.0, as a line segment does not curve.
     */
    [[nodiscard]] double computeCurvature(double t) const override {
        return 0.0;  // Curvature of a line segment is zero
    }
    /**
     * @brief Plots the boundary and its normals using matplot++.
     *
     * This method plots the boundary as a continuous line and overlays normal vectors at specified intervals.
     *
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the boundary.
     * @param plotNormals If the normals are to be plotted
     * @param plotCoordinateAxes If we want to plot the coordinate axes
     */
    void plot(const matplot::axes_handle &ax, const int numPoints, const bool plotNormals, const bool plotCoordinateAxes) const override {
        using namespace matplot;
        std::vector<double> x(numPoints + 1), y(numPoints + 1);
        std::vector<double> u(numPoints + 1), v(numPoints + 1);
        const double step = 1.0 / numPoints;
        ax->hold(matplot::on);

        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);

            x[i] = p.x;
            y[i] = p.y;
            if (plotNormals) {
                ax->arrow(p.x, p.y, p.x + 0.1 * n.x, p.y + 0.1 * n.y)->color("red");
            }
        }

        // Plot the boundary
        ax->plot(x, y, "-k");

        if (plotCoordinateAxes) {
            // Plot the coordinate system (x and y axes)
            double minX = *std::ranges::min_element(x);
            double maxX = *std::ranges::max_element(x);
            double minY = *std::ranges::min_element(y);
            double maxY = *std::ranges::max_element(y);

            // Extend axes limits a bit for better visualization
            const double xPadding = (maxX - minX) * 0.05;
            const double yPadding = (maxY - minY) * 0.05;

            minX -= xPadding;
            maxX += xPadding;
            minY -= yPadding;
            maxY += yPadding;

            // Plot x-axis
            ax->plot({minX, maxX}, std::vector<double>{0, 0}, "--k");

            // Plot y-axis
            ax->plot({0, 0}, {minY, maxY}, "--k");
        }

        ax->hold(matplot::off);
        ax->xlabel("x");
        ax->ylabel("y");
    }
    [[nodiscard]] bool supportsIsInside() const override {
        return false;
    }
    /**
     * @brief Calculates the bounding box that completely encloses the line segment.
     * @param minX Reference to store the minimum x-coordinate.
     * @param maxX Reference to store the maximum x-coordinate.
     * @param minY Reference to store the minimum y-coordinate.
     * @param maxY Reference to store the maximum y-coordinate.
     *
     * @details
     * - The bounding box for a line segment is the smallest rectangle that completely contains both the start and end points.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        minX = std::min(start.x, end.x);
        maxX = std::max(start.x, end.x);
        minY = std::min(start.y, end.y);
        maxY = std::max(start.y, end.y);
    }

    [[nodiscard]] double calculateArcParameter(const double t) const override {
        return t * calculateArcLength();
    };
};

/**
 * @class SemiCircle
 * @brief Represents a semicircle boundary in a 2D plane.
 *
 * The SemiCircle class provides functionality to represent a semicircular boundary
 * with a specified center, radius, direction, and orientation (clockwise or counterclockwise).
 * It includes methods for calculating the curve parametrization, normal vectors, arc length,
 * curvature, and for plotting the semicircle using the sciplot library.
 */
// ReSharper disable once CppClassCanBeFinal
class SemiCircle : public AbstractBoundary {
    Point center;
    double radius;
    Point direction;  // This indicates where the flat part of the semicircle faces
    bool clockwise;

public:
    /**
     * @brief Constructor for SemiCircle.
     *
     * @param center The center point of the semicircle.
     * @param radius The radius of the semicircle.
     * @param direction The direction vector indicating where the flat part of the semicircle faces.
     * @param clockwise Optional parameter to set the sweep direction (true for clockwise, false for counterclockwise). Defaults to true.
     */
    SemiCircle(const Point& center, const double radius, const Point& direction, const bool clockwise = true)
        : center(center), radius(radius), direction(direction.normalized()), clockwise(clockwise) {}
    /**
     * @brief Computes the point on the semicircle at parameter t.
     *
     * @param t The parameter, ranging from 0 to 1, indicating the position along the semicircle.
     * @return The point on the semicircle at parameter t.
     */
    [[nodiscard]] Point curveParametrization(const double t) const override {
        // Calculate the starting angle for the semicircle based on the direction vector
        const double startAngle = atan2(direction.y, direction.x) - M_PI / 2;
        // Adjust the start angle based on the orientation (clockwise or counterclockwise)
        const double angleAdjustment = clockwise ? M_PI : 0;
        const double angle = startAngle + (clockwise ? -1 : 1) * t * M_PI + angleAdjustment;

        return {center.x + radius * cos(angle), center.y + radius * sin(angle)};
    }
    /**
     * @brief Calculates the normal vector at a given parameter t.
     *
     * @param t The parameter, ranging from 0 to 1, indicating the position along the semicircle.
     * @return The normal vector at parameter t.
     */
    [[nodiscard]] Point calculateNormal(const double t) const override {
        // Calculate the base angle based on the semicircle direction and adjust according to clockwise/anticlockwise
        const double startAngle = atan2(direction.y, direction.x) - M_PI / 2;
        const double angle = startAngle + (clockwise ? -1 : 1) * t * M_PI;
        // Adjust the normal angle based on the sweep direction
        const double normalAngle = angle + (clockwise ? -M_PI : M_PI);
        return {cos(normalAngle), sin(normalAngle)};
    }
    /**
     * @brief Computes the arc length of the semicircle.
     *
     * @return The arc length of the semicircle.
     */
    [[nodiscard]] double calculateArcLength() const override {
        return M_PI * radius;
    }
    [[nodiscard]] double getArea() const override {
        return M_PI * radius * radius / 2.0;
    };
    /**
     * @brief Computes the curvature of the semicircle at a given parameter t.
     *
     * @param t The parameter, ranging from 0 to 1, indicating the position along the semicircle.
     * @return The curvature at parameter t.
     */
    [[nodiscard]] double computeCurvature(double t) const override {
        return 1 / radius;
    }
    /**
     * @brief Plots the boundary and its normals using matplot++.
     *
     * This method plots the boundary as a continuous line and overlays normal vectors at specified intervals.
     *
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the boundary.
     * @param plotNormals If the normals are to be plotted
     * @param plotCoordinateAxes If we want to plot the coordinate axes
     */
    void plot(const matplot::axes_handle &ax, const int numPoints, const bool plotNormals, const bool plotCoordinateAxes) const override {
        using namespace matplot;
        std::vector<double> x(numPoints + 1), y(numPoints + 1);
        std::vector<double> u(numPoints + 1), v(numPoints + 1);
        const double step = 1.0 / numPoints;
        ax->hold(matplot::on);

        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);

            x[i] = p.x;
            y[i] = p.y;
            if (plotNormals) {
                ax->arrow(p.x, p.y, p.x + 0.1 * n.x, p.y + 0.1 * n.y)->color("red");
            }
        }

        // Plot the boundary
        ax->plot(x, y, "-k");

        if (plotCoordinateAxes) {
            // Plot the coordinate system (x and y axes)
            double minX = *std::ranges::min_element(x);
            double maxX = *std::ranges::max_element(x);
            double minY = *std::ranges::min_element(y);
            double maxY = *std::ranges::max_element(y);

            // Extend axes limits a bit for better visualization
            const double xPadding = (maxX - minX) * 0.05;
            const double yPadding = (maxY - minY) * 0.05;

            minX -= xPadding;
            maxX += xPadding;
            minY -= yPadding;
            maxY += yPadding;

            // Plot x-axis
            ax->plot({minX, maxX}, std::vector<double>{0, 0}, "--k");

            // Plot y-axis
            ax->plot({0, 0}, {minY, maxY}, "--k");
        }

        ax->hold(matplot::off);
        ax->xlabel("x");
        ax->ylabel("y");
    }
    [[nodiscard]] bool supportsIsInside() const override {
        return false;
    }
    /**
     * @brief Computes the bounding box of the semicircle.
     *
     * @param minX Reference to the minimum x-coordinate of the bounding box.
     * @param maxX Reference to the maximum x-coordinate of the bounding box.
     * @param minY Reference to the minimum y-coordinate of the bounding box.
     * @param maxY Reference to the maximum y-coordinate of the bounding box.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        // Determine the full angle range based on the direction
        const double startAngle = atan2(direction.y, direction.x) - M_PI / 2;
        const double endAngle = startAngle + M_PI;

        const std::vector<double> angles = {startAngle, endAngle, startAngle + M_PI / 2, endAngle + M_PI / 2};
        minX = maxX = center.x + radius * cos(angles[0]);
        minY = maxY = center.y + radius * sin(angles[0]);

        for (const double angle : angles) {
            double x = center.x + radius * cos(angle);
            double y = center.y + radius * sin(angle);
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }

        // Ensure flat side coordinates are considered
        const double flatX = center.x + radius * cos(startAngle + M_PI);
        const double flatY = center.y + radius * sin(startAngle + M_PI);
        minX = std::min(minX, flatX);
        maxX = std::max(maxX, flatX);
        minY = std::min(minY, flatY);
        maxY = std::max(maxY, flatY);
    }

    [[nodiscard]] double calculateArcParameter(double t) const override {
        return t * calculateArcLength();
    };
};

// ReSharper disable once CppClassCanBeFinal
/**
 * @class QuarterCircle
 * @brief Represents a quarter circle, defined by a center point, radius, direction, and orientation (clockwise or counterclockwise).
 *
 * This class provides methods for geometric computations related to a quarter-circle,
 * including point parametrization along the arc, normal calculation at any given point,
 * arc length, curvature, plotting capabilities, and bounding box calculations.
 */
class QuarterCircle : public AbstractBoundary {
    Point center;
    double radius;
    Point direction;  // Direction where the flat part of the quarter-circle faces
    bool clockwise;

    public:
    QuarterCircle(const Point& center, const double radius, const Point& direction, const bool clockwise = true)
        : center(center), radius(radius), direction(direction.normalized()), clockwise(clockwise) {}


    /**
     * @brief Calculates the parametric representation of the quarter-circle at a given parameter t.
     *
     * The curveParametrization function defines the quarter-circle's shape, factoring in its orientation and starting direction.
     * This method translates a normalized parametric value t (ranging from 0 to 1) into a point on the quarter-circle.
     *
     * @param t A parameter from 0 to 1 where 0 starts at the direction vector, and 1 ends 90 degrees from the start, according to the specified direction (clockwise or counterclockwise).
     * @return Point on the quarter-circle corresponding to the parameter t.
     *
     * @details
     * The function computes the parametric point as follows:
     * 1. Compute the base angle from the direction vector, which defines the starting point of the quarter-circle.
     * 2. Depending on the 'clockwise' flag, adjust the base angle to map the progression of t from 0 to 1 to either a clockwise or counterclockwise movement along the circle's arc.
     * 3. The final point is calculated using the radius and the computed angle to provide the x and y coordinates.
     *
     * Example use includes generating points along the quarter-circle for plotting or collision detection in graphical applications.
     */
    [[nodiscard]] Point curveParametrization(const double t) const override {
        // Calculate the starting angle based on the direction vector and restrict to a quarter (90 degrees)
        const double startAngle = atan2(direction.y, direction.x) - M_PI_2;
        // Adjust the start angle based on the orientation (clockwise or counterclockwise)
        const double angleAdjustment = clockwise ? M_PI : 0;
        const double angle = startAngle + (clockwise ? -1 : 1) * t * M_PI_2 + angleAdjustment;

        return {center.x + radius * cos(angle), center.y + radius * sin(angle)};
    }
    /**
     * @brief Calculates the normal vector at a point on the quarter-circle defined by the parameter t.
     *
     * This method provides the outward normal vector to the quarter-circle at any point specified by the parametric value t.
     * The direction of the normal is influenced by the orientation of the quarter-circle (whether it is drawn clockwise or counterclockwise).
     *
     * @param t A parameter from 0 to 1 that specifies the point along the quarter-circle arc at which to calculate the normal.
     * @return The outward normal vector at the quarter-circle point corresponding to t.
     *
     * @details
     * The normal vector calculation involves:
     * 1. Identifying the angle of the tangent at the point determined by t, using the circle's parametric equations.
     * 2. Adjusting the angle based on the quarter-circle's orientation to ensure the normal is always outward:
     *    - If clockwise, the normal vector is rotated 90 degrees clockwise from the tangent.
     *    - If counterclockwise, it is rotated 90 degrees counterclockwise.
     * 3. The normal vector is then constructed using standard trigonometric functions to find its components.
     *
     * This method is crucial for applications involving physics calculations, collision detection, or graphics where knowing the surface normal is necessary.
     */
    [[nodiscard]] Point calculateNormal(const double t) const override {
        // Calculate the angle for the normal based on the parametric angle
        const double startAngle = atan2(direction.y, direction.x) - M_PI_2;
        const double angle = startAngle + (clockwise ? -1 : 1) * t * M_PI_2;
        const double normalAngle = angle + (clockwise ? -M_PI : M_PI);

        return {cos(normalAngle), sin(normalAngle)};
    }
    /**
     * @brief Calculates the arc length of the quarter-circle.
     *
     * @return The length of the quarter-circle's arc, which is a fixed value, π/2 times the radius.
     */
    [[nodiscard]] double calculateArcLength() const override {
        return M_PI_2 * radius;  // Quarter of the full circumference of the circle
    }
    [[nodiscard]] double getArea() const override {
        return M_PI * radius * radius / 4.0;
    };
    /**
     * @brief Computes the curvature of the quarter-circle, which is constant.
     *
     * Since a circle's curvature is defined as the reciprocal of the radius, this method returns the constant curvature based on the circle's radius and orientation.
     *
     * @param t The parameter at which to evaluate the curvature (unused in uniform curvature).
     * @return The curvature of the quarter-circle, positive if counterclockwise and negative if clockwise.
     */
    [[nodiscard]] double computeCurvature(double t) const override {
        return 1 / radius;
    }
    /**
     * @brief Plots the boundary and its normals using matplot++.
     *
     * This method plots the boundary as a continuous line and overlays normal vectors at specified intervals.
     *
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the boundary.
     * @param plotNormals If the normals are to be plotted
     * @param plotCoordinateAxes If we want to plot the coordinate axes
     */
    void plot(const matplot::axes_handle &ax, const int numPoints, const bool plotNormals, const bool plotCoordinateAxes) const override {
        using namespace matplot;
        std::vector<double> x(numPoints + 1), y(numPoints + 1);
        std::vector<double> u(numPoints + 1), v(numPoints + 1);
        const double step = 1.0 / numPoints;
        ax->hold(matplot::on);

        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);

            x[i] = p.x;
            y[i] = p.y;
            if (plotNormals) {
                ax->arrow(p.x, p.y, p.x + 0.1 * n.x, p.y + 0.1 * n.y)->color("red");
            }
        }

        // Plot the boundary
        ax->plot(x, y, "-k");

        if (plotCoordinateAxes) {
            // Plot the coordinate system (x and y axes)
            double minX = *std::ranges::min_element(x);
            double maxX = *std::ranges::max_element(x);
            double minY = *std::ranges::min_element(y);
            double maxY = *std::ranges::max_element(y);

            // Extend axes limits a bit for better visualization
            const double xPadding = (maxX - minX) * 0.05;
            const double yPadding = (maxY - minY) * 0.05;

            minX -= xPadding;
            maxX += xPadding;
            minY -= yPadding;
            maxY += yPadding;

            // Plot x-axis
            ax->plot({minX, maxX}, std::vector<double>{0, 0}, "--k");

            // Plot y-axis
            ax->plot({0, 0}, {minY, maxY}, "--k");
        }

        ax->hold(matplot::off);
        ax->xlabel("x");
        ax->ylabel("y");
    }
    /**
     * @brief Computes the bounding box that contains the quarter-circle.
     *
     * This method calculates the minimum and maximum x and y coordinates that enclose the quarter-circle.
     *
     * @param minX Reference to store the minimum x-coordinate of the bounding box.
     * @param maxX Reference to store the maximum x-coordinate of the bounding box.
     * @param minY Reference to store the minimum y-coordinate of the bounding box.
     * @param maxY Reference to store the maximum y-coordinate of the bounding box.
     *
     * @details
     * - Considers the critical points along the quarter-circle (start, end, and the highest/lowest points depending on orientation).
     * - Adjusts for the quarter-circle's direction and radius to determine the extents of the bounding box.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        // The base angle is determined by the direction vector
        double baseAngle = atan2(direction.y, direction.x);
        // Adjust the base angle based on the direction and whether the circle is clockwise or counterclockwise
        baseAngle -= (clockwise ? 0 : M_PI_2);
        const std::vector<double> checkAngles = {
            baseAngle,
            baseAngle + (clockwise ? M_PI_2 : -M_PI_2),  // 90 degrees away
        };
        // Initialize the bounding box at the center, expanded by the radius
        minX = maxX = center.x + radius * cos(baseAngle);
        minY = maxY = center.y + radius * sin(baseAngle);
        // Expand the bounding box to include all critical points around the quarter circle
        for (const double angle : checkAngles) {
            double x = center.x + radius * cos(angle);
            double y = center.y + radius * sin(angle);
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }
        // Explicit check for the vertical maximum when quarter circle is in standard position
        if (clockwise) {
            maxY = std::max(maxY, center.y + radius);  // Ensure topmost point is considered
        } else {
            minY = std::min(minY, center.y - radius);  // If clockwise, ensure the lowest point is considered
        }
    }

    [[nodiscard]] double calculateArcParameter(const double t) const override {
        return t * calculateArcLength();
    };

    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    };
};

// ReSharper disable once CppClassCanBeFinal
/**
 * @class SemiEllipse
 * @brief Represents a semi-ellipse defined by a center, semi-major axis, semi-minor axis, and a direction indicating the flat side.
 *
 * This class provides methods for geometric computations related to a semi-ellipse,
 * including point parametrization along the arc, normal calculation at any given point,
 * arc length, curvature, plotting capabilities, and bounding box calculations.
 */
class SemiEllipse final : public AbstractBoundary {
    Point center;   // Center of the ellipse
    double a;       // Semi-major axis length
    double b;       // Semi-minor axis length
    Point direction; // Direction vector that indicates where the flat part faces

public:
    SemiEllipse(const Point &center, const double a, const double b, const Point &direction)
        : center(center), a(a), b(b), direction(direction.normalized()) {}

    /**
     * @brief Computes the coordinates of a point on the semi-ellipse for a given parameter t.
     *
     * Parametrizes the upper half of an ellipse based on a normalized parameter t (0 to 1),
     * adjusting the orientation based on a direction vector. The semi-ellipse spans from the
     * "left" to the "right" side of the ellipse as defined by the direction vector.
     *
     * @param t A normalized parameter (from 0 to 1) representing the position along the semi-ellipse.
     * @return Point representing the coordinates on the semi-ellipse at parameter t.
     *
     * @details
     * - Calculates the base angle from the direction vector, adjusting for a half-circle (π) to orient the flat side properly.
     * - Maps the parameter t to an angle that represents a sweep from 0 to π across the ellipse.
     * - Uses the ellipse parametric equations to compute x and y coordinates:
     *   x(t) = a * cos(angle)
     *   y(t) = b * sin(angle)
     * where a and b are the semi-major and semi-minor axes, respectively.
     */
    [[nodiscard]] Point curveParametrization(const double t) const override {
        // Calculate the base angle of the direction vector
        const double baseAngle = atan2(direction.y, direction.x) - M_PI / 2;  // Adjust to point "flat side"
        // The parametrization will always be for the top half, adjusted for direction
        const double angle = baseAngle + t * M_PI;  // t from 0 to 1 spans half the ellipse
        // Calculate the coordinates using the parametric equations of the ellipse
        const double x = a * cos(angle);
        const double y = b * sin(angle);
        return {center.x + x, center.y + y};
    }
    /**
     * @brief Calculates the normal vector at a specific point on the semi-ellipse, defined by the parameter t.
     *
     * Determines the outward-facing normal vector, ensuring that it points away from the ellipse's center,
     * which is necessary for certain geometric computations such as collision detection or boundary conditions.
     *
     * @param t A normalized parameter (from 0 to 1) that specifies the point along the semi-ellipse arc.
     * @return Point representing the outward normal vector at the specified point on the semi-ellipse.
     *
     * @details
     * - Computes the tangent angle at the point given by t and adjusts it by π (180 degrees) to ensure the normal points outward.
     * - The normal vector is then calculated by:
     *   nx = -cos(angle + π)
     *   ny = -sin(angle + π)
     * These adjustments are based on the normal right-hand rule for curves, ensuring the normal vector is orthogonal to the tangent
     * and points towards the expected direction based on the ellipse orientation.
     */
    [[nodiscard]] Point calculateNormal(const double t) const override {
        const double baseAngle = atan2(direction.y, direction.x) - M_PI / 2;
        const double angle = baseAngle + t * M_PI;
        const double dx = -a * sin(angle);
        const double dy = b * cos(angle);
        const double length = std::hypot(dy, dx); // Normalize the normal vector
        return {dy / length, -dx / length}; // Correctly assigning dy and dx for the normal vector
    }
    /**
     * @brief Calculates the arc length of the semi-ellipse using elliptic integrals.
     *
     * Computes the length of the semi-ellipse's perimeter from one end to the other along the arc.
     * This computation is essential for precise physical modeling and graphical representation.
     *
     * @return The arc length of the semi-ellipse.
     *
     * @details
     * - Utilizes the elliptic integral of the second kind, which is required for ellipses with different semi-major and semi-minor axes.
     * - The computation involves the elliptic modulus k, derived from the ratio of the axes:
     *   k = sqrt(1 - (b^2 / a^2))
     * - Uses GSL's implementation of the elliptic integral for high accuracy.
     */
    [[nodiscard]] double calculateArcLength() const override {
        // Calculate elliptic integral of the second kind for the arc length
        const double k = std::sqrt(1 - (b * b) / (a * a));
        // For a semi-ellipse, the total arc length is given by 2 * a * Ecomp(k)
        return 2 * a * gsl_sf_ellint_Ecomp(k, GSL_PREC_DOUBLE);
    }

    /**
     * @brief Computes the arc length parameter for a given normalized parameter t.
     *
     * This method distributes the parameter t along the semi-ellipse based on its arc length,
     * ensuring accurate representation of the boundary's geometry using elliptic integrals.
     *
     * @param t A double value where 0 <= t <= 1, representing a normalized distance along the semi-ellipse.
     * @return The arc length parameter corresponding to t, adjusted for the semi-ellipse.
     *
     * @details
     * - The calculation involves the elliptic integral of the second kind, which computes the arc length
     *   of the semi-ellipse up to the angle φ corresponding to t.
     * - The angle φ is determined by scaling t by π/2, as t ranges from 0 to 1 over the semi-ellipse.
     */
    [[nodiscard]] double calculateArcParameter(const double t) const override {
        // Calculate the elliptic modulus k
        const double k = std::sqrt(1 - (b * b) / (a * a));
        // Compute the angle φ corresponding to t
        const double phi = t * M_PI;
        // Calculate the arc length parameter using the elliptic integral of the second kind
        return a * gsl_sf_ellint_E(phi, k, GSL_PREC_DOUBLE);
    }
    /**
     * @brief Computes the curvature of the semi-ellipse at a specified parameter t.
     *
     * Provides a measure of how sharply the semi-ellipse curves at any point, which is crucial for various
     * analytical and simulation tasks in physics and graphics.
     *
     * @param t The parameter at which to evaluate the curvature, normalized between 0 and 1.
     * @return The curvature of the semi-ellipse at the specified parameter t.
     *
     * @details
     * - The curvature formula for a parametric curve is:
     *   kappa = |x' * y'' - y' * x''| / (x'^2 + y'^2)^(3/2)
     * - Derivatives are calculated based on the parametric equations of the ellipse, considering the orientation and the axes lengths.
     */
    [[nodiscard]] double computeCurvature(const double t) const override {
        // Calculate curvature using derivatives of parametric equations
        const double baseAngle = atan2(direction.y, direction.x) - M_PI / 2;
        const double angle = baseAngle + t * M_PI;
        const double dx_dt = -a * sin(angle);
        const double dy_dt = b * cos(angle);
        const double ddx_dt = -a * cos(angle);
        const double ddy_dt = -b * sin(angle);
        const double numerator = dx_dt * ddy_dt - dy_dt * ddx_dt;
        const double denominator = pow(dx_dt * dx_dt + dy_dt * dy_dt, 1.5);
        return std::abs(numerator) / denominator;
    }
    /**
     * @brief Plots the boundary and its normals using matplot++.
     *
     * This method plots the boundary as a continuous line and overlays normal vectors at specified intervals.
     *
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the boundary.
     * @param plotNormals If the normals are to be plotted
     * @param plotCoordinateAxes If we want to plot the coordinate axes
     */
    void plot(const matplot::axes_handle &ax, const int numPoints, const bool plotNormals, const bool plotCoordinateAxes) const override {
        using namespace matplot;
        std::vector<double> x(numPoints + 1), y(numPoints + 1);
        std::vector<double> u(numPoints + 1), v(numPoints + 1);
        const double step = 1.0 / numPoints;
        ax->hold(matplot::on);

        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);

            x[i] = p.x;
            y[i] = p.y;
            if (plotNormals) {
                ax->arrow(p.x, p.y, p.x + 0.1 * n.x, p.y + 0.1 * n.y)->color("red");
            }
        }

        // Plot the boundary
        ax->plot(x, y, "-k");

        if (plotCoordinateAxes) {
            // Plot the coordinate system (x and y axes)
            double minX = *std::ranges::min_element(x);
            double maxX = *std::ranges::max_element(x);
            double minY = *std::ranges::min_element(y);
            double maxY = *std::ranges::max_element(y);

            // Extend axes limits a bit for better visualization
            const double xPadding = (maxX - minX) * 0.05;
            const double yPadding = (maxY - minY) * 0.05;

            minX -= xPadding;
            maxX += xPadding;
            minY -= yPadding;
            maxY += yPadding;

            // Plot x-axis
            ax->plot({minX, maxX}, std::vector<double>{0, 0}, "--k");

            // Plot y-axis
            ax->plot({0, 0}, {minY, maxY}, "--k");
        }

        ax->hold(matplot::off);
        ax->xlabel("x");
        ax->ylabel("y");
    }
    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    }
    /**
     * @brief Calculates the axis-aligned bounding box for the semi-ellipse.
     *
     * This method determines the minimum and maximum coordinates in both x and y directions,
     * which define the smallest rectangle that fully contains the semi-ellipse.
     *
     * @param minX Reference to store the minimum x-coordinate of the bounding box.
     * @param maxX Reference to store the maximum x-coordinate of the bounding box.
     * @param minY Reference to store the minimum y-coordinate of the bounding box.
     * @param maxY Reference to store the maximum y-coordinate of the bounding box.
     *
     * @details
     * - The bounding box is calculated by evaluating the semi-ellipse at critical points where
     *   the derivative of the parametric equations is zero (the vertices of the semi-ellipse) and at the ends.
     * - Because the semi-ellipse is symmetrical about its major axis, the critical points include:
     *   - The endpoints of the semi-major axis at t = 0 and t = 1 (start and end points of the semi-ellipse arc).
     *   - The topmost and bottommost points of the semi-ellipse, occurring at t = 0.5 (the midpoint of the arc).
     * - These points are computed directly from the ellipse's parametric equations:
     *   - x(t) = center.x + a * cos(angle)
     *   - y(t) = center.y + b * sin(angle)
     *   where `angle` is adjusted based on the `t` value to cover the top half of the ellipse.
     * - The angles evaluated are at t = 0, t = 0.5, and t = 1, which correspond to the rightmost, topmost, and leftmost points respectively.
     * - Additionally, checks the semi-major and semi-minor axes' ends to ensure all critical points are considered.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        // Base angle determined by direction vector
        const double baseAngle = atan2(direction.y, direction.x) - M_PI / 2;
        // Angles corresponding to critical points in the normalized [0, 1] range
        const std::vector<double> angles {
            baseAngle,                      // angle at t=0
            baseAngle + M_PI,               // angle at t=1
            baseAngle + M_PI / 2            // angle at y max/min
        };
        // Compute extremal points
        std::vector<double> xs, ys;
        for (const double angle : angles) {
            xs.push_back(center.x + a * cos(angle));
            ys.push_back(center.y + b * sin(angle));
        }
        // Find the min and max of computed coordinates
        minX = *std::ranges::min_element(xs);
        maxX = *std::ranges::max_element(xs);
        minY = *std::ranges::min_element(ys);
        maxY = *std::ranges::max_element(ys);
    }

    /**
     * @brief Prints detailed information about the semi-ellipse boundary at specified points.
     * @param numPoints The number of points at which to print the boundary information.
     */
    void printBoundaryInfo(const int numPoints) const {
        const double step = 1.0 / numPoints;
        std::cout << "SemiEllipse Boundary Info (" << numPoints << " points):" << std::endl;
        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);
            const double s = calculateArcParameter(t);
            const double curvature = computeCurvature(t);
            std::cout << "t: " << t
                      << ", Point: [" << p.x << ", " << p.y << "]"
                      << ", Normal: [" << n.x << ", " << n.y << "]"
                      << ", Arc parameter: " << s
                      << ", Curvature: " << curvature
                      << std::endl;
        }
    }
};

class Ellipse final : public AbstractBoundary {
    Point center;
    double a; // Semi-major axis
    double b; // Semi-minor axis
public:
    /**
     * @brief Constructs an Ellipse with the given center, semi-major axis, and semi-minor axis.
     * @param center The center point of the ellipse.
     * @param a The length of the semi-major axis.
     * @param b The length of the semi-minor axis.
     */
    Ellipse(const Point &center, const double a, const double b) : center(center), a(a), b(b) {}

    /**
     * @brief Computes the point on the ellipse for a given parameter t.
     * @param t A parameter ranging from 0 to 1, where 0 and 1 correspond to the same point on the ellipse.
     * @return The point on the ellipse at parameter t.
     */
    [[nodiscard]] Point curveParametrization(const double t) const override {
        const double angle = 2 * M_PI * t;
        return {center.x + a * cos(angle), center.y + b * sin(angle)};
    }

    /**
     * @brief Calculates the outward normal vector at a given point on the ellipse.
     * @param t A parameter ranging from 0 to 1.
     * @return The normal vector at the point corresponding to parameter t.
     */
    [[nodiscard]] Point calculateNormal(const double t) const override {
        const double angle = 2 * M_PI * t;
        const double dx = -a * sin(angle);
        const double dy = b * cos(angle);
        const double length = std::hypot(dx, dy);
        return {dy / length, -dx / length};
    }

    /**
     * @brief Computes the total arc length of the ellipse.
     * @return The total arc length of the ellipse.
     */
    [[nodiscard]] double calculateArcLength() const override {
        const double k = std::sqrt(1 - (b * b) / (a * a));
        return 4 * a * gsl_sf_ellint_Ecomp(k, GSL_PREC_DOUBLE);
    }

    /**
     * @brief Computes the arc length parameter for a given normalized parameter t.
     * @param t A parameter ranging from 0 to 1.
     * @return The arc length parameter corresponding to t.
     */
    [[nodiscard]] double calculateArcParameter(const double t) const override {
        const double k = std::sqrt(1 - (b * b) / (a * a));
        const double phi = 2 * M_PI * t;
        return a * gsl_sf_ellint_E(phi, k, GSL_PREC_DOUBLE);
    }

    /**
     * @brief Computes the curvature of the ellipse at a given point.
     * @param t A parameter ranging from 0 to 1.
     * @return The curvature at the point corresponding to parameter t.
     */
    [[nodiscard]] double computeCurvature(const double t) const override {
        const double angle = 2 * M_PI * t;
        const double dx = -a * sin(angle);
        const double dy = b * cos(angle);
        const double ddx = -a * cos(angle);
        const double ddy = -b * sin(angle);
        const double numerator = std::abs(dx * ddy - dy * ddx);
        const double denominator = std::pow(dx * dx + dy * dy, 1.5);
        return numerator / denominator;
    }

    /**
     * @brief Plots the ellipse and its normals using matplot++.
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the ellipse.
     * @param plotNormals If the normals are to be plotted.
     * @param plotCoordinateAxes If the coordinate axes are to be plotted.
     */
    void plot(const matplot::axes_handle &ax, const int numPoints, const bool plotNormals, const bool plotCoordinateAxes) const override {
        using namespace matplot;
        std::vector<double> x(numPoints + 1), y(numPoints + 1);
        std::vector<double> u(numPoints + 1), v(numPoints + 1);
        const double step = 1.0 / numPoints;
        ax->hold(matplot::on);

        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);

            x[i] = p.x;
            y[i] = p.y;
            if (plotNormals) {
                ax->arrow(p.x, p.y, p.x + 0.1 * n.x, p.y + 0.1 * n.y)->color("red");
            }
        }

        // Plot the ellipse
        ax->plot(x, y, "-k");

        if (plotCoordinateAxes) {
            // Plot the coordinate system (x and y axes)
            double minX = *std::ranges::min_element(x);
            double maxX = *std::ranges::max_element(x);
            double minY = *std::ranges::min_element(y);
            double maxY = *std::ranges::max_element(y);

            // Extend axes limits a bit for better visualization
            const double xPadding = (maxX - minX) * 0.05;
            const double yPadding = (maxY - minY) * 0.05;

            minX -= xPadding;
            maxX += xPadding;
            minY -= yPadding;
            maxY += yPadding;

            // Plot x-axis
            ax->plot({minX, maxX}, std::vector<double>{0, 0}, "--k");

            // Plot y-axis
            ax->plot({0, 0}, {minY, maxY}, "--k");
        }

        ax->hold(matplot::off);
        ax->xlabel("x");
        ax->ylabel("y");
    }

    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    };

    /**
     * @brief Determines if a point is inside the ellipse.
     * @param point The point to check.
     * @return True if the point is inside the ellipse, false otherwise.
     */
    [[nodiscard]] bool isInside(const Point& point) const override {
        const double dx = (point.x - center.x) / a;
        const double dy = (point.y - center.y) / b;
        return (dx * dx + dy * dy) <= 1.0;
    }

    /**
     * @brief Calculates the bounding box that completely encloses the ellipse.
     * @param minX Reference to store the minimum x-coordinate.
     * @param maxX Reference to store the maximum x-coordinate.
     * @param minY Reference to store the minimum y-coordinate.
     * @param maxY Reference to store the maximum y-coordinate.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        minX = center.x - a;
        maxX = center.x + a;
        minY = center.y - b;
        maxY = center.y + b;
    }

    /**
     * @brief Prints detailed information about the ellipse boundary at specified points.
     * @param numPoints The number of points at which to print the boundary information.
     */
    void printBoundaryInfo(const int numPoints) const {
        const double step = 1.0 / numPoints;
        std::cout << "Ellipse Boundary Info (" << numPoints << " points):" << std::endl;
        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);
            const double s = calculateArcParameter(t);
            const double curvature = computeCurvature(t);
            std::cout << "t: " << t
                      << ", Point: [" << p.x << ", " << p.y << "]"
                      << ", Normal: [" << n.x << ", " << n.y << "]"
                      << ", Arc parameter: " << s
                      << ", Curvature: " << curvature
                      << std::endl;
        }
    }
};

/**
 * @class ProsenBilliard
 * @brief Represents a Prosen billiard boundary defined by the equation r(φ) = 1 + a cos(4φ).
 *
 * This class inherits from AbstractBoundary and defines a Prosen billiard using parametric equations.
 * It provides methods to calculate normal vectors, arc length, arc parameters, and curvature.
 *
 * @param a The amplitude parameter that modifies the shape of the billiard.
 */
class ProsenBilliard final : public AbstractBoundary {
    double a; // Amplitude parameter
public:
    /**
     * @brief Constructs a ProsenBilliard with the given amplitude parameter.
     * @param a The amplitude parameter.
     */
    explicit ProsenBilliard(const double a) : a(a) {}

    /**
     * @brief Computes the point on the Prosen billiard for a given parameter t.
     * @param t A parameter ranging from 0 to 1, where 0 and 1 correspond to the same point on the billiard.
     * @return The point on the Prosen billiard at parameter t.
     */
    [[nodiscard]] Point curveParametrization(const double t) const override {
        const double phi = 2 * M_PI * t;
        const double r = 1 + a * std::cos(4 * phi);
        return {r * std::cos(phi), r * std::sin(phi)};
    }

    /**
     * @brief Calculates the outward normal vector at a given point on the Prosen billiard.
     * @param t A parameter ranging from 0 to 1.
     * @return The normal vector at the point corresponding to parameter t.
     */
    [[nodiscard]] Point calculateNormal(const double t) const override {
        double dx_dt, dy_dt;
        calculateTangents(t, dx_dt, dy_dt);
        const double length = std::hypot(dx_dt, dy_dt);
        return {dy_dt / length, -dx_dt / length};
    }

    /**
     * @brief Computes the total arc length of the Prosen billiard.
     * @return The total arc length of the Prosen billiard.
     */
    [[nodiscard]] double calculateArcLength() const override {
        return calculateArcParameter(1.0);
    }

    /**
     * @brief Computes the arc length parameter for a given normalized parameter t.
     * @param t A parameter ranging from 0 to 1.
     * @return The arc length parameter corresponding to t.
     */
    [[nodiscard]] double calculateArcParameter(const double t) const override {
        gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(1000);
        double result, error;

        auto integrand = [](const double t, void *params) -> double {
            const auto *prosen = static_cast<ProsenBilliard *>(params);
            double dx_dt, dy_dt;
            prosen->calculateTangents(t, dx_dt, dy_dt);
            return std::hypot(dx_dt, dy_dt);
        };

        gsl_function F;
        F.function = integrand;
        F.params = const_cast<ProsenBilliard *>(this);

        size_t limit = 1000;
        constexpr double start = 0;
        const double end = t;

        while (gsl_integration_qags(&F, start, end, 0, 1e-8, limit, workspace, &result, &error) == GSL_ERANGE) {
            limit += 1000;
            gsl_integration_workspace_free(workspace);
            workspace = gsl_integration_workspace_alloc(limit);
        }

        gsl_integration_workspace_free(workspace);
        return result;
    }

    /**
     * @brief Computes the curvature of the Prosen billiard at a given point.
     * @param t A parameter ranging from 0 to 1.
     * @return The curvature at the point corresponding to parameter t.
     */
    [[nodiscard]] double computeCurvature(const double t) const override {
        double dx_dt, dy_dt;
        calculateTangents(t, dx_dt, dy_dt);

        double d2x_dt2, d2y_dt2;
        calculateSecondDerivatives(t, d2x_dt2, d2y_dt2);

        const double numerator = std::abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2);
        const double denominator = std::pow(dx_dt * dx_dt + dy_dt * dy_dt, 1.5);
        return numerator / denominator;
    }

    /**
     * @brief Plots the Prosen billiard and its normals using matplot++.
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the billiard.
     * @param plotNormals If the normals are to be plotted.
     * @param plotCoordinateAxes If the coordinate axes are to be plotted.
     */
    void plot(const matplot::axes_handle &ax, const int numPoints, const bool plotNormals, const bool plotCoordinateAxes) const override {
        using namespace matplot;
        std::vector<double> x(numPoints + 1), y(numPoints + 1);
        std::vector<double> u(numPoints + 1), v(numPoints + 1);
        const double step = 1.0 / numPoints;
        ax->hold(matplot::on);

        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);

            x[i] = p.x;
            y[i] = p.y;
            if (plotNormals) {
                ax->arrow(p.x, p.y, p.x + 0.1 * n.x, p.y + 0.1 * n.y)->color("red");
            }
        }

        // Plot the Prosen billiard
        ax->plot(x, y, "-k");

        if (plotCoordinateAxes) {
            // Plot the coordinate system (x and y axes)
            double minX = *std::ranges::min_element(x);
            double maxX = *std::ranges::max_element(x);
            double minY = *std::ranges::min_element(y);
            double maxY = *std::ranges::max_element(y);

            // Extend axes limits a bit for better visualization
            const double xPadding = (maxX - minX) * 0.05;
            const double yPadding = (maxY - minY) * 0.05;

            minX -= xPadding;
            maxX += xPadding;
            minY -= yPadding;
            maxY += yPadding;

            // Plot x-axis
            ax->plot({minX, maxX}, std::vector<double>{0, 0}, "--k");

            // Plot y-axis
            ax->plot({0, 0}, {minY, maxY}, "--k");
        }

        ax->hold(matplot::off);
        ax->xlabel("x");
        ax->ylabel("y");
    }

    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    };

    /**
     * @brief Determines if a point is inside the Prosen billiard.
     * @param point The point to check.
     * @return True if the point is inside the Prosen billiard, false otherwise.
     */
    [[nodiscard]] bool isInside(const Point& point) const override {
        const double t = std::atan2(point.y, point.x) / (2 * M_PI);
        const double r = 1 + a * std::cos(4 * t * 2 * M_PI);
        return std::hypot(point.x, point.y) <= r;
    }

    /**
     * @brief Calculates the bounding box that completely encloses the Prosen billiard.
     * @param minX Reference to store the minimum x-coordinate.
     * @param maxX Reference to store the maximum x-coordinate.
     * @param minY Reference to store the minimum y-coordinate.
     * @param maxY Reference to store the maximum y-coordinate.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        minX = -1 - a;
        maxX = 1 + a;
        minY = -1 - a;
        maxY = 1 + a;
    }

    /**
     * @brief Prints detailed information about the Prosen billiard boundary at specified points.
     * @param numPoints The number of points at which to print the boundary information.
     */
    void printBoundaryInfo(const int numPoints) const {
        const double step = 1.0 / numPoints;
        std::cout << "Prosen Billiard Boundary Info (" << numPoints << " points):" << std::endl;
        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);
            const double s = calculateArcParameter(t);
            const double curvature = computeCurvature(t);
            std::cout << "t: " << t
                      << ", Point: [" << p.x << ", " << p.y << "]"
                      << ", Normal: [" << n.x << ", " << n.y << "]"
                      << ", Arc parameter: " << s
                      << ", Curvature: " << curvature
                      << std::endl;
        }
    }

private:
    /**
    * @brief Calculates the first derivatives of x and y with respect to t.
    * @param t The parameter t.
    * @param dx_dt The first derivative of x with respect to t.
    * @param dy_dt The first derivative of y with respect to t.
    */
    void calculateTangents(double t, double &dx_dt, double &dy_dt) const {
        dx_dt = -2 * M_PI * (1 + a * std::cos(8 * M_PI * t)) * std::sin(2 * M_PI * t) - 8 * a * M_PI * std::cos(2 * M_PI * t) * std::sin(8 * M_PI * t);
        dy_dt = 2 * M_PI * (1 + a * std::cos(8 * M_PI * t)) * std::cos(2 * M_PI * t) - 8 * a * M_PI * std::sin(2 * M_PI * t) * std::sin(8 * M_PI * t);
    }

    /**
    * @brief Calculates the second derivatives of x and y with respect to t.
    * @param t The parameter t.
    * @param d2x_dt2 The second derivative of x with respect to t.
    * @param d2y_dt2 The second derivative of y with respect to t.
    */
    void calculateSecondDerivatives(double t, double &d2x_dt2, double &d2y_dt2) const {
        d2x_dt2 = -4 * M_PI * M_PI * (1 + a * std::cos(8 * M_PI * t)) * std::cos(2 * M_PI * t) - 64 * a * M_PI * M_PI * std::cos(2 * M_PI * t) * std::cos(8 * M_PI * t) + 32 * a * M_PI * M_PI * std::sin(2 * M_PI * t) * std::sin(8 * M_PI * t);
        d2y_dt2 = -4 * M_PI * M_PI * (1 + a * std::cos(8 * M_PI * t)) * std::sin(2 * M_PI * t) - 64 * a * M_PI * M_PI * std::sin(2 * M_PI * t) * std::cos(8 * M_PI * t) - 32 * a * M_PI * M_PI * std::cos(2 * M_PI * t) * std::sin(8 * M_PI * t);
    }
};

class QuarterProsenBilliard final : public AbstractBoundary {
    double a; // Amplitude parameter
public:
    /**
     * @brief Constructs a Quarter Prosen billiard with the given parameter a.
     * @param a The parameter that affects the shape of the billiard.
     */
    explicit QuarterProsenBilliard(const double a) : a(a) {}

    /**
     * @brief Computes the point on the Quarter Prosen billiard for a given parameter t.
     * @param t A parameter ranging from 0 to 1.
     * @return The point on the billiard at parameter t.
     */
    [[nodiscard]] Point curveParametrization(const double t) const override {
        const double phi = M_PI * t / 2;
        const double r = 1 + a * cos(4 * phi);
        return {r * cos(phi), r * sin(phi)};
    }

    /**
     * @brief Calculates the outward normal vector at a given point on the Prosen billiard.
     * @param t A parameter ranging from 0 to 1.
     * @return The normal vector at the point corresponding to parameter t.
     */
    [[nodiscard]] Point calculateNormal(const double t) const override {
        double dx_dt, dy_dt;
        calculateTangents(t, dx_dt, dy_dt);
        const double length = std::hypot(dx_dt, dy_dt);
        return {dy_dt / length, -dx_dt / length};
    }

    /**
     * @brief Computes the total arc length of the Prosen billiard.
     * @return The total arc length of the Prosen billiard.
     */
    [[nodiscard]] double calculateArcLength() const override {
        return calculateArcParameter(1.0);
    }

    /**
     * @brief Computes the arc length parameter for a given normalized parameter t.
     * @param t A parameter ranging from 0 to 1.
     * @return The arc length parameter corresponding to t.
     */
    [[nodiscard]] double calculateArcParameter(const double t) const override {
        gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(1000);
        double result, error;

        auto integrand = [](const double t, void *params) -> double {
            const auto *prosen = static_cast<QuarterProsenBilliard *>(params);
            double dx_dt, dy_dt;
            prosen->calculateTangents(t, dx_dt, dy_dt);
            return std::hypot(dx_dt, dy_dt);
        };

        gsl_function F;
        F.function = integrand;
        F.params = const_cast<QuarterProsenBilliard*>(this);

        size_t limit = 1000;
        constexpr double start = 0;
        const double end = t;

        while (gsl_integration_qags(&F, start, end, 0, 1e-8, limit, workspace, &result, &error) == GSL_ERANGE) {
            limit += 1000;
            gsl_integration_workspace_free(workspace);
            workspace = gsl_integration_workspace_alloc(limit);
        }

        gsl_integration_workspace_free(workspace);
        return result;
    }

    /**
     * @brief Computes the curvature of the Prosen billiard at a given point.
     * @param t A parameter ranging from 0 to 1.
     * @return The curvature at the point corresponding to parameter t.
     */
    [[nodiscard]] double computeCurvature(const double t) const override {
        double dx_dt, dy_dt;
        calculateTangents(t, dx_dt, dy_dt);

        double d2x_dt2, d2y_dt2;
        calculateSecondDerivatives(t, d2x_dt2, d2y_dt2);

        const double numerator = std::abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2);
        const double denominator = std::pow(dx_dt * dx_dt + dy_dt * dy_dt, 1.5);
        return numerator / denominator;
    }

    /**
     * @brief Plots the Quarter Prosen billiard and its normals using matplot++.
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the billiard.
     * @param plotNormals If the normals are to be plotted.
     * @param plotCoordinateAxes If the coordinate axes are to be plotted.
     */
    void plot(const matplot::axes_handle &ax, const int numPoints, const bool plotNormals, const bool plotCoordinateAxes) const override {
        using namespace matplot;
        std::vector<double> x(numPoints + 1), y(numPoints + 1);
        std::vector<double> u(numPoints + 1), v(numPoints + 1);
        const double step = 1.0 / numPoints;
        ax->hold(matplot::on);

        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);

            x[i] = p.x;
            y[i] = p.y;
            if (plotNormals) {
                ax->arrow(p.x, p.y, p.x + 0.1 * n.x, p.y + 0.1 * n.y)->color("red");
            }
        }

        // Plot the Prosen billiard
        ax->plot(x, y, "-k");

        if (plotCoordinateAxes) {
            // Plot the coordinate system (x and y axes)
            double minX = *std::ranges::min_element(x);
            double maxX = *std::ranges::max_element(x);
            double minY = *std::ranges::min_element(y);
            double maxY = *std::ranges::max_element(y);

            // Extend axes limits a bit for better visualization
            const double xPadding = (maxX - minX) * 0.05;
            const double yPadding = (maxY - minY) * 0.05;

            minX -= xPadding;
            maxX += xPadding;
            minY -= yPadding;
            maxY += yPadding;

            // Plot x-axis
            ax->plot({minX, maxX}, std::vector<double>{0, 0}, "--k");

            // Plot y-axis
            ax->plot({0, 0}, {minY, maxY}, "--k");
        }

        ax->hold(matplot::off);
        ax->xlabel("x");
        ax->ylabel("y");
    }

    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    };

    /**
     * @brief Determines if a point is inside the Quarter Prosen billiard.
     * @param point The point to check.
     * @return True if the point is inside the billiard, false otherwise.
     */
    [[nodiscard]] bool isInside(const Point& point) const override {
        if (point.x < 0 || point.y < 0) return false;
        const double t = std::atan2(point.y, point.x) / (M_PI / 2);
        const Point p = curveParametrization(t);
        return point.x * point.x + point.y * point.y <= p.x * p.x + p.y * p.y;
    }

    /**
     * @brief Calculates the bounding box that completely encloses the Quarter Prosen billiard.
     * @param minX Reference to store the minimum x-coordinate.
     * @param maxX Reference to store the maximum x-coordinate.
     * @param minY Reference to store the minimum y-coordinate.
     * @param maxY Reference to store the maximum y-coordinate.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        minX = 0;
        maxX = 1 + a;
        minY = 0;
        maxY = 1 + a;
    }

    /**
     * @brief Prints detailed information about the Quarter Prosen billiard boundary at specified points.
     * @param numPoints The number of points at which to print the boundary information.
     */
    void printBoundaryInfo(const int numPoints) const {
        const double step = 1.0 / numPoints;
        std::cout << "Quarter Prosen Billiard Boundary Info (" << numPoints << " points):" <<std::endl;
        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);
            const double s = calculateArcParameter(t);
            const double curvature = computeCurvature(t);
            std::cout << "t: " << t
                      << ", Point: [" << p.x << ", " << p.y << "]"
                      << ", Normal: [" << n.x << ", " << n.y << "]"
                      << ", Arc parameter: " << s
                      << ", Curvature: " << curvature
                      << std::endl;
        }
    }

private:
    /**
    * @brief Calculates the first derivatives of x and y with respect to t.
    * @param t The parameter t.
    * @param dx_dt The first derivative of x with respect to t.
    * @param dy_dt The first derivative of y with respect to t.
    */
    void calculateTangents(const double t, double &dx_dt, double &dy_dt) const {
        const double phi = M_PI * t / 2;
        constexpr double dphi_dt = M_PI / 2;
        const double cosPhi = std::cos(phi);
        const double sinPhi = std::sin(phi);
        const double cos4Phi = std::cos(4 * phi);
        const double sin4Phi = std::sin(4 * phi);
        dx_dt = dphi_dt * (-a * 4 * sin4Phi * cosPhi + (1 + a * cos4Phi) * (-sinPhi));
        dy_dt = dphi_dt * (-a * 4 * sin4Phi * sinPhi + (1 + a * cos4Phi) * cosPhi);
    }

    /**
     * @brief Calculates the second derivatives of x and y with respect to t.
     * @param t The parameter t.
     * @param d2x_dt2 The second derivative of x with respect to t.
     * @param d2y_dt2 The second derivative of y with respect to t.
     */
    void calculateSecondDerivatives(const double t, double &d2x_dt2, double &d2y_dt2) const {
        const double phi = M_PI * t / 2;
        const double cos_phi = std::cos(phi);
        const double sin_phi = std::sin(phi);
        const double cos_2pi_t = std::cos(2 * M_PI * t);
        const double sin_2pi_t = std::sin(2 * M_PI * t);

        d2x_dt2 = 2 * M_PI * M_PI * a * sin_phi * sin_2pi_t - 4 * M_PI * M_PI * a * cos_phi * cos_2pi_t
                  - (M_PI * M_PI / 4) * (a * cos_2pi_t + 1) * cos_phi;

        d2y_dt2 = -4 * M_PI * M_PI * a * sin_phi * cos_2pi_t - 2 * M_PI * M_PI * a * sin_2pi_t * cos_phi
                  - (M_PI * M_PI / 4) * (a * cos_2pi_t + 1) * sin_phi;
    }
};

/**
 * @class RobnikBilliard
 * @brief Represents a Robnik billiard boundary defined by the parameter ε.
 *
 * This class inherits from AbstractBoundary and defines a Robnik billiard using parametric equations.
 * It provides methods to calculate normal vectors, arc length, arc parameters, and curvature.
 *
 * @param epsilon The parameter that modifies the shape of the Robnik billiard.
 */
class RobnikBilliard final : public AbstractBoundary {
    double epsilon; // Shape parameter
public:
    /**
     * @brief Constructs a RobnikBilliard with the given epsilon parameter.
     * @param epsilon The shape parameter.
     */
    explicit RobnikBilliard(const double epsilon) : epsilon(epsilon) {}

    /**
     * @brief Computes the point on the Robnik billiard for a given parameter t.
     * @param t A parameter ranging from 0 to 1, where 0 and 1 correspond to the same point on the billiard.
     * @return The point on the Robnik billiard at parameter t.
     */
    [[nodiscard]] Point curveParametrization(const double t) const override {
        const double theta = 2 * M_PI * t;
        const double X = (1 + epsilon * std::cos(theta)) * std::cos(theta);
        const double Y = (1 + epsilon * std::cos(theta)) * std::sin(theta);
        return {X, Y};
    }

    /**
     * @brief Calculates the outward normal vector at a given point on the Robnik billiard.
     * @param t A parameter ranging from 0 to 1.
     * @return The normal vector at the point corresponding to parameter t.
     */
    [[nodiscard]] Point calculateNormal(const double t) const override {
        double dx_dt, dy_dt;
        calculateTangents(t, dx_dt, dy_dt);
        const double length = std::hypot(dx_dt, dy_dt);
        return {dy_dt / length, -dx_dt / length};
    }

    /**
     * @brief Computes the total arc length of the Robnik billiard.
     * @return The total arc length of the Robnik billiard.
     */
    [[nodiscard]] double calculateArcLength() const override {
        return calculateArcParameter(1.0);
    }

    /**
     * @brief Computes the arc length parameter for a given normalized parameter t.
     * @param t A parameter ranging from 0 to 1.
     * @return The arc length parameter corresponding to t.
     */
    [[nodiscard]] double calculateArcParameter(const double t) const override {
        gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(1000);
        double result, error;

        auto integrand = [](const double t, void *params) -> double {
            const auto *robnik = static_cast<RobnikBilliard *>(params);
            double dx_dt, dy_dt;
            robnik->calculateTangents(t, dx_dt, dy_dt);
            return std::hypot(dx_dt, dy_dt);
        };

        gsl_function F;
        F.function = integrand;
        F.params = const_cast<RobnikBilliard *>(this);

        size_t limit = 1000;
        constexpr double start = 0;
        const double end = t;

        while (gsl_integration_qags(&F, start, end, 0, 1e-8, limit, workspace, &result, &error) == GSL_ERANGE) {
            limit += 1000;
            gsl_integration_workspace_free(workspace);
            workspace = gsl_integration_workspace_alloc(limit);
        }

        gsl_integration_workspace_free(workspace);
        return result;
    }

    /**
     * @brief Computes the curvature of the Robnik billiard at a given point.
     * @param t A parameter ranging from 0 to 1.
     * @return The curvature at the point corresponding to parameter t.
     */
    [[nodiscard]] double computeCurvature(const double t) const override {
        double dx_dt, dy_dt;
        calculateTangents(t, dx_dt, dy_dt);

        double d2x_dt2, d2y_dt2;
        calculateSecondDerivatives(t, d2x_dt2, d2y_dt2);

        const double numerator = std::abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2);
        const double denominator = std::pow(dx_dt * dx_dt + dy_dt * dy_dt, 1.5);
        return numerator / denominator;
    }

    /**
     * @brief Plots the Robnik billiard and its normals using matplot++.
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the billiard.
     * @param plotNormals If the normals are to be plotted.
     * @param plotCoordinateAxes If the coordinate axes are to be plotted.
     */
    void plot(const matplot::axes_handle &ax, const int numPoints, const bool plotNormals, const bool plotCoordinateAxes) const override {
        using namespace matplot;
        std::vector<double> x(numPoints + 1), y(numPoints + 1);
        std::vector<double> u(numPoints + 1), v(numPoints + 1);
        const double step = 1.0 / numPoints;
        ax->hold(matplot::on);

        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);

            x[i] = p.x;
            y[i] = p.y;
            if (plotNormals) {
                ax->arrow(p.x, p.y, p.x + 0.1 * n.x, p.y + 0.1 * n.y)->color("red");
            }
        }

        // Plot the Robnik billiard
        ax->plot(x, y, "-k");

        if (plotCoordinateAxes) {
            // Plot the coordinate system (x and y axes)
            double minX = *std::ranges::min_element(x);
            double maxX = *std::ranges::max_element(x);
            double minY = *std::ranges::min_element(y);
            double maxY = *std::ranges::max_element(y);

            // Extend axes limits a bit for better visualization
            const double xPadding = (maxX - minX) * 0.05;
            const double yPadding = (maxY - minY) * 0.05;

            minX -= xPadding;
            maxX += xPadding;
            minY -= yPadding;
            maxY += yPadding;

            // Plot x-axis
            ax->plot({minX, maxX}, std::vector<double>{0, 0}, "--k");

            // Plot y-axis
            ax->plot({0, 0}, {minY, maxY}, "--k");
        }

        ax->hold(matplot::off);
        ax->xlabel("x");
        ax->ylabel("y");
    }

    /**
     * @brief Determines if a point is inside the Robnik billiard.
     * @param point The point to check.
     * @return True if the point is inside the Robnik billiard, false otherwise.
     */
    [[nodiscard]] bool isInside(const Point& point) const override {
        const double t = std::atan2(point.y, point.x) / (2 * M_PI);
        const double r = (1 + epsilon * std::cos(2 * M_PI * t));
        return std::hypot(point.x, point.y) <= r;
    }

    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    };

    /**
     * @brief Calculates the bounding box that completely encloses the Robnik billiard. This is a very crude approach.
     * @param minX Reference to store the minimum x-coordinate.
     * @param maxX Reference to store the maximum x-coordinate.
     * @param minY Reference to store the minimum y-coordinate.
     * @param maxY Reference to store the maximum y-coordinate.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        constexpr int numPoints = 1000000;
        std::vector<Point> points;
        for (int i = 0; i <= numPoints; ++i) {
            const double t = static_cast<double>(i) / numPoints;
            points.push_back(curveParametrization(t));
        }

        minX = points[0].x;
        maxX = points[0].x;
        minY = points[0].y;
        maxY = points[0].y;

        for (const auto& point : points) {
            if (point.x < minX) minX = point.x;
            if (point.x > maxX) maxX = point.x;
            if (point.y < minY) minY = point.y;
            if (point.y > maxY) maxY = point.y;
        }
    }

    /**
     * @brief Prints detailed information about the Robnik billiard boundary at specified points.
     * @param numPoints The number of points at which to print the boundary information.
     */
    void printBoundaryInfo(const int numPoints) const {
        const double step = 1.0 / numPoints;
        std::cout << "Robnik Billiard Boundary Info (" << numPoints << " points):" << std::endl;
        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);
            const double s = calculateArcParameter(t);
            const double curvature = computeCurvature(t);
            std::cout << "t: " << t
                        << ", Point: [" << p.x << ", " << p.y << "]"
                        << ", Normal: [" << n.x << ", " << n.y << ", Normal: [" << n.x << ", " << n.y << "]"
                        << ", Arc parameter: " << s
                        << ", Curvature: " << curvature
                        << std::endl;
        }
    }

private:
    /**
    * @brief Calculates the first derivatives of x and y with respect to t.
    * @param t The parameter t.
    * @param dx_dt The first derivative of x with respect to t.
    * @param dy_dt The first derivative of y with respect to t.
    */
    void calculateTangents(const double t, double &dx_dt, double &dy_dt) const {
        const double theta = 2 * M_PI * t;
        dx_dt = -2 * M_PI * epsilon * std::sin(theta) * std::cos(theta) - 2 * M_PI * (epsilon * std::cos(theta) + 1) * std::sin(theta);
        dy_dt = -2 * M_PI * epsilon * std::sin(theta) * std::sin(theta) + 2 * M_PI * (epsilon * std::cos(theta) + 1) * std::cos(theta);
    }

    /**
    * @brief Calculates the second derivatives of x and y with respect to t.
    * @param t The parameter t.
    * @param d2x_dt2 The second derivative of x with respect to t.
    * @param d2y_dt2 The second derivative of y with respect to t.
    */
    void calculateSecondDerivatives(const double t, double &d2x_dt2, double &d2y_dt2) const {
        const double theta = 2 * M_PI * t;
        d2x_dt2 = 8 * M_PI * M_PI * epsilon * std::sin(theta) * std::sin(theta) - 4 * M_PI * M_PI * epsilon * std::cos(theta) * std::cos(theta) - 4 * M_PI * M_PI * (epsilon * std::cos(theta) + 1) * std::cos(theta);
        d2y_dt2 = -12 * M_PI * M_PI * epsilon * std::sin(theta) * std::cos(theta) - 4 * M_PI * M_PI * (epsilon * std::cos(theta) + 1) * std::sin(theta);
    }
};

class C3CurveFull final : public AbstractBoundary {
    double a; // Amplitude parameter

public:
    /**
     * @brief Constructs a C3Curve with the given parameter a.
     * @param a The parameter that affects the shape of the curve.
     */
    explicit C3CurveFull(const double a) : a(a) {}

    /**
     * @brief Computes the point on the C3Curve for a given parameter t.
     * @param t A parameter ranging from 0 to 1.
     * @return The point on the curve at parameter t.
     */
    [[nodiscard]] Point curveParametrization(const double t) const override {
        const double phi = 2 * M_PI * t;
        const double r = 0.5 * (1 + a * (std::cos(3 * phi) - std::sin(6 * phi)));
        return {r * std::cos(phi), r * std::sin(phi)};
    }

    /**
     * @brief Calculates the outward normal vector at a given point on the C3Curve.
     * @param t A parameter ranging from 0 to 1.
     * @return The normal vector at the point corresponding to parameter t.
     */
    [[nodiscard]] Point calculateNormal(const double t) const override {
        double dx_dt, dy_dt;
        calculateTangents(t, dx_dt, dy_dt);
        const double length = std::hypot(dx_dt, dy_dt);
        return {dy_dt / length, -dx_dt / length};
    }

    /**
     * @brief Computes the total arc length of the C3Curve.
     * @return The total arc length of the C3Curve.
     */
    [[nodiscard]] double calculateArcLength() const override {
        return calculateArcParameter(1.0);
    }

    /**
     * @brief Computes the arc length parameter for a given normalized parameter t.
     * @param t A parameter ranging from 0 to 1.
     * @return The arc length parameter corresponding to t.
     */
    [[nodiscard]] double calculateArcParameter(const double t) const override {
        gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(1000);
        double result, error;

        auto integrand = [](const double t, void *params) -> double {
            const auto *curve = static_cast<C3CurveFull *>(params);
            double dx_dt, dy_dt;
            curve->calculateTangents(t, dx_dt, dy_dt);
            return std::hypot(dx_dt, dy_dt);
        };

        gsl_function F;
        F.function = integrand;
        F.params = const_cast<C3CurveFull*>(this);

        size_t limit = 1000;
        constexpr double start = 0;
        const double end = t;

        while (gsl_integration_qags(&F, start, end, 0, 1e-8, limit, workspace, &result, &error) == GSL_ERANGE) {
            limit += 1000;
            gsl_integration_workspace_free(workspace);
            workspace = gsl_integration_workspace_alloc(limit);
        }

        gsl_integration_workspace_free(workspace);
        return result;
    }

    /**
     * @brief Computes the curvature of the C3Curve at a given point.
     * @param t A parameter ranging from 0 to 1.
     * @return The curvature at the point corresponding to parameter t.
     */
    [[nodiscard]] double computeCurvature(const double t) const override {
        double dx_dt, dy_dt;
        calculateTangents(t, dx_dt, dy_dt);

        double d2x_dt2, d2y_dt2;
        calculateSecondDerivatives(t, d2x_dt2, d2y_dt2);

        const double numerator = std::abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2);
        const double denominator = std::pow(dx_dt * dx_dt + dy_dt * dy_dt, 1.5);
        return numerator / denominator;
    }

    /**
     * @brief Plots the C3Curve and its normals using matplot++.
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the curve.
     * @param plotNormals If the normals are to be plotted.
     * @param plotCoordinateAxes If the coordinate axes are to be plotted.
     */
    void plot(const matplot::axes_handle &ax, const int numPoints, const bool plotNormals, const bool plotCoordinateAxes) const override {
        using namespace matplot;
        std::vector<double> x(numPoints + 1), y(numPoints + 1);
        std::vector<double> u(numPoints + 1), v(numPoints + 1);
        const double step = 1.0 / numPoints;
        ax->hold(matplot::on);

        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);

            x[i] = p.x;
            y[i] = p.y;
            if (plotNormals) {
                ax->arrow(p.x, p.y, p.x + 0.1 * n.x, p.y + 0.1 * n.y)->color("red");
            }
        }

        // Plot the curve
        ax->plot(x, y, "-k");

        if (plotCoordinateAxes) {
            // Plot the coordinate system (x and y axes)
            double minX = *std::ranges::min_element(x);
            double maxX = *std::ranges::max_element(x);
            double minY = *std::ranges::min_element(y);
            double maxY = *std::ranges::max_element(y);

            // Extend axes limits a bit for better visualization
            const double xPadding = (maxX - minX) * 0.05;
            const double yPadding = (maxY - minY) * 0.05;

            minX -= xPadding;
            maxX += xPadding;
            minY -= yPadding;
            maxY += yPadding;

            // Plot x-axis
            ax->plot({minX, maxX}, std::vector<double>{0, 0}, "--k");

            // Plot y-axis
            ax->plot({0, 0}, {minY, maxY}, "--k");
        }

        ax->hold(matplot::off);
        ax->xlabel("x");
        ax->ylabel("y");
    }

    /**
     * @brief Determines if a point is inside the C3Curve.
     * @param point The point to check.
     * @return True if the point is inside the curve, false otherwise.
     */
    [[nodiscard]] bool isInside(const Point& point) const override {
        const double t = std::atan2(point.y, point.x) / (2 * M_PI);
        const Point p = curveParametrization(t);
        return point.x * point.x + point.y * point.y <= p.x * p.x + p.y * p.y;
    }

    /**
    * @brief Finds the extreme values of the curve to determine the bounding box.
    * @param minX Reference to store the minimum x-coordinate.
    * @param minY Reference to store the minimum y-coordinate.
    * @param maxX Reference to store the maximum x-coordinate.
    * @param maxY Reference to store the maximum y-coordinate.
    */
    void findExtreme(double &minX, double &minY, double &maxX, double &maxY) const {
        constexpr int num_points = 100000;
        minX = std::numeric_limits<double>::max();
        maxX = std::numeric_limits<double>::lowest();
        minY = std::numeric_limits<double>::max();
        maxY = std::numeric_limits<double>::lowest();

        for (int i = 0; i < num_points; ++i) {
            const double t = static_cast<double>(i) / (num_points - 1);
            const Point p = curveParametrization(t);
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        }
    }

    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    };

    /**
     * @brief Prints detailed information about the C3Curve boundary at specified points.
     * @param numPoints The number of points at which to print the boundary information.
     */
    void printBoundaryInfo(const int numPoints) const {
        const double step = 1.0 / numPoints;
        std::cout << "C3CurveFull Boundary Info (" << numPoints << " points):" << std::endl;
        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);
            const double s = calculateArcParameter(t);
            const double curvature = computeCurvature(t);
            std::cout << "t: " << t
                      << ", Point: [" << p.x << ", " << p.y << "]"
                      << ", Normal: [" << n.x << ", " << n.y << "]"
                      << ", Arc parameter: " << s
                      << ", Curvature: " << curvature
                      << std::endl;
        }
    }

private:
    /**
    * @brief Calculates the first derivatives of x and y with respect to t.
    * @param t The parameter t.
    * @param dx_dt The first derivative of x with respect to t.
    * @param dy_dt The first derivative of y with respect to t.
    */
    void calculateTangents(const double t, double &dx_dt, double &dy_dt) const {
        dx_dt = M_PI * (-3.0 * a * (sin(6 * M_PI * t) + 2 * cos(12 * M_PI * t)) * cos(2 * M_PI * t) + (a * (sin(12 * M_PI * t) - cos(6 * M_PI * t)) - 1) * sin(2 * M_PI * t));
        dy_dt = M_PI * (-3.0 * a * (sin(6 * M_PI * t) + 2 * cos(12 * M_PI * t)) * sin(2 * M_PI * t) - (a * (sin(12 * M_PI * t) - cos(6 * M_PI * t)) - 1) * cos(2 * M_PI * t));
    }

    /**
    * @brief Calculates the second derivatives of x and y with respect to t.
    * @param t The parameter t.
    * @param d2x_dt2 The second derivative of x with respect to t.
    * @param d2y_dt2 The second derivative of y with respect to t.
    */
    void calculateSecondDerivatives(const double t, double &d2x_dt2, double &d2y_dt2) const {
        d2x_dt2 = M_PI * M_PI * (12.0 * a * (sin(6 * M_PI * t) + 2 * cos(12 * M_PI * t)) * sin(2 * M_PI * t) + 18.0 * a * (4 * sin(12 * M_PI * t) - cos(6 * M_PI * t)) * cos(2 * M_PI * t) + 2.0 * (a * (sin(12 * M_PI * t) - cos(6 * M_PI * t)) - 1) * cos(2 * M_PI * t));
        d2y_dt2 = M_PI * M_PI * (-12.0 * a * (sin(6 * M_PI * t) + 2 * cos(12 * M_PI * t)) * cos(2 * M_PI * t) + 18.0 * a * (4 * sin(12 * M_PI * t) - cos(6 * M_PI * t)) * sin(2 * M_PI * t) + 2.0 * (a * (sin(12 * M_PI * t) - cos(6 * M_PI * t)) - 1) * sin(2 * M_PI * t));
    }

public:
    void getBoundingBox(double &minX, double &maxX, double &minY, double &maxY) const override {
        findExtreme(minX, minY, maxX, maxY);
    };
};

class C3CurveDesymmetrized final : public AbstractBoundary {
    double a; // Amplitude parameter

public:
    /**
     * @brief Constructs a C3CurveDesymmetrized with the given parameter a.
     * @param a The parameter that affects the shape of the curve.
     */
    explicit C3CurveDesymmetrized(const double a) : a(a) {}

    /**
     * @brief Computes the point on the C3CurveDesymmetrized for a given parameter t.
     * @param t A parameter ranging from 0 to 1.
     * @return The point on the curve at parameter t.
     */
    [[nodiscard]] Point curveParametrization(const double t) const override {
        const double phi = 2 * M_PI * t / 3.0;
        const double r = 0.5 * (1 + a * (std::cos(3 * phi) - std::sin(6 * phi)));
        return {r * std::cos(phi), r * std::sin(phi)};
    }

    /**
     * @brief Calculates the outward normal vector at a given point on the C3CurveDesymmetrized.
     * @param t A parameter ranging from 0 to 1.
     * @return The normal vector at the point corresponding to parameter t.
     */
    [[nodiscard]] Point calculateNormal(const double t) const override {
        double dx_dt, dy_dt;
        calculateTangents(t, dx_dt, dy_dt);
        const double length = std::hypot(dx_dt, dy_dt);
        return {dy_dt / length, -dx_dt / length};
    }

    /**
     * @brief Computes the total arc length of the C3CurveDesymmetrized.
     * @return The total arc length of the C3CurveDesymmetrized.
     */
    [[nodiscard]] double calculateArcLength() const override {
        return calculateArcParameter(1.0);
    }

    /**
     * @brief Computes the arc length parameter for a given normalized parameter t.
     * @param t A parameter ranging from 0 to 1.
     * @return The arc length parameter corresponding to t.
     */
    [[nodiscard]] double calculateArcParameter(const double t) const override {
        gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(1000);
        double result, error;

        auto integrand = [](const double t, void *params) -> double {
            const auto *curve = static_cast<C3CurveDesymmetrized *>(params);
            double dx_dt, dy_dt;
            curve->calculateTangents(t, dx_dt, dy_dt);
            return std::hypot(dx_dt, dy_dt);
        };

        gsl_function F;
        F.function = integrand;
        F.params = const_cast<C3CurveDesymmetrized*>(this);

        size_t limit = 1000;
        constexpr double start = 0;
        const double end = t;

        while (gsl_integration_qags(&F, start, end, 0, 1e-8, limit, workspace, &result, &error) == GSL_ERANGE) {
            limit += 1000;
            gsl_integration_workspace_free(workspace);
            workspace = gsl_integration_workspace_alloc(limit);
        }

        gsl_integration_workspace_free(workspace);
        return 1.0/3.0*result;
    }

    /**
     * @brief Computes the curvature of the C3CurveDesymmetrized at a given point.
     * @param t A parameter ranging from 0 to 1.
     * @return The curvature at the point corresponding to parameter t.
     */
    [[nodiscard]] double computeCurvature(const double t) const override {
        double dx_dt, dy_dt;
        calculateTangents(t, dx_dt, dy_dt);

        double d2x_dt2, d2y_dt2;
        calculateSecondDerivatives(t, d2x_dt2, d2y_dt2);

        const double numerator = std::abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2);
        const double denominator = std::pow(dx_dt * dx_dt + dy_dt * dy_dt, 1.5);
        return numerator / denominator;
    }

    /**
     * @brief Plots the C3CurveDesymmetrized and its normals using matplot++.
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the curve.
     * @param plotNormals If the normals are to be plotted.
     * @param plotCoordinateAxes If the coordinate axes are to be plotted.
     */
    void plot(const matplot::axes_handle &ax, const int numPoints, const bool plotNormals, const bool plotCoordinateAxes) const override {
        using namespace matplot;
        std::vector<double> x(numPoints + 1), y(numPoints + 1);
        std::vector<double> u(numPoints + 1), v(numPoints + 1);
        const double step = 1.0 / numPoints;
        ax->hold(matplot::on);

        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);

            x[i] = p.x;
            y[i] = p.y;
            if (plotNormals) {
                ax->arrow(p.x, p.y, p.x + 0.1 * n.x, p.y + 0.1 * n.y)->color("red");
            }
        }
        // Plot the curve
        ax->plot(x, y, "-k");
        if (plotCoordinateAxes) {
            // Plot the coordinate system (x and y axes)
            double minX = *std::ranges::min_element(x);
            double maxX = *std::ranges::max_element(x);
            double minY = *std::ranges::min_element(y);
            double maxY = *std::ranges::max_element(y);

            // Extend axes limits a bit for better visualization
            const double xPadding = (maxX - minX) * 0.05;
            const double yPadding = (maxY - minY) * 0.05;

            minX -= xPadding;
            maxX += xPadding;
            minY -= yPadding;
            maxY += yPadding;

            // Plot x-axis
            ax->plot({minX, maxX}, std::vector<double>{0, 0}, "--k");

            // Plot y-axis
            ax->plot({0, 0}, {minY, maxY}, "--k");
        }

        ax->hold(matplot::off);
        ax->xlabel("x");
        ax->ylabel("y");
    }

    /**
     * @brief Determines if a point is inside the C3CurveDesymmetrized.
     * @param point The point to check.
     * @return Always returns false as this class does not support point-in-boundary checks.
     */
    [[nodiscard]] bool isInside(const Point& point) const override {
        return false;
    }

    /**
     * @brief Indicates that this class does not support point-in-boundary checks.
     * @return Always returns false.
     */
    [[nodiscard]] bool supportsIsInside() const override {
        return false;
    }

    void getBoundingBox(double &minX, double &maxX, double &minY, double &maxY) const override {
        findExtreme(minX, minY, maxX, maxY);
    };

    /**
     * @brief Prints detailed information about the desymmetrized C3Curve boundary at specified points.
     * @param numPoints The number of points at which to print the boundary information.
     */
    void printBoundaryInfo(const int numPoints) const {
        const double step = 1.0 / numPoints;
        std::cout << "C3 Desymmetrized Boundary Info (" << numPoints << " points):" << std::endl;
        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);
            const double s = calculateArcParameter(t);
            const double curvature = computeCurvature(t);
            std::cout << "t: " << t
                      << ", Point: [" << p.x << ", " << p.y << "]"
                      << ", Normal: [" << n.x << ", " << n.y << "]"
                      << ", Arc parameter: " << s
                      << ", Curvature: " << curvature
                      << std::endl;
        }
    }

    /**
    * @brief Calculates the first derivatives of x and y with respect to t.
    * @param t The parameter t.
    * @param dx_dt The first derivative of x with respect to t.
    * @param dy_dt The first derivative of y with respect to t.
    */
    void calculateTangents(const double t, double &dx_dt, double &dy_dt) const {
        const double scaled_t = t / 3.0;
        dx_dt = M_PI * (-3.0 * a * (sin(6 * M_PI * scaled_t) + 2 * cos(12 * M_PI * scaled_t)) * cos(2 * M_PI * scaled_t) + (a * (sin(12 * M_PI * scaled_t) - cos(6 * M_PI * scaled_t)) - 1) * sin(2 * M_PI * scaled_t));
        dy_dt = M_PI * (-3.0 * a * (sin(6 * M_PI * scaled_t) + 2 * cos(12 * M_PI * scaled_t)) * sin(2 * M_PI * scaled_t) - (a * (sin(12 * M_PI * scaled_t) - cos(6 * M_PI * scaled_t)) - 1) * cos(2 * M_PI * scaled_t));
    }

private:

    /**
     * @brief Calculates the second derivatives of x and y with respect to t.
     * @param t The parameter t.
     * @param d2x_dt2 The second derivative of x with respect to t.
     * @param d2y_dt2 The second derivative of y with respect to t.
     */
    void calculateSecondDerivatives(const double t, double &d2x_dt2, double &d2y_dt2) const {
        const double scaled_t = t / 3.0;
        d2x_dt2 = M_PI * M_PI * (12.0 * a * (sin(6 * M_PI * scaled_t) + 2 * cos(12 * M_PI * scaled_t)) * sin(2 * M_PI * scaled_t) + 18.0 * a * (4 * sin(12 * M_PI * scaled_t) - cos(6 * M_PI * scaled_t)) * cos(2 * M_PI * scaled_t) + 2.0 * (a * (sin(12 * M_PI * scaled_t) - cos(6 * M_PI * scaled_t)) - 1) * cos(2 * M_PI * scaled_t));
        d2y_dt2 = M_PI * M_PI * (-12.0 * a * (sin(6 * M_PI * scaled_t) + 2 * cos(12 * M_PI * scaled_t)) * cos(2 * M_PI * scaled_t) + 18.0 * a * (4 * sin(12 * M_PI * scaled_t) - cos(6 * M_PI * scaled_t)) * sin(2 * M_PI * scaled_t) + 2.0 * (a * (sin(12 * M_PI * scaled_t) - cos(6 * M_PI * scaled_t)) - 1) * sin(2 * M_PI * scaled_t));
    }

    /**
    * @brief Finds the extreme values of the curve to determine the bounding box.
    * @param minX Reference to store the minimum x-coordinate.
    * @param minY Reference to store the minimum y-coordinate.
    * @param maxX Reference to store the maximum x-coordinate.
    * @param maxY Reference to store the maximum y-coordinate.
    */
    void findExtreme(double &minX, double &minY, double &maxX, double &maxY) const {
        constexpr int num_points = 30000;
        minX = std::numeric_limits<double>::max();
        maxX = std::numeric_limits<double>::lowest();
        minY = std::numeric_limits<double>::max();
        maxY = std::numeric_limits<double>::lowest();

        for (int i = 0; i < num_points; ++i) {
            const double t = static_cast<double>(i) / (num_points - 1);
            const Point p = curveParametrization(t);
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        }
    }
};

/**
 * @brief Represents a perfect circle and provides methods to calculate its properties.
 *
 * This class encapsulates a circle defined by a center point and a radius. It offers methods to
 * parametrize the circle, compute normals, calculate curvature, and other properties.
 */
class Circle final : public AbstractBoundary {
    Point center;
    double radius;

public:
    /**
     * @brief Constructor for the Circle class.
     * @param center The center point of the circle.
     * @param radius The radius of the circle.
     */
    Circle(const Point &center, const double radius) : center(center), radius(radius) {}

    /**
     * @brief Computes the point on the circle corresponding to a given parametric value t.
     * @param t A parameter ranging from 0 to 1, where 0 and 1 represent the start and end points of the circle, respectively.
     * @return Point at the parametric position t on the circle.
     *
     * @details
     * - The parametric equation of the circle in terms of t is:
     *   x(t) = center.x + radius * cos(2 * PI * t)
     *   y(t) = center.y + radius * sin(2 * PI * t)
     * - This representation ensures that the circle is traversed exactly once as t moves from 0 to 1.
     */
    [[nodiscard]] Point curveParametrization(const double t) const override {
        return Point(center.x + radius * cos(2*M_PI*t), center.y + radius * sin(2*M_PI*t)); // NOLINT(*-return-braced-init-list)
    }
    /**
     * @brief Calculates the outward normal at a point on the circle defined by the parametric value t.
     * @param t A parameter ranging from 0 to 1.
     * @return The outward normal vector at the parametric position t on the circle.
     *
     * @details
     * - The normal vector at any point on a circle points directly away from the center.
     * - It can be calculated directly from the parametric position:
     *   nx(t) = cos(2 * PI * t)
     *   ny(t) = sin(2 * PI * t)
     * - These calculations assume a counterclockwise traversal of the circle.
     */
    [[nodiscard]] Point calculateNormal(const double t) const override {
        return Point(cos(2*M_PI*t), sin(2*M_PI*t)); // Normal vector (outward) NOLINT(*-return-braced-init-list)
    }
    /**
     * @brief Computes the total arc length of the circle.
     * @return The circumference of the circle.
     */
    [[nodiscard]] double calculateArcLength() const override {
        return 2 * M_PI * radius;
    }
    [[nodiscard]] double getArea() const override {
        return M_PI * radius * radius;
    };
    /**
     * @brief Computes the curvature of the circle.
     * @param t Parametric value, unused because the curvature of a circle is constant.
     * @return The constant curvature of the circle, which is the reciprocal of the radius.
     */
    [[nodiscard]] double computeCurvature(double t) const override {
        return 1 / radius; // Constant curvature of a circle
    }
    void printBoundaryInfo(const int numPoints) const {
        const double step = 1.0 / numPoints;
        std::cout << "Circle Boundary Info (" << numPoints << " points):" << std::endl;
        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);
            const double curvature = computeCurvature(t);
            std::cout << "t: " << t
                      << ", Point: [" << p.x << ", " << p.y << "]"
                      << ", Normal: [" << n.x << ", " << n.y << "]"
                      << ", Curvature: " << curvature
                      << std::endl;
        }
    }
    /**
     * @brief Plots the boundary and its normals using matplot++.
     *
     * This method plots the boundary as a continuous line and overlays normal vectors at specified intervals.
     *
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the boundary.
     * @param plotNormals If the normals are to be plotted
     * @param plotCoordinateAxes If we want to plot the coordinate axes
     */
    void plot(const matplot::axes_handle &ax, const int numPoints, const bool plotNormals, const bool plotCoordinateAxes) const override {
        using namespace matplot;
        std::vector<double> x(numPoints + 1), y(numPoints + 1);
        std::vector<double> u(numPoints + 1), v(numPoints + 1);
        const double step = 1.0 / numPoints;
        ax->hold(matplot::on);

        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);

            x[i] = p.x;
            y[i] = p.y;
            if (plotNormals) {
                ax->arrow(p.x, p.y, p.x + 0.1 * n.x, p.y + 0.1 * n.y)->color("red");
            }
        }

        // Plot the boundary
        ax->plot(x, y, "-k");

        if (plotCoordinateAxes) {
            // Plot the coordinate system (x and y axes)
            double minX = *std::ranges::min_element(x);
            double maxX = *std::ranges::max_element(x);
            double minY = *std::ranges::min_element(y);
            double maxY = *std::ranges::max_element(y);

            // Extend axes limits a bit for better visualization
            const double xPadding = (maxX - minX) * 0.05;
            const double yPadding = (maxY - minY) * 0.05;

            minX -= xPadding;
            maxX += xPadding;
            minY -= yPadding;
            maxY += yPadding;

            // Plot x-axis
            ax->plot({minX, maxX}, std::vector<double>{0, 0}, "--k");

            // Plot y-axis
            ax->plot({0, 0}, {minY, maxY}, "--k");
        }

        ax->hold(matplot::off);
        ax->xlabel("x");
        ax->ylabel("y");
    }
    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    }
    /**
     * @brief Calculates the bounding box that completely encloses the circle.
     * @param minX Reference to store the minimum x-coordinate.
     * @param maxX Reference to store the maximum x-coordinate.
     * @param minY Reference to store the minimum y-coordinate.
     * @param maxY Reference to store the maximum y-coordinate.
     *
     * @details
     * - The bounding box for a circle is straightforward to calculate based on its center and radius.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        minX = center.x - radius;
        maxX = center.x + radius;
        minY = center.y - radius;
        maxY = center.y + radius;
    }
    /**
     * @brief Checks if a given point lies inside the circle.
     * @param point The point to check.
     * @return True if the point is inside the circle, false otherwise.
     *
     * @details
     * - Determines if a point is inside the circle by checking if the distance from the point
     *   to the center of the circle is less than or equal to the radius.
     */
    [[nodiscard]] bool isInside(const Point& point) const override {
        const double dx = point.x - center.x;
        const double dy = point.y - center.y;
        return (dx * dx + dy * dy) <= (radius * radius);
    }

    [[nodiscard]] double calculateArcParameter(const double t) const override {
        return t * calculateArcLength();
    };
};

/**
 * @brief Represents a composite boundary consisting of multiple boundary segments.
 *        This class facilitates modeling complex geometries by allowing multiple different
 *        shapes to form a unified boundary. The behavior of functions like parametrization and
 *        normal calculations takes into account the composite nature, handling transitions between segments.
 */
class CompositeBoundary : public AbstractBoundary {
    std::vector<std::shared_ptr<AbstractBoundary>> segments;
    bool clockwise;

public:
    /**
     * Constructs a CompositeBoundary with an optional orientation flag.
     *
     * @param clockwise Specifies the direction in which the boundary segments are ordered.
     *                  True for clockwise, which affects the normal vector orientation.
     *                  Affects calculations of inside/outside, normals, and arc parameterizations.
     */
    explicit CompositeBoundary(const bool clockwise = true) : clockwise(clockwise) {}

    /**
     * Adds a boundary segment to the composite boundary. Order of addition matters as it determines
     * the traversal order in operations like plotting or calculating normals.
     *
     * @param segment A shared pointer to an AbstractBoundary object representing the segment to add.
     */
    void addSegment(const std::shared_ptr<AbstractBoundary>& segment) {
        segments.push_back(segment);
    }

    /**
     * Retrieves all segments making up the composite boundary.
     * Useful for operations that need to process each segment individually, like collision detection.
     *
     * @return A vector of shared pointers to AbstractBoundary objects.
     */
    [[nodiscard]] std::vector<std::shared_ptr<AbstractBoundary>> getSegments() const {
        return segments;
    }

    /**
     * Retrieves a specific segment from the boundary by its index.
     * Throws an exception if the index is out of range, ensuring robust error handling.
     *
     * @param i The index of the segment to retrieve.
     * @return A shared pointer to the AbstractBoundary segment if found; nullptr if the index is out of range.
     */
    [[nodiscard]] std::shared_ptr<AbstractBoundary> getSegment(const int i) const {
        try {
            return segments.at(i);
        } catch (std::exception& e) {
            std::cerr << "Segment get exception " << e.what() << std::endl;
        }
        return nullptr;
    }

    /**
     * Computes the point on the composite boundary corresponding to the parameter t.
     * Handles the distribution of t across segments based on their relative arc lengths.
     *
     * @param t A double value where 0 <= t <= 1, representing a normalized distance along the boundary.
     * @return The point on the boundary corresponding to t.
     */
    [[nodiscard]] Point curveParametrization(const double t) const override {
        const double totalLength = calculateArcLength();
        double accumulatedLength = 0;
        for (const auto& segment : segments) {
            const double segmentLength = segment->calculateArcLength();
            if (t * totalLength < accumulatedLength + segmentLength) {
                // Calculate local parameter for the segment
                const double local_t = (t * totalLength - accumulatedLength) / segmentLength;
                return segment->curveParametrization(local_t);
            }
            accumulatedLength += segmentLength;
        }
        // Edge case for t=1 (or very close to), which might not be exactly covered due to floating-point precision
        return segments.back()->curveParametrization(1.0);
    }

    /**
     * Calculates the normal at the boundary point corresponding to the parameter t.
     * Adjusts the normal's direction based on the clockwise attribute to ensure it always points outward.
     *
     * @param t A double value where 0 <= t <= 1.
     * @return The outward normal at the boundary point.
     */
    [[nodiscard]] Point calculateNormal(const double t) const override {
        const double totalLength = calculateArcLength();
        double accumulatedLength = 0;
        for (const auto& segment : segments) {
            const double segmentLength = segment->calculateArcLength();
            if (t * totalLength < accumulatedLength + segmentLength) {
                const Point normal = segment->calculateNormal((t * totalLength - accumulatedLength) / segmentLength);
                return Point(normal.x, normal.y);  // Ensure normals are outward. Here we did not again change the y, x orientation and flipped the sings b/c this is already done in each individual segment NOLINT(*-return-braced-init-list)
            }
            accumulatedLength += segmentLength;
        }
        return segments.back()->calculateNormal(1.0);
    }

    /**
     * Computes the speed at a given parametric value t on the boundary. It is overridden in the CompositeBoundary class due to the fact we have discontinuities when gluing segments together
     * Useful for arc length calculations and determining how fast a point moves along the boundary.
     *
     * @param t A double value where 0 <= t <= 1, representing the parametric position along the boundary.
     * @return The speed at parameter t.
     */
    [[nodiscard]] double speedAt(const double t) const override {
        const double totalLength = calculateArcLength();
        double accumulatedLength = 0;
        for (const auto& segment : segments) {
            const double segmentLength = segment->calculateArcLength();
            if (t * totalLength < accumulatedLength + segmentLength) {
                // Calculate local parameter for the segment
                const double local_t = (t * totalLength - accumulatedLength) / segmentLength;
                return segment->speedAt(local_t);
            }
            accumulatedLength += segmentLength;
        }
        // Handle edge case for t=1 (or very close to), which might not be exactly covered due to floating-point precision
        return segments.back()->speedAt(1.0);
    }

    /**
     * Computes the total arc length of the boundary by summing up the arc lengths of all segments.
     * This method is computationally intensive as it may require integration over each segment.
     *
     * @return The total arc length of the composite boundary.
     */
    [[nodiscard]] double calculateArcLength() const override {
        double totalLength = 0;
        for (const auto& segment : segments) {
            totalLength += segment->calculateArcLength();
        }
        return totalLength;
    }

    /**
     * Computes the arc length parameter for a given normalized parameter t.
     * Distributes the parameter t across the segments based on their relative arc lengths.
     *
     * @param t A double value where 0 <= t <= 1, representing a normalized distance along the composite boundary.
     * @return The arc length parameter corresponding to t, adjusted for the composite boundary.
     */
    [[nodiscard]] double calculateArcParameter(const double t) const override {
        const double totalLength = calculateArcLength();
        double accumulatedLength = 0.0;
        const double targetLength = t * totalLength;

        for (const auto& segment : segments) {
            const double segmentLength = segment->calculateArcLength();
            if (targetLength <= accumulatedLength + segmentLength) {
                const double segmentT = (targetLength - accumulatedLength) / segmentLength;
                return segment->calculateArcParameter(segmentT) + accumulatedLength;
            }
            accumulatedLength += segmentLength;
        }
        // Edge case for t=1 (or very close to), which might not be exactly covered due to floating-point precision
        return accumulatedLength;
    }

    /**
     * Prints detailed boundary information at a specified number of points along the boundary.
     * Each point includes position, normal, and curvature, giving a comprehensive overview of the boundary's shape.
     *
     * @param numPoints The number of points at which to print the boundary information.
     */
    void printBoundaryInfo(const int numPoints) const {
        if (numPoints <= 0) {
            std::cerr << "Number of points must be positive." << std::endl;
            return;
        }
        const double step = 1.0 / numPoints;
        std::cout << "Boundary Info (" << numPoints << " points):" << std::endl;
        for (int i = 0; i <= numPoints; ++i) {
            const double t = i * step;
            const Point p = curveParametrization(t);
            const Point n = calculateNormal(t);
            const double s = calculateArcParameter(t);
            const double curvature = computeCurvature(t);
            std::cout << "t: " << t
                      << ", Point: [" << p.x << ", " << p.y << "]"
                      << ", Normal: [" << n.x << ", " << n.y << "]"
                        << ", Arc parameter: " << s
                      << ", Curvature: " << curvature
                      << std::endl;
        }
    }

    /**
     * Calculates the curvature at a specific point along the boundary.
     * This method delegates to individual segments, taking into account their specific curvature calculations.
     *
     * @param t A parametric value where 0 <= t <= 1.
     * @return The curvature at the point t.
     */
    [[nodiscard]] double computeCurvature(const double t) const override {
        const double totalLength = calculateArcLength();
        double accumulatedLength = 0;
        for (const auto& segment : segments) {
            const double segmentLength = segment->calculateArcLength();
            if (t * totalLength < accumulatedLength + segmentLength) {
                return segment->computeCurvature((t * totalLength - accumulatedLength) / segmentLength);
            }
            accumulatedLength += segmentLength;
        }
        return 0; // Curvature is zero if we somehow reach the end
    }
    /**
     * @brief Plots the composite boundary and its normals using matplot++.
     *
     * This method plots each segment of the composite boundary as a continuous line
     * and overlays normal vectors at specified intervals. Each segment is held on the plot.
     *
     * @param ax The matplot++ axes handle where the plot will be drawn.
     * @param numPoints The number of discretization points for plotting the boundary.
     * @param plotNormals A boolean flag to control whether normals are plotted. Defaults to true.
     * @param plotCoordinateAxes If the coordinate axes be plotted
     */
    void plot(const matplot::axes_handle &ax, const int numPoints, const bool plotNormals, const bool plotCoordinateAxes) const override {
        ax->hold(matplot::on);
        for (const auto& segment : segments) {
            segment->plot(ax, numPoints, plotNormals, plotCoordinateAxes);
        }
        ax->hold(matplot::off);
    }

    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    }

    /**
     * @brief Determines if a point is inside the composite boundary.
     *
     * This method employs the ray-casting algorithm, also known as the even-odd rule algorithm.
     * The algorithm casts a ray from the point in question to infinity in a fixed direction (typically the positive x-axis)
     * and counts how many times the ray intersects the boundary segments. A point is considered inside if and only if
     * this count is odd, implying the point is within the boundary rather than outside or on an edge.
     *
     * The method handles the composite nature of the boundary by iterating through each segment and checking intersections.
     * Intersections are detected by analyzing if the horizontal ray from the point crosses the line segment formed by each boundary part.
     *
     * @param point The point for which to determine if it lies inside the boundary.
     * @return True if the point is inside the boundary based on the ray-casting algorithm; otherwise, false.
     *
     * @details
     * The ray-casting check involves:
     * 1. For each segment of the boundary, determining if the line segment from the start to the end of the segment crosses the y-coordinate of the point.
     * 2. If it does, compute the x-coordinate where the crossing occurs.
     * 3. If this x-coordinate is greater than the x-coordinate of the point (meaning the crossing is to the right of the point), increment the intersection count.
     * 4. After checking all segments, the point is inside if the intersection count is odd.
     *
     * This method does not correctly handle points exactly on the boundary edge and may behave unpredictably for complex boundaries
     * that intersect themselves or for non-manifold edges.
     */
    [[nodiscard]] bool isInside(const Point& point) const override {
        int count = 0;
        auto checkIntersect = [&](const Point& p1, const Point& p2) {
            if ((p1.y > point.y) != (p2.y > point.y)) {
                if (const double xIntersect = p1.x + (point.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y); point.x < xIntersect) {
                    count++;
                }
            }
        };
        for (const auto& segment : segments) {
            Point start = segment->curveParametrization(0);
            Point end = segment->curveParametrization(1);
            checkIntersect(start, end);
        }
        return (count % 2) == 1; // Inside if odd number of intersections
    }

    /**
     * Calculates the bounding box of the composite boundary by evaluating the bounding boxes of individual segments.
     *
     * @param minX Reference to store the minimum x-coordinate of the bounding box.
     * @param maxX Reference to store the maximum x-coordinate of the bounding box.
     * @param minY Reference to store the minimum y-coordinate of the bounding box.
     * @param maxY Reference to store the maximum y-coordinate of the bounding box.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        minX = minY = std::numeric_limits<double>::infinity();
        maxX = maxY = -std::numeric_limits<double>::infinity();
        double segMinX, segMaxX, segMinY, segMaxY;
        for (const auto& segment : segments) {
            segment->getBoundingBox(segMinX, segMaxX, segMinY, segMaxY);
            minX = std::min(minX, segMinX);
            maxX = std::max(maxX, segMaxX);
            minY = std::min(minY, segMinY);
            maxY = std::max(maxY, segMaxY);
        }
    }

    /**
     * Prints the bounding boxes of each individual segment within the composite boundary.
     */
    void printSegmentsBoundingBox() const {
        std::cout << "Composite Boundary Segments Bounding Boxes:" << std::endl;
        for (size_t i = 0; i < segments.size(); ++i) {
            double minX, maxX, minY, maxY;
            segments[i]->getBoundingBox(minX, maxX, minY, maxY);
            std::cout << "Segment " << i << ": x_min = " << minX << ", x_max = " << maxX
                      << ", y_min = " << minY << ", y_max = " << maxY << std::endl;
        }
    }

    /**
     * Prints the bounding box of a specific segment identified by its index.
     * Provides detailed information about the spatial extent of individual segments.
     *
     * @param index The index of the segment whose bounding box will be printed.
     */
    void getSegmentBoundingBox(const int index) const {
        if (index < 0 || index >= segments.size()) {
            std::cerr << "Error: Index out of range for boundary segments." << std::endl;
            return;
        }
        double minX, maxX, minY, maxY;
        segments[index]->getBoundingBox(minX, maxX, minY, maxY);
        std::cout << "Bounding box for segment " << index << ": x_min = " << minX
                  << ", x_max = " << maxX << ", y_min = " << minY << ", y_max = " << maxY << std::endl;
    }

    /**
     * Prints the overall bounding box of the composite boundary.
     * Useful for spatial queries and preliminary collision detection optimizations.
     */
    void printCompositeBoundingBox() const {
        double minX = std::numeric_limits<double>::infinity();
        double maxX = -std::numeric_limits<double>::infinity();
        double minY = std::numeric_limits<double>::infinity();
        double maxY = -std::numeric_limits<double>::infinity();

        double segMinX, segMaxX, segMinY, segMaxY;
        for (const auto& segment : segments) {
            segment->getBoundingBox(segMinX, segMaxX, segMinY, segMaxY);
            minX = std::min(minX, segMinX);
            maxX = std::max(maxX, segMaxX);
            minY = std::min(minY, segMinY);
            maxY = std::max(maxY, segMaxY);
        }

        std::cout << "Composite Boundary Bounding Box:" << std::endl;
        std::cout << "x_min = " << minX << ", x_max = " << maxX
                  << ", y_min = " << minY << ", y_max = " << maxY << std::endl;
    }
};

/**
 * @class ParabolicBoundary
 * @brief Represents a segment of a parabola defined between two points with a specific curvature.
 *
 * This class inherits from ParametricBoundary and defines a parabolic curve using parametric equations.
 * The parabola is defined such that it passes through the start and end points provided in the constructor,
 * with a curvature determined by the coefficient parameter. This setup ensures flexibility and precision in defining
 * boundary shapes for various applications.
 *
 * @param start The starting point of the parabolic segment.
 * @param end The ending point of the parabolic segment.
 * @param coefficient The curvature coefficient of the parabola. Positive values yield a concave up parabola,
 *                    negative values yield a concave down parabola.
 */
class ParabolicBoundary final : public ParametricBoundary {

public:
    ParabolicBoundary(Point start, Point end, double coefficient)
        : ParametricBoundary(
            [start, end](const double t) -> double {
                // Linear interpolation between start.x and end.x
                return start.x + t * (end.x - start.x);
            },
            [start, end, coefficient](const double t) -> double {
                // Quadratic interpolation for y based on coefficient and x(t)
                const double x = start.x + t * (end.x - start.x);
                return start.y + coefficient * (x - start.x) * (x - end.x); // Ensures the curve passes through start and end
            }
          ), start(start), end(end), coefficient(coefficient) {}

    /**
     * @brief Calculates the arc length of the parabolic segment using numerical integration.
     *
     * Computes the length of the parabolic segment's perimeter from the start to the end point.
     * This computation is essential for precise physical modeling and graphical representation.
     *
     * @return The arc length of the parabolic segment.
     *
     * @details
     * - Uses GSL for numerical integration to compute the arc length.
     * - Employs adaptive integration with GSL to handle varying curvature.
     * - Increases workspace size dynamically if the default size is insufficient.
     */
    [[nodiscard]] double calculateArcLength() const override {
        gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(2000);
        double result, error;

        auto integrand = [](const double x, void* params) -> double {
            const auto* parabolaParams = static_cast<ParabolicBoundary*>(params);
            const double dy_dx = 2 * parabolaParams->coefficient * x - parabolaParams->coefficient * (parabolaParams->start.x + parabolaParams->end.x);
            return std::sqrt(1 + dy_dx * dy_dx);
        };

        gsl_function F;
        F.function = integrand;
        F.params = const_cast<ParabolicBoundary*>(this);

        size_t limit = 2000;
        const double startX = start.x;
        const double endX = end.x;

        while (gsl_integration_qags(&F, startX, endX, 0, 1e-8, limit, workspace, &result, &error) == GSL_ERANGE) {
            limit *= 2;
            gsl_integration_workspace_free(workspace);
            workspace = gsl_integration_workspace_alloc(limit);
        }

        gsl_integration_workspace_free(workspace);

        return result;
    }

    /**
     * @brief Computes the arc length parameter for a given normalized parameter t.
     *
     * This method distributes the parameter t along the parabola based on its arc length,
     * ensuring accurate representation of the boundary's geometry.
     *
     * @param t A double value where 0 <= t <= 1, representing a normalized distance along the parabola.
     * @return The arc length parameter corresponding to t, adjusted for the parabola.
     *
     * @details
     * - Uses numerical integration to find the arc length corresponding to the given t.
     * - Employs adaptive integration with GSL to handle varying curvature.
     * - Increases workspace size dynamically if the default size is insufficient.
     */
    [[nodiscard]] double calculateArcParameter(const double t) const override {
        gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(2000);
        double result, error;

        auto integrand = [](const double x, void* params) -> double {
            const auto* parabolaParams = static_cast<ParabolicBoundary*>(params);
            const double dy_dx = 2 * parabolaParams->coefficient * x - parabolaParams->coefficient * (parabolaParams->start.x + parabolaParams->end.x);
            return std::sqrt(1 + dy_dx * dy_dx);
        };

        gsl_function F;
        F.function = integrand;
        F.params = const_cast<ParabolicBoundary*>(this);

        size_t limit = 2000;
        const double startX = start.x;
        const double endX = start.x + t * (end.x - start.x); // Here is where we used the parameter t

        while (gsl_integration_qags(&F, startX, endX, 0, 1e-7, limit, workspace, &result, &error) == GSL_ERANGE) {
            limit *= 2;
            gsl_integration_workspace_free(workspace);
            workspace = gsl_integration_workspace_alloc(limit);
        }

        gsl_integration_workspace_free(workspace);

        return result;
    }

    /**
     * @brief Determines if a point is inside the ParabolicBoundary.
     *
     * This method checks if a given point lies within the boundary of the parabolic segment.
     * The boundary is defined by the parabolic curve from the start point to the end point.
     *
     * @param point The point to check.
     * @return True if the point is inside the boundary, false otherwise.
     *
     * @details
     * - The method first checks if the point is within the x-range of the parabolic segment.
     * - For a downward-facing parabola (positive coefficient), it checks if the point's y-coordinate is below the y-coordinates of the start and end points and above the parabolic curve.
     * - For an upward-facing parabola (negative coefficient), it checks if the point's y-coordinate is above the y-coordinates of the start and end points and below the parabolic curve.
     */
    [[nodiscard]] bool isInside(const Point& point) const override {
        // Check if the point's x-coordinate is within the x-range of the parabolic segment.
        // The x-range is defined by the start and end x-coordinates of the parabola.
        if (point.x < start.x || point.x > end.x) {
            return false;
        }

        // Calculate the y-coordinate of the parabolic curve at the given x-coordinate of the point.
        // The parabolic equation is given by: y = start.y + coefficient * (point.x - start.x) * (point.x - end.x)
        const double parabolaY = start.y + coefficient * (point.x - start.x) * (point.x - end.x);

        if (coefficient > 0) {
            // For a downward-facing parabola (positive coefficient)
            // Check if the point's y-coordinate is below the start and end y-coordinates and above the parabola
            return point.y <= start.y && point.y <= end.y && point.y >= parabolaY;
        } else {
            // For an upward-facing parabola (negative coefficient)
            // Check if the point's y-coordinate is above the start and end y-coordinates and below the parabola
            return point.y >= start.y && point.y >= end.y && point.y <= parabolaY;
        }
    }

    /**
     * @brief Calculates the bounding box that completely encloses the ParabolicBoundary.
     *
     * This method computes the minimum and maximum x and y coordinates that enclose the parabolic segment.
     *
     * @param minX Reference to store the minimum x-coordinate.
     * @param maxX Reference to store the maximum x-coordinate.
     * @param minY Reference to store the minimum y-coordinate.
     * @param maxY Reference to store the maximum y-coordinate.
     *
     * @details
     * - The x-range is straightforward as it is between the start and end points of the parabola.
     * - The y-range is determined by evaluating the y-coordinates of the parabola at the start, end, and vertex.
     * - The vertex of the parabola occurs at the midpoint of the x-range.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        // The x-range is directly between the start and end x-coordinates
        minX = start.x;
        maxX = end.x;

        // Calculate the y-coordinates of the start and end points of the parabola
        const double yStart = start.y;
        const double yEnd = end.y;

        // Calculate the x-coordinate of the vertex of the parabola
        const double vertexX = (start.x + end.x) / 2.0;

        // Calculate the y-coordinate of the vertex of the parabola
        const double vertexY = start.y + coefficient * (vertexX - start.x) * (vertexX - end.x);

        // Determine minY and maxY based on the direction of the parabola
        if (coefficient > 0) {
            // Downward-facing parabola (positive coefficient)
            minY = std::min(yStart, std::min(yEnd, vertexY));
            maxY = std::max(yStart, yEnd);
        } else {
            // Upward-facing parabola (negative coefficient)
            minY = std::min(yStart, yEnd);
            maxY = std::max(yStart, std::max(yEnd, vertexY));
        }
    }

    /**
     * @brief Calculates the outward normal vector at a given point on the parabolic boundary.
     * @param t A parameter ranging from 0 to 1.
     * @return The normal vector at the point corresponding to parameter t.
     */
    [[nodiscard]] Point calculateNormal(const double t) const override {
        double dx_dt, dy_dt;
        calculateTangents(t, dx_dt, dy_dt);
        const double length = std::hypot(dx_dt, dy_dt);
        return {-dy_dt / length, dx_dt / length};
    }

    /**
     * @brief Computes the curvature of the parabolic boundary at a given point.
     * @param t A parameter ranging from 0 to 1.
     * @return The curvature at the point corresponding to parameter t.
     */
    [[nodiscard]] double computeCurvature(const double t) const override {
        double dx_dt, dy_dt;
        calculateTangents(t, dx_dt, dy_dt);

        double d2x_dt2, d2y_dt2;
        calculateSecondDerivatives(t, d2x_dt2, d2y_dt2);

        const double numerator = std::abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2);
        const double denominator = std::pow(dx_dt * dx_dt + dy_dt * dy_dt, 1.5);
        return numerator / denominator;
    }

private:
    Point start;
    Point end;
    double coefficient;

    /**
     * @brief Calculates the tangents (first derivatives) of the parabolic curve at a given parameter t.
     * @param t The parameter t.
     * @param dx_dt The first derivative of x with respect to t.
     * @param dy_dt The first derivative of y with respect to t.
     */
    void calculateTangents(const double t, double &dx_dt, double &dy_dt) const {
        dx_dt = end.x - start.x;
        dy_dt = coefficient * (2 * (start.x + t * (end.x - start.x)) - start.x - end.x) * (end.x - start.x);
    }

    /**
     * @brief Calculates the second derivatives of x and y with respect to t.
     * @param t The parameter t.
     * @param d2x_dt2 The second derivative of x with respect to t.
     * @param d2y_dt2 The second derivative of y with respect to t.
     */
    void calculateSecondDerivatives(const double t, double &d2x_dt2, double &d2y_dt2) const {
        d2x_dt2 = 0;
        d2y_dt2 = 2 * coefficient * std::pow(end.x - start.x, 2);
    }
};

/**
 * @class SquareWithParabolicTop
 * @brief Constructs a square boundary with a parabolic curve replacing the top edge.
 *
 * This class demonstrates the use of CompositeBoundary to create a complex shape by combining
 * linear and curved segments. The square is oriented such that one of its sides is replaced by
 * a downward-facing parabolic curve, specified by a coefficient.
 *
 * @param center The center point of the square.
 * @param sideLength The length of each side of the square.
 * @param parabolaCoefficient The coefficient defining the curvature of the parabolic top.
 */
class SquareWithParabolicTop final : public CompositeBoundary {
public:
    SquareWithParabolicTop(const Point center, const double sideLength, const double parabolaCoefficient) : sideLength(sideLength), parabolaCoefficient(parabolaCoefficient)
    {
        // Points for square
        leftTop = Point(center.x - sideLength / 2, center.y + sideLength / 2);
        rightTop = Point(center.x + sideLength / 2, center.y + sideLength / 2);
        rightBottom = Point(center.x + sideLength / 2, center.y - sideLength / 2);
        leftBottom = Point(center.x - sideLength / 2, center.y - sideLength / 2);
        // Parabola between leftTop and rightTop
        addSegment(std::make_shared<ParabolicBoundary>(leftTop, rightTop, parabolaCoefficient));
        // Right vertical side
        addSegment(std::make_shared<LineSegment>(rightTop, rightBottom, false));
        // Bottom horizontal side
        addSegment(std::make_shared<LineSegment>(rightBottom, leftBottom, false));
        // Left vertical side
        addSegment(std::make_shared<LineSegment>(leftBottom, leftTop, false));
    }

    /**
     * Calculates the area of the parabolic top square. It is divied into 2 parts. The first one calculates the area of the elevated parabola. The second part just calculates the square area adn then we add the areas together
     * @return The area of the geometric shape
     */
    [[nodiscard]] double getArea() const override {
        const double rightX = rightTop.x;
        const double leftX = leftTop.x;
        const double parabolaArea = parabolaCoefficient * ((std::pow(rightX, 3) - std::pow(leftX, 3)) / 3 - (rightTop.x + leftTop.x) * (std::pow(rightX, 2) - std::pow(leftX, 2)) / 2 + rightTop.x * leftTop.x * (rightX - leftX));
        const double squareArea = std::pow(sideLength, 2);
        if (parabolaCoefficient < 0) { // Depending whether we have a outward cap or inward
            return squareArea + parabolaArea;
        } else {
            return squareArea - parabolaArea;
        }
    };

    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    };

    /**
     * @brief Determines if a point is inside the SquareWithParabolicTop boundary.
     *
     * This method checks if a given point lies within the boundary of the SquareWithParabolicTop.
     * The boundary is defined by the rectangular area and the parabolic curve at the top.
     *
     * @param point The point to check.
     * @return True if the point is inside the boundary, false otherwise.
     *
     * @details
     * - First, the method checks if the point is within the bounding box of the rectangle.
     * - For a downward-facing parabola (positive coefficient), the method checks if the point's y-coordinate is below the parabolic curve.
     * - For an upward-facing parabola (negative coefficient), the method checks if the point's y-coordinate is below the parabolic curve.
     * - If the point is within the x-range of the parabolic section, it must be below the parabola to be inside.
     * - If the point is not within the parabolic x-range, the method returns true as the point lies within the rectangular bounds.
     */
    [[nodiscard]] bool isInside(const Point& point) const override {
        // Check if the point is within the bounding box of the rectangle
        if (point.x < leftBottom.x || point.x > rightBottom.x || point.y < leftBottom.y || point.y > leftTop.y) {
            return false;
        }

        //Possibly unused
        if (parabolaCoefficient > 0) {
            // Check if the point is within the x-range of the rectangle and below the parabolic curve
            if (point.x >= leftTop.x && point.x <= rightTop.x) {
                const double parabolaY = leftTop.y + parabolaCoefficient * (point.x - leftTop.x) * (point.x - rightTop.x);
                return point.y <= parabolaY;
            }
        } else {
            // For upward-facing parabola (negative coefficient)
            // Check if the point is within the x-range of the rectangle and below the parabolic curve
            if (point.x >= leftTop.x && point.x <= rightTop.x) {
                const double parabolaY = leftTop.y + parabolaCoefficient * (point.x - leftTop.x) * (point.x - rightTop.x);
                return point.y <= parabolaY;
            }
        }

        // If the point is not within the parabolic x-range, check if it is within the general rectangle bounds
        return true;
    }

    /**
     * @brief Calculates the bounding box that completely encloses the SquareWithParabolicTop.
     *
     * The bounding box is determined by the leftmost, rightmost, lowest, and highest points
     * of the shape, which includes the rectangular base and the parabolic top.
     *
     * For an upward-facing parabola (negative coefficient), the maximum y-coordinate will
     * be the y-coordinate of the vertex of the parabola if it exceeds the top of the square.
     * For a downward-facing parabola (positive coefficient), the maximum y-coordinate will
     * remain the top of the square.
     *
     * @param minX Reference to store the minimum x-coordinate.
     * @param maxX Reference to store the maximum x-coordinate.
     * @param minY Reference to store the minimum y-coordinate.
     * @param maxY Reference to store the maximum y-coordinate.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        minX = leftBottom.x;
        maxX = rightBottom.x;
        minY = leftBottom.y;

        // Determine the maximum y based on the parabola
        // ReSharper disable once CppDFAUnusedValue
        double maxYParabola = leftTop.y;  // Default to the top of the square
        const double vertexX = (leftTop.x + rightTop.x) / 2;
        const double vertexY = leftTop.y + parabolaCoefficient * (vertexX - leftTop.x) * (vertexX - rightTop.x);
        if (parabolaCoefficient < 0) {
            // Vertex of the upward parabola (maximum y value)
            maxYParabola = vertexY;
        } else {
            // Vertex of the downward parabola (minimum y value, should not change maxY)
            maxYParabola = leftTop.y;
        }
        maxY = std::max(leftTop.y, maxYParabola);
    }

private:
    double sideLength;
    double parabolaCoefficient;
    Point leftTop;
    Point rightTop;
    Point rightBottom;
    Point leftBottom;
};

/**
 * @class Rectangle
 * @brief Constructs a rectangle using four line segments, with configurable dimensions.
 *
 * This class extends CompositeBoundary to form a rectangle defined by a top-left corner point,
 * width, and height. It showcases a basic use of line segments to construct a simple closed shape.
 *
 * @param topLeft The top-left corner of the rectangle.
 * @param width The width of the rectangle.
 * @param height The height of the rectangle.
 */
class Rectangle final : public CompositeBoundary {
public:
    Rectangle(const Point& topLeft, const double width, const double height) : topLeft(topLeft), width(width), height(height) {
        // Define points for rectangle corners
        Point topRight(topLeft.x + width, topLeft.y);
        Point bottomRight(topLeft.x + width, topLeft.y + height);
        Point bottomLeft(topLeft.x, topLeft.y + height);

        // Add segments in clockwise order
        addSegment(std::make_shared<LineSegment>(topLeft, topRight, true));  // Top edge
        addSegment(std::make_shared<LineSegment>(topRight, bottomRight, true));  // Right edge
        addSegment(std::make_shared<LineSegment>(bottomRight, bottomLeft, true));  // Bottom edge
        addSegment(std::make_shared<LineSegment>(bottomLeft, topLeft, true));  // Left edge
    }

    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        minX = std::min(topLeft.x, topLeft.x + width);
        maxX = std::max(topLeft.x, topLeft.x + width);
        minY = std::min(topLeft.y, topLeft.y + height);
        maxY = std::max(topLeft.y, topLeft.y + height);
    }
    /**
     * @brief Calculates the area of the rectangle.
     *
     * This method computes the area of the rectangle using the formula:
     * Area = width * height.
     *
     * @return The area of the rectangle.
     */
    [[nodiscard]] double getArea() const override {
        return width * height;
    }

    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    };
private:
    Point topLeft;
    double width;
    double height;
};

/**
 * @class QuarterRectangle
 * @brief Constructs a quarter rectangle in the first quadrant using two line segments.
 *
 * This class extends CompositeBoundary to form a quarter rectangle defined by a bottom-left corner point,
 * width, and height. It only includes the top and right edges to form a quarter of a full rectangle.
 *
 * @param bottomLeft The bottom-left corner of the quarter rectangle.
 * @param width The width of the quarter rectangle.
 * @param height The height of the quarter rectangle.
 */
class QuarterRectangle final : public CompositeBoundary {
public:
    QuarterRectangle(const Point& bottomLeft, const double width, const double height)
        : bottomLeft(bottomLeft), width(width), height(height) {
        // Define points for quarter rectangle corners
        Point topLeft(bottomLeft.x, bottomLeft.y + height / 2);
        Point topRight(bottomLeft.x + width / 2, bottomLeft.y + height / 2); // Midpoint of the top edge
        Point bottomRight(bottomLeft.x + width / 2, bottomLeft.y); // Midpoint of the right edge

        // Add segments for the quarter rectangle
        addSegment(std::make_shared<LineSegment>(topLeft, topRight, false));  // Upper half of the right edge
        addSegment(std::make_shared<LineSegment>(topRight, bottomRight, false));  // Right half of the top edge
    }

    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        minX = bottomLeft.x;
        maxX = bottomLeft.x + width / 2;
        minY = bottomLeft.y;
        maxY = bottomLeft.y + height / 2;
    }

    /**
     * @brief Calculates the area of the quarter rectangle.
     *
     * This method computes the area of the quarter rectangle using the formula:
     * Area = (width / 2) * (height / 2).
     *
     * @return The area of the quarter rectangle.
     */
    [[nodiscard]] double getArea() const override {
        return (width / 2) * (height / 2);
    }

private:
    Point bottomLeft;
    double width;
    double height;
};

/**
 * @class XSymmetryHalfRectangle
 * @brief Represents a half rectangle in the positive y-plane with its center at the origin.
 *
 * This class defines a half rectangle boundary by specifying the center, width, and height.
 * The rectangle is open at the bottom and includes the left, top, and right edges, constructed in a clockwise fashion.
 */
class XSymmetryHalfRectangle final : public CompositeBoundary {
public:
    XSymmetryHalfRectangle(const Point& center, const double width, const double height)
        : center(center), width(width), height(height) {
        // Define points for half rectangle in the positive y plane, in a clockwise fashion
        Point bottomLeft(center.x - width / 2, center.y);
        Point topLeft(center.x - width / 2, center.y + height / 2);
        Point topRight(center.x + width / 2, center.y + height / 2);
        Point bottomRight(center.x + width / 2, center.y);

        // Add segments for the half rectangle
        addSegment(std::make_shared<LineSegment>(bottomLeft, topLeft, false));  // Left edge
        addSegment(std::make_shared<LineSegment>(topLeft, topRight, false));  // Top edge
        addSegment(std::make_shared<LineSegment>(topRight, bottomRight, false));  // Right edge
    }

    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        minX = center.x - width / 2;
        maxX = center.x + width / 2;
        minY = center.y;
        maxY = center.y + height / 2;
    }

    /**
     * @brief Calculates the area of the half rectangle.
     * @return The area of the half rectangle.
     */
    [[nodiscard]] double getArea() const override {
        return width * height / 2;
    }

private:
    Point center;
    double width;
    double height;
};

/**
 * @class YSymmetryHalfRectangle
 * @brief Represents a half rectangle in the positive x-plane with its center at the origin.
 *
 * This class defines a half rectangle boundary by specifying the center, width, and height.
 * The rectangle is open at the left and includes the top, right, and bottom edges, constructed in a clockwise fashion.
 */
class YSymmetryHalfRectangle final : public CompositeBoundary {
public:
    YSymmetryHalfRectangle(const Point& center, const double width, const double height)
        : center(center), width(width), height(height) {
        // Define points for half rectangle in the positive x plane, in a clockwise fashion
        Point topLeft(center.x, center.y + height / 2);
        Point topRight(center.x + width / 2, center.y + height / 2);
        Point bottomRight(center.x + width / 2, center.y - height / 2);
        Point bottomLeft(center.x, center.y - height / 2);

        // Add segments for the half rectangle
        addSegment(std::make_shared<LineSegment>(topLeft, topRight, false));  // Top edge
        addSegment(std::make_shared<LineSegment>(topRight, bottomRight, false));  // Right edge
        addSegment(std::make_shared<LineSegment>(bottomRight, bottomLeft, false));  // Bottom edge
    }

    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        minX = center.x;
        maxX = center.x + width / 2;
        minY = center.y - height / 2;
        maxY = center.y + height / 2;
    }

    /**
     * @brief Calculates the area of the half rectangle.
     * @return The area of the half rectangle.
     */
    [[nodiscard]] double getArea() const override {
        return width * height / 2;
    }

private:
    Point center;
    double width;
    double height;
};

/**
 * @brief Represents a Bunimovich stadium, a dynamical billiard with a geometry that consists of two semicircles connected by straight segments.
 *
 * The Bunimovich stadium is widely used in chaos theory and dynamical systems for exploring chaotic trajectories and their statistical properties.
 *
 * @param center1 The center of the left semicircle.
 * @param radius Radius of the semicircles.
 * @param length Length of the straight segment connecting the semicircles.
 */
class BunimovichStadium final : public CompositeBoundary {
public:
    BunimovichStadium(const Point &center1, double radius, const double length) : radius(radius), length(length) {
        Point left(center1.x - length / 2, center1.y);
        Point right(center1.x + length / 2, center1.y);

        addSegment(std::make_shared<SemiCircle>(right, radius, Point{1, 0}, true));  // Right semicircle
        addSegment(std::make_shared<LineSegment>(Point(left.x, left.y - radius), Point(right.x, right.y - radius), true)); // Bottom edge
        addSegment(std::make_shared<SemiCircle>(left, radius, Point{-1, 0}, true));  // Left semicircle
        addSegment(std::make_shared<LineSegment>(Point(left.x, left.y + radius), Point(right.x, right.y + radius), false)); // Top edge
    }

    /**
     * @brief Calculates the area of the Bunimovich stadium.
     *
     * This method computes the area of the stadium using the formula:
     * Area = (π * radius^2) + (length * 2 * radius).
     *
     * @return The area of the Bunimovich stadium.
     */
    [[nodiscard]] double getArea() const override {
        return (M_PI * radius * radius) + (length * 2 * radius);
    }

    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    };

private:
    double radius;
    double length;
};

/**
 * @brief Represents one quarter of a Bunimovich stadium, primarily used for studies involving symmetrical properties of the stadium.
 *
 * This class simplifies the geometry to a quarter circle combined with a straight edge, making it suitable for partial system analyses.
 *
 * @param center Center point at the base of the quarter circle.
 * @param radius Radius of the quarter circle.
 * @param length Length of the horizontal segment from the center to the edge of the quarter circle.
 */
class QuarterBunimovichStadium final : public CompositeBoundary {
public:
    QuarterBunimovichStadium(const Point &center, double radius, const double length) : center(center), radius(radius), length(length) {
        // Assume 'center' is the midpoint of the base line at the bottom
        Point quarterCircleCenter(center.x + length / 2, center.y);  // Center of the quarter circle
        Point lineSegmentStart(quarterCircleCenter.x - length / 2, quarterCircleCenter.y + radius);
        Point lineSegmentEnd(quarterCircleCenter.x, quarterCircleCenter.y + radius);
        // Add base line segment
        addSegment(std::make_shared<LineSegment>(lineSegmentStart,
                                                 lineSegmentEnd, false));  // Horizontal base line
        // Add the quarter circle facing upwards, which is a 90-degree segment
        addSegment(std::make_shared<QuarterCircle>(quarterCircleCenter, radius, Point{1, 0}, true));  // Counter-clockwise
    }

    /**
     * @brief Calculates the area of the quarter Bunimovich stadium.
     *
     * This method computes the area of the quarter stadium using the formula:
     * Area = (π / 4 * radius^2) + (length * radius / 2).
     *
     * @return The area of the quarter Bunimovich stadium.
     */
    [[nodiscard]] double getArea() const override {
        return (M_PI / 4 * radius * radius) + (length * radius / 2);
    }

    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    };
    /**
     * @brief Computes the bounding box of the quarter Bunimovich stadium.
     *
     * @param minX Reference to store the minimum x-coordinate.
     * @param maxX Reference to store the maximum x-coordinate.
     * @param minY Reference to store the minimum y-coordinate.
     * @param maxY Reference to store the maximum y-coordinate.
     */
    void getBoundingBox(double& minX, double& maxX, double& minY, double& maxY) const override {
        minX = center.x;
        maxX = center.x + length / 2 + radius;
        minY = center.y;
        maxY = center.y + radius;
    }

private:
    Point center;
    double radius;
    double length;
};

/**
 * @brief Models a mushroom-shaped billiard, a popular shape in dynamical systems used to explore properties of chaotic and integrable regions.
 *
 * The Mushroom billiard combines a semicircular cap with a rectangular stem, providing a mixed phase space for dynamic studies.
 *
 * @param center Center of the semicircular cap.
 * @param capRadius Radius of the cap.
 * @param stemHeight Height of the stem.
 * @param stemWidth Width of the stem.
 */
class MushroomBilliard final : public CompositeBoundary {
public:
    MushroomBilliard(Point center, double capRadius, const double stemHeight, const double stemWidth) : capRadius(capRadius), stemHeight(stemHeight), stemWidth(stemWidth) {
        // Points for the stem and the connections between cap and stem
        Point leftConnection(center.x - capRadius, center.y);   // Left connection to the cap
        Point rightConnection(center.x + capRadius, center.y);  // Right connection to the cap
        Point leftStemTop(center.x - stemWidth / 2, center.y - stemHeight);  // Top left of the stem
        Point rightStemTop(center.x + stemWidth / 2, center.y - stemHeight); // Top right of the stem
        Point leftStemBottom(center.x - stemWidth / 2, center.y); // Bottom left of the stem (connects to cap)
        Point rightStemBottom(center.x + stemWidth / 2, center.y); // Bottom right of the stem (connects to cap)

        // Add the semi-circle at the top, oriented upward
        addSegment(std::make_shared<SemiCircle>(center, capRadius, Point{0, 1}, true));  // Upward facing semicircle
        // Right segment connecting semicircle to stem
        addSegment(std::make_shared<LineSegment>(rightConnection, rightStemBottom, false)); // Right side from cap to stem
        // Right side of the stem
        addSegment(std::make_shared<LineSegment>(rightStemBottom, rightStemTop, false)); // Right side of the stem
        // Bottom of the stem
        addSegment(std::make_shared<LineSegment>(rightStemTop, leftStemTop, false)); // Bottom horizontal line of the stem
        // Left side of the stem
        addSegment(std::make_shared<LineSegment>(leftStemTop, leftStemBottom, false)); // Left side of the stem
        // Left segment connecting stem to semicircle
        addSegment(std::make_shared<LineSegment>(leftStemBottom, leftConnection, false)); // Left side from stem to cap
    }

    /**
    •	@brief Calculates the area of the mushroom billiard.

    •	This method computes the area of the mushroom billiard using the formula:
    •	Area = (π * radius^2 / 2) + (stemHeight * stemWidth).

    •	@return The area of the mushroom billiard.
*/
    [[nodiscard]] double getArea() const override {
        return (M_PI * capRadius * capRadius / 2) + (stemHeight * stemWidth);
    }

    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    };

private:
    double capRadius;
    double stemHeight;
    double stemWidth;
};

/**
 * @brief Represents half of a mushroom billiard table, reflecting symmetry and simplifying simulations in systems where only half of the structure is relevant.
 *
 * This class is particularly useful in studies that take advantage of the symmetry of the mushroom shape to reduce computational overhead.
 *
 * @param center Geometric center at the base of the semicircular cap.
 * @param capRadius Radius of the semicircular cap.
 * @param stemHeight Height of the stem.
 * @param stemWidth Width of the stem extending from the center to the edge.
 */
class HalfMushroomBilliard final : public CompositeBoundary {
public:
    HalfMushroomBilliard(Point center, double capRadius, const double stemHeight, const double stemWidth) : capRadius(capRadius), stemHeight(stemHeight), stemWidth(stemWidth) {
        // Points for the stem
        Point rightConnection(center.x + capRadius, center.y);  // Right connection to the cap
        Point rightStemTop(center.x + stemWidth / 2, center.y - stemHeight); // Top right of the stem
        Point rightStemBottom(center.x + stemWidth / 2, center.y); // Bottom right of the stem (connects to cap)
        Point stemEndBottom(center.x, center.y - stemHeight); // End point for the bottom line
        // Add the quarter-circle at the top, oriented to cover the right half of a full circle
        addSegment(std::make_shared<QuarterCircle>(center, capRadius, Point{1, 0}, true)); // Quarter-circle facing right
        // Add line segments for the stem, ensuring normals point outward and maintaining a clockwise arrangement
        addSegment(std::make_shared<LineSegment>(rightConnection, rightStemBottom, false)); // Right side from cap to stem
        addSegment(std::make_shared<LineSegment>(rightStemBottom, rightStemTop, false)); // Right side of the stem
        addSegment(std::make_shared<LineSegment>(rightStemTop, stemEndBottom, false)); // Top horizontal line of the stem back to center
    }

    /**
    •	@brief Calculates the area of the half mushroom billiard.
    •	This method computes the area of the half mushroom billiard using the formula:
    •	Area = (π * radius^2 / 4) + (stemHeight * stemWidth / 2)
    •	@return The area of the half mushroom billiard.
*/
    [[nodiscard]] double getArea() const override {
        return (M_PI * capRadius * capRadius / 4) + (stemHeight * stemWidth / 2);
    }

private:
    double capRadius;
    double stemHeight;
    double stemWidth;
};

/**
 * @brief Represents half of a mushroom billiard table, reflecting symmetry and simplifying simulations in systems where only half of the structure is relevant. THIS ONE HAS THE REAL EDGE ON THE Y-AXIS WHICH MEANS THAT IT IS A CLOSED BOUNDARY. So use the default integration strategy
 *
 * This class is particularly useful in studies that take advantage of the symmetry of the mushroom shape to reduce computational overhead.
 *
 * @param center Geometric center at the base of the semicircular cap.
 * @param capRadius Radius of the semicircular cap.
 * @param stemHeight Height of the stem.
 * @param stemWidth Width of the stem extending from the center to the edge.
 */
class HalfMushroomBilliardWithDirichletEdge final : public CompositeBoundary {
public:
    HalfMushroomBilliardWithDirichletEdge(Point center, double capRadius, const double stemHeight, const double stemWidth) : capRadius(capRadius), stemHeight(stemHeight), stemWidth(stemWidth) {
        // Points for the stem
        Point quarterCircleStart(center.x, center.y + capRadius); // Top of the quarter circle
        Point rightConnection(center.x + capRadius, center.y);  // Right connection to the cap
        Point rightStemTop(center.x + stemWidth / 2, center.y - stemHeight); // Top right of the stem
        Point rightStemBottom(center.x + stemWidth / 2, center.y); // Bottom right of the stem (connects to cap)
        Point stemEndBottom(center.x, center.y - stemHeight); // End point for the bottom line
        // Add the quarter-circle at the top, oriented to cover the right half of a full circle
        addSegment(std::make_shared<QuarterCircle>(center, capRadius, Point{1, 0}, true)); // Quarter-circle facing right
        // Add line segments for the stem, ensuring normals point outward and maintaining a clockwise arrangement
        addSegment(std::make_shared<LineSegment>(rightConnection, rightStemBottom, false)); // Right side from cap to stem
        addSegment(std::make_shared<LineSegment>(rightStemBottom, rightStemTop, false)); // Right side of the stem
        addSegment(std::make_shared<LineSegment>(rightStemTop, stemEndBottom, false)); // Top horizontal line of the stem back to center
        addSegment(std::make_shared<LineSegment>(stemEndBottom, quarterCircleStart, false)); // CLOSE THE BOUNDARY
    }

    /**
    •	@brief Calculates the area of the half mushroom billiard.
    •	This method computes the area of the half mushroom billiard using the formula:
    •	Area = (π * radius^2 / 4) + (stemHeight * stemWidth / 2)
    •	@return The area of the half mushroom billiard.
*/
    [[nodiscard]] double getArea() const override {
        return (M_PI * capRadius * capRadius / 4) + (stemHeight * stemWidth / 2);
    }

private:
    double capRadius;
    double stemHeight;
    double stemWidth;
};

/**
 * @class HalfMushroomBilliardNoStemWithDirichletEdge
 * @brief Represents the limiting case of a half mushroom billiard with no stem, consisting of a quarter circle and a line segment connecting the end of the quarter circle to the center. THIS IS A FULL BOUNDARY
 *
 * This class constructs the boundary of a half mushroom billiard without the stem. The boundary consists of a quarter circle
 * and a line segment connecting the end of the quarter circle back to the center point. It inherits from CompositeBoundary to combine these two segments.
 */
class HalfMushroomBilliardNoStemWithDirichletEdge final : public CompositeBoundary {
public:
    HalfMushroomBilliardNoStemWithDirichletEdge(Point center, double capRadius) : capRadius(capRadius) {
        Point quarterCircleStart(center.x, center.y + capRadius); // Top of the quarter circle
        Point rightConnection(center.x + capRadius, center.y);  // Right connection to the cap
        Point rightStemBottom(center.x, center.y); // Bottom right of the stem (connects to cap)

        // Add the quarter-circle at the top, oriented to cover the right half of a full circle
        addSegment(std::make_shared<QuarterCircle>(center, capRadius, Point{1, 0}, true)); // Quarter-circle facing right
        // Add line segments for the stem, ensuring normals point outward and maintaining a clockwise arrangement
        addSegment(std::make_shared<LineSegment>(rightConnection, rightStemBottom, false)); // Right side from cap to center
        addSegment(std::make_shared<LineSegment>(rightStemBottom, quarterCircleStart, false)); // CLOSE THE BOUNDARY
    }

    /**
    •	@brief Calculates the area of the half mushroom billiard with no stem.
    •	This method computes the area of the half mushroom billiard using the formula:
    •	Area = (π * radius^2 / 4).
    •	@return The area of the half mushroom billiard with no stem.
*/
    [[nodiscard]] double getArea() const override {
        return (M_PI * capRadius * capRadius / 4);
    }

private:
    double capRadius;
};

/**
 * @class HalfMushroomBilliardFullStemWidthWithDirichletEdge
 * @brief Represents a half mushroom billiard with the width of the stem equal to the radius of the cap.
 *
 * This class constructs the boundary of a half mushroom billiard where the width of the stem is equal to the radius of the quarter circle cap.
 * The boundary consists of a quarter circle and three line segments forming a closed shape.
 */
class HalfMushroomBilliardFullStemWidthWithDirichletEdge final : public CompositeBoundary {
public:
    HalfMushroomBilliardFullStemWidthWithDirichletEdge(Point center, double capRadius, const double stemHeight) : capRadius(capRadius), stemHeight(stemHeight){
        Point quarterCircleStart(center.x, center.y + capRadius); // Top of the quarter circle
        Point rightStemTop(center.x + capRadius, center.y - stemHeight); // Top right of the stem
        Point rightStemBottom(center.x + capRadius, center.y); // Bottom right of the stem (connects to cap)
        Point stemEndBottom(center.x, center.y - stemHeight); // End point for the bottom line
        // Add the quarter-circle at the top, oriented to cover the right half of a full circle
        addSegment(std::make_shared<QuarterCircle>(center, capRadius, Point{1, 0}, true)); // Quarter-circle facing right
        addSegment(std::make_shared<LineSegment>(rightStemBottom, rightStemTop, false)); // Right side of the stem
        addSegment(std::make_shared<LineSegment>(rightStemTop, stemEndBottom, false)); // Top horizontal line of the stem back to center
        addSegment(std::make_shared<LineSegment>(stemEndBottom, quarterCircleStart, false)); // CLOSE THE BOUNDARY
    }
    /**
    •	@brief Calculates the area of the half mushroom billiard with the width of the stem the same as the radius of the cap.
    •	@return The area of the half mushroom billiard with this geometry.
*/
    [[nodiscard]] double getArea() const override {
        return (M_PI * capRadius * capRadius / 4) + stemHeight * capRadius;
    }
private:
    double capRadius;
    double stemHeight;
};

class C3DesymmetrizedBilliard final : public CompositeBoundary {
public:
    /**
     * @brief Constructs a C3DesymmetrizedBilliard by combining a desymmetrized C3 curve with line segments to the origin.
     * @param a Amplitude parameter that affects the shape of the desymmetrized C3 curve.
     */
    explicit C3DesymmetrizedBilliard(const double a) : a_(a) {
        // Construct the desymmetrized C3 curve
        const auto c3Curve = std::make_shared<C3CurveDesymmetrized>(a);
        // Get the start and end points of the desymmetrized C3 curve
        Point start = c3Curve->curveParametrization(0.0);
        Point end = c3Curve->curveParametrization(1.0);
        // Add the desymmetrized C3 curve segment
        addSegment(c3Curve);
        // Add line segment from the end of the C3 curve to the origin
        addSegment(std::make_shared<LineSegment>(end, Point{0, 0}, true));
        // Add line segment from the origin back to the start of the C3 curve
        addSegment(std::make_shared<LineSegment>(Point{0, 0}, start, true));
        c3Curve_ = c3Curve;
        end_ = end;
        start_ = start;
    }

    /**
     * @brief Indicates that this class supports point-in-boundary checks.
     * @return Always returns true.
     */
    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    }

    /**
     * @brief Determines if a point is inside the C3DesymmetrizedBilliard.
     * @param point The point to check.
     * @return True if the point is inside the boundary, false otherwise.
     *
     * This method uses the ray-casting algorithm to determine if the point is inside the boundary.
     * The algorithm counts how many times a horizontal ray from the point intersects the boundary.
     * If the count is odd, the point is inside; otherwise, it is outside.
     */
    [[nodiscard]] bool isInside(const Point& point) const override {
        int count = 0;
        auto checkIntersect = [&](const Point& p1, const Point& p2) {
            if ((p1.y > point.y) != (p2.y > point.y)) {
                if (const double xIntersect = p1.x + (point.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y); point.x < xIntersect) {
                    count++;
                }
            }
        };
        for (const auto& segment : this->getSegments()) {
            Point start = segment->curveParametrization(0.0);
            Point end = segment->curveParametrization(1.0);
            checkIntersect(start, end);
        }
        return (count % 2) == 1; // Inside if odd number of intersections
    }

    /**
     * @brief Calculates the area of the C3DesymmetrizedBilliard.
     *
     * This method calculates the area of the desymmetrized C3 curve and subtracts the
     * area of the triangle formed by the end of the curve and the origin (0,0).
     * The integration is performed using the GNU Scientific Library (GSL) to numerically
     * integrate the area under the curve.
     *
     * @return The total area of the C3DesymmetrizedBilliard.
     */
    [[nodiscard]] double getArea() const override {
        // Allocate workspace for the GSL integration
        gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(1000);
        double result, error;

        // Define the integrand function to calculate the area under the parametric curve
        auto integrand = [](const double t, void* params) -> double {
            // Cast the params to C3CurveDesymmetrized pointer
            const auto* c3Curve = static_cast<C3CurveDesymmetrized*>(params);
            // Get the point on the curve at parameter t
            const Point p = c3Curve->curveParametrization(t);
            double dx_dt, dy_dt;
            // Calculate the tangents at parameter t
            c3Curve->calculateTangents(t, dx_dt, dy_dt);
            // Return the integrand value: y(t) * sqrt((dx/dt)^2 + (dy/dt)^2)
            return p.y * std::hypot(dx_dt, dy_dt);
        };

        // Set up the GSL function for integration
        gsl_function F;
        F.function = integrand;
        F.params = c3Curve_.get();

        // Perform the integration over the interval [0, 1]
        gsl_integration_qags(&F, 0, 1.0, 0, 1e-7, 1000, workspace, &result, &error);
        // Free the allocated workspace
        gsl_integration_workspace_free(workspace);
        // The result of the integration is the area under the C3 desymmetrized curve
        const double c3CurveArea = result;
        // Calculate the area of the triangle formed by (end.x, end.y), (end.x, 0), and (0,0)
        const double triangleArea = std::abs(0.5 * end_.x * end_.y);
        // Return the total area of the billiard, which is the area under the curve
        // minus the area of the triangle
        return c3CurveArea - triangleArea;
    }

private:
    double a_;
    std::shared_ptr<C3CurveDesymmetrized> c3Curve_;
    Point end_;
    Point start_;
};

/**
 * @brief Defines a convex polygon billiard table by connecting a series of vertices in a specified order.
 *
 * This class constructs a billiard boundary in the shape of a polygon, with vertices sorted in a clockwise manner to
 * maintain a consistent definition of interior and exterior. The polygonx must be convex for the billiard dynamics
 * to be mathematically well-defined.
 *
 * @param vertices A list of points defining the vertices of the polygon, which will be sorted to ensure they are in
 *                 clockwise order. The sorting is based on their angles relative to the calculated centroid.
 */
class PolygonBilliard final : public CompositeBoundary {
public:
    explicit PolygonBilliard(const std::vector<Point>& vertices) : vertices(vertices) {
        std::vector<Point> sortedVertices = vertices;
        sortVertices(sortedVertices);  // Ensure vertices are in clockwise order

        for (size_t i = 0; i < sortedVertices.size(); i++) {
            const size_t next = (i + 1) % sortedVertices.size();
            addSegment(std::make_shared<LineSegment>(sortedVertices[i], sortedVertices[next], false));
        }
    }

    /**
     * @brief Calculates the area of the polygon.
     *
     * This method uses the shoelace formula (Gauss's area formula) to calculate the area of the polygon
     * defined by the vertices.
     *
     * @return The area of the polygon.
     */
    [[nodiscard]] double getArea() const override {
        return std::abs(signedArea(vertices));
    }

    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    };


private:
    std::vector<Point> vertices;
    /**
     * @brief Sorts the vertices of the polygon in clockwise order.
     *
     * This method calculates the centroid of the vertices and sorts them based on their angles relative to
     * the centroid. It ensures the vertices are in clockwise order by checking the signed area.
     *
     * @param vertices A list of points defining the vertices of the polygon.
     */
    static void sortVertices(std::vector<Point>& vertices) {
        Point centroid = findCentroid(vertices);
        std::ranges::sort(vertices, [&centroid](const Point& a, const Point& b) {
            const double angleA = atan2(a.y - centroid.y, a.x - centroid.x);
            const double angleB = atan2(b.y - centroid.y, b.x - centroid.x);
            return angleA < angleB;
        });

        if (signedArea(vertices) > 0) {
            std::ranges::reverse(vertices);  // Ensure clockwise order
        }
    }

    /**
     * @brief Finds the centroid of the given vertices.
     *
     * This method calculates the centroid (geometric center) of the vertices by averaging their coordinates.
     *
     * @param vertices A list of points defining the vertices of the polygon.
     * @return The centroid of the vertices.
     */
    static Point findCentroid(const std::vector<Point>& vertices) {
        Point centroid{0, 0};
        for (const auto& v : vertices) {
            centroid.x += v.x;
            centroid.y += v.y;
        }
        centroid.x /= vertices.size(); // NOLINT(*-narrowing-conversions)
        centroid.y /= vertices.size(); // NOLINT(*-narrowing-conversions)
        return centroid;
    }

    /**
     * @brief Calculates the signed area of the polygon.
     *
     * This method calculates the signed area of the polygon using the shoelace formula. The sign of the area
     * indicates the order of the vertices (positive for counterclockwise, negative for clockwise).
     *
     * @param vertices A list of points defining the vertices of the polygon.
     * @return The signed area of the polygon.
     */
    static double signedArea(const std::vector<Point>& vertices) {
        double area = 0.0;
        const size_t n = vertices.size();
        for (size_t i = 0; i < n; i++) {
            const size_t j = (i + 1) % n;
            area += vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y;
        }
        return area / 2.0;
    }
};

/**
 * @class RightTriangle
 * @brief Represents a right triangle boundary composed of three line segments.
 *
 * This class inherits from CompositeBoundary and defines a right triangle using three line segments.
 * The right triangle is defined by its two perpendicular sides k1 and k2.
 */
class RightTriangle final : public CompositeBoundary {
public:
    /**
     * @brief Constructs a RightTriangle.
     * @param k1 Length of the first perpendicular side.
     * @param k2 Length of the second perpendicular side.
     */
    RightTriangle(const double k1, const double k2) : k1(k1), k2(k2){
        // Define the vertices of the right triangle
        Point p1{-0.5, -0.5};      // First vertex at the origin
        Point p2{k1 -0.5, -0.5};     // Second vertex along the x-axis
        Point p3{-0.5, k2 - 0.5};     // Third vertex along the y-axis

        // Add line segments to form the right triangle
        addSegment(std::make_shared<LineSegment>(p1, p2));
        addSegment(std::make_shared<LineSegment>(p2, p3));
        addSegment(std::make_shared<LineSegment>(p3, p1));
    }

    /**
     * @brief Calculates the area of the right triangle.
     *
     * This method computes the area of the right triangle using the formula:
     * Area = 0.5 * base * height, where base is k1 and height is k2.
     *
     * @return The area of the right triangle.
     */
    [[nodiscard]] double getArea() const override {
        return 0.5 * k1 * k2;
    }

    [[nodiscard]] bool supportsIsInside() const override {
        return true;
    };

private:
    double k1;
    double k2;
};

}

#endif //BOUNDARY_HPP
