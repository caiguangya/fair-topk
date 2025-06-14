/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#include <iostream>
#include <algorithm>
#include <random>
#include <boost/mp11/algorithm.hpp>

#include <Eigen/Dense>

#include "utility.h"
#include "experiments.h"
#include "data_loader.h"
#include "bsp_tree.h"

template<int dimension>
FairTopK::Plane<dimension - 1> getProjectedIntersection(const FairTopK::Plane<dimension>& plane1, 
    const FairTopK::Plane<dimension>& plane2) {
    using ProjectedVector = Eigen::Matrix<double, dimension - 1, 1>;
    using Matrix = Eigen::Matrix<double, 2, dimension - 1>;
    using RowVector = Eigen::Matrix<double, 1, dimension - 1>;

    //Find an itersection point with x_d = 1.0 and project it
    ProjectedVector normal1 = ProjectedVector::Map(plane1.normal.data());
    ProjectedVector normal2 = ProjectedVector::Map(plane2.normal.data());

    Matrix mat;
    mat.row(0) = RowVector::Map(normal1.data());
    mat.row(1) = RowVector::Map(normal2.data());

    Eigen::Vector2d rhs(plane1.constant - plane1.normal(dimension - 1),
        plane2.constant - plane2.normal(dimension - 1));
    ProjectedVector projectedPt = mat.colPivHouseholderQr().solve(rhs);

    double factor = (normal1.dot(projectedPt) - plane1.constant) / 
        (normal2.dot(projectedPt) - plane2.constant);
    
    FairTopK::Plane<dimension - 1> projectedPlane;
    projectedPlane.normal = (factor * normal1) - normal2;
    projectedPlane.constant = (factor * plane1.constant) - plane2.constant;

    return projectedPlane;
}

template<int dimension>
std::pair<Eigen::Matrix<double, dimension, -1>, Eigen::VectorXd> initLPConstraints(const Eigen::VectorXd& weights, double margin) {
    int constrsCount = 2 * dimension;
    constexpr int addConstrsCount = 2;

    using LPConstrsMat = Eigen::Matrix<double, dimension, -1>;
    using ColVector = Eigen::Matrix<double, dimension - 1, 1>;

    LPConstrsMat mat = LPConstrsMat::Zero(dimension, constrsCount + addConstrsCount);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(constrsCount + addConstrsCount);

    int offset = addConstrsCount; //Reserve first two cols

    for (int i = 0; i < dimension - 1; i++) {
        double lb = std::max(0.0, weights(i) - margin);
        double ub = std::min(1.0, weights(i) + margin);
            
        mat(i, offset + 2 * i) = 1.0;
        rhs(offset + 2 * i) = ub;
        mat(i, offset + 2 * i + 1) = -1.0;
        rhs(offset + 2 * i + 1) = -lb;
    }

    {
        int lastTwoOffset = offset + 2 * (dimension - 1);

        double lb = std::max(0.0, weights(dimension - 1) - margin);
        double ub = std::min(1.0, weights(dimension - 1) + margin);

        mat.col(lastTwoOffset).template head<dimension - 1>() = -ColVector::Ones();
        rhs(lastTwoOffset) = ub - 1.0;
        mat.col(lastTwoOffset + 1).template head<dimension - 1>() = ColVector::Ones();
        rhs(lastTwoOffset + 1) = 1.0 - lb;
    }

    mat.row(dimension - 1).tail(constrsCount).setOnes();

    return { std::move(mat), std::move(rhs) };
}

template<int dimension>
bool testIntersection(const FairTopK::Plane<dimension> plane,
    Eigen::Matrix<double, dimension + 1, -1>& mat, Eigen::VectorXd& rhs) {
    using LPVector = Eigen::Matrix<double, dimension + 1, 1>;
    using ColVector = Eigen::Matrix<double, dimension, 1>;

    LPVector objCoeffs = LPVector::Zero();
    objCoeffs(dimension) = -1.0;
    LPVector results;

    mat.col(0).template head<dimension>() = -ColVector::Map(plane.normal.data());
    rhs(0) = -plane.constant;
    mat.col(1).template head<dimension>() = ColVector::Map(plane.normal.data());
    rhs(1) = plane.constant;

    double val = sdlp::linprog<dimension + 1>(objCoeffs, mat, rhs, results);

    return val <= 0.0;
}

template<int projDimension>
bool findValidWeightVector(const std::vector<std::pair<FairTopK::Plane<projDimension>, bool> >& halfSpaces,
    const Eigen::VectorXd& refWeights, double margin, Eigen::VectorXd& weights) {
    using LPConstrsMat = Eigen::Matrix<double, projDimension + 1, -1>;
    using LPVector = Eigen::Matrix<double, projDimension + 1, 1>;
    using ColVector = Eigen::Matrix<double, projDimension, 1>;

    int count = halfSpaces.size();
    int addConstrsCount = 2 * (projDimension + 1);
    LPConstrsMat mat = LPConstrsMat::Zero(projDimension + 1, count + addConstrsCount);
    Eigen::VectorXd rhs(count + addConstrsCount);

    LPVector objCoeffs = LPVector::Zero();
    objCoeffs(projDimension) = -1.0;
    LPVector results;

    for (int i = 0; i < projDimension; i++) {
        double lb = std::max(0.0, refWeights(i) - margin);
        double ub = std::min(1.0, refWeights(i) + margin);
            
        mat(i, 2 * i) = 1.0;
        rhs(2 * i) = ub;
        mat(i, 2 * i + 1) = -1.0;
        rhs(2 * i + 1) = -lb;
    }

    {
        int lastTwoOffset = 2 * projDimension;

        double lb = std::max(0.0, refWeights(projDimension) - margin);
        double ub = std::min(1.0, refWeights(projDimension) + margin);

        mat.col(lastTwoOffset).template head<projDimension>() = -ColVector::Ones();
        rhs(lastTwoOffset) = ub - 1.0;
        mat.col(lastTwoOffset + 1).template head<projDimension>() = ColVector::Ones();
        rhs(lastTwoOffset + 1) = 1.0 - lb;
    }

    int offset = addConstrsCount;
    for (int i = 0; i < count; i++) {
        const auto& [halfSpacePlane, isPositive] = halfSpaces[i];
        int colIdx = offset + i;
        if (isPositive) {
            mat.col(colIdx).template head<projDimension>() = 
                -ColVector::Map(halfSpacePlane.normal.data());
            rhs(colIdx) = -halfSpacePlane.constant;
        }
        else {
            mat.col(colIdx).template head<projDimension>() = 
                ColVector::Map(halfSpacePlane.normal.data());
            rhs(colIdx) = halfSpacePlane.constant;
        }
        mat(projDimension, colIdx) = 1.0;
    }

    double val = sdlp::linprog<projDimension + 1>(objCoeffs, mat, rhs, results);

    if (val >= 0.0) 
        return false;

    results(projDimension) = 1.0 - results.template head<projDimension>().sum();
    weights = Eigen::VectorXd::Map(results.data(), projDimension + 1);
    return true;
}

template<int dimension>
bool solve(const std::vector<Eigen::VectorXd> &points,
const std::vector<int>& groups, int k, int pGroup, int pGroupLowerBound, 
int pGroupUpperBound, double margin, Eigen::VectorXd& weights) {
    using DualPlane = FairTopK::Plane<dimension>;
    using PlaneNormalVector = FairTopK::Plane<dimension>::NormalVector;
    using ProjectedVector = Eigen::Matrix<double, dimension - 1, 1>;

    constexpr double epsilon = 1e-8;

    int count = points.size();

    auto sortedPoints = points;
    std::sort(sortedPoints.begin(), sortedPoints.end(), [](const auto& p1, const auto& p2) {
        for (int i = 0; i < dimension; i++) {
            if (p1(i) < p2(i)) return true;
            else if (p1(i) > p2(i)) return false;
        }
        return false;
    });

    std::vector<std::pair<Eigen::VectorXd, DualPlane> > pointPlanePairs;
    pointPlanePairs.reserve(count);
    for (int i = 0; i < count; i++) {
        const auto& pt = sortedPoints[i];

        if (i > 0 && (pt.array() == sortedPoints[i - 1].array()).all())
            continue;

        DualPlane plane;
        plane.normal = PlaneNormalVector::Map(pt.data());
        plane.normal -= pt(dimension - 1) * PlaneNormalVector::Ones();
        plane.normal(dimension - 1) = -1.0;
        plane.constant = -pt(dimension - 1);

        pointPlanePairs.emplace_back(pt, std::move(plane));
    }

    std::default_random_engine rand(2024);
    std::shuffle(pointPlanePairs.begin(), pointPlanePairs.end(), rand);

    auto [mat, rhs] = initLPConstraints<dimension>(weights, margin);

    auto fairnessChecker = [&points, &groups, &weights, margin, pGroup, k,       
                            pGroupLowerBound, pGroupUpperBound
                           ](const std::vector<std::pair<FairTopK::Plane<dimension - 1>, bool> >& halfSpaces, 
                             Eigen::VectorXd& results) {
        bool found = findValidWeightVector(halfSpaces, weights, margin, results);
        if (found && FairTopK::checkFairness(points, groups, results, k, pGroup, pGroupLowerBound, pGroupUpperBound, epsilon)) {
            return true;
        }

        return false;
    };

    Eigen::VectorXd trialWeights(dimension);
    FairTopK::BSPTree<dimension - 1> tree;

    int distinctCount = pointPlanePairs.size();
    for (int i = 0; i < distinctCount - 1; i++) {
        const auto &[pt1, plane1] = pointPlanePairs[i];

        for (int j = i + 1; j < distinctCount; j++) {
            const auto &[pt2, plane2] = pointPlanePairs[j];
            auto diff = pt1 - pt2;
            const auto& diffArray = diff.array();
            if ((diffArray >= 0.0).all() || (diffArray <= 0.0).all()) {
                continue;
            }

            auto prjoectedPlane = getProjectedIntersection(plane1, plane2);

            if (testIntersection(prjoectedPlane, mat, rhs)) {
                bool found = tree.insert(prjoectedPlane, trialWeights, fairnessChecker);

                if (found) {
                    weights = std::move(trialWeights);
                    return true;
                }
            }
        }
    }

    return false;
}

int main(int argc, char* argv[]) {
    std::vector<Eigen::VectorXd> points;
    std::vector<int> groups;
    int protectedGroup = -1;

    auto [fileName, params] = FairTopK::parseCommandLine(argc, argv);

    FairTopK::DataLoader::readPreprocessedDataset(fileName, points, groups, protectedGroup);

    int dimension = points[0].rows();
    constexpr int minDimension = 3;
    constexpr int maxDimension = 6;

    if (dimension < minDimension || dimension > maxDimension) {
        std::cerr << "Only support datasets with 3 <= dimensions <= 6" << std::endl;
        return -1;
    }

    constexpr int dimCount = maxDimension - minDimension + 1;

    auto solveFunc = boost::mp11::mp_with_index<dimCount>(dimension - minDimension,
        [](auto dimDiff) { return solve<dimDiff() + minDimension>; });

    FairTopK::fairTopkExperiments(points, groups, protectedGroup, params, solveFunc);

    return 0;
}
