/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#ifndef FAIR_TOPK_BSPTREE_H
#define FAIR_TOPK_BSPTREE_H

#include <stack>
#include <iterator>
#include <concepts>
#include <algorithm>
#include <cstddef>

#include <Eigen/Dense>
#include <sdlp/sdlp.hpp>

#include "utility.h"
#include "memory.h"

namespace FairTopK {

template <class Func, int dimension>
concept LegitFairnessChecker = requires(Func func, const std::vector<std::pair<Plane<dimension>, bool> >& halfSpaces,
                                        Eigen::VectorXd& weights) {
    { func(halfSpaces, weights) } -> std::convertible_to<bool>;
};

template <int dimension>
class BSPTree {
public:
    BSPTree() = default;
    template <class Func> requires LegitFairnessChecker<Func, dimension>
    bool insert(const Plane<dimension>& plane, Eigen::VectorXd& weights, Func&& fairnessChecker);

    ~BSPTree() = default;
    BSPTree(const BSPTree&) = delete;
    BSPTree(BSPTree&&) = delete;
    BSPTree& operator=(const BSPTree&) = delete;
    BSPTree& operator=(BSPTree&&) = delete;
private:
    using BSPPlane = Plane<dimension>;
    struct Node {
        BSPPlane plane;
        Node *left = nullptr; //positive
        Node *right = nullptr; //negative
    };

    bool testIntersection(const BSPPlane& plane, const std::vector<std::pair<BSPPlane, bool> >& halfSpaces);

    MemoryArena<Node, CacheLineAlign, std::max(alignof(Node), (std::size_t)4)> nodePool;
    Node *root = nullptr;
};

template <int dimension>
template <class Func> requires LegitFairnessChecker<Func, dimension>
bool BSPTree<dimension>::insert(const Plane<dimension>& plane, Eigen::VectorXd& weights, Func&& fairnessChecker) {
    std::vector<std::pair<BSPPlane, bool> > halfSpaces;
    if (root == nullptr) {
        root = nodePool.Alloc();
        root->plane = plane;     
        halfSpaces.emplace_back(root->plane, true);
        if (std::forward<Func>(fairnessChecker)(halfSpaces, weights)) {
            return true;
        }
        halfSpaces[0].second = false;
        if (std::forward<Func>(fairnessChecker)(halfSpaces, weights)) {
            return true;
        }
        return false;
    }

    constexpr std::uintptr_t mask = 0x3;
    std::stack<std::uintptr_t> nodeStack;

    nodeStack.push((std::uintptr_t)root);
    while (!nodeStack.empty()) {
        std::uintptr_t markedNode = nodeStack.top();
        Node *node = (Node *)(markedNode & (~mask));
        unsigned int visitCounter = markedNode & mask;

        if (visitCounter >= 2) {
            nodeStack.pop();
            halfSpaces.pop_back();
            continue;
        }
        nodeStack.top() = ((std::uintptr_t)node | (visitCounter + 1));

        bool isPositive = (visitCounter == 0);
        if (isPositive) {
            halfSpaces.emplace_back(node->plane, true);
        }
        else {
            halfSpaces.back().second = false;
        }

        if (testIntersection(plane, halfSpaces)) {
            Node *nextNode = isPositive ? node->left : node->right;
            if (nextNode) {
                nodeStack.push((std::uintptr_t)nextNode);
            }
            else {
                if (std::forward<Func>(fairnessChecker)(halfSpaces, weights)) {
                    return true;
                }

                Node *newNode = nodePool.Alloc();
                newNode->plane = plane;
                if (isPositive) node->left = newNode;
                else node->right = newNode;
            }
        }
    }

    return false;
}

template <int dimension>
bool BSPTree<dimension>::testIntersection(const BSPPlane& plane, const std::vector<std::pair<BSPPlane, bool> >& halfSpaces) {
    constexpr double eps = 1e-10;
    if (halfSpaces.empty()) return true;
    if (halfSpaces.size() == 1) {
        const auto& [halfSpacePlane, isPositive] = halfSpaces[0];
        double dotProd = plane.normal.normalized().dot(halfSpacePlane.normal.normalized());
        if (std::abs(1.0 - dotProd) <= eps) {
            return isPositive ? plane.constant >= halfSpacePlane.constant :
                                plane.constant <= halfSpacePlane.constant;
        }
        return true;
    }
    
    using LPConstrsMat = Eigen::Matrix<double, dimension + 1, -1>;
    using LPVector = Eigen::Matrix<double, dimension + 1, 1>;
    using ColVector = Eigen::Matrix<double, dimension, 1>;

    int count = halfSpaces.size();
    constexpr int addConstrsCount = 2;

    LPConstrsMat mat = LPConstrsMat::Ones(dimension + 1, count + addConstrsCount);
    Eigen::VectorXd rhs(count + addConstrsCount);

    LPVector objCoeffs = LPVector::Zero();
    objCoeffs(dimension) = -1.0;
    LPVector results;

    for (int i = 0; i < count; i++) {
        const auto& [halfSpacePlane, isPositive] = halfSpaces[i];
        if (isPositive) {
            mat.col(i).template head<dimension>() = 
                -ColVector::Map(halfSpacePlane.normal.data());
            rhs(i) = -halfSpacePlane.constant;
        }
        else {
            mat.col(i).template head<dimension>() = 
                ColVector::Map(halfSpacePlane.normal.data());
            rhs(i) = halfSpacePlane.constant;
        }
    }

    mat.col(count).template head<dimension>() = -ColVector::Map(plane.normal.data());
    mat(dimension, count) = 0.0;
    rhs(count) = -plane.constant;
    mat.col(count + 1).template head<dimension>() = ColVector::Map(plane.normal.data());
    mat(dimension, count + 1) = 0.0;
    rhs(count + 1) = plane.constant;

    double val = sdlp::linprog<dimension + 1>(objCoeffs, mat, rhs, results);

    return val < 0.0;
}

}

#endif