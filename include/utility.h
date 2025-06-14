/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#ifndef FAIR_TOPK_UTILITY_H
#define FAIR_TOPK_UTILITY_H

#include <vector>
#include <random>

#include <Eigen/Dense>

namespace FairTopK {

bool checkFairness(const std::vector<Eigen::VectorXd> &points,
const std::vector<int>& groups, const Eigen::VectorXd& weights,
int k, int pGroup, int pGroupLowerBound, int pGroupUpperBound, double epsilon = 1e-8);

Eigen::VectorXd getRandomWeightVector(int dimension, std::default_random_engine& rand);
std::vector<Eigen::VectorXd> getRandomWeightVectors(int count, int dimension);
std::vector<Eigen::VectorXd> getRandomWeightVectors(int count, const std::vector<Eigen::VectorXd> &points,
const std::vector<int>& groups, int k, int pGroup, int pGroupLowerBound, int pGroupUpperBound);

template <int d>
struct Plane {
    using NormalVector = Eigen::Matrix<double, d, 1>;
    NormalVector normal; //Unnormalized
    double constant;
};

struct InputParams {
    int k = 0;
    int pGroupLowerBound = 0;
    int pGroupUpperBound = 0;
    double margin = 0.0;
    int threadCount = 0;
    int sampleCount = 0;
    bool uniformSampling = false;
    bool runtime = false;
    bool quality = false;
    bool unoptimized = false;
};

std::pair<std::string, InputParams> parseCommandLine(int argc, char* argv[]);

}

#endif
