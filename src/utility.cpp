#include "utility.h"
#include <random>
#include <limits>
#include <iostream>

namespace FairTopK {

bool checkFairness(const std::vector<Eigen::VectorXd> &points,
const std::vector<int>& groups, const Eigen::VectorXd& weights,
int k, int pGroup, int pGroupLowerBound, int pGroupUpperBound, double epsilon) {
    std::vector<std::pair<double, int> > pts;
    int count = points.size();

    pts.reserve(count);
    for (int i = 0; i < count; i++) {
        pts.emplace_back(weights.dot(points[i]), groups[i]);
    }

    std::nth_element(pts.begin(), pts.begin() + (k - 1), pts.end(),
        [](const auto& p0, const auto& p1) { return p0.first > p1.first; });

    double kthScore = pts[k - 1].first;

    int pGroupBaseCount = 0;
    int vacant = 0;
    int tieProtected = 0;
    int tieOther = 0;
    for (int i = 0; i < k; i++) {
        const auto [score, group] = pts[i];

        int isProtected = (group == pGroup);
        if (score - kthScore > epsilon) {
            pGroupBaseCount += isProtected;
        }
        else {
            vacant += 1;
            tieProtected += isProtected;
            tieOther += 1 - isProtected;
        }
    }

    for (int i = k; i < count; i++) {
        const auto [score, group] = pts[i];

        if (kthScore - score <= epsilon) {
            int isProtected = (group == pGroup);
            tieProtected += isProtected;
            tieOther += 1 - isProtected;
        }
    }

    int pGroupLowerCount = pGroupBaseCount + std::max(0, vacant - tieOther);
    int pGroupUpperCount = pGroupBaseCount + vacant - std::max(0, vacant - tieProtected);

    return std::max(pGroupLowerCount, pGroupLowerBound) <= std::min(pGroupUpperCount, pGroupUpperBound);
}

Eigen::VectorXd getRandomWeightVector(int dimension, std::default_random_engine& rand) {
    std::vector<double> rngGenNums(dimension, 0.0);
    std::uniform_real_distribution<double> dis(0.0, 1.0 + std::numeric_limits<double>::epsilon());
    for (int i = 0; i < dimension; i++) {
        rngGenNums[i] = dis(rand);
    }
    std::sort(rngGenNums.begin(), rngGenNums.end());

    Eigen::VectorXd weights(dimension);
    weights(0) = rngGenNums[0];
    for (int i = 1; i < dimension - 1; i++) {
        weights(i) = rngGenNums[i] - rngGenNums[i - 1];
    }
    weights(dimension - 1) = 1.0 - rngGenNums[dimension - 1];

    weights = weights.cwiseAbs();
    weights /= weights.sum();

    return weights;
}

std::vector<Eigen::VectorXd> getRandomWeightVectors(int count, int dimension) {
    std::vector<Eigen::VectorXd> vectors;
    vectors.reserve(count);

    std::default_random_engine rand(2024);

    for (int i = 0; i < count; i++) {
        Eigen::VectorXd weights = getRandomWeightVector(dimension, rand);

        vectors.push_back(std::move(weights));
    }

    return vectors;
}

std::vector<Eigen::VectorXd> getRandomWeightVectors(int count, const std::vector<Eigen::VectorXd> &points, 
const std::vector<int>& groups, int k, int pGroup, int pGroupLowerBound, int pGroupUpperBound) {
    std::vector<Eigen::VectorXd> vectors;
    vectors.reserve(count);

    std::default_random_engine rand(2024);

    int dimension = points[0].rows();
    while (vectors.size() < count) {
        Eigen::VectorXd weights = getRandomWeightVector(dimension, rand);
        if (!checkFairness(points, groups, weights, k, pGroup, pGroupLowerBound, pGroupUpperBound)) {
            vectors.push_back(std::move(weights));
        }
    }

    return vectors;
}

void printInputInfos(int k, double pGroupLowerBound, double pGroupUpperBound, double margin, int threadCount) {
    std::cout << "k: " << k << 
                 " | Protected Group Proportion Lower Bound: " << pGroupLowerBound <<
                 " | Protected Group Proportion Upper Bound: " << pGroupUpperBound <<
                 " | Epsilon: " << margin;

    if (threadCount > 0)
        std::cout << " | Number of Threads: " << threadCount;
        
    std::cout << std::endl;
}

}