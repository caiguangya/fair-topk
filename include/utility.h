#ifndef FAIR_TOPK_UTILITY_H
#define FAIR_TOPK_UTILITY_H

#include <vector>
#include <chrono>
#include <random>
#include <iostream>
#include <concepts>

#include <Eigen/Dense>

namespace FairTopK {

inline long long getElapsedTime(const std::chrono::steady_clock::time_point& start) noexcept {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
}

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

template <typename Func>
concept SolveExecutable = requires(Func solve, const std::vector<Eigen::VectorXd> &points, const std::vector<int>& groups, 
int k, int pGroup, int pGroupLowerBound, int pGroupUpperBound, double margin, Eigen::VectorXd& weights) {
    { solve(points, groups, k, pGroup, pGroupLowerBound, pGroupUpperBound, margin, weights) } -> std::convertible_to<bool>;
};

template <SolveExecutable Func>
void fairTopkMarginTimeProfiling(const std::vector<Eigen::VectorXd> &weightSamples, const std::vector<Eigen::VectorXd> &points,
const std::vector<int>& groups, int k, int pGroup, int pGroupLowerBound, int pGroupUpperBound, double margin, Func solve) {
    int dimension = points[0].rows();
	
    Eigen::VectorXd weights(dimension);
	
    auto startTime = std::chrono::steady_clock::now();

    int validCount = 0;
    for (const auto& sampleWeight : weightSamples) {
        weights = sampleWeight;
        bool valid = solve(points, groups, k, pGroup, pGroupLowerBound, pGroupUpperBound, margin, weights);
		
        validCount += valid;
    }

    auto elapsedTime = getElapsedTime(startTime);
    double avgTime = ((double)elapsedTime / 1000) / weightSamples.size();
	
    std::cout << validCount << "/" << weightSamples.size() << " fair weight vectors are found\n";
    std::cout << "Average run time: " << std::scientific << avgTime << std::endl;
}

void printInputInfos(int k, double pGroupLowerBound, double pGroupUpperBound, double margin, int threadCount = 0);

}

#endif
