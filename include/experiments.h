/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#ifndef FAIR_TOPK_EXPERIMENTS_H
#define FAIR_TOPK_EXPERIMENTS_H

#include <vector>
#include <iostream>
#include <chrono>
#include <concepts>
#include <utility>

#include <Eigen/Dense>
#include <boost/predef.h>

#include "utility.h"

#if (BOOST_COMP_GNUC || BOOST_COMP_CLANG)
#define BENCHMARK_ALWAYS_INLINE __attribute__((always_inline))
#elif BOOST_COMP_MSVC
#define BENCHMARK_ALWAYS_INLINE __forceinline
#else
#define BENCHMARK_ALWAYS_INLINE
#endif

namespace FairTopK {

template <class T>
inline BENCHMARK_ALWAYS_INLINE void doNotOptimize(T *p) {
#if BOOST_COMP_GNUC
    asm volatile("" : "+m,r"(p) : : "memory");
#elif BOOST_COMP_CLANG
    asm volatile("" : "+r,m"(p) : : "memory");
#else
    //Empty
#endif
}

inline BENCHMARK_ALWAYS_INLINE void clobberMemory() {
#if (BOOST_COMP_GNUC || BOOST_COMP_CLANG)
    asm volatile("" : : : "memory");
#else
    //Empty
#endif
}

inline BENCHMARK_ALWAYS_INLINE long long getElapsedTime(const std::chrono::steady_clock::time_point& start) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
}

template <typename Func>
concept SolveExecutable = requires(Func solve, const std::vector<Eigen::VectorXd> &points, const std::vector<int>& groups, 
    int k, int pGroup, int pGroupLowerBound, int pGroupUpperBound, double margin, Eigen::VectorXd& weights) {
    { solve(points, groups, k, pGroup, pGroupLowerBound, pGroupUpperBound, margin, weights) } -> std::convertible_to<bool>;
};

void testInputWeightVectors(const std::vector<Eigen::VectorXd> &points, const std::vector<int>& groups, int pGroup, 
    const InputParams& params, std::vector<Eigen::VectorXd> &weightVectorSamples);

void evaluateQuality(const std::vector<Eigen::VectorXd> &points, const std::vector<int>& groups, int pGroup, 
    const InputParams& params, const std::vector<std::pair<int, Eigen::VectorXd> >& fairVectors, 
    const std::vector<Eigen::VectorXd> &unfairVectors);

template <SolveExecutable Func, bool benchmarking>
std::vector<std::pair<int, Eigen::VectorXd> > findFairWeightVectors(const std::vector<Eigen::VectorXd> &points, 
    const std::vector<int>& groups, int pGroup, const InputParams& params, const std::vector<Eigen::VectorXd> &unfairVectors,
    Func&& solve) {
    int size = unfairVectors.size();
    std::vector<std::pair<int, Eigen::VectorXd> > fairVectors;
    fairVectors.reserve(size);

    int dimension = points[0].rows();
    
    Eigen::VectorXd weightVector(dimension);
    int foundCount = 0;

    std::chrono::steady_clock::time_point startTime;
    if constexpr (benchmarking) startTime = std::chrono::steady_clock::now();

    if constexpr (benchmarking) doNotOptimize(fairVectors.data());
    for (int i = 0; i < size; i++) {
        weightVector = unfairVectors[i];
        bool found = std::forward<Func>(solve)(points, groups, params.k, pGroup, 
            params.pGroupLowerBound, params.pGroupUpperBound, params.margin, weightVector);
		
        foundCount += found;
        if (found) fairVectors.emplace_back(i, weightVector);
    }
    if constexpr (benchmarking) clobberMemory();

    long long elapsedTime = 0;
    if constexpr (benchmarking) elapsedTime = getElapsedTime(startTime);

    double avgTime = 0.0;
    if constexpr (benchmarking) avgTime = ((double)elapsedTime / 1000) / size;
    
    std::cout << foundCount << "/" << size << " fair weight vectors are found" << std::endl;

    if constexpr (benchmarking) std::cout << "Average run time: " << std::scientific << avgTime << std::endl;

    return fairVectors;
}

template <SolveExecutable Func>
void fairTopkExperiments(const std::vector<Eigen::VectorXd> &points, const std::vector<int>& groups, int pGroup, 
    const InputParams& params, Func solve) {
    std::vector<Eigen::VectorXd> weightVectorSamples;
    if (params.uniformSampling) {
        int dimension = points[0].rows();
        weightVectorSamples = getRandomWeightVectors(params.sampleCount, dimension);
    }
    else {
        weightVectorSamples = FairTopK::getRandomWeightVectors(params.sampleCount, points, groups, params.k, pGroup, 
            params.pGroupLowerBound, params.pGroupUpperBound);
    }

    if (params.uniformSampling)
        testInputWeightVectors(points, groups, pGroup, params, weightVectorSamples);

    auto fairVectors = params.runtime ? 
        findFairWeightVectors<Func, true>(points, groups, pGroup, params, weightVectorSamples, std::move(solve)) :
        findFairWeightVectors<Func, false>(points, groups, pGroup, params, weightVectorSamples, std::move(solve));

    if (params.quality)
        evaluateQuality(points, groups, pGroup, params, fairVectors, weightVectorSamples);
}

}


#endif
