/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#include <iostream>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <thread>
#include <type_traits>
#include <boost/functional/hash.hpp>
#include <boost/mp11/algorithm.hpp>
#include <Eigen/Dense>

#include <cds/gc/nogc.h>
#include <cds/container/michael_list_nogc.h>
#include <cds/container/split_list_set_nogc.h>
#include <xenium/kirsch_kfifo_queue.hpp>
#include <xenium/reclamation/generic_epoch_based.hpp>
#include <sdlp/sdlp.hpp>

#include "data_loader.h"
#include "memory.h"
#include "utility.h"
#include "experiments.h"

struct KSetComparator {
    int operator ()(const int *left, const int *right) const {
        for (int i = 0; i < num; i++) {
            if (left[i] < right[i]) return -1;
            else if (left[i] > right[i]) return 1;
        }
        return 0;
    }
    static void Configurate(int k) noexcept { num = k; }
private:
    static int num;
};
int KSetComparator::num = 0;

struct KSetHash {
    std::size_t operator ()(const int* const kset) const {
        return boost::hash_range(kset, kset + num);
    }
    static void Configurate(int k) noexcept { num = k; }
private:
    static int num;
};
int KSetHash::num = 0;

void insertionSort(int *indices, int k, int substIdx) {
    int val = indices[substIdx];
    for (int i = substIdx; i < k - 1; i++) {
        indices[i] = indices[i + 1];
    }

    int idx = k - 1;
    while (idx > 0 && indices[idx - 1] > val) {
        indices[idx] = indices[idx - 1];
        idx -= 1;
    }
    indices[idx] = val;
}

template<int dimension>
std::pair<Eigen::Matrix<double, dimension, -1>, Eigen::VectorXd> initLPConstraints(
    const std::vector<Eigen::VectorXd> &points, const Eigen::VectorXd& weights, double margin) {
    int count = points.size();
    constexpr int addConstrsCount = 2 * dimension;

    using LPConstrsMat = Eigen::Matrix<double, dimension, -1>;
    using ColVec = Eigen::Matrix<double, dimension - 1, 1>;

    LPConstrsMat mat = LPConstrsMat::Zero(dimension, count + addConstrsCount);
    Eigen::VectorXd rhs(count + addConstrsCount);

    for (int i = 0; i < count; i++) {
        const auto &pt = points[i];
        mat.col(i).template head<dimension - 1>() = 
            ColVec::Map(pt.data()) - (pt(dimension - 1) * ColVec::Ones());
        mat(dimension - 1, i) = -1.0;
        rhs(i) = -pt(dimension - 1);
    }

    int offset = count;

    for (int i = 0; i < dimension - 1; i++) {
        double lb = std::max(0.0, weights(i) - margin);
        double ub = std::min(1.0, weights(i) + margin);
            
        mat(i, offset + 2 * i) = 1.0;
        rhs(offset + 2 * i) = ub;
        mat(i, offset + 2 * i + 1) = -1.0;
        rhs(offset + 2 * i + 1) = -lb;
    }

    {
        int lastTwoOffset =  offset + 2 * (dimension - 1);

        double lb = std::max(0.0, weights(dimension - 1) - margin);
        double ub = std::min(1.0, weights(dimension - 1) + margin);

        mat.col(lastTwoOffset).template head<dimension - 1>() = -ColVec::Ones();
        rhs(lastTwoOffset) = ub - 1.0;
        mat.col(lastTwoOffset + 1).template head<dimension - 1>() = ColVec::Ones();
        rhs(lastTwoOffset + 1) = 1.0 - lb;
    }

    return { std::move(mat), std::move(rhs) };
}

void getFirstKSet(int k, const std::vector<Eigen::VectorXd> &points, const Eigen::VectorXd& weights, int *firstKSet) {
    int count = points.size();
    std::vector<std::pair<double, int> > scores(count);

    for (int i = 0; i < count; i++) {
        scores[i] = { weights.dot(points[i]), i };
    }

    std::nth_element(scores.begin(), scores.begin() + (k - 1), scores.end(),
        [](const auto& p0, const auto& p1) { return p0.first > p1.first; });

    for (int i = 0; i < k; i++) {
        firstKSet[i] = scores[i].second;
    }
}

template<int dimension>
inline bool solveLP(const int *potKSet, int k, 
    const Eigen::Matrix<double, dimension, 1>& objCoeffs, 
    Eigen::Matrix<double, dimension, -1>& mat,
    Eigen::VectorXd& rhs, 
    Eigen::Matrix<double, dimension, 1>& vars) {
    for (int i = 0; i < k; i++) {
        mat.col(potKSet[i]) *= -1.0;
        rhs(potKSet[i]) = -rhs(potKSet[i]);
    }

    double val = sdlp::linprog<dimension>(objCoeffs, mat, rhs, vars);

    for (int i = 0; i < k; i++) {
        mat.col(potKSet[i]) *= -1.0;
        rhs(potKSet[i]) = -rhs(potKSet[i]);
    }

    return val != INFINITY;
}

template<int dimension>
inline bool solveLP(int preIdx, 
    const Eigen::Matrix<double, dimension, 1>& objCoeffs,
    Eigen::Matrix<double, dimension, -1>& mat,
    Eigen::VectorXd& rhs, 
    Eigen::Matrix<double, dimension, 1>& vars) {
    mat.col(preIdx) *= -1.0;
    rhs(preIdx) = -rhs(preIdx);

    double val = sdlp::linprog<dimension>(objCoeffs, mat, rhs, vars);

    mat.col(preIdx) *= -1.0;
    rhs(preIdx) = -rhs(preIdx);

    return val != INFINITY;
}

template <int dimension>
bool sequentialSolve(const std::vector<Eigen::VectorXd> &points, const std::vector<int>& groups, int k, int pGroup, 
int pGroupLowerBound, int pGroupUpperBound, double margin, Eigen::VectorXd& weights) {
    int count = points.size();
    
    auto [mat, rhs] = initLPConstraints<dimension>(points, weights, margin);

    using LPVector = Eigen::Matrix<double, dimension, 1>;

    LPVector vars = LPVector::Zero();
    LPVector objCoeffs = LPVector::Zero(); 
    objCoeffs(dimension - 1) = -1.0;

    auto hash = [k](const int* const kset) -> std::size_t {
        return boost::hash_range(kset, kset + k);
    };

    auto comp = [k](const int *left, const int *right) {
        for (int i = 0; i < k; i++) {
            if (left[i] != right[i]) return false;
        }
        
        return true;
    };

    std::queue<std::pair<const int*, int> > kSetQueue;
    std::unordered_set<int *, decltype(hash), decltype(comp)> kSets(std::max(count, 16384), hash, comp);

    FairTopK::MemoryArena<int, alignof(int)> pool(128 * k);

    int *firstKSet = pool.Alloc(k, std::false_type{});
    getFirstKSet(k, points, weights, firstKSet);
    std::sort(firstKSet, firstKSet + k);
    int firstPGroupCount = 0;
    for (int i = 0; i < k; i++)
        firstPGroupCount += (groups[firstKSet[i]] == pGroup);

    kSets.insert(firstKSet);
    kSetQueue.emplace(firstKSet, firstPGroupCount);

    while (!kSetQueue.empty()) {
        const auto [kSet, pGroupCount] = kSetQueue.front();
        kSetQueue.pop();

        for (int i = 0; i < k; i++) {
            mat.col(kSet[i]) *= -1.0;
            rhs(kSet[i]) = -rhs(kSet[i]);
        }

        int *potKSet = nullptr;
        int kSetEleIdx = 0;
        for (int i = 0; i < count; i++) {
            if (kSetEleIdx < k && kSet[kSetEleIdx] == i) {
                kSetEleIdx += 1;
            }
            else {
                mat.col(i) *= -1.0;
                rhs(i) = -rhs(i);
                const auto& newPt = points[i].array();
                for (int j = 0; j < k; j++) {
                    int preEleIdx = kSet[j];
                    const auto& prePt = points[preEleIdx].array();
                    if ((newPt < prePt).all()) continue;

                    if (potKSet == nullptr) potKSet = pool.Alloc(k, std::false_type{});
                    
                    std::copy(kSet, kSet + k, potKSet);
                    potKSet[j] = i;
                    insertionSort(potKSet, k, j);

                    if (kSets.find(potKSet) != kSets.cend()) continue;

                    bool valid = solveLP<dimension>(preEleIdx, objCoeffs, mat, rhs, vars);

                    if (!valid) continue;

                    int potPGroupCount = pGroupCount - (groups[preEleIdx] == pGroup) + 
                        (groups[i] == pGroup);

                    if (potPGroupCount >= pGroupLowerBound && potPGroupCount <= pGroupUpperBound) {
                        vars(dimension - 1) = 1.0 - vars.template head<dimension - 1>().sum();
                        weights = Eigen::VectorXd::Map(vars.data(), dimension);
                        return true;
                    }

                    kSets.insert(potKSet);
                    kSetQueue.emplace(potKSet, potPGroupCount);
                    potKSet = nullptr;
                }
                mat.col(i) *= -1.0;
                rhs(i) = -rhs(i);
            }
        }

        for (int i = 0; i < k; i++) {
            mat.col(kSet[i]) *= -1.0;
            rhs(kSet[i]) = -rhs(kSet[i]);
        }
    }

    return false;
}

template<int dimension>
bool parallelSolve(int totalThreadCount,
    const std::vector<Eigen::VectorXd> &points, const std::vector<int>& groups, int k, int pGroup, 
    int pGroupLowerBound, int pGroupUpperBound, double margin, Eigen::VectorXd& weights) {
    KSetComparator::Configurate(k);
    KSetHash::Configurate(k);

    struct PotKSet {
        const int *kSet = nullptr;
        std::int32_t substIdx = -1;
        std::int32_t newEle = -1;
    };

    struct split_list_traits : public cds::container::split_list::traits {
        typedef KSetHash hash;
        struct ordered_list_traits : public cds::container::michael_list::traits {
            typedef KSetComparator compare;
        };
    };

    using KSetsHashSet = cds::container::SplitListSet<cds::gc::nogc, int *, split_list_traits>;

    KSetsHashSet kSets(std::max(points.size(), (std::size_t)16384));

    using reclaimer = xenium::policy::reclaimer<xenium::reclamation::debra<> >;
    using padding_bytes = xenium::policy::padding_bytes<0>;
    using KSetQueue = xenium::kirsch_kfifo_queue<PotKSet*, padding_bytes, reclaimer>;

    KSetQueue kSetQueue(std::clamp(totalThreadCount * 4, 32, 512));

    using KsetArena = FairTopK::MemoryArena<int, FairTopK::CacheLineAlign, FairTopK::CacheLineAlign>;
    using PotKSetPool = FairTopK::MemoryPool<PotKSet>;

    //Keeps everything relevant in memory until completion of all spawned threads
    KsetArena **arenas = new KsetArena* [totalThreadCount];
    PotKSetPool **pools = new PotKSetPool* [totalThreadCount];

    std::atomic<int> workingThreadCount;
    std::atomic_flag found = ATOMIC_FLAG_INIT;

    workingThreadCount.store(totalThreadCount, std::memory_order_relaxed);

    auto refWeights = weights;

    auto func = [&points, margin, k, &kSetQueue, &kSets, &workingThreadCount,
                 &groups, pGroup, pGroupLowerBound, pGroupUpperBound,
                 arenas, pools, &refWeights, &weights, &found]<bool init>(int threadIdx, std::bool_constant<init>) {
        bool decremented = false;
        int count = points.size();

        int nodeSize = k + 1;
        constexpr std::size_t alignComplement = FairTopK::CacheLineAlign - 1;
        std::size_t alignedSize = ((nodeSize * sizeof(int) + alignComplement) & (~alignComplement)) / sizeof(int);
        KsetArena *arena = new KsetArena(128 * alignedSize);

        PotKSetPool *pool = new PotKSetPool(std::min(count * k, 4194304));
        arenas[threadIdx] = arena;
        pools[threadIdx] = pool;

        if constexpr (init) {
            int *firstKSet = arena->Alloc(nodeSize, std::false_type{});
            getFirstKSet(k, points, refWeights, firstKSet);
            std::sort(firstKSet, firstKSet + k);
            int firstPGroupCount = 0;
            for (int i = 0; i < k; i++)
                firstPGroupCount += (groups[firstKSet[i]] == pGroup);
            firstKSet[k] = firstPGroupCount;
            
            kSets.insert(firstKSet);

            int kSetEleIdx = 0;
            for (int i = 0; i < count; i++) {
                if (kSetEleIdx < k && firstKSet[kSetEleIdx] == i) {
                    kSetEleIdx += 1;
                }
                else {
                    const auto& newPt = points[i].array();
                    for (int j = 0; j < k; j++) {
                        const auto& prePt = points[firstKSet[j]].array();
                        if ((newPt >= prePt).any())
                            kSetQueue.push(pool->Alloc(firstKSet, j, i));
                    }
                }
            }

            if (found.test(std::memory_order_relaxed)) return;
        }

        auto [mat, rhs] = initLPConstraints<dimension>(points, refWeights, margin);

        using LPVector = Eigen::Matrix<double, dimension, 1>;

        LPVector vars = LPVector::Zero();
        LPVector objCoeffs = LPVector::Zero();
        objCoeffs(dimension - 1) = -1.0;

        int *potKSetEles = nullptr;

        //Increment the workingThreadCount between two linearization points and right before the linearization point of pop()
        auto foundBeforeFunc = [&decremented, &workingThreadCount]() noexcept {
            if (decremented) {
                workingThreadCount.fetch_add(1, std::memory_order_relaxed);
                decremented = false;
            }
        };

        while (!found.test(std::memory_order_relaxed)) {
            auto potKSetPtOpt = kSetQueue.pop(foundBeforeFunc);
            if (potKSetPtOpt) {
                PotKSet potKSet = **potKSetPtOpt;
                pool->Dealloc(*potKSetPtOpt);
                const int *kSet = potKSet.kSet;

                if (potKSetEles == nullptr) potKSetEles = arena->Alloc(nodeSize, std::false_type{});
                    
                std::copy(kSet, kSet + k, potKSetEles);
                potKSetEles[potKSet.substIdx] = potKSet.newEle;
                insertionSort(potKSetEles, k, potKSet.substIdx);

                int pGroupCount = kSet[k];
                    
                if (kSets.contains(potKSetEles) != kSets.end()) continue;

                bool valid = solveLP<dimension>(potKSetEles, k, objCoeffs, mat, rhs, vars);

                if (!valid) continue;

                pGroupCount = pGroupCount - (groups[kSet[potKSet.substIdx]] == pGroup) + 
                    (groups[potKSet.newEle] == pGroup);
                    
                if (pGroupCount >= pGroupLowerBound && pGroupCount <= pGroupUpperBound) {
                    if (!found.test_and_set(std::memory_order_acquire)) {
                        vars(dimension - 1) = 1.0 - vars.template head<dimension - 1>().sum();
                        weights = Eigen::VectorXd::Map(vars.data(), dimension);
                    }
                    return;
                }

                if (found.test(std::memory_order_relaxed)) return;
                    
                potKSetEles[k] = pGroupCount;
                auto iter = kSets.insert(potKSetEles);

                if (iter == kSets.end()) continue;
                    
                int kSetEleIdx = 0;
                for (int i = 0; i < count; i++) {
                    if (kSetEleIdx < k && potKSetEles[kSetEleIdx] == i) {
                        kSetEleIdx += 1;
                    }
                    else {
                        const auto& newPt = points[i].array();
                        for (int j = 0; j < k; j++) {
                            if (found.test(std::memory_order_relaxed)) return;

                            const auto& prePt = points[potKSetEles[j]].array();
                            if ((newPt >= prePt).any())
                                kSetQueue.push(pool->Alloc(potKSetEles, j, i));
                        }
                    }
                }

                potKSetEles = nullptr;
            }
            else {
                if (!decremented) {
                    workingThreadCount.fetch_sub(1, std::memory_order_relaxed);
                    decremented = true;
                }
                if (workingThreadCount.load(std::memory_order_relaxed) <= 0)
                    break;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(totalThreadCount);
    threads.push_back(std::thread(func, 0, std::true_type{}));
    for (int i = 1; i < totalThreadCount; i++)
        threads.push_back(std::thread(func, i, std::false_type{}));

    for (auto& thread : threads) thread.join();

    for (int i = 0; i < totalThreadCount; i++) {
        delete arenas[i];
        delete pools[i];
    }
    delete[] arenas;
    delete[] pools;
    
    return found.test(std::memory_order_relaxed);
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
    if (params.threadCount <= 0) {
        std::cerr << "The number of threads must be greater than or equal to 1" << std::endl;
        return -1;
    }

    constexpr int dimCount = maxDimension - minDimension + 1;
    int dimDiff = dimension - minDimension;

    if (params.threadCount > 1) {
        auto solveFunc = boost::mp11::mp_with_index<dimCount>(dimDiff,
            [](auto dimDiff) { return parallelSolve<dimDiff() + minDimension>; });

        FairTopK::fairTopkExperiments(points, groups, protectedGroup, params, 
            [threadCount = params.threadCount, solveFunc]<class... Args>(Args&&... params) { 
                return solveFunc(threadCount, std::forward<Args>(params)...);
        });
    }
    else {        
        auto solveFunc = boost::mp11::mp_with_index<dimCount>(dimDiff,
            [](auto dimDiff) { return sequentialSolve<dimDiff() + minDimension>; });

        FairTopK::fairTopkExperiments(points, groups, protectedGroup, params, solveFunc);
    }

    return 0;
}
