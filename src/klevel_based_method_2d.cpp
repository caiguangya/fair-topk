/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#include <iostream>
#include <limits>

#include <Eigen/Dense>

#include "utility.h"
#include "data_loader.h"
#include "experiments.h"
#include "tourney_tree.h"

struct GroupedLine {
    double k;
    double b;
    int group;
};

bool isFair(const std::vector<GroupedLine>& lines, int k, double time, int pGroup,
    int pGroupLowerBound, int pGroupUpperBound, double epsilon) {
    const GroupedLine& kthLine = lines[k - 1];
    double kthScore = kthLine.k * time + kthLine.b;

    int pGroupBaseCount = 0;
    int vacant = 1;
    int tieProtected = (kthLine.group == pGroup);
    int tieOther = 1 - tieProtected;

    for (int i = 0; i < k - 1; i++) {
        const GroupedLine& line = lines[i];
        double score = line.k * time + line.b;
        int group = line.group;

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

    int count = lines.size();
    for (int i = k; i < count; i++) {
        const GroupedLine& line = lines[i];
        double score = line.k * time + line.b;
        int group = line.group;

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

template <bool Opt = true>
bool solve(const std::vector<Eigen::VectorXd> &points, const std::vector<int>& groups, int k, int pGroup, 
    int pGroupLowerBound, int pGroupUpperBound, double margin, Eigen::VectorXd& weights) {
    double timeLower = std::max({ 0.0, weights(0) - margin, 1.0 - weights(1) - margin });
    double timeUpper = std::min({ 1.0, weights(0) + margin, 1.0 - weights(1) + margin });

    constexpr double epsilon = 1e-8;
    int count = points.size();

    std::vector<GroupedLine> lines;
    lines.reserve(count);

    for (int i = 0; i < count; i++) {
        const auto &point = points[i];
        lines.emplace_back(point(0) - point(1), point(1), groups[i]);
    }

    std::nth_element(lines.begin(), lines.begin() + (k - 1), lines.end(),
        [timeLower](const auto& l1, const auto& l2) { 
            double score1 = l1.k * timeLower + l1.b;
            double score2 = l2.k * timeLower + l2.b;
            double diff = score1 - score2;

            return std::abs(diff) > epsilon ? (diff > 0.0) : 
                (l1.k != l2.k ? l1.k > l2.k : l1.b > l2.b);
    });

    if (isFair(lines, k, timeLower, pGroup, pGroupLowerBound, pGroupUpperBound, epsilon)) {
        weights(0) = timeLower; 
        weights(1) = 1.0 - timeLower;
        
        return true;
    }

    auto less = [](const GroupedLine& left, const GroupedLine& right, double time) noexcept {
        double score1 = left.k * time + left.b;
        double score2 = right.k * time + right.b;
        double diff = score1 - score2;

        return std::abs(diff) > epsilon ? (diff < 0.0) : 
            (left.k != right.k ? left.k < right.k : left.b < right.b);
    };

    auto greater = [](const GroupedLine& left, const GroupedLine& right, double time) noexcept {
        double score1 = left.k * time + left.b;
        double score2 = right.k * time + right.b;
        double diff = score1 - score2;

        return std::abs(diff) > epsilon ? (diff > 0.0) : 
            (left.k != right.k ? left.k > right.k : left.b > right.b);
    };

    auto crossCompute = [](const GroupedLine& left, const GroupedLine& right) noexcept -> double {
        if (left.k == right.k) return std::numeric_limits<double>::infinity();

        return (right.b - left.b) / (left.k - right.k);
    };

    using TopKTreeType = FairTopK::KineticTourneyLineTree<GroupedLine, Opt, decltype(less), decltype(crossCompute)>;
    using OtherTreeType = FairTopK::KineticTourneyLineTree<GroupedLine, Opt, decltype(greater), decltype(crossCompute)>;

    TopKTreeType topKTree(lines.cbegin(), lines.cbegin() + k, timeLower, timeUpper, less, crossCompute);
    OtherTreeType otherTree(lines.cbegin() + k, lines.cend(), timeLower, timeUpper, greater, crossCompute);

    int pGroupCount = 0;
    for (int i = 0; i < k; i++)
        pGroupCount += (lines[i].group == pGroup);

    enum class AdvanceType { Exchange, TopK, Other };

    double curTime = timeLower;
    bool exchanged = false;
    while (true) {
        auto topKMin = topKTree.Top();
        auto otherMax = otherTree.Top();

        double exchangeTime = std::numeric_limits<double>::max();
        if (!exchanged && topKMin.k != otherMax.k && greater(otherMax, topKMin, timeUpper)) {
            exchangeTime = std::min((otherMax.b - topKMin.b) / (topKMin.k - otherMax.k), timeUpper);
        }

        double topKNextEventTime = topKTree.getNextEventTime();
        double otherNextEventTime = otherTree.getNextEventTime();

        double nextTime = exchangeTime;
        AdvanceType advanceType = AdvanceType::Exchange;
        if (topKNextEventTime < nextTime && !greater(otherMax, topKMin, topKNextEventTime)) {
            nextTime = topKNextEventTime;
            advanceType = AdvanceType::TopK;
        }
        if (otherNextEventTime < nextTime && !greater(otherMax, topKMin, otherNextEventTime)) {
            nextTime = otherNextEventTime;
            advanceType = AdvanceType::Other;
        }

        if (nextTime > timeUpper) break;

        double prevTime = curTime;

        curTime = std::max(curTime, nextTime);

        if (advanceType == AdvanceType::Exchange) {
            int tieProtected = 0;
            int tieOther = 0;
            int vacant = 0;
            int pGroupBaseCount = pGroupCount;
      
            while (topKTree.getNextEventTime() <= curTime + epsilon) {
                topKTree.Advance();
            }
            
            while (otherTree.getNextEventTime() <= curTime + epsilon) {
                otherTree.Advance();
            }

            auto testEquiv = [curTime](const GroupedLine& left, const GroupedLine& right) noexcept {
                double score1 = left.k * curTime + left.b;
                double score2 = right.k * curTime + right.b;
                
                return std::abs(score1 - score2) <= epsilon;
            };

            auto handleTopKTops = [pGroup, &tieProtected, &tieOther, &vacant, &pGroupBaseCount](const GroupedLine& line) noexcept {
                int isProtected = (line.group == pGroup);
                tieProtected += isProtected;
                tieOther += 1 - isProtected;
                pGroupBaseCount -= isProtected;
                vacant += 1;
            };

            auto handleOtherTops = [pGroup, &tieProtected, &tieOther](const GroupedLine& line) noexcept {
                int isProtected = (line.group == pGroup);
                tieProtected += isProtected;
                tieOther += 1 - isProtected;
            };

            topKTree.applyToTopEquivs(testEquiv, handleTopKTops);
            otherTree.applyToTopEquivs(testEquiv, handleOtherTops);
            
            int pGroupLowerCount = pGroupBaseCount + std::max(0, vacant - tieOther);
            int pGroupUpperCount = pGroupBaseCount + vacant - std::max(0, vacant - tieProtected);

            if (std::max(pGroupLowerCount, pGroupLowerBound) <= std::min(pGroupUpperCount, pGroupUpperBound)) {
                weights(0) = curTime; 
                weights(1) = 1.0 - weights(0);

                return true;
            }

            topKMin = topKTree.Top();
            otherMax = otherTree.Top();
 
            pGroupCount -= (topKMin.group == pGroup);
            pGroupCount += (otherMax.group == pGroup);

            bool topKMinChange = topKTree.replaceTop(otherMax);
            bool otherMaxChange = otherTree.replaceTop(topKMin);

            exchanged = true;

            bool topKTreeChange = true;
            bool otherTreeChange = true;
            while (topKTreeChange || otherTreeChange) {
                while (topKMinChange || otherMaxChange) {
                    topKMin = topKTree.Top();
                    otherMax = otherTree.Top();
                    if (greater(otherMax, topKMin, curTime)) {
                        pGroupCount -= (topKMin.group == pGroup);
                        pGroupCount += (otherMax.group == pGroup);

                        topKMinChange = topKTree.replaceTop(otherMax);
                        otherMaxChange = otherTree.replaceTop(topKMin);
                    }
                    else {
                        topKMinChange = false;
                        otherMaxChange = false;
                    }
                }

                topKTreeChange = false;
                if (topKTree.getNextEventTime() <= curTime + epsilon) {
                    topKMinChange = topKTree.Advance();
                    topKTreeChange = true;
                }
                otherTreeChange = false;
                if (otherTree.getNextEventTime() <= curTime + epsilon) {
                    otherMaxChange = otherTree.Advance();
                    otherTreeChange = true;
                }
            }
            
            curTime += epsilon;
        }
        else if (advanceType == AdvanceType::TopK) {
            bool topChanged = topKTree.Advance();
            if (exchanged && topChanged) exchanged = false;
        }
        else {
            bool topChanged = otherTree.Advance();
            if (exchanged && topChanged) exchanged = false;
        }
    }

    return false;
}

int main(int argc, char* argv[]) {
    std::vector<Eigen::VectorXd> points;
    std::vector<int> groups;
    int protectedGroup = -1;

    auto [fileName, params] = FairTopK::parseCommandLine(argc, argv);

    bool success = FairTopK::DataLoader::readPreprocessedDataset(fileName, points, groups, protectedGroup);
    if (!success) return -1;

    int dimension = points[0].rows();
    if (dimension != 2) {
        std::cerr << "Do not support datasets with dimensions != 2" << std::endl;
        return -1;
    }

    auto solveFunc = params.unoptimized ? solve<false> : solve<true>;

    FairTopK::fairTopkExperiments(points, groups, protectedGroup, params, solveFunc);

    return 0;
}
