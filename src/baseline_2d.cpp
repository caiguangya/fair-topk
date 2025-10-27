/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#include <iostream>
#include <queue>
#include <tuple>
#include <limits>
#include <cmath>

#include <Eigen/Dense>

#include "utility.h"
#include "data_loader.h"
#include "experiments.h"
#include "memory.h"

struct GroupedLine {
    double k;
    double b;
    int rank;
    int count;
    int pGroupCount;
};

inline double computeIntersect(const GroupedLine& front, const GroupedLine& back, double upperLimit) noexcept {
    double scoreFront = front.k * upperLimit + front.b;
    double scoreBack = back.k * upperLimit + back.b;
    if (scoreFront < scoreBack && front.k != back.k) {
        return (front.b - back.b) / (back.k - front.k);
    }

    return std::numeric_limits<double>::infinity();
}

bool isDegenerated(const std::vector<GroupedLine *>& lines, int frontRank, int backRank, double sweepLinePos, double epsilon) {
    if (backRank - frontRank > 1) return true;

    const GroupedLine *kthLine = lines[frontRank];
    double kthScore = kthLine->k * sweepLinePos + kthLine->b;

    if (frontRank > 0) {
        const GroupedLine *line = lines[frontRank - 1];
        double score =  line->k * sweepLinePos + line->b;
        if (std::abs(score - kthScore) <= epsilon)
            return true;
    }
    if (backRank < lines.size() - 1) {
        const GroupedLine *line = lines[backRank + 1];
        double score =  line->k * sweepLinePos + line->b;
        if (std::abs(score - kthScore) <= epsilon)
            return true;
    }

    return false;
}

bool isFair(const std::vector<GroupedLine *>& lines, int k, double sweepLinePos, int kthIdx, int kthTieCount,
int pGroupBaseCount, int pGroupLowerBound, int pGroupUpperBound, double epsilon) {
    const GroupedLine* kthLine = lines[kthIdx];

    double kthScore = kthLine->k * sweepLinePos + kthLine->b;
    int tieProtected = kthLine->pGroupCount;
    int tieOther = kthLine->count - tieProtected;

    int vacant = kthTieCount;
    int idx = kthIdx - 1;
    while (idx >= 0) {
        const GroupedLine* line = lines[idx];
        double score = line->k * sweepLinePos + line->b;
        if (std::abs(score - kthScore) > epsilon) {
            break;
        }
        tieProtected += line->pGroupCount;
        tieOther += (line->count - line->pGroupCount);

        pGroupBaseCount -= line->pGroupCount;
        vacant += line->count;
        idx -= 1;
    }

    idx = kthIdx + 1;
    int count = lines.size();
    while (idx < count) {
        const GroupedLine* line = lines[idx];
        double score = line->k * sweepLinePos + line->b;
        if (std::abs(score - kthScore) > epsilon) {
            break;
        }

        tieProtected += line->pGroupCount;
        tieOther += (line->count - line->pGroupCount);

        idx += 1;
    }

    int pGroupLowerCount = pGroupBaseCount + std::max(0, vacant - tieOther);
    int pGroupUpperCount = pGroupBaseCount + vacant - std::max(0, vacant - tieProtected);

    return std::max(pGroupLowerCount, pGroupLowerBound) <= std::min(pGroupUpperCount, pGroupUpperBound);
}

bool solve(const std::vector<Eigen::VectorXd> &points, const std::vector<int>& groups, int k, int pGroup, int pGroupLowerBound, 
int pGroupUpperBound, double margin, Eigen::VectorXd& weights) {
    constexpr double epsilon = 1e-8;
    double sweepLower = std::max({ 0.0, weights(0) - margin, 1.0 - weights(1) - margin });
    double sweepUpper = std::min({ 1.0, weights(0) + margin, 1.0 - weights(1) + margin });

    int pointCount = points.size();

    std::vector<std::tuple<double, double, int> > rawLines;
    rawLines.reserve(pointCount);

    for (int i = 0; i < pointCount; i++) {
        const auto &point = points[i];
        rawLines.emplace_back(point(0) - point(1), point(1), i);
    }

    std::sort(rawLines.begin(), rawLines.end(),
        [sweepLower](const auto& l1, const auto& l2) { 
            double k1 = std::get<0>(l1);
            double b1 = std::get<1>(l1);
            double k2 = std::get<0>(l2);
            double b2 = std::get<1>(l2);

            double s1 = k1 * sweepLower + b1;
            double s2 = k2 * sweepLower + b2;

            return s1 != s2 ? (s1 > s2) : (k1 != k2 ? (k1 > k2) : (b1 > b2));
    });

    FairTopK::MemoryArena<GroupedLine> pool(pointCount);
    std::vector<GroupedLine *> lines;
    lines.reserve(pointCount);

    {
        auto [k, b, idx] = rawLines[0];
        lines.emplace_back(pool.Alloc(1, std::true_type{}, k, b, 0, 1, groups[idx] == pGroup));
    }

    for (int i = 1; i < pointCount; i++) {
        auto [k, b, idx] = rawLines[i];
        auto preLine = lines.back();

        if (k == preLine->k && b == preLine->b) {
            preLine->count += 1;
            preLine->pGroupCount += (groups[idx] == pGroup);
        }
        else {
            lines.push_back(pool.Alloc(1, std::true_type{}, k, b, preLine->rank + 1, 1, groups[idx] == pGroup));
        }
    }

    int kthIdx = 0;
    int kthTieCount = 0;
    {
        int count = lines[0]->count;
        while (count < k) {
            count += lines[++kthIdx]->count;
        }
        kthTieCount = k - (count - lines[kthIdx]->count); 
    }
    
    int pGroupBaseCount = 0;
    for (int i = 0; i < kthIdx; i++) {
        pGroupBaseCount += lines[i]->pGroupCount;
    }

    if (isFair(lines, k, sweepLower, kthIdx, kthTieCount, pGroupBaseCount, pGroupLowerBound, pGroupUpperBound, epsilon)) {
        weights(0) = sweepLower; 
        weights(1) = 1.0 - weights(0);
        
        return true;
    }

    auto compare = [](const auto& intersect0, const auto& intersect1) noexcept {
        return std::get<0>(intersect0) > std::get<0>(intersect1);
    };

    int lineCount = lines.size();

    using IntersectTuple = std::tuple<double, GroupedLine*, GroupedLine*>;
    using IntPriorityQueue = std::priority_queue<IntersectTuple, std::vector<IntersectTuple>, decltype(compare)>;
    std::vector<IntersectTuple> queueContainer;
    queueContainer.reserve(2 * lineCount);

    IntPriorityQueue intersectQueue(compare, std::move(queueContainer));

    for (int i = 0; i < lineCount - 1; i++) {
        double intersect = computeIntersect(*lines[i], *lines[i + 1], sweepUpper);
        if (!std::isinf(intersect)) {
            intersectQueue.emplace(intersect, lines[i], lines[i + 1]);
        }
    }

    auto insertNewOrderingChange = [&intersectQueue, sweepUpper](GroupedLine* front, GroupedLine* back) {
        double intersect = computeIntersect(*front, *back, sweepUpper);
        if (!std::isinf(intersect)) {
            intersectQueue.emplace(intersect, front, back);
        }
    };

    while (!intersectQueue.empty()) {
        auto [intersect, frontLine, backLine] = intersectQueue.top();
        intersectQueue.pop();

        int frontRank = frontLine->rank;
        int backRank = backLine->rank;

        if (backRank < frontRank) {
            continue;
        }

        std::swap(lines[frontLine->rank], lines[backLine->rank]);
        std::swap(frontLine->rank, backLine->rank);

        if (frontRank <= kthIdx && backRank > kthIdx) {
            int kBaseCount = k - kthTieCount;
            if (frontRank != kthIdx) {
                pGroupBaseCount -= frontLine->pGroupCount;
                pGroupBaseCount += backLine->pGroupCount;

                kBaseCount -= frontLine->count;
                kBaseCount += backLine->count;
            }
            kthTieCount = k - kBaseCount;
            
            int count = kBaseCount + lines[kthIdx]->count;
            if (kBaseCount >= k) {
                int newKthIdx = kthIdx - 1;
                count = kBaseCount;
                while (newKthIdx >= 0 && count >= k) {
                    count -= lines[newKthIdx--]->count;
                }
                newKthIdx += 1;
                
                for (int i = newKthIdx; i < kthIdx; i++)
                    pGroupBaseCount -= lines[i]->pGroupCount;

                kthIdx = newKthIdx;
                kthTieCount = k - count;
            }
            else if (count < k) {
                int newKthIdx = kthIdx;
                do {
                    count += lines[++newKthIdx]->count;
                } while (count < k);

                for (int i = kthIdx; i < newKthIdx; i++)
                    pGroupBaseCount += lines[i]->pGroupCount;

                kthIdx = newKthIdx;
                kthTieCount = k - (count - lines[kthIdx]->count);
            }

            bool fair = false;
            if (isDegenerated(lines, frontRank, backRank, intersect, epsilon)) {
                fair = isFair(lines, k, intersect, kthIdx, kthTieCount, pGroupBaseCount, pGroupLowerBound, pGroupUpperBound, epsilon);
            }
            else {
                int tieProtected = frontLine->pGroupCount + backLine->pGroupCount;
                int tieOther = (frontLine->count - frontLine->pGroupCount) + 
                    (backLine->count - backLine->pGroupCount);

                int pGroupLowerCount = pGroupBaseCount + std::max(0, kthTieCount - tieOther);
                int pGroupUpperCount = pGroupBaseCount + kthTieCount - std::max(0, kthTieCount - tieProtected);

                fair = (std::max(pGroupLowerCount, pGroupLowerBound) <= std::min(pGroupUpperCount, pGroupUpperBound));
            }

            if (fair) {
                weights(0) = intersect;
                weights(1) = 1.0 - weights(0);

                return true;
            }
        }
        else if (backRank == kthIdx) {
            int kBaseCount = k - kthTieCount;

            pGroupBaseCount -= frontLine->pGroupCount;
            pGroupBaseCount += backLine->pGroupCount;

            kBaseCount -= frontLine->count;
            kBaseCount += backLine->count;

            kthTieCount = k - kBaseCount;

            if (kBaseCount >= k) {
                int newKthIdx = kthIdx - 1;
                int count = kBaseCount;
                while (newKthIdx >= 0 && count >= k) {
                    count -= lines[newKthIdx--]->count;
                }
                newKthIdx += 1;

                for (int i = newKthIdx; i < kthIdx; i++)
                    pGroupBaseCount -= lines[i]->pGroupCount;

                kthIdx = newKthIdx;
                kthTieCount = k - count;
            }
            
            if (isFair(lines, k, intersect, kthIdx, kthTieCount, pGroupBaseCount, pGroupLowerBound, pGroupUpperBound, epsilon)) {
                weights(0) = intersect;
                weights(1) = 1.0 - weights(0);

                return true;
            }
        }

        if (frontRank > 0) {
            insertNewOrderingChange(lines[frontRank - 1], backLine);
        }
        if (backRank < lineCount - 1) {
            insertNewOrderingChange(frontLine, lines[backRank + 1]);
        }

        if (backRank - frontRank > 1) {
            insertNewOrderingChange(backLine, lines[frontRank + 1]);
            insertNewOrderingChange(lines[backRank - 1], frontLine);
        }
    }

    if (isFair(lines, k, sweepUpper, kthIdx, kthTieCount, pGroupBaseCount, pGroupLowerBound, pGroupUpperBound, epsilon)) {
        weights(0) = sweepUpper; 
        weights(1) = 1.0 - weights(0);
        
        return true;
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

    FairTopK::fairTopkExperiments(points, groups, protectedGroup, params, solve);

    return 0;
}
