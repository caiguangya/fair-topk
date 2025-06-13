#include "experiments.h"
#include <algorithm>

namespace FairTopK {

void testInputWeightVectors(const std::vector<Eigen::VectorXd> &points, const std::vector<int>& groups, int pGroup, 
    const InputParams& params, std::vector<Eigen::VectorXd> &weightVectorSamples) {
    int fairCount = 0;
    std::vector<Eigen::VectorXd> unfairSamples;
    unfairSamples.reserve(weightVectorSamples.size());

    for (const auto& sampleVector : weightVectorSamples) {
        bool isFair = checkFairness(points, groups, sampleVector, params.k, 
            pGroup, params.pGroupLowerBound, params.pGroupUpperBound);
        if (isFair) fairCount += 1;
        else unfairSamples.push_back(sampleVector);
    }

    std::cout << fairCount << "/" << weightVectorSamples.size() << " input weight vectors are fair" << std::endl;
    std::swap(unfairSamples, weightVectorSamples);
}

int getPGroupCount(const std::vector<Eigen::VectorXd> &points, int k,
    const std::vector<int>& groups, int pGroup, int pGroupLowerBound, int pGroupUpperBound,
    const Eigen::VectorXd& weightVector) {
    constexpr double epsilon = 1e-8;
    std::vector<std::pair<double, int> > pts;
    int count = points.size();
    
    pts.reserve(count);
    for (int i = 0; i < count; i++) {
        pts.emplace_back(weightVector.dot(points[i]), i);
    }
    
    std::stable_sort(pts.begin(), pts.end(),
        [](const auto& p0, const auto& p1) { return p0.first > p1.first; });
    
    double kthScore = pts[k - 1].first;
    
    int vacant = 0;
    int pGroupBaseCount = 0;
    int tieProtected = 0;
    int tieOther = 0;
    for (int i = 0; i < k; i++) {
        const auto [score, idx] = pts[i];
                
        int isProtected = (groups[idx] == pGroup);
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
        const auto [score, idx] = pts[i];
    
        if (kthScore - score <= epsilon) {
            int isProtected = (groups[idx] == pGroup);
            tieProtected += isProtected;
            tieOther += 1 - isProtected;
        }
        else {
            break;
        }
    }

    int pGroupLowerCount = pGroupBaseCount + std::max(0, vacant - tieOther);
    int pGroupUpperCount = pGroupBaseCount + vacant - std::max(0, vacant - tieProtected);

    int topKPGroupCount = pGroupLowerBound + (pGroupUpperBound - pGroupLowerBound) / 2;
    if (topKPGroupCount < pGroupLowerCount) {
        topKPGroupCount = pGroupLowerCount;
    }
    else if (topKPGroupCount > pGroupUpperCount) {
        topKPGroupCount = pGroupUpperCount;
    }

    return topKPGroupCount;
}

double computeTopKUtility(const std::vector<Eigen::VectorXd> &points, int k, 
    const std::vector<int>& groups, int pGroup, int pGroupLowerBound, int pGroupUpperBound, 
    const std::vector<double>& scores, const Eigen::VectorXd& weightVector, bool arbTieBreaking) {
    constexpr double epsilon = 1e-8;
    std::vector<std::pair<double, int> > pts;
    int count = points.size();

    pts.reserve(count);
    for (int i = 0; i < count; i++) {
        pts.emplace_back(weightVector.dot(points[i]), i);
    }

    std::stable_sort(pts.begin(), pts.end(),
        [](const auto& p0, const auto& p1) { return p0.first > p1.first; });

    double utility = 0.0;
    if (arbTieBreaking) {
        for (int i = 0; i < k; i++) utility += scores[pts[i].second];
    }
    else {
        std::vector<int> tieProtectedIndices;
        std::vector<int> tieOtherIndices;

        double kthScore = pts[k - 1].first;

        int vacant = 0;
        int pGroupBaseCount = 0;
        for (int i = 0; i < k; i++) {
            const auto [score, idx] = pts[i];

            bool isProtected = (groups[idx] == pGroup);
            if (score - kthScore > epsilon) {
                pGroupBaseCount += (int)isProtected;
                utility += scores[idx];
            }
            else {
                vacant += 1;
                if (isProtected) tieProtectedIndices.push_back(idx);
                else tieOtherIndices.push_back(idx);
            }
        }

        for (int i = k; i < count; i++) {
            const auto [score, idx] = pts[i];

            if (kthScore - score <= epsilon) {
                if (groups[idx] == pGroup) tieProtectedIndices.push_back(idx);
                else tieOtherIndices.push_back(idx);
            }
            else {
                break;
            }
        }

        int pGroupLowerCount = pGroupBaseCount + std::max(0, vacant - (int)tieOtherIndices.size());
        int pGroupUpperCount = pGroupBaseCount + vacant - std::max(0, vacant - (int)tieProtectedIndices.size());

        int topKPGroupCount = pGroupLowerBound + (pGroupUpperBound - pGroupLowerBound) / 2;
        if (topKPGroupCount < pGroupLowerCount) {
            topKPGroupCount = pGroupLowerCount;
        }
        else if (topKPGroupCount > pGroupUpperCount) {
            topKPGroupCount = pGroupUpperCount;
        }

        int addPGroupCount = topKPGroupCount - pGroupBaseCount;
        for (int i = 0; i < addPGroupCount; i++) {
            utility += scores[tieProtectedIndices[i]];
        }

        int addOtherCount = vacant - addPGroupCount;
        for (int i = 0; i < addOtherCount; i++) {
            utility += scores[tieOtherIndices[i]];
        }
    }

    return utility;
}

void evaluateQuality(const std::vector<Eigen::VectorXd> &points, const std::vector<int>& groups, int pGroup, 
    const InputParams& params, const std::vector<std::pair<int, Eigen::VectorXd> >& fairVectors, 
    const std::vector<Eigen::VectorXd> &unfairVectors) {
    if (fairVectors.empty()) {
        std::cout << "Average weight vector difference: N/A\n";
        std::cout << "Average protected group fraction: N/A\n";
        std::cout << "Average utility loss: N/A" << std::endl;
        return;
    }

    int size = fairVectors.size();
    int dimension = points[0].rows();

    double diffs = 0;
    for (int i = 0; i < size; i++) {
        const auto& [idx, fairVector] = fairVectors[i];
        diffs += (fairVector - unfairVectors[idx]).cwiseAbs().sum();
    }
    diffs /= size;

    std::cout << "Average weight vector difference: " << std::scientific << diffs << "\n";

    int pGroupCount = 0;
    for (int i = 0; i < size; i++) {
        pGroupCount += getPGroupCount(points, params.k, 
            groups, pGroup, params.pGroupLowerBound, params.pGroupUpperBound, fairVectors[i].second);
    }
    double avgPGroupFraction = ((double)pGroupCount / params.k) / size;

    std::cout << "Average protected group proportion: " <<  std::scientific << avgPGroupFraction << "\n";

    double avgUtilityLoss = 0.0; 
    std::vector<double> scores(points.size(), 0.0);
    for (int i = 0; i < size; i++) {
        const auto& [idx, fairVector] = fairVectors[i];
        const Eigen::VectorXd& oriVector = unfairVectors[idx];

        std::transform(points.cbegin(), points.cend(), scores.begin(),
            [&oriVector](const auto& point) -> double { return point.dot(oriVector); } );
            
        double oriUtility = computeTopKUtility(points, params.k, groups, 
            pGroup, params.pGroupLowerBound, params.pGroupUpperBound, scores, oriVector, true);
        double newUtility = computeTopKUtility(points, params.k, groups,
            pGroup, params.pGroupLowerBound, params.pGroupUpperBound, scores, fairVector, false);
        double utilityLoss = 1.0 - newUtility / oriUtility;
        avgUtilityLoss += utilityLoss;
    }
    avgUtilityLoss /= size;

    std::cout << "Average utility loss: " << std::scientific << avgUtilityLoss << std::endl;
}

}