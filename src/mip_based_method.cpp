#include <iostream>
#include <algorithm>
#include <vector>

#include <Eigen/Dense>
#include <gurobi/gurobi_c++.h>

#include "utility.h"
#include "data_loader.h"

bool solve(int threadCount, const std::vector<Eigen::VectorXd> &points,
const std::vector<int>& groups, int k, int pGroup, int pGroupLowerBound, 
int pGroupUpperBound, double margin, Eigen::VectorXd& weights) {
    GRBEnv env = GRBEnv(true);
    env.set("OutputFlag", "0");
    env.set(GRB_IntParam_Threads, threadCount);
    env.start();
    
    GRBModel model = GRBModel(env);
    model.set("MIPFocus", "1");
    model.set("SolutionLimit", "1");

    int count = points.size();
    int dimension = points[0].rows();

    std::vector<double> ones(count, 1.0);

    std::vector<GRBVar> scoreVars;
    std::vector<GRBVar> indicatorVars;
    std::vector<GRBVar> pGroupIndVars;

    scoreVars.reserve(dimension + 1);
    indicatorVars.reserve(count);
    pGroupIndVars.reserve(count);

    for (int i = 0; i < dimension; i++) {
        double lb = std::max(0.0, weights(i) - margin);
        double ub = std::min(1.0, weights(i) + margin);
        GRBVar var = model.addVar(lb, ub, 0.0, GRB_CONTINUOUS);
        scoreVars.push_back(var);
    }

    {
        GRBVar var = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
        scoreVars.push_back(var);
    }

    for (int i = 0; i < count; i++) {
        GRBVar var = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
        indicatorVars.push_back(var);
        if (groups[i] == pGroup) {
            pGroupIndVars.push_back(var);
        }
    }

    //sum_{i=1}^{d} w_i = 1
    {
        GRBLinExpr expr = 0;
        expr.addTerms(&ones[0], &scoreVars[0], dimension);
        model.addConstr(expr == 1.0);
    }


    for (int i = 0; i < count; i++) {
        GRBLinExpr expr = 0;  

        expr.addTerms(points[i].data(), &scoreVars[0], dimension);
        expr -= scoreVars[dimension];
        expr -= indicatorVars[i];

        model.addRange(expr, -1.0, 0.0);
    }

    {
        GRBLinExpr expr = 0;
        expr.addTerms(&ones[0], &pGroupIndVars[0], pGroupIndVars.size());
        model.addRange(expr, pGroupLowerBound, pGroupUpperBound);
    }

    {
        GRBLinExpr expr = 0;
        expr.addTerms(&ones[0], &indicatorVars[0], count);
        model.addConstr(expr == k);
    }

    model.optimize();

    int status = model.get(GRB_IntAttr_Status);
    if (status == GRB_INFEASIBLE || status == GRB_INF_OR_UNBD) {
        return false;
    }
    else {
        for (int i = 0; i < dimension; i++)
            weights(i) = scoreVars[i].get(GRB_DoubleAttr_X);

        return true;
    }
}

int main(int argc, char* argv[]) {
    std::vector<Eigen::VectorXd> points;
    std::vector<int> groups;
    int protectedGroup = -1;

    FairTopK::DataLoader::readPreprocessedDataset(argv[1], points, groups, protectedGroup);

    int k = 0;
    double pGroupLowerBound = 0;
    double pGroupUpperBound = 0;
    double margin = 0.0;
    int sampleCount = 0;
    int threadCount = 0;

    try {
        k = std::stoi(std::string(argv[2]));
        pGroupLowerBound = std::stod(std::string(argv[3]));
        pGroupUpperBound = std::stod(std::string(argv[4]));
        margin = std::stod(std::string(argv[5]));
        sampleCount = std::stoi(std::string(argv[6]));
        threadCount = std::stoi(std::string(argv[7]));
    } catch (const std::exception& e) {
        std::cerr << "Invalid input parameters" << std::endl;
        return -1;
    }

    FairTopK::printInputInfos(k, pGroupLowerBound, pGroupUpperBound, margin, threadCount);

    if (threadCount <= 0) {
        std::cerr << "The number of threads must be greater than or equal to 1" << std::endl;
        return -1;
    }

    int lowerBoundInt = (int)std::floor(pGroupLowerBound * k);
    int upperBoundInt = (int)std::ceil(pGroupUpperBound * k);

    auto samples = FairTopK::getRandomWeightVectors(sampleCount, points, groups, k, protectedGroup, 
        lowerBoundInt, upperBoundInt);

    FairTopK::fairTopkMarginTimeProfiling(samples, points, groups, k, protectedGroup, 
        lowerBoundInt, upperBoundInt, margin, 
        [threadCount]<class... Args>(Args&&... params) { 
            return solve(threadCount, std::forward<Args>(params)...);
    });

    return 0;
}