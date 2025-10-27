/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>

#include <Eigen/Dense>
#include <gurobi/gurobi_c++.h>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include "utility.h"
#include "data_loader.h"
#include "experiments.h"

bool checkGurobiLicense() {
    try {
        GRBEnv env = GRBEnv(true);
        env.set("OutputFlag", "0");
        env.start();
        return true;
    } catch (const GRBException& e) {
        std::cerr << e.getMessage() << std::endl;
        return false;
    }
}

bool solveGurobi(int threadCount, const std::vector<Eigen::VectorXd> &points,
    const std::vector<int>& groups, int k, int pGroup, int pGroupLowerBound, 
    int pGroupUpperBound, double margin, Eigen::VectorXd& weights) {
    GRBEnv env = GRBEnv(true);
    env.set("OutputFlag", "0");
    if (threadCount > 0) 
        env.set(GRB_IntParam_Threads, threadCount);
    env.start();
    
    GRBModel model = GRBModel(env);
    model.set("MIPFocus", "1");
    model.set("SolutionLimit", "1");
    model.set("IntFeasTol", "1e-6");

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
        expr.addTerms(ones.data(), scoreVars.data(), dimension);
        model.addConstr(expr == 1.0);
    }


    for (int i = 0; i < count; i++) {
        GRBLinExpr expr = 0;  

        expr.addTerms(points[i].data(), scoreVars.data(), dimension);
        expr -= scoreVars[dimension];
        expr -= indicatorVars[i];

        model.addRange(expr, -1.0, 0.0);
    }

    {
        GRBLinExpr expr = 0;
        expr.addTerms(ones.data(), pGroupIndVars.data(), pGroupIndVars.size());
        model.addRange(expr, pGroupLowerBound, pGroupUpperBound);
    }

    {
        GRBLinExpr expr = 0;
        expr.addTerms(ones.data(), indicatorVars.data(), count);
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

bool solveSCIP([[maybe_unused]] int threadCount, const std::vector<Eigen::VectorXd> &points,
    const std::vector<int>& groups, int k, int pGroup, int pGroupLowerBound, 
    int pGroupUpperBound, double margin, Eigen::VectorXd& weights) {
    SCIP *scip = nullptr;

    SCIP_CALL(SCIPcreate(&scip));
    SCIP_CALL(SCIPincludeDefaultPlugins(scip));
    SCIPmessagehdlrSetQuiet(SCIPgetMessagehdlr(scip), true);

    SCIP_CALL(SCIPcreateProbBasic(scip, ""));

    SCIP_CALL(SCIPsetIntParam(scip, "limits/solutions", 1));
    SCIP_CALL(SCIPsetIntParam(scip, "limits/maxsol", 1));
    SCIP_CALL(SCIPsetRealParam(scip, "numerics/feastol", 1e-6));

    int count = points.size();
    int dimension = points[0].rows();

    std::vector<double> ones(count, 1.0);

    std::vector<SCIP_VAR *> scoreVars;
    std::vector<SCIP_VAR *> indicatorVars;
    std::vector<SCIP_VAR *> pGroupIndVars;
    std::vector<SCIP_CONS *> constraints;

    scoreVars.reserve(dimension + 1);
    indicatorVars.reserve(count);
    pGroupIndVars.reserve(count);
    constraints.reserve(count + 3);

    for (int i = 0; i < dimension; i++) {
        double lb = std::max(0.0, weights(i) - margin);
        double ub = std::min(1.0, weights(i) + margin);

        SCIP_VAR *var = nullptr;
        SCIP_CALL(SCIPcreateVarBasic(scip, &var, nullptr, lb, ub, 0.0, SCIP_VARTYPE_CONTINUOUS));
        SCIP_CALL(SCIPaddVar(scip, var));

        scoreVars.push_back(var);
    }

    {
        SCIP_VAR *var = nullptr;
        SCIP_CALL(SCIPcreateVarBasic(scip, &var, nullptr, 0.0, 1.0, 0.0, SCIP_VARTYPE_CONTINUOUS));
        SCIP_CALL(SCIPaddVar(scip, var));

        scoreVars.push_back(var);
    }

    for (int i = 0; i < count; i++) {
        SCIP_VAR *var = nullptr;
        SCIP_CALL(SCIPcreateVarBasic(scip, &var, nullptr, 0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY));
        SCIP_CALL(SCIPaddVar(scip, var));

        indicatorVars.push_back(var);
        if (groups[i] == pGroup) {
            pGroupIndVars.push_back(var);
        }
    }

    {
        SCIP_CONS *cons = nullptr;
        SCIP_CALL(SCIPcreateConsBasicLinear(scip, &cons, "", dimension, scoreVars.data(), ones.data(), 1.0, 1.0));
        SCIP_CALL(SCIPaddCons(scip, cons));

        constraints.push_back(cons);
    }

    {
        std::vector<SCIP_VAR *> consVars = scoreVars;
        Eigen::VectorXd consVals = -Eigen::VectorXd::Ones(dimension + 2);

        consVars.push_back(nullptr);

        for (int i = 0; i < count; i++) {
            consVars[dimension + 1] = indicatorVars[i];
            consVals.head(dimension) = points[i];

            SCIP_CONS *cons = nullptr;
            SCIP_CALL(SCIPcreateConsBasicLinear(scip, &cons, "", dimension + 2, consVars.data(), consVals.data(), 
                -1.0, 0.0));
            SCIP_CALL(SCIPaddCons(scip, cons));

            constraints.push_back(cons);
        }
    }

    {
        SCIP_CONS *cons = nullptr;
        SCIP_CALL(SCIPcreateConsBasicLinear(scip, &cons, "", pGroupIndVars.size(), pGroupIndVars.data(), ones.data(), 
            (SCIP_Real)pGroupLowerBound, (SCIP_Real)pGroupUpperBound));
        SCIP_CALL(SCIPaddCons(scip, cons));

        constraints.push_back(cons);
    }

    {
        SCIP_CONS *cons = nullptr;
        SCIP_CALL(SCIPcreateConsBasicLinear(scip, &cons, "", count, indicatorVars.data(), ones.data(), 
            (SCIP_Real)k, (SCIP_Real)k));
        SCIP_CALL(SCIPaddCons(scip, cons));

        constraints.push_back(cons);
    }

    SCIP_CALL(SCIPsolve(scip));

    bool found = (SCIPgetStatus(scip) == SCIP_STATUS_OPTIMAL);
    if (found)  {
        SCIP_SOL *sol = SCIPgetBestSol(scip);

        for (int i = 0; i < dimension; i++) {
            weights(i) = SCIPgetSolVal(scip, sol, scoreVars[i]);
        }
    }

    for (auto var : scoreVars) {
        SCIP_CALL(SCIPreleaseVar(scip, &var));
    }
    scoreVars.clear();
    for (auto var: indicatorVars) {
        SCIP_CALL(SCIPreleaseVar(scip, &var));
    }
    indicatorVars.clear();
    pGroupIndVars.clear();

    for (auto cons: constraints) {
        SCIP_CALL(SCIPreleaseCons(scip, &cons));
    }
    constraints.clear();

    SCIP_CALL(SCIPfree(&scip));

    return found;
}

int main(int argc, char* argv[]) {
    std::vector<Eigen::VectorXd> points;
    std::vector<int> groups;
    int protectedGroup = -1;

    auto [fileName, params] = FairTopK::parseCommandLine(argc, argv);

    if (params.solver != "gurobi" && params.solver != "scip") {
        std::cerr << "Unknown solver: " << params.solver << std::endl;
        return -1;
    }

    bool isGurobi = (params.solver == "gurobi");
    if (isGurobi && !checkGurobiLicense()) {
        return -1;
    }

    bool success = FairTopK::DataLoader::readPreprocessedDataset(fileName, points, groups, protectedGroup);
    if (!success) return -1;

    if (isGurobi) {
        FairTopK::fairTopkExperiments(points, groups, protectedGroup, params, 
            [threadCount = params.threadCount]<class... Args>(Args&&... params) { 
                return solveGurobi(threadCount, std::forward<Args>(params)...);
        });
    }
    else {
        FairTopK::fairTopkExperiments(points, groups, protectedGroup, params, 
            [threadCount = params.threadCount]<class... Args>(Args&&... params) { 
                return solveSCIP(threadCount, std::forward<Args>(params)...);
        });
    }

    return 0;
}
