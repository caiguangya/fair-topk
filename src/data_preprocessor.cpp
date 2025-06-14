/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#include "data_preprocessor.h"
#include <gurobi/gurobi_c++.h>

namespace FairTopK {

namespace DataPreprocessor {

std::vector<int> getSkyband(const std::vector<Eigen::VectorXd>& points, int k) {
    int count = points.size();
    int dimension = points[0].rows();

    std::vector<int> indices;

    for (int i = 0; i < count; i++) {
        const auto& pt = points[i];

        int dominatedCount = 0;
        for (int j = 0; j < count; j++) {
            if (i == j) continue;

            Eigen::VectorXd diff = points[j] - pt;

            if ((diff.array() > 0.0).all()) dominatedCount += 1;
            if (dominatedCount >= k) break;
        }

        if (dominatedCount < k) 
            indices.push_back(i);

    }

    return indices;
}

std::vector<int> getSubset(const std::vector<Eigen::VectorXd>& points, int cutoff) {
    GRBEnv env = GRBEnv(true);
    env.set("OutputFlag", "0");
    env.start();

    int count = points.size();
    int dimension = points[0].rows();

    std::vector<GRBVar> scoreVars;
    std::vector<GRBVar> indVars;

    scoreVars.reserve(dimension + 1);
    indVars.reserve(count);

    std::vector<int> indices;

    for (int i = 0; i < count; i++) {
        GRBModel model = GRBModel(env);
        model.set("MIPFocus", "1");
        model.set("SolutionLimit", "1");
        model.set("TimeLimit", "1000");

        const auto& pt = points[i];

        for (int j = 0; j < dimension; j++) {
            GRBVar var = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
            scoreVars.push_back(var);
        }

        for (int j = 0; j < count - 1; j++) {
            GRBVar var = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
            indVars.push_back(var);
        }

        {
            GRBLinExpr expr = 0;
            for (int j = 0; j < dimension; j++) 
                expr += scoreVars[j];
            model.addConstr(expr == 1.0);
        }

        int idx = 0;
        for (int j = 0; j < count; j++) {
            if (j == i) continue;

            GRBLinExpr expr = 0;
            auto diff = pt - points[j];
            
            for (int k = 0; k < dimension; k++) {
                expr += diff(k) * scoreVars[k];
            }

            model.addGenConstrIndicator(indVars[idx++], 0, expr >= 0.0);
        }

        {
            GRBLinExpr expr = 0;
            for (int j = 0; j < count - 1; j++) 
                expr += indVars[j];
                
            model.addConstr(expr <= (cutoff - 1));
        }

        model.optimize();
        
        int status = model.get(GRB_IntAttr_Status);
        
        if (status == GRB_TIME_LIMIT ||
            (status != GRB_INF_OR_UNBD && status != GRB_INFEASIBLE)) {
            indices.push_back(i);
        }

        scoreVars.clear();
        indVars.clear();
    }

    return indices;
}

}

}