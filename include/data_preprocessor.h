/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef FAIR_TOPK_DATA_PREPROCESS
#define FAIR_TOPK_DATA_PREPROCESS

#include <vector>
#include <Eigen/Dense>

namespace FairTopK {

namespace DataPreprocessor {

std::vector<int> getSkyband(const std::vector<Eigen::VectorXd>& points, int k);

std::vector<int> getSubset(const std::vector<Eigen::VectorXd>& points, int k);

}

}

#endif