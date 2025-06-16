/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#ifndef FAIR_TOPK_DATA_LOADER
#define FAIR_TOPK_DATA_LOADER

#include <vector>
#include <string>
#include <Eigen/Dense>

namespace FairTopK {

namespace DataLoader {

void readCompasData(const std::string& file, std::vector<Eigen::VectorXd>& points, 
    std::vector<int>& genders, std::vector<int>& races);

void readJEEData(const std::string& file, std::vector<Eigen::VectorXd>& points, std::vector<int>& genders);

bool readPreprocessedDataset(const std::string& file, std::vector<Eigen::VectorXd>& points, std::vector<int>& groups, 
    int& protectedGroup);

}

}

#endif
