#ifndef FAIR_TOPK_DATA_LOADER
#define FAIR_TOPK_DATA_LOADER

#include <vector>
#include <string>
#include <Eigen/Dense>

namespace FairTopK {

namespace DataLoader {

void readPreprocessedDataset(const std::string& file, std::vector<Eigen::VectorXd>& points, std::vector<int>& groups, 
    int& protectedGroup);

}

}

#endif