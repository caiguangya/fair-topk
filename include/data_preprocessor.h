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