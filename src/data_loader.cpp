#include "data_loader.h"
#include <string>
#include <fstream>

#include <xtensor/xcsv.hpp>
#include <xtensor/xview.hpp>

namespace FairTopK {

namespace DataLoader {

namespace {
void readPreprocessedCompasData(const std::string& file, std::vector<Eigen::VectorXd>& points,  
    std::vector<int>& genders, std::vector<int>& races);
void readPreprocessedJEEData(const std::string& file, std::vector<Eigen::VectorXd>& points,std::vector<int>& genders);
}

void readPreprocessedDataset(const std::string& file, std::vector<Eigen::VectorXd>& points, std::vector<int>& groups, 
    int& protectedGroup) {
    if (file.find("compas") != std::string::npos) {
        std::vector<int> dummy;
        readPreprocessedCompasData(file, points, dummy, groups);
        protectedGroup = 0;
    }
    else if (file.find("jee") != std::string::npos) {
        readPreprocessedJEEData(file, points, groups);
        protectedGroup = 1;
    }
}

namespace {

void readPreprocessedCompasData(const std::string& file, std::vector<Eigen::VectorXd>& points, 
    std::vector<int>& genders, std::vector<int>& races) {
    std::ifstream inf(file, std::ifstream::in);
    
    auto data = xt::load_csv<double>(inf);
    auto shape = data.shape();

    int dimension = shape[1] - 2;

    int count = shape[0];

    Eigen::VectorXd normalizer(dimension);
    {
        Eigen::VectorXd cwiseMinPt(dimension), cwiseMaxPt(dimension);
        auto cwiseMinRow = xt::row(data, 0);
        for (int i = 0; i < dimension; i++) {
            cwiseMinPt(i) = cwiseMinRow(i);
        }
        auto cwiseMaxRow = xt::row(data, 1);
        for (int i = 0; i < dimension; i++) {
            cwiseMaxPt(i) = cwiseMaxRow(i);
        }
        normalizer = (cwiseMaxPt - cwiseMinPt).cwiseInverse();
    }

    for (int i = 2; i < count; i++) {
        auto row = xt::row(data, i);

        Eigen::VectorXd point(dimension);
        for (int j = 0; j < dimension; j++) {
            point(j) = row(j);
        }
        point = point.cwiseProduct(normalizer);

        points.push_back(std::move(point));
        genders.push_back((int)row(dimension));
        races.push_back((int)row(dimension + 1));
    }
}

void readPreprocessedJEEData(const std::string& file, std::vector<Eigen::VectorXd>& points,std::vector<int>& genders) {
    std::ifstream inf(file, std::ifstream::in);
    
    auto data = xt::load_csv<double>(inf);
    auto shape = data.shape();

    int dimension = shape[1] - 1;

    int count = shape[0];

    Eigen::VectorXd normalizer(dimension);
    {
        Eigen::VectorXd cwiseMinPt(dimension), cwiseMaxPt(dimension);
        auto cwiseMinRow = xt::row(data, 0);
        for (int i = 0; i < dimension; i++) {
            cwiseMinPt(i) = cwiseMinRow(i);
        }
        auto cwiseMaxRow = xt::row(data, 1);
        for (int i = 0; i < dimension; i++) {
            cwiseMaxPt(i) = cwiseMaxRow(i);
        }
        normalizer = (cwiseMaxPt - cwiseMinPt).cwiseInverse();
    }

    for (int i = 2; i < count; i++) {
        auto row = xt::row(data, i);

        Eigen::VectorXd point(dimension);
        for (int j = 0; j < dimension; j++) {
            point(j) = row(j);
        }
        point = point.cwiseProduct(normalizer);

        points.push_back(std::move(point));
        genders.push_back((int)row(dimension));
    }
}

}

}

}