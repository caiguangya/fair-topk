#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <random>

#include <Eigen/Dense>
#include <xtensor/xcsv.hpp>
#include <xtensor/xarray.hpp>

#include "data_loader.h"
#include "data_preprocessor.h"

void outputCOMPAS(const std::vector<int>& indices, 
    const std::vector<Eigen::VectorXd>& points, const std::vector<int>& genders, const std::vector<int>& races,
    const Eigen::VectorXd& cwiseMinPt, const Eigen::VectorXd& cwiseMaxPt, const std::string& outputFile) {
    int dimension = points[0].rows();
    xt::xarray<double, xt::layout_type::dynamic> mat({ indices.size() + 2, dimension + 2 }, xt::layout_type::row_major);

    for (int j = 0; j < dimension; j++) {
        mat(0, j) = cwiseMinPt(j);
    }
    mat(0, dimension) = -1;
    mat(0, dimension + 1) = -1;
    for (int j = 0; j < dimension; j++) {
        mat(1, j) = cwiseMaxPt(j);
    }
    mat(1, dimension) = -1;
    mat(1, dimension + 1) = -1;
    for (std::size_t i = 0; i < indices.size(); i++) {
        int index = indices[i];
        const auto& pt = points[index];

        for (int j = 0; j < dimension; j++) {
            mat(i + 2, j) = pt(j);
        }
        mat(i + 2, dimension) = genders[index];
        mat(i + 2, dimension + 1) = races[index];
    }
    
    std::ofstream outf(outputFile, std::ofstream::out);
    xt::dump_csv(outf, mat);
}

void outputJEE(const std::vector<int>& indices,
    const std::vector<Eigen::VectorXd>& points, const std::vector<int>& genders, 
    const Eigen::VectorXd& cwiseMinPt, const Eigen::VectorXd& cwiseMaxPt, const std::string& outputFile) {
    int dimension = points[0].rows();
    xt::xarray<double, xt::layout_type::dynamic> mat({ indices.size() + 2, dimension + 1 }, xt::layout_type::row_major);

    for (int j = 0; j < dimension; j++) {
        mat(0, j) = cwiseMinPt(j);
    }
    mat(0, dimension) = -1;
    for (int j = 0; j < dimension; j++) {
        mat(1, j) = cwiseMaxPt(j);
    }
    mat(1, dimension) = -1;
    for (std::size_t i = 0; i < indices.size(); i++) {
        int index = indices[i];
        const auto& pt = points[index];

        for (int j = 0; j < dimension; j++) {
            mat(i + 2, j) = pt(j);
        }
        mat(i + 2, dimension) = genders[index];
    }
    
    std::ofstream outf(outputFile, std::ofstream::out);
    xt::dump_csv(outf, mat);
}

int main(int argc, char* argv[]) {
    std::vector<Eigen::VectorXd> points;
    std::vector<int> genders;
    std::vector<int> races;

    std::string file(argv[1]);
    bool isCompas = true;
    if (file.find("compas") != std::string::npos) {
        FairTopK::DataLoader::readCompasData(file, points, genders, races);
    }
    else if (file.find("jee") != std::string::npos) {
        isCompas = false;
        FairTopK::DataLoader::readJEEData(file, points, genders);
    }
    else {
        std::cerr << "File not supported" << std::endl;
        return -1;
    }

    int k = 0;
    double ratio = 1.0;
    std::string outputFile;

    try {
        k = std::stoi(std::string(argv[2]));
        ratio = std::stod(std::string(argv[3]));
        outputFile = std::string(argv[4]);
    } catch (const std::exception& e) {
        std::cerr << "Invalid input parameters" << std::endl;
        return -1;
    }

    std::cout << k << " " << ratio << std::endl;

    int count = points.size();
    int dimension = points[0].rows();

    double infty = std::numeric_limits<double>::max();

    Eigen::VectorXd cwiseMinPt(dimension), cwiseMaxPt(dimension);
    for (int i = 0; i < dimension; i++) {
        cwiseMinPt(i) = infty;
        cwiseMaxPt(i) = -infty;
    }

    for (int i = 0; i < count; i++) {
        cwiseMinPt = cwiseMinPt.cwiseMin(points[i]);
        cwiseMaxPt = cwiseMaxPt.cwiseMax(points[i]);
    }
    
    Eigen::VectorXd normalizer = cwiseMaxPt - cwiseMinPt;
    normalizer = normalizer.cwiseInverse();

    if (normalizer.array().isInf().any()) {
        std::cerr << "Divided by 0" << std::endl;
        return -1;
    }

    std::vector<int> reducedIndices;
    reducedIndices.reserve(count);
    for (int i = 0; i < count; i++) reducedIndices.push_back(i);

    std::default_random_engine rand(2024);
    std::shuffle(reducedIndices.begin(), reducedIndices.end(), rand);

    int reducedCount = std::clamp((int)std::round(count * ratio), 0, count);
    std::sort(reducedIndices.begin(), reducedIndices.begin() + reducedCount);

    std::vector<Eigen::VectorXd> nomalizedPoints;
    nomalizedPoints.reserve(reducedCount);
    for (int i = 0; i < reducedCount; i++) {
        Eigen::VectorXd p = (points[reducedIndices[i]] - cwiseMinPt).cwiseProduct(normalizer);
        nomalizedPoints.push_back(std::move(p));
    }

    std::vector<Eigen::VectorXd> skybandPoints;

    auto skybandIndices = FairTopK::DataPreprocessor::getSkyband(nomalizedPoints, k);

    std::vector<int> remainingIdx;
    remainingIdx.reserve(skybandIndices.size());

    for (auto index : skybandIndices)
        remainingIdx.push_back(reducedIndices[index]);

    if (isCompas) {
        for (auto index : skybandIndices) 
            skybandPoints.push_back(nomalizedPoints[index]);

        auto subsetIndices = FairTopK::DataPreprocessor::getSubset(skybandPoints, k);

        remainingIdx.clear();
        for (auto index : subsetIndices)
            remainingIdx.push_back(reducedIndices[skybandIndices[index]]);
    }

    if (isCompas) {
        outputCOMPAS(remainingIdx, points, genders, races, cwiseMinPt, cwiseMaxPt, outputFile);
    }
    else {
        outputJEE(remainingIdx, points, genders, cwiseMinPt, cwiseMaxPt, outputFile);
    }

    return 0;
}