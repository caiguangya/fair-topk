#include "data_loader.h"
#include <string>
#include <fstream>
#include <unordered_map>
#include <ctime>

#include <boost/tokenizer.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xview.hpp>

namespace FairTopK {

namespace DataLoader {

void readCompasData(const std::string& file, std::vector<Eigen::VectorXd>& points, 
    std::vector<int>& genders, std::vector<int>& races) {

    std::ifstream inf(file, std::ifstream::in);
    std::string line;
    std::getline(inf, line);
    
    boost::char_separator<char> sep(",", "", boost::keep_empty_tokens);
    boost::tokenizer<boost::char_separator<char> > tok(line, sep);

    constexpr int dimension = 6;

    constexpr int genderIdx = 5;
    constexpr int raceIdx = 9;

    constexpr int juryOtherCourtIdx = 13;
    constexpr int priorsCountIdx = 14;
    constexpr int daysFromCompasIdx = 21;
    constexpr int startIdx = 49;
    constexpr int endIdx = 50;

    constexpr int jailInIdx = 16;
    constexpr int jailOutIdx = 17;

    constexpr int coordIndices[dimension - 1] = 
        { juryOtherCourtIdx, priorsCountIdx, daysFromCompasIdx, startIdx, endIdx };

    std::unordered_map<int, int> pointCoordIdxLookupTable;
    for (int i = 0; i < dimension - 1; i++) {
        pointCoordIdxLookupTable.emplace(coordIndices[i], i);
    }

    while (std::getline(inf, line)) {
        tok = boost::tokenizer<boost::char_separator<char> >(line, sep);
        
        bool hasMissingValue = false;

        int gender = -1;
        int race = -1;
        Eigen::VectorXd point = Eigen::VectorXd::Zero(dimension);

        std::time_t jailInDate = 0;
        std::time_t jailOutDate = 0;

        int idx = 0;
        for (auto token : tok) {
            if (idx == genderIdx) {
                if (token.empty()) {
                    hasMissingValue = true;
                    break;
                }

                if (token == "Female") {
                    gender = 0;
                }
                else {
                    gender = 1;
                }
            }
            else if (idx == raceIdx) {
                if (token.empty()) {
                    hasMissingValue = true;
                    break;
                }

                if (token == "African-American") {
                    race = 0;
                }
                else if (token == "Caucasian") {
                    race = 1;
                }
                else {
                    race = 2;
                }
            }
            else if (std::find(coordIndices, coordIndices + dimension - 1, idx) != coordIndices + dimension - 1) {              
                if (token.empty()) {
                    hasMissingValue = true;
                    break;
                }

                point(pointCoordIdxLookupTable[idx]) = stoi(token);
            }
            else if (idx == jailInIdx) {
                if (token.empty()) {
                    hasMissingValue = true;
                    break;
                }

                std::tm t = {};
                std::istringstream ss(token);
                ss >> std::get_time(&t, "%Y-%m-%d %H:%M:%S");

                if (ss.fail()) {
                    hasMissingValue = true;
                    break;
                }
                t.tm_hour = 0;
                t.tm_min = 0;
                t.tm_sec = 0;
                jailInDate = std::mktime(&t);
            }
            else if (idx == jailOutIdx) {
                 if (token.empty()) {
                    hasMissingValue = true;
                    break;
                }

                std::tm t = {};
                std::istringstream ss(token);
                ss >> std::get_time(&t, "%Y-%m-%d %H:%M:%S");

                if (ss.fail()) {
                    hasMissingValue = true;
                    break;
                }
                t.tm_hour = 0;
                t.tm_min = 0;
                t.tm_sec = 0;
                jailOutDate = std::mktime(&t);
            }

            idx += 1;
        }

        if (!hasMissingValue) {
            point(dimension - 1) = (std::difftime(jailOutDate, jailInDate) / (60 * 60 * 24)) + 1;
            points.push_back(std::move(point));
            genders.push_back(gender);
            races.push_back(race);
        }
    }
}

void readJEEData(const std::string& file, std::vector<Eigen::VectorXd>& points, std::vector<int>& genders) {
    std::ifstream inf(file, std::ifstream::in);
    std::string line;
    std::getline(inf, line);
    
    boost::char_separator<char> sep(",", "", boost::keep_empty_tokens);boost::tokenizer<boost::char_separator<char> > tok(line, sep);

    constexpr int dimension = 3;

    constexpr int genderIdx = 4;
    constexpr int mathIdx = 7;
    constexpr int physIdx = 8;
    constexpr int chemIdx = 9;

    constexpr int scoreIndices[dimension] = { mathIdx, physIdx, chemIdx };

    std::unordered_map<int, int> pointCoordIdxLookupTable;
    for (int i = 0; i < dimension; i++) {
        pointCoordIdxLookupTable.emplace(scoreIndices[i], i);
    }

    int lineCount = 0;
    while (std::getline(inf, line)) {
        tok = boost::tokenizer<boost::char_separator<char> >(line, sep);
        lineCount += 1;

        int gender = -1;
        Eigen::VectorXd point = Eigen::VectorXd::Zero(dimension);

        bool hasMissingValue = false;

        int idx = 0;
        for (auto token : tok) {
            if (idx == genderIdx) {
                if (token.empty()) {
                    hasMissingValue = true;
                    break;
                }

                gender = (token == "M" ? 0 : 1);
            }

            else if (std::find(scoreIndices, scoreIndices + dimension, idx) !=
                     scoreIndices + dimension) {
                if (token.empty()) {
                    hasMissingValue = true;
                    break;
                }
                point(pointCoordIdxLookupTable[idx]) = stoi(token);
            }

            idx += 1;
        }

        if (!hasMissingValue) {
            points.push_back(std::move(point));
            genders.push_back(gender);
        }
    }
}

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