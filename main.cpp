#include "simulator.hpp"
#include "src.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class DataLoader {
public:
  bool loadDataFromFile(const std::string &filename, std::vector<float> &data) {
    std::ifstream file(filename);

    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      return false;
    }

    std::string line;
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      float value;
      while (ss >> value) {
        data.push_back(value);
      }
    }

    file.close();
    return true;
  }
};

int main() {
  sjtu::GpuSimulator gpu_sim;
  sjtu::MatrixMemoryAllocator matrix_memory_allocator;
  std::vector<sjtu::Matrix *> keys;
  std::vector<sjtu::Matrix *> values;
  std::vector<sjtu::Matrix *> queries;
  std::vector<sjtu::Matrix *> answers;

  // Read Data from File
  std::string filename = "./data/keys.txt";
  std::vector<float> data;

  DataLoader loader;

  if (loader.loadDataFromFile(filename, data)) {
    std::cerr << "Data loaded successfully!" << std::endl;
  } else {
    std::cerr << "Failed to load data from file." << std::endl;
  }

  for (int i = 0; i < 32; ++i) {
    keys.push_back(
        new sjtu::Matrix(1, 512,
                         std::vector<float>(data.begin() + i * 512,
                                            data.begin() + (i + 1) * 512),
                         gpu_sim));
    matrix_memory_allocator.Bind(keys.back(), "key_" + std::to_string(i));
  }

  data.clear();

  filename = "./data/values.txt";

  if (loader.loadDataFromFile(filename, data)) {
    std::cerr << "Data loaded successfully!" << std::endl;
  } else {
    std::cerr << "Failed to load data from file." << std::endl;
  }

  for (int i = 0; i < 32; ++i) {
    values.push_back(
        new sjtu::Matrix(1, 512,
                         std::vector<float>(data.begin() + i * 512,
                                            data.begin() + (i + 1) * 512),
                         gpu_sim));
    matrix_memory_allocator.Bind(values.back(), "value_" + std::to_string(i));
  }

  data.clear();

  filename = "./data/queries.txt";

  if (loader.loadDataFromFile(filename, data)) {
    std::cerr << "Data loaded successfully!" << std::endl;
  } else {
    std::cerr << "Failed to load data from file." << std::endl;
  }

  for (int i = 0; i < 32; ++i) {
    queries.push_back(new sjtu::Matrix(
        i + 1, 512,
        std::vector<float>(data.begin() + i * (i + 1) * 512 / 2,
                           data.begin() + (i + 1) * (i + 2) * 512 / 2),
        gpu_sim));
    matrix_memory_allocator.Bind(queries.back(), "query_" + std::to_string(i));
  }

  data.clear();

  filename = "./data/ans.txt";

  if (loader.loadDataFromFile(filename, data)) {
    std::cerr << "Data loaded successfully!" << std::endl;
  } else {
    std::cerr << "Failed to load data from file." << std::endl;
  }
  for (int i = 0; i < 32; ++i) {
    answers.push_back(new sjtu::Matrix(
        i + 1, 512,
        std::vector<float>(data.begin() + i * (i + 1) * 512 / 2,
                           data.begin() + (i + 1) * (i + 2) * 512 / 2),
        gpu_sim));
    matrix_memory_allocator.Bind(answers.back(), "answer_" + std::to_string(i));
  }

  sjtu::Rater rater(keys, values, queries, answers);
  sjtu::Test(rater, gpu_sim, matrix_memory_allocator);
  return 0;
}