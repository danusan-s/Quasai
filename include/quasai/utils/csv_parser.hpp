#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace quasai {

inline std::unordered_map<std::string, std::vector<std::string>>
parse_csv(const std::string &filename) {
  std::unordered_map<std::string, std::vector<std::string>> data;
  std::fstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  std::string line;
  std::vector<std::string> headers;

  std::cout << "Parsing CSV file: " << filename << std::endl;

  // Read the header line
  if (std::getline(file, line)) {
    size_t pos = 0;
    while ((pos = line.find(',')) != std::string::npos) {
      headers.push_back(line.substr(0, pos));
      line.erase(0, pos + 1);
    }
    headers.push_back(line); // Add the last header
  }

  std::cout << "Found headers: ";
  for (const std::string &header : headers) {
    std::cout << header << " ";
  }

  // Read the data lines
  while (std::getline(file, line)) {
    size_t pos = 0;
    size_t header_index = 0;
    while ((pos = line.find(',')) != std::string::npos) {
      if (header_index < headers.size()) {
        data[headers[header_index]].push_back(line.substr(0, pos));
      }
      line.erase(0, pos + 1);
      header_index++;
    }
    if (header_index < headers.size()) {
      data[headers[header_index]].push_back(line); // Add the last value
    }
  }

  std::cout << "Finished parsing CSV file. Parsed " << data.size()
            << " columns." << std::endl;

  for (const auto &pair : data) {
    std::cout << "Column: " << pair.first
              << ", Number of values: " << pair.second.size() << std::endl;
  }

  return data;
}

inline bool clean_is_float(std::vector<std::string> &values) {
  bool is_float = true;
  for (std::string &value : values) {
    try {
      if (value.empty()) {
        value = "0.0"; // Treat empty strings as zero
      }
      std::stof(value);
    } catch (const std::invalid_argument &) {
      is_float = false;
      break;
    }
  }
  return is_float;
}

} // namespace quasai
