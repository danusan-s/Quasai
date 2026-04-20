#pragma once

#include <iostream>
#define LOG_INFO(message) Logger::log(message, __FILE__, __LINE__)
#define LOG_ERROR(message) Logger::log_error(message, __FILE__, __LINE__)

class Logger {
public:
  static void log_info(const char *message, const char *file, int line) {
    std::cout << "[INFO] [" << file << ":" << line << "] " << message
              << std::endl;
  }
  static void log_error(const char *message, const char *file, int line) {
    std::cerr << "[ERROR] [" << file << ":" << line << "] " << message
              << std::endl;
  }
};
