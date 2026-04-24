#pragma once

#include <iostream>
#define LOG_INFO(message)                                                      \
  Logger::log_info(message, __FILE__, __LINE__, __func__)
#define LOG_DEBUG(message)                                                     \
  Logger::log_debug(message, __FILE__, __LINE__, __func__)
#define LOG_ERROR(message)                                                     \
  Logger::log_error(message, __FILE__, __LINE__, __func__)

class Logger {
public:
  static void log_info(const char *message, const char *file, int line,
                       const char *func) {
    std::cout << "[INFO] [" << file << ":" << func << ":" << line << "] "
              << message << std::endl;
  }

  static void log_debug(const char *message, const char *file, int line,
                        const char *func) {
    std::cout << "[DEBUG] [" << file << ":" << func << ":" << line << "] "
              << message << std::endl;
  }

  static void log_error(const char *message, const char *file, int line,
                        const char *func) {
    std::cerr << "[ERROR] [" << file << ":" << func << ":" << line << "] "
              << message << std::endl;
  }
};
