#pragma once

#include <iostream>
#include <string>

namespace quasai::utils {

#define LOG_INFO(message)                                                      \
  utils::Logger::log_info(message, __FILE__, __LINE__, __func__)

#ifdef QUASAI_DEBUG
#define LOG_DEBUG(message)                                                     \
  utils::Logger::log_debug(message, __FILE__, __LINE__, __func__)
#else
#define LOG_DEBUG(message)
#endif

#ifdef QUASAI_SUPPRESS_WARNINGS
#define LOG_WARNING(message)
#else
#define LOG_WARNING(message)                                                   \
  utils::Logger::log_warning(message, __FILE__, __LINE__, __func__)
#endif

#define LOG_ERROR(message)                                                     \
  utils::Logger::log_error(message, __FILE__, __LINE__, __func__)

class Logger {
public:
  static void log_info(const char *message, const char *file, int line,
                       const char *func) {
    std::cout << "[INFO] [" << trim_path(file) << ":" << func << ":" << line
              << "] " << message << std::endl;
  }

  static void log_debug(const char *message, const char *file, int line,
                        const char *func) {
    std::cout << "[DEBUG] [" << trim_path(file) << ":" << func << ":" << line
              << "] " << message << std::endl;
  }

  static void log_warning(const char *message, const char *file, int line,
                          const char *func) {
    std::cerr << "[WARNING] [" << trim_path(file) << ":" << func << ":" << line
              << "] " << message << std::endl;
  }

  static void log_error(const char *message, const char *file, int line,
                        const char *func) {
    std::cerr << "[ERROR] [" << trim_path(file) << ":" << func << ":" << line
              << "] " << message << std::endl;
  }

private:
  static std::string trim_path(const char *path) {
    std::string s(path);

    // This is a bit hacky and will depend on what folder the user has the
    // quasai repo in
    std::string key = "Quasai/";
    auto pos = s.find(key);

    if (pos != std::string::npos) {
      return s.substr(pos);
    }

    // fallback: just filename
    auto slash = s.find_last_of("/\\");
    if (slash != std::string::npos) {
      return s.substr(slash + 1);
    }

    return s;
  }
};

} // namespace quasai::utils
