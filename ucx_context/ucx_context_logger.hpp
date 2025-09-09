/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.

Licensed under the Apache License Version 2.0 with LLVM Exceptions
(the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

    https://llvm.org/LICENSE.txt

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#ifndef UCX_CONTEXT_UCX_CONTEXT_LOGGER_HPP_
#define UCX_CONTEXT_UCX_CONTEXT_LOGGER_HPP_

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace stdexe_ucx_runtime {

// Log levels
/**
 * @enum LogLevel
 * @brief Defines the severity levels for log messages.
 */
enum class LogLevel {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  FATAL = 5
};

// Base logger interface
/**
 * @class UcxLogger
 * @brief An abstract interface for logging messages.
 *
 * This class defines the contract for logger implementations, supporting
 * different log levels, stream-based logging, and flushing.
 */
class UcxLogger {
 public:
  /**
   * @brief Virtual destructor for the logger interface.
   */
  virtual ~UcxLogger() = default;

  // Core logging methods
  /**
   * @brief Logs a message at a specific level.
   * @param level The log level of the message.
   * @param message The message to log.
   */
  virtual void log(LogLevel level, std::string_view message) = 0;
  /**
   * @brief Flushes any buffered log messages to the underlying output.
   */
  virtual void flush() = 0;

  // Log level control
  /**
   * @brief Sets the minimum log level for messages to be recorded.
   * @param level The new minimum log level.
   */
  virtual void set_level(LogLevel level) = 0;
  /**
   * @brief Gets the current minimum log level.
   * @return The current log level.
   */
  virtual LogLevel get_level() const = 0;
  /**
   * @brief Checks if a given log level is enabled.
   * @param level The log level to check.
   * @return True if the level is enabled, false otherwise.
   */
  virtual bool should_log(LogLevel level) const = 0;

  // Direct stream access for zero-copy logging
  /**
   * @brief Gets the output stream for a specific log level.
   * @param level The log level for which to get the stream.
   * @return A reference to the output stream (e.g., `std::cout` or
   * `std::cerr`).
   */
  virtual std::ostream& get_stream(LogLevel level) = 0;

  // State management for streaming
  /**
   * @brief Sets the log level for the current streaming operation.
   * @param level The log level to set for the current stream.
   */
  virtual void set_current_stream_level(LogLevel level) = 0;
  /**
   * @brief Gets the log level of the current streaming operation.
   * @return The current stream's log level.
   */
  virtual LogLevel get_current_stream_level() const = 0;
  /**
   * @brief Gets the current output stream based on the current stream level.
   * @return A reference to the current output stream.
   */
  virtual std::ostream& get_current_stream() = 0;
};

// Default logger implementation using std::cout/cerr
/**
 * @class DefaultUcxLogger
 * @brief A default implementation of UcxLogger that writes to `std::cout` and
 * `std::cerr`.
 *
 * Logs ERROR and FATAL messages to `std::cerr`, and all other levels to
 * `std::cout`.
 */
class DefaultUcxLogger : public UcxLogger {
 public:
  /**
   * @brief Constructs a DefaultUcxLogger with an initial level of INFO.
   */
  DefaultUcxLogger()
    : current_level_(LogLevel::INFO), current_stream_level_(LogLevel::INFO) {}

  /**
   * @brief Logs a message if the level is sufficient.
   * @param level The message's log level.
   * @param message The message content.
   */
  void log(LogLevel level, std::string_view message) override {
    if (should_log(level)) {
      get_stream(level) << message;
    }
  }

  /**
   * @brief Flushes `std::cout` and `std::cerr`.
   */
  void flush() override {
    std::cout.flush();
    std::cerr.flush();
  }

  /**
   * @brief Sets the logger's minimum level.
   * @param level The new minimum level.
   */
  void set_level(LogLevel level) override { current_level_ = level; }

  /**
   * @brief Gets the logger's current minimum level.
   * @return The current minimum level.
   */
  LogLevel get_level() const override { return current_level_; }

  /**
   * @brief Determines if a log level should be processed.
   * @param level The level to check.
   * @return True if the level is greater than or equal to the current minimum
   * level.
   */
  bool should_log(LogLevel level) const override {
    return static_cast<int>(level) >= static_cast<int>(current_level_);
  }

  class NullStreambuf : public std::streambuf {
   protected:
    int overflow(int c) override {
      return traits_type::not_eof(c);  // Return non-EOF to indicate success
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
      return n;  // Report all characters as "written"
    }
  };

  std::ostream& get_null_stream() {
    static NullStreambuf s_null_streambuf;
    static std::ostream s_null_stream(&s_null_streambuf);
    return s_null_stream;
  }

  /**
   * @brief Gets the appropriate stream for the log level.
   * @param level The log level.
   * @return `std::cerr` for ERROR and FATAL, null stream for other levels.
   */
  std::ostream& get_stream(LogLevel level) override {
    if (static_cast<int>(level) >= static_cast<int>(LogLevel::ERROR)) {
      return std::cerr;
    }
    return get_null_stream();
  }

  /**
   * @brief Sets the level for the current streaming operation.
   * @param level The level to set.
   */
  void set_current_stream_level(LogLevel level) override {
    current_stream_level_ = level;
  }

  /**
   * @brief Gets the level of the current streaming operation.
   * @return The current stream's log level.
   */
  LogLevel get_current_stream_level() const override {
    return current_stream_level_;
  }

  /**
   * @brief Gets the stream for the current streaming operation.
   * @return The output stream for the current level.
   */
  std::ostream& get_current_stream() override {
    return get_stream(current_stream_level_);
  }

 private:
  LogLevel current_level_;
  LogLevel current_stream_level_;
};

// Logger manager singleton
/**
 * @class UcxLoggerManager
 * @brief A singleton to manage the global logger instance.
 */
class UcxLoggerManager {
 public:
  /**
   * @brief Gets the singleton instance of the logger manager.
   * @return A reference to the UcxLoggerManager.
   */
  static UcxLoggerManager& get_instance() {
    static UcxLoggerManager instance;
    return instance;
  }

  /**
   * @brief Sets the global logger instance.
   * @param logger A pointer to the logger to use.
   */
  void set_logger(UcxLogger* logger) { logger_ = logger; }

  /**
   * @brief Gets the current global logger instance.
   * @return A pointer to the current logger.
   */
  UcxLogger* get_logger() { return logger_; }

  // Create and set default logger
  /**
   * @brief Creates and sets a `DefaultUcxLogger` as the global logger.
   */
  void set_default_logger() {
    static DefaultUcxLogger default_logger;
    set_logger(&default_logger);
  }

 private:
  UcxLoggerManager() : logger_(nullptr) { set_default_logger(); }
  UcxLogger* logger_;
};

// Default stream operator implementation
/**
 * @brief Overloads the stream insertion operator for logging arbitrary types.
 * @tparam T The type of the value to log.
 * @param logger The logger instance.
 * @param value The value to log.
 * @return A reference to the logger.
 */
template <typename T>
UcxLogger& operator<<(UcxLogger& logger, const T& value) {
  if (logger.should_log(logger.get_current_stream_level())) {
    logger.get_current_stream() << value;
  }
  return logger;
}

// Support for std::endl and other stream manipulators
/**
 * @brief Overloads the stream insertion operator for stream manipulators like
 * `std::endl`.
 * @param logger The logger instance.
 * @param manip The stream manipulator.
 * @return A reference to the logger.
 */
inline UcxLogger& operator<<(
  UcxLogger& logger, std::ostream& (*manip)(std::ostream&)) {
  if (manip == static_cast<std::ostream& (*)(std::ostream&)>(std::endl)) {
    logger.get_current_stream() << manip;
    logger.flush();
  }
  return logger;
}

// Log level stream operator
/**
 * @brief Overloads the stream insertion operator for setting the log level.
 * @param logger The logger instance.
 * @param level The log level to set for the following stream operations.
 * @return A reference to the logger.
 */
inline UcxLogger& operator<<(UcxLogger& logger, LogLevel level) {
  if (logger.should_log(level)) {
    logger.set_current_stream_level(level);
  }
  return logger;
}

// Global logger macro
#define UCX_CONN_LOG (*UcxLoggerManager::get_instance().get_logger())

// Convenience macros for different log levels
#define UCX_CONN_TRACE UCX_CONN_LOG << LogLevel::TRACE << log_prefix_
#define UCX_CONN_DEBUG UCX_CONN_LOG << LogLevel::DEBUG << log_prefix_
#define UCX_CONN_INFO UCX_CONN_LOG << LogLevel::INFO << log_prefix_
#define UCX_CONN_WARN UCX_CONN_LOG << LogLevel::WARN << log_prefix_
#define UCX_CONN_ERROR UCX_CONN_LOG << LogLevel::ERROR << log_prefix_
#define UCX_CONN_FATAL UCX_CONN_LOG << LogLevel::FATAL << log_prefix_
#define UCX_CTX_TRACE UCX_CONN_LOG << LogLevel::TRACE
#define UCX_CTX_DEBUG UCX_CONN_LOG << LogLevel::DEBUG
#define UCX_CTX_INFO UCX_CONN_LOG << LogLevel::INFO
#define UCX_CTX_WARN UCX_CONN_LOG << LogLevel::WARN
#define UCX_CTX_ERROR UCX_CONN_LOG << LogLevel::ERROR
#define UCX_CTX_FATAL UCX_CONN_LOG << LogLevel::FATAL

}  // namespace stdexe_ucx_runtime

#endif  // UCX_CONTEXT_UCX_CONTEXT_LOGGER_HPP_
