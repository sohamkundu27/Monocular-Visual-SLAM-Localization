// Exception hierarchy mirroring the Python built-ins the original code relies on.
//
// The CLI entry points map exception *categories* to exit codes exactly as the
// Python ones do (DatasetError -> 2, ValueError/RuntimeError -> 1, ...), so
// the categories are preserved here rather than collapsed into one type:
//
//   Python                          C++
//   ValueError                      mvslam::ValueError
//     config.ConfigError              mvslam::ConfigError
//     calibration.CalibrationError    mvslam::CalibrationError
//     json.JSONDecodeError            mvslam::JsonDecodeError
//   RuntimeError                    mvslam::RuntimeError
//   OSError                         mvslam::OSError
//     FileNotFoundError               mvslam::FileNotFoundError
//       kitti.DatasetError              mvslam::DatasetError
//   KeyError                        mvslam::KeyError
//   IndexError                      mvslam::IndexError
#pragma once

#include <stdexcept>
#include <string>

namespace mvslam {

/// Base of every exception thrown deliberately by this project.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ValueError : public Error {
public:
    using Error::Error;
};

class RuntimeError : public Error {
public:
    using Error::Error;
};

class OSError : public Error {
public:
    using Error::Error;
};

class FileNotFoundError : public OSError {
public:
    using OSError::OSError;
};

class KeyError : public Error {
public:
    using Error::Error;
};

class IndexError : public Error {
public:
    using Error::Error;
};

/// Raised when a configuration file or override is malformed.
class ConfigError : public ValueError {
public:
    using ValueError::ValueError;
};

/// Raised when a calibration file cannot be parsed or is inconsistent.
class CalibrationError : public ValueError {
public:
    using ValueError::ValueError;
};

/// Raised when a JSON document cannot be parsed.
class JsonDecodeError : public ValueError {
public:
    using ValueError::ValueError;
};

/// Raised when the dataset layout is missing or malformed.
class DatasetError : public FileNotFoundError {
public:
    using FileNotFoundError::FileNotFoundError;
};

}  // namespace mvslam
