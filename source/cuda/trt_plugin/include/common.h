#pragma once
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logger.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Global logger instance
static Logger gLogger{nvinfer1::ILogger::Severity::kINFO};

// Convenience macros for logging
#define gLogVerbose LOG_VERBOSE(gLogger)
#define gLogInfo    LOG_INFO(gLogger)
#define gLogWarning LOG_WARN(gLogger)
#define gLogError   LOG_ERROR(gLogger)
#define gLogFatal   LOG_FATAL(gLogger)

// Simple error reporting function
inline void reportError(const char* msg, const char* file = nullptr, int line = -1) noexcept
{
    gLogError << "Error: " << msg;
    if (file != nullptr)
    {
        gLogError << " (at " << file << ":" << line << ")";
    }
    gLogError << std::endl;
}

#undef ASSERT
#define ASSERT(condition)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            gLogError << "Assertion failure: " << #condition << std::endl;                                     \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

inline int64_t volume(nvinfer1::Dims const& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

inline int64_t volume(nvinfer1::Dims const& dims, int32_t start, int32_t stop)
{
    ASSERT(start >= 0);
    ASSERT(start <= stop);
    ASSERT(stop <= dims.nbDims);
    ASSERT(std::all_of(dims.d + start, dims.d + stop, [](int32_t x) { return x >= 0; }));
    return std::accumulate(dims.d + start, dims.d + stop, int64_t{1}, std::multiplies<int64_t>{});
}