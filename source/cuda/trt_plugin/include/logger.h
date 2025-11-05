#ifndef LOGGER_H
#define LOGGER_H

#include <NvInfer.h>
#include <cassert>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

// Very small LogStreamConsumer for stream-style logging
class LogStreamConsumer
{
public:
    LogStreamConsumer(nvinfer1::ILogger::Severity reportableSeverity,
                      nvinfer1::ILogger::Severity msgSeverity) noexcept
        : mMsgSeverity(msgSeverity)
        , mShouldEmit(msgSeverity <= reportableSeverity)
    {
    }

    LogStreamConsumer(LogStreamConsumer&& other) noexcept
        : mMsgSeverity(other.mMsgSeverity)
        , mShouldEmit(other.mShouldEmit)
    {
        mOss.str(std::move(other.mOss.str()));
        other.mShouldEmit = false;
    }

    ~LogStreamConsumer() noexcept
    {
        if (!mShouldEmit) return;
        std::lock_guard<std::mutex> lk(getMutex());
        std::ostringstream out;
        out << severityPrefix(mMsgSeverity) << mOss.str();
        if (mMsgSeverity >= nvinfer1::ILogger::Severity::kINFO)
        {
            std::cout << out.str();
            std::cout.flush();
        }
        else
        {
            std::cerr << out.str();
            std::cerr.flush();
        }
    }

    template <typename T>
    LogStreamConsumer& operator<<(T const& v) noexcept
    {
        if (mShouldEmit) mOss << v;
        return *this;
    }

    typedef std::ostream& (*ManipFn)(std::ostream&);
    LogStreamConsumer& operator<<(ManipFn manip) noexcept
    {
        if (mShouldEmit) manip(mOss);
        return *this;
    }

private:
    static std::mutex& getMutex() noexcept
    {
        static std::mutex s_mutex;
        return s_mutex;
    }
    static const char* severityPrefix(nvinfer1::ILogger::Severity s) noexcept
    {
        switch (s)
        {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "[F] ";
            case nvinfer1::ILogger::Severity::kERROR: return "[E] ";
            case nvinfer1::ILogger::Severity::kWARNING: return "[W] ";
            case nvinfer1::ILogger::Severity::kINFO: return "[I] ";
            case nvinfer1::ILogger::Severity::kVERBOSE: return "[V] ";
            default: return "[?] ";
        }
    }

    nvinfer1::ILogger::Severity mMsgSeverity;
    bool mShouldEmit{false};
    std::ostringstream mOss;
};

// Minimal Logger implementing nvinfer1::ILogger
class Logger : public nvinfer1::ILogger
{
public:
    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING)
        : mReportableSeverity(severity) {}

    nvinfer1::ILogger& getTRTLogger() noexcept { return *this; }

    void log(Severity severity, const char* msg) noexcept override
    {
        LogStreamConsumer(mReportableSeverity, severity) << "[TRT] " << std::string(msg) << std::endl;
    }

    void setReportableSeverity(Severity s) noexcept { mReportableSeverity = s; }
    Severity getReportableSeverity() const noexcept { return mReportableSeverity; }

private:
    Severity mReportableSeverity{Severity::kWARNING};
};



// Stream helpers using a Logger instance
inline LogStreamConsumer LOG_VERBOSE(const Logger& logger) { return LogStreamConsumer(logger.getReportableSeverity(), nvinfer1::ILogger::Severity::kVERBOSE); }
inline LogStreamConsumer LOG_INFO(const Logger& logger)    { return LogStreamConsumer(logger.getReportableSeverity(), nvinfer1::ILogger::Severity::kINFO); }
inline LogStreamConsumer LOG_WARN(const Logger& logger)    { return LogStreamConsumer(logger.getReportableSeverity(), nvinfer1::ILogger::Severity::kWARNING); }
inline LogStreamConsumer LOG_ERROR(const Logger& logger)   { return LogStreamConsumer(logger.getReportableSeverity(), nvinfer1::ILogger::Severity::kERROR); }
inline LogStreamConsumer LOG_FATAL(const Logger& logger)   { return LogStreamConsumer(logger.getReportableSeverity(), nvinfer1::ILogger::Severity::kINTERNAL_ERROR); }

#endif // LOGGER_H