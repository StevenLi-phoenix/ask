#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <pwd.h>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <curl/curl.h>
#include "vendor/cjson/cJSON.h"

namespace {

constexpr const char *DEFAULT_MODEL = "gpt-5.5";
constexpr size_t MAX_BUFFER_SIZE = 8192;
constexpr int DEFAULT_TOKEN_LIMIT = 128000;
constexpr const char *MODELS_CACHE_FILE = "~/.cache/ask_models_cache.json";
constexpr time_t MODELS_CACHE_EXPIRY = 86400; // 24 hours
constexpr long CONNECT_TIMEOUT = 10L;
constexpr long REQUEST_TIMEOUT = 60L;
constexpr int MAX_RETRIES = 1; // total attempts = MAX_RETRIES + 1
constexpr const char *LAST_CONTEXT_FILE = "~/.cache/ask_last_context.json";
constexpr const char *DEFAULT_SYSTEM_PROMPT =
    "Terminal assistant. Give shortest answer with directly copyable commands. "
    "No emoji, no filler. For errors: brief cause then fix. "
    "Style: man page, not chatbot.";

volatile sig_atomic_t g_shutdownSignal = 0;
volatile sig_atomic_t g_requestInFlight = 0;
std::atomic_bool g_interruptNoticePrinted{false};

void resetSignalState() {
    g_shutdownSignal = 0;
    g_requestInFlight = 0;
    g_interruptNoticePrinted.store(false);
}

bool shutdownRequested() {
    return g_shutdownSignal != 0;
}

int interruptedExitCode() {
    return shutdownRequested() ? 128 + g_shutdownSignal : 1;
}

void setRequestInFlight(bool inFlight) {
    g_requestInFlight = inFlight ? 1 : 0;
}

void printInterruptNotice() {
    bool expected = false;
    if (g_interruptNoticePrinted.compare_exchange_strong(expected, true)) {
        std::fprintf(stderr, "Interrupted.\n");
    }
}

extern "C" void handleTerminationSignal(int signum) {
    g_shutdownSignal = signum;

    const char newline = '\n';
    if (g_requestInFlight != 0) {
        (void)::write(STDERR_FILENO, &newline, 1);
    }
}

bool installSignalHandlers() {
    struct sigaction action {};
    action.sa_handler = handleTerminationSignal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;

    return sigaction(SIGINT, &action, nullptr) == 0 &&
        sigaction(SIGTERM, &action, nullptr) == 0;
}

std::string detectOS() {
    struct utsname info;
    if (uname(&info) != 0) return "Unknown OS";
    std::string sysname = info.sysname;
    std::string os = (sysname == "Darwin") ? "macOS" : sysname;
    return os + " " + info.sysname + " " + info.release + " " + info.machine;
}

enum class LogLevel {
    None = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4
};

struct Message {
    std::string role;
    std::string content;
};

struct ModelInfo {
    std::string id;
    time_t created{};
};

struct ModelsCacheData {
    std::vector<ModelInfo> models;
    time_t lastUpdated{};
};

struct ResponseBuffer {
    std::string data;
    std::string responseContent;
    bool streamEnabled{false};
    bool sawStreamData{false};
    bool rawMode{false};
    std::atomic_bool *firstTokenFlag{nullptr};
    class Logger *logger{nullptr};
};

class Logger {
public:
    Logger() = default;
    ~Logger() { close(); }

    void configure(LogLevel level, bool debug, bool logToFile, std::string filePath) {
        level_ = level;
        debug_ = debug;
        logToFile_ = logToFile;
        filePath_ = std::move(filePath);
        if (logToFile_) {
            openFile();
        }
    }

    void log(LogLevel level, const char *format, ...) {
        if (level > level_) return;

        char levelStr[10];
        switch (level) {
            case LogLevel::Error: std::strcpy(levelStr, "ERROR"); break;
            case LogLevel::Warn: std::strcpy(levelStr, "WARN"); break;
            case LogLevel::Info: std::strcpy(levelStr, "INFO"); break;
            case LogLevel::Debug: std::strcpy(levelStr, "DEBUG"); break;
            default: std::strcpy(levelStr, "NONE"); break;
        }

        std::time_t now = std::time(nullptr);
        std::tm localTime{};
        if (localtime_r(&now, &localTime) == nullptr) {
            std::memset(&localTime, 0, sizeof(localTime));
        }
        char timeStr[20];
        std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &localTime);

        va_list args;
        va_start(args, format);

        char message[MAX_BUFFER_SIZE];
        std::vsnprintf(message, sizeof(message), format, args);

        if (level <= LogLevel::Warn || debug_) {
            std::fprintf(stderr, "[%s] %s: %s\n", timeStr, levelStr, message);
        }

        if (logToFile_ && file_) {
            std::fprintf(file_, "[%s] %s: %s\n", timeStr, levelStr, message);
            std::fflush(file_);
        }

        va_end(args);
    }

    LogLevel level() const { return level_; }
    bool debug() const { return debug_; }

private:
    void openFile() {
        file_ = std::fopen(filePath_.c_str(), "a");
        if (!file_) {
            std::fprintf(stderr, "Failed to open log file %s\n", filePath_.c_str());
            logToFile_ = false;
        }
    }

    void close() {
        if (file_) {
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    LogLevel level_{LogLevel::Info};
    bool debug_{false};
    bool logToFile_{false};
    std::string filePath_{"ask.log"};
    FILE *file_{nullptr};
};

struct Settings {
    std::string apiKey;
    std::string model;
    std::string systemPrompt;
    int tokenLimit = DEFAULT_TOKEN_LIMIT;
    int fileSizeLimit = 10000;
    bool debugMode{false};
    LogLevel logLevel{LogLevel::Info};
    bool logToFile{false};
    std::string logFilePath{"ask.log"};
};

class FileUtil {
public:
    static std::string expandHomePath(const std::string &path, Logger &logger) {
        if (path.empty() || path[0] != '~') {
            return path;
        }

        struct passwd *pw = getpwuid(getuid());
        if (pw == nullptr) {
            logger.log(LogLevel::Error, "Could not determine home directory");
            return path;
        }

        if (pw->pw_dir == nullptr) {
            logger.log(LogLevel::Error, "Home directory path is null");
            return path;
        }

        std::string expanded = pw->pw_dir;
        expanded += path.substr(1);

        auto lastSlash = expanded.find_last_of('/');
        if (lastSlash != std::string::npos) {
            std::string dir = expanded.substr(0, lastSlash);
            if (mkdir(dir.c_str(), 0700) == -1 && errno != EEXIST) {
                logger.log(LogLevel::Error, "Failed to create directory: %s", dir.c_str());
            }
        }

        logger.log(LogLevel::Debug, "Expanded path: %s", expanded.c_str());
        return expanded;
    }

    static bool isPlainTextFile(const std::string &filename, Logger &logger) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            logger.log(LogLevel::Warn, "Cannot open file: %s", filename.c_str());
            return false;
        }

        std::vector<unsigned char> buffer(1024);
        file.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
        size_t bytesRead = static_cast<size_t>(file.gcount());

        if (bytesRead == 0) {
            return true;
        }

        size_t nullCount = 0;
        size_t controlCount = 0;
        for (size_t i = 0; i < bytesRead; ++i) {
            if (buffer[i] == 0) {
                ++nullCount;
            } else if (buffer[i] < 32 && buffer[i] != '\n' && buffer[i] != '\r' && buffer[i] != '\t') {
                ++controlCount;
            }
        }

        if (nullCount > 0 || controlCount > bytesRead / 20) {
            return false;
        }
        return true;
    }

    static std::optional<std::string> readFileContent(const std::string &filename, Logger &logger, int fileSizeLimit = 10000, bool interactive = false) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            logger.log(LogLevel::Error, "Failed to open file: %s", filename.c_str());
            return std::nullopt;
        }

        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        if (fileSize <= 0) {
            return std::string();
        }

        if (fileSize > fileSizeLimit) {
            if (interactive) {
                std::fprintf(stderr, "File '%s' is %ldKB (limit: %dKB). Include anyway? [y/N]: ",
                    filename.c_str(), static_cast<long>(fileSize) / 1000, fileSizeLimit / 1000);
                std::fflush(stderr);

                std::string answer;
                if (!std::getline(std::cin, answer)) {
                    return std::nullopt;
                }

                if (answer.empty() || (answer[0] != 'y' && answer[0] != 'Y')) {
                    return std::nullopt;
                }
            } else {
                logger.log(LogLevel::Warn, "File too large (>%dKB): %s", fileSizeLimit / 1000, filename.c_str());
                return std::nullopt;
            }
        }

        std::string content(static_cast<size_t>(fileSize), '\0');
        file.read(content.data(), fileSize);
        content.resize(static_cast<size_t>(file.gcount()));
        logger.log(LogLevel::Debug, "Read %zu bytes from file: %s", content.size(), filename.c_str());
        return content;
    }

    static bool writeSecureFile(const std::string &path, const std::string &content, Logger &logger) {
        int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd == -1) {
            logger.log(LogLevel::Error, "Failed to open file for secure writing: %s", path.c_str());
            return false;
        }
        size_t totalWritten = 0;
        while (totalWritten < content.size()) {
            ssize_t written = ::write(fd, content.data() + totalWritten, content.size() - totalWritten);
            if (written <= 0) {
                ::close(fd);
                logger.log(LogLevel::Error, "Failed to write to file: %s", path.c_str());
                return false;
            }
            totalWritten += static_cast<size_t>(written);
        }
        ::close(fd);
        return true;
    }

    static std::string processFileReferences(const std::string &input, Logger &logger, int fileSizeLimit = 10000, bool interactive = false) {
        if (input.empty()) return {};

        std::string result = input;
        size_t searchPos = 0;

        while ((searchPos = result.find('@', searchPos)) != std::string::npos) {
            if (searchPos > 0) {
                char prev = result[searchPos - 1];
                if (!std::isspace(static_cast<unsigned char>(prev)) && prev != '(' && prev != '[' && prev != '{' && prev != '\n') {
                    searchPos += 1;
                    continue;
                }
            }

            size_t filenameStart = searchPos + 1;
            if (filenameStart >= result.size()) break;

            bool quoted = false;
            char quoteChar = '\0';
            if (result[filenameStart] == '"' || result[filenameStart] == '\'') {
                quoted = true;
                quoteChar = result[filenameStart];
                ++filenameStart;
            }

            size_t filenameEnd = filenameStart;
            if (quoted) {
                while (filenameEnd < result.size() && result[filenameEnd] != quoteChar) {
                    ++filenameEnd;
                }
            } else {
                while (filenameEnd < result.size()) {
                    char c = result[filenameEnd];
                    if (std::isspace(static_cast<unsigned char>(c)) ||
                        c == '?' || c == '!' || c == ';' ||
                        (c == '.' && (filenameEnd + 1 >= result.size() || std::isspace(static_cast<unsigned char>(result[filenameEnd + 1])))) ||
                        c == ',' || c == ')' || c == '}') {
                        break;
                    }
                    ++filenameEnd;
                }
            }

            if (filenameEnd == filenameStart) {
                searchPos = filenameStart;
                continue;
            }

            std::string filename = result.substr(filenameStart, filenameEnd - filenameStart);
            size_t suffixStart = filenameEnd;
            while (!filename.empty() && (filename.back() == '"' || filename.back() == '\'' || filename.back() == '`')) {
                filename.pop_back();
            }
            if (filename.empty()) {
                searchPos = suffixStart;
                continue;
            }
            if (quoted && suffixStart < result.size() && result[suffixStart] == quoteChar) {
                ++suffixStart;
            }

            std::string prefix = result.substr(0, searchPos);
            std::string suffix = result.substr(suffixStart);
            std::string replacement;

            if (access(filename.c_str(), F_OK) == 0 && isPlainTextFile(filename, logger)) {
                auto contentOpt = readFileContent(filename, logger, fileSizeLimit, interactive);
                if (contentOpt.has_value()) {
                    replacement = prefix + "\nFile: " + filename + "\n```\n" + *contentOpt + "\n```" + suffix;
                    logger.log(LogLevel::Info, "Attached file content: %s (%zu bytes)", filename.c_str(), contentOpt->size());
                } else {
                    replacement = prefix + "[Skipped: " + filename + " (too large)]" + suffix;
                }
            } else {
                logger.log(LogLevel::Warn, "File not found or not plain text: %s", filename.c_str());
                replacement = prefix + "[File not found: " + filename + "]" + suffix;
            }

            result.swap(replacement);
            searchPos = result.size() - suffix.size();
        }

        return result;
    }
};

class ModelsCacheManager {
public:
    ModelsCacheData &data() { return data_; }

    bool load(Logger &logger) {
        std::string cachePath = FileUtil::expandHomePath(MODELS_CACHE_FILE, logger);
        std::ifstream cacheFile(cachePath);
        if (!cacheFile.is_open()) {
            logger.log(LogLevel::Debug, "No models cache file found");
            return false;
        }

        cacheFile.seekg(0, std::ios::end);
        std::streampos fileSize = cacheFile.tellg();
        cacheFile.seekg(0, std::ios::beg);
        if (fileSize <= 0) {
            logger.log(LogLevel::Warn, "Empty models cache file");
            return false;
        }

        std::string content(static_cast<size_t>(fileSize), '\0');
        cacheFile.read(content.data(), fileSize);
        content.resize(static_cast<size_t>(cacheFile.gcount()));

        if (content.empty()) {
            logger.log(LogLevel::Warn, "Empty models cache file");
            return false;
        }

        cJSON *root = cJSON_Parse(content.c_str());
        if (!root) {
            logger.log(LogLevel::Error, "Failed to parse models cache file: %s", cJSON_GetErrorPtr());
            return false;
        }

        cJSON *timestamp = cJSON_GetObjectItem(root, "timestamp");
        if (!timestamp || !cJSON_IsNumber(timestamp)) {
            logger.log(LogLevel::Error, "Invalid timestamp in models cache");
            cJSON_Delete(root);
            return false;
        }
        data_.lastUpdated = static_cast<time_t>(timestamp->valuedouble);

        time_t now = time(nullptr);
        if (now - data_.lastUpdated > MODELS_CACHE_EXPIRY) {
            logger.log(LogLevel::Info, "Models cache is expired (older than 24 hours)");
            cJSON_Delete(root);
            return false;
        }

        cJSON *modelsArray = cJSON_GetObjectItem(root, "models");
        if (!modelsArray || !cJSON_IsArray(modelsArray)) {
            logger.log(LogLevel::Error, "Invalid models array in cache");
            cJSON_Delete(root);
            return false;
        }

        data_.models.clear();
        int modelCount = cJSON_GetArraySize(modelsArray);
        data_.models.reserve(static_cast<size_t>(modelCount));
        for (int i = 0; i < modelCount; ++i) {
            cJSON *model = cJSON_GetArrayItem(modelsArray, i);
            if (!model || !cJSON_IsObject(model)) continue;

            cJSON *id = cJSON_GetObjectItem(model, "id");
            cJSON *created = cJSON_GetObjectItem(model, "created");
            if (!id || !cJSON_IsString(id) || !created || !cJSON_IsNumber(created)) continue;

            data_.models.push_back({id->valuestring, static_cast<time_t>(created->valuedouble)});
        }

        cJSON_Delete(root);
        logger.log(LogLevel::Info, "Loaded %zu models from cache", data_.models.size());
        return !data_.models.empty();
    }

    bool save(Logger &logger) {
        if (data_.models.empty()) {
            logger.log(LogLevel::Warn, "No models to save to cache");
            return false;
        }

        cJSON *root = cJSON_CreateObject();
        if (!root) {
            logger.log(LogLevel::Error, "Failed to create JSON object for cache");
            return false;
        }

        cJSON_AddNumberToObject(root, "timestamp", static_cast<double>(data_.lastUpdated));
        cJSON *modelsArray = cJSON_CreateArray();
        if (!modelsArray) {
            logger.log(LogLevel::Error, "Failed to create models array for cache");
            cJSON_Delete(root);
            return false;
        }
        cJSON_AddItemToObject(root, "models", modelsArray);

        for (const auto &model : data_.models) {
            cJSON *node = cJSON_CreateObject();
            if (!node) continue;
            cJSON_AddStringToObject(node, "id", model.id.c_str());
            cJSON_AddNumberToObject(node, "created", static_cast<double>(model.created));
            cJSON_AddItemToArray(modelsArray, node);
        }

        char *jsonStr = cJSON_PrintUnformatted(root);
        if (!jsonStr) {
            logger.log(LogLevel::Error, "Failed to print JSON for cache");
            cJSON_Delete(root);
            return false;
        }

        std::string cachePath = FileUtil::expandHomePath(MODELS_CACHE_FILE, logger);
        std::ofstream cacheFile(cachePath);
        if (!cacheFile.is_open()) {
            logger.log(LogLevel::Error, "Failed to open cache file for writing");
            std::free(jsonStr);
            cJSON_Delete(root);
            return false;
        }

        cacheFile << jsonStr;
        cacheFile.close();

        std::free(jsonStr);
        cJSON_Delete(root);
        logger.log(LogLevel::Info, "Saved %zu models to cache", data_.models.size());
        return true;
    }

private:
    ModelsCacheData data_{};
};

class ContextManager {
public:
    void save(const std::vector<Message> &messages, Logger &logger) {
        cJSON *arr = cJSON_CreateArray();
        if (!arr) return;
        for (const auto &msg : messages) {
            cJSON *obj = cJSON_CreateObject();
            if (!obj) continue;
            cJSON_AddStringToObject(obj, "role", msg.role.c_str());
            cJSON_AddStringToObject(obj, "content", msg.content.c_str());
            cJSON_AddItemToArray(arr, obj);
        }
        char *jsonStr = cJSON_PrintUnformatted(arr);
        cJSON_Delete(arr);
        if (!jsonStr) return;
        std::string path = FileUtil::expandHomePath(LAST_CONTEXT_FILE, logger);
        FileUtil::writeSecureFile(path, jsonStr, logger);
        std::free(jsonStr);
        logger.log(LogLevel::Debug, "Saved %zu messages to context file", messages.size());
    }

    std::vector<Message> load(Logger &logger) {
        std::string path = FileUtil::expandHomePath(LAST_CONTEXT_FILE, logger);
        std::ifstream file(path);
        if (!file.is_open()) {
            logger.log(LogLevel::Debug, "No context file found");
            return {};
        }
        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        if (fileSize <= 0) return {};

        std::string content(static_cast<size_t>(fileSize), '\0');
        file.read(content.data(), fileSize);
        content.resize(static_cast<size_t>(file.gcount()));
        if (content.empty()) return {};

        cJSON *arr = cJSON_Parse(content.c_str());
        if (!arr || !cJSON_IsArray(arr)) {
            if (arr) cJSON_Delete(arr);
            logger.log(LogLevel::Warn, "Failed to parse context file");
            return {};
        }

        std::vector<Message> messages;
        int n = cJSON_GetArraySize(arr);
        messages.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            cJSON *obj = cJSON_GetArrayItem(arr, i);
            if (!obj) continue;
            cJSON *role = cJSON_GetObjectItem(obj, "role");
            cJSON *cont = cJSON_GetObjectItem(obj, "content");
            if (role && cJSON_IsString(role) && cont && cJSON_IsString(cont)) {
                messages.push_back({role->valuestring, cont->valuestring});
            }
        }
        cJSON_Delete(arr);
        logger.log(LogLevel::Debug, "Loaded %zu messages from context file", messages.size());
        return messages;
    }
};

class CurlHandle {
public:
    CurlHandle() : handle_(curl_easy_init()) {}
    ~CurlHandle() {
        if (handle_) curl_easy_cleanup(handle_);
    }
    CURL *get() { return handle_; }
    bool valid() const { return handle_ != nullptr; }
private:
    CURL *handle_{nullptr};
};

class ApiClient {
public:
    ApiClient(Logger &logger, Settings &settings, ModelsCacheManager &cache)
        : logger_(logger), settings_(settings), cache_(cache) {}

    bool validateModel(CURL *curl, const std::string &model) {
        bool cacheLoaded = cache_.load(logger_);
        if (!cacheLoaded || cache_.data().models.empty()) {
            if (!fetchModelsList(curl)) {
                if (shutdownRequested()) {
                    return false;
                }
                logger_.log(LogLevel::Warn, "Failed to fetch models list, continuing without validation");
                return true;
            }
        }

        for (const auto &info : cache_.data().models) {
            if (info.id == model) {
                logger_.log(LogLevel::Debug, "Model '%s' is valid", model.c_str());
                return true;
            }
        }

        suggestSimilarModel(model);
        return false;
    }

    std::string sendChat(CURL *curl, std::vector<Message> &messages, double temperature, bool noStream, int tokenLimit, bool rawMode = false) {
        if (messages.empty()) {
            logger_.log(LogLevel::Warn, "No messages to send to API");
            return {};
        }

        if (shutdownRequested()) {
            return {};
        }

        logger_.log(LogLevel::Info, "Sending request (model: %s, temp: %.2f, stream: %s)", settings_.model.c_str(), temperature, noStream ? "disabled" : "enabled");

        if (!trimMessagesToTokenLimit(messages, tokenLimit)) {
            return {};
        }

        cJSON *root = cJSON_CreateObject();
        if (!root) {
            logger_.log(LogLevel::Error, "Failed to allocate request JSON object");
            return {};
        }
        cJSON_AddStringToObject(root, "model", settings_.model.c_str());
        cJSON_AddNumberToObject(root, "temperature", temperature);
        cJSON_AddBoolToObject(root, "stream", !noStream);

        cJSON *messageArray = cJSON_CreateArray();
        if (!messageArray) {
            logger_.log(LogLevel::Error, "Failed to allocate request messages array");
            cJSON_Delete(root);
            return {};
        }
        for (const auto &msg : messages) {
            cJSON *message = cJSON_CreateObject();
            if (!message) {
                logger_.log(LogLevel::Error, "Failed to allocate request message object");
                cJSON_Delete(root);
                return {};
            }
            cJSON_AddStringToObject(message, "role", msg.role.c_str());
            cJSON_AddStringToObject(message, "content", msg.content.c_str());
            cJSON_AddItemToArray(messageArray, message);
        }
        cJSON_AddItemToObject(root, "messages", messageArray);

        char *jsonStr = cJSON_PrintUnformatted(root);
        if (!jsonStr) {
            logger_.log(LogLevel::Error, "Failed to serialize request JSON");
            cJSON_Delete(root);
            return {};
        }
        logger_.log(LogLevel::Debug, "Request: model=%s, messages=%zu, temperature=%.2f, stream=%s",
            settings_.model.c_str(), messages.size(), temperature, noStream ? "false" : "true");

        struct curl_slist *headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        std::string authHeader = "Authorization: Bearer " + settings_.apiKey;
        headers = curl_slist_append(headers, authHeader.c_str());
        if (!noStream) {
            headers = curl_slist_append(headers, "Accept: text/event-stream");
        }

        int attempt = 0;
        std::string result;

        while (attempt <= MAX_RETRIES) {
            attempt++;
            logger_.log(LogLevel::Info, "Attempt %d/%d", attempt, MAX_RETRIES + 1);

            std::atomic_bool spinnerStop(false);
            std::atomic_bool firstTokenReceived(false);

            ResponseBuffer buffer;
            buffer.streamEnabled = !noStream;
            buffer.rawMode = rawMode;
            buffer.firstTokenFlag = &firstTokenReceived;
            buffer.logger = &logger_;

            std::thread spinnerThread;
            if (!rawMode) {
                spinnerThread = std::thread(spinnerLoop, std::ref(spinnerStop), std::ref(firstTokenReceived));
            }

            curl_easy_reset(curl);
            curl_easy_setopt(curl, CURLOPT_URL, "https://api.openai.com/v1/chat/completions");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, transferAbortCallback);
            curl_easy_setopt(curl, CURLOPT_XFERINFODATA, nullptr);
            curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, CONNECT_TIMEOUT);
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, REQUEST_TIMEOUT);

            logger_.log(LogLevel::Info, "Sending request to API...");
            setRequestInFlight(true);
            CURLcode res = curl_easy_perform(curl);
            setRequestInFlight(false);
            long httpCode = 0;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

            spinnerStop.store(true);
            if (spinnerThread.joinable()) spinnerThread.join();

            if (res != CURLE_OK) {
                if (res == CURLE_ABORTED_BY_CALLBACK && shutdownRequested()) {
                    logger_.log(LogLevel::Warn, "Request interrupted by signal");
                    printInterruptNotice();
                    result.clear();
                    break;
                }

                logger_.log(LogLevel::Error, "curl_easy_perform() failed: %s", curl_easy_strerror(res));
                bool shouldRetry = (res == CURLE_OPERATION_TIMEDOUT) && attempt <= MAX_RETRIES;
                if (shouldRetry) {
                    std::cout << "Request timed out, retrying (" << attempt << "/" << MAX_RETRIES + 1 << ")...\n";
                    continue;
                } else {
                    std::fprintf(stderr, "Request failed: %s\n", curl_easy_strerror(res));
                }
            } else {
                logger_.log(LogLevel::Info, "Request completed successfully");
                logger_.log(LogLevel::Debug, "Response size: %zu bytes", buffer.data.size());
                if (httpCode >= 400) {
                    printApiError(httpCode, buffer.data);
                } else if (noStream) {
                    buffer.responseContent = extractAndPrintContent(buffer.data);
                } else {
                    if (!buffer.sawStreamData && !buffer.data.empty()) {
                        buffer.responseContent = extractAndPrintContent(buffer.data);
                        if (buffer.responseContent.empty()) {
                            printApiError(httpCode, buffer.data);
                        }
                    }
                    if (!rawMode) std::cout << std::endl;
                }
            }

            result = buffer.responseContent;
            break;
        }

        curl_slist_free_all(headers);
        cJSON_Delete(root);
        std::free(jsonStr);
        return result;
    }

private:
    static int transferAbortCallback(void *, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
        return shutdownRequested() ? 1 : 0;
    }

    static size_t writeCallback(void *contents, size_t size, size_t nmemb, void *userp) {
        size_t realSize = size * nmemb;
        auto *buffer = static_cast<ResponseBuffer *>(userp);
        if (buffer->firstTokenFlag) {
            buffer->firstTokenFlag->store(true);
        }
        buffer->data.append(static_cast<char *>(contents), realSize);

        if (buffer->streamEnabled) {
            while (true) {
                size_t sepPos = buffer->data.find("\n\n");
                if (sepPos == std::string::npos) break;

                std::string line = buffer->data.substr(0, sepPos);
                buffer->data.erase(0, sepPos + 2);

                if (line.rfind("data: ", 0) == 0 && line.compare(6, std::string::npos, "[DONE]") != 0) {
                    std::string jsonPayload = line.substr(6);
                    cJSON *json = cJSON_Parse(jsonPayload.c_str());
                    if (json) {
                        cJSON *choices = cJSON_GetObjectItem(json, "choices");
                        if (choices && cJSON_IsArray(choices) && cJSON_GetArraySize(choices) > 0) {
                            cJSON *choice = cJSON_GetArrayItem(choices, 0);
                            cJSON *delta = cJSON_GetObjectItem(choice, "delta");
                            if (delta) {
                                cJSON *content = cJSON_GetObjectItem(delta, "content");
                                if (content && cJSON_IsString(content) && content->valuestring) {
                                    buffer->responseContent += content->valuestring;
                                    if (!buffer->sawStreamData && !buffer->rawMode) {
                                        std::cout << "\n";
                                    }
                                    std::cout << content->valuestring;
                                    std::cout.flush();
                                    buffer->sawStreamData = true;
                                    if (buffer->firstTokenFlag) {
                                        buffer->firstTokenFlag->store(true);
                                    }
                                }
                            }
                        }
                        cJSON_Delete(json);
                    }
                }
            }
        }

        if (buffer->logger) {
            buffer->logger->log(LogLevel::Debug, "Received %zu bytes from API", realSize);
        }
        return realSize;
    }

    bool fetchModelsList(CURL *curl) {
        logger_.log(LogLevel::Info, "Fetching available models from OpenAI API");

        ResponseBuffer buffer;
        buffer.streamEnabled = false;
        buffer.sawStreamData = false;
        buffer.logger = &logger_;

        curl_easy_reset(curl);

        struct curl_slist *headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        std::string authHeader = "Authorization: Bearer " + settings_.apiKey;
        headers = curl_slist_append(headers, authHeader.c_str());

        curl_easy_setopt(curl, CURLOPT_URL, "https://api.openai.com/v1/models");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, transferAbortCallback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, nullptr);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, CONNECT_TIMEOUT);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, REQUEST_TIMEOUT);

        setRequestInFlight(true);
        CURLcode res = curl_easy_perform(curl);
        setRequestInFlight(false);
        if (res != CURLE_OK) {
            if (res == CURLE_ABORTED_BY_CALLBACK && shutdownRequested()) {
                logger_.log(LogLevel::Warn, "Model fetch interrupted by signal");
                curl_slist_free_all(headers);
                return false;
            }
            logger_.log(LogLevel::Error, "Failed to fetch models: %s", curl_easy_strerror(res));
            curl_slist_free_all(headers);
            return false;
        }

        long httpCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
        if (httpCode != 200) {
            logger_.log(LogLevel::Error, "API returned HTTP %ld when fetching models", httpCode);
            curl_slist_free_all(headers);
            return false;
        }

        cJSON *root = cJSON_Parse(buffer.data.c_str());
        if (!root) {
            logger_.log(LogLevel::Error, "Failed to parse API response: %s", cJSON_GetErrorPtr());
            curl_slist_free_all(headers);
            return false;
        }

        cJSON *data = cJSON_GetObjectItem(root, "data");
        if (!data || !cJSON_IsArray(data)) {
            logger_.log(LogLevel::Error, "Invalid response format: 'data' array not found");
            cJSON_Delete(root);
            curl_slist_free_all(headers);
            return false;
        }

        cache_.data().models.clear();
        cache_.data().lastUpdated = time(nullptr);

        int modelCount = cJSON_GetArraySize(data);
        for (int i = 0; i < modelCount; ++i) {
            cJSON *model = cJSON_GetArrayItem(data, i);
            if (!model || !cJSON_IsObject(model)) continue;

            cJSON *id = cJSON_GetObjectItem(model, "id");
            cJSON *created = cJSON_GetObjectItem(model, "created");
            if (!id || !cJSON_IsString(id)) continue;

            ModelInfo info;
            info.id = id->valuestring;
            info.created = created && cJSON_IsNumber(created) ? static_cast<time_t>(created->valuedouble) : time(nullptr);
            cache_.data().models.push_back(info);
        }

        cJSON_Delete(root);
        curl_slist_free_all(headers);

        logger_.log(LogLevel::Info, "Fetched %zu models from API", cache_.data().models.size());
        if (!cache_.data().models.empty()) {
            cache_.save(logger_);
        }

        return !cache_.data().models.empty();
    }

    void suggestSimilarModel(const std::string &invalidModel) {
        if (cache_.data().models.empty()) return;

        int minDistance = std::numeric_limits<int>::max();
        std::string closestModel;

        for (const auto &info : cache_.data().models) {
            int distance = levenshteinDistance(invalidModel, info.id);
            if (distance < minDistance) {
                minDistance = distance;
                closestModel = info.id;
            }
        }

        if (minDistance <= 5) {
            std::printf("Model '%s' not found. Did you mean '%s'?\n", invalidModel.c_str(), closestModel.c_str());
            logger_.log(LogLevel::Info, "Suggested alternative model: %s (distance: %d)", closestModel.c_str(), minDistance);
        } else {
            std::printf("Model '%s' not found. Available models include: gpt-5.5, gpt-5.4, gpt-5.4-mini\n", invalidModel.c_str());
        }
    }

    static std::string extractAndPrintContent(const std::string &body, bool print = true) {
        cJSON *json = cJSON_Parse(body.c_str());
        if (!json) {
            return {};
        }

        std::string result;
        cJSON *choices = cJSON_GetObjectItem(json, "choices");
        if (choices && cJSON_IsArray(choices) && cJSON_GetArraySize(choices) > 0) {
            cJSON *choice = cJSON_GetArrayItem(choices, 0);
            cJSON *message = cJSON_GetObjectItem(choice, "message");
            if (message) {
                cJSON *content = cJSON_GetObjectItem(message, "content");
                if (content && cJSON_IsString(content) && content->valuestring) {
                    result = content->valuestring;
                    if (print) {
                        std::cout << result << "\n";
                    }
                }
            }
        }
        cJSON_Delete(json);
        return result;
    }

    static void printApiError(long httpCode, const std::string &body) {
        cJSON *err = cJSON_Parse(body.c_str());
        if (err) {
            cJSON *errorObj = cJSON_GetObjectItem(err, "error");
            cJSON *msg = errorObj ? cJSON_GetObjectItem(errorObj, "message") : nullptr;
            if (msg && cJSON_IsString(msg) && msg->valuestring) {
                std::fprintf(stderr, "API error (HTTP %ld): %s\n", httpCode, msg->valuestring);
            } else {
                std::fprintf(stderr, "API error (HTTP %ld).\n", httpCode);
            }
            cJSON_Delete(err);
        } else {
            std::fprintf(stderr, "API error (HTTP %ld).\n", httpCode);
        }
    }

public:
    static int countTokens(const std::vector<Message> &messages, const std::string &model) {
        (void)model;
        int tokens = 3;
        for (const auto &message : messages) {
            tokens += countMessageTokens(message);
        }
        return tokens;
    }

private:
    static int countMessageTokens(const Message &message) {
        int tokens = 3;
        tokens += static_cast<int>(message.content.size()) / 4;
        if (!message.role.empty()) tokens += 1;
        return tokens;
    }

    bool trimMessagesToTokenLimit(std::vector<Message> &messages, int tokenLimit) {
        int tokenCount = countTokens(messages, settings_.model) + 100;
        size_t droppedCount = 0;
        while (tokenCount > tokenLimit) {
            if (messages.size() <= 2) {
                logger_.log(LogLevel::Error, "Prompt exceeds token limit even after dropping history");
                std::fprintf(stderr, "Error: prompt exceeds token limit. Shorten the input or increase --tokenLimit.\n");
                return false;
            }

            size_t eraseCount = 1;
            if (messages.size() > 3 &&
                messages[1].role == "user" &&
                messages[2].role == "assistant") {
                eraseCount = 2;
            }

            for (size_t i = 1; i < 1 + eraseCount; ++i) {
                tokenCount -= countMessageTokens(messages[i]);
            }
            messages.erase(messages.begin() + 1, messages.begin() + 1 + eraseCount);
            droppedCount += eraseCount;
        }

        if (droppedCount > 0) {
            std::fprintf(stderr, "Warning: dropped %zu message(s) to fit within token limit.\n", droppedCount);
            logger_.log(LogLevel::Debug, "Dropped %zu messages (token counting is approximate)", droppedCount);
        }

        return true;
    }

    static int levenshteinDistance(const std::string &s1, const std::string &s2) {
        size_t len1 = s1.size();
        size_t len2 = s2.size();
        std::vector<std::vector<int>> matrix(len1 + 1, std::vector<int>(len2 + 1));

        for (size_t i = 0; i <= len1; ++i) matrix[i][0] = static_cast<int>(i);
        for (size_t j = 0; j <= len2; ++j) matrix[0][j] = static_cast<int>(j);

        for (size_t i = 1; i <= len1; ++i) {
            for (size_t j = 1; j <= len2; ++j) {
                int cost = s1[i - 1] == s2[j - 1] ? 0 : 1;
                int deletion = matrix[i - 1][j] + 1;
                int insertion = matrix[i][j - 1] + 1;
                int substitution = matrix[i - 1][j - 1] + cost;
                matrix[i][j] = std::min({deletion, insertion, substitution});
            }
        }
        return matrix[len1][len2];
    }

    static void spinnerLoop(std::atomic_bool &stopFlag, std::atomic_bool &firstTokenFlag) {
        const char frames[] = {'|', '/', '-', '\\'};
        size_t idx = 0;
        std::cout << "thinking... " << std::flush;
        while (!stopFlag.load()) {
            if (shutdownRequested()) {
                std::cout << std::endl << std::flush;
                return;
            }
            if (firstTokenFlag.load()) {
                std::cout << "\n" << std::flush; // newline to separate spinner from response output
                return;
            }
            std::cout << "\rthinking... " << frames[idx % 4] << std::flush;
            idx++;
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
        std::cout << std::endl << std::flush;
    }

    Logger &logger_;
    Settings &settings_;
    ModelsCacheManager &cache_;
};

struct ParseOutcome {
    bool continueMode{false};
    bool noStream{false};
    bool rawMode{false};
    bool contextLast{false};
    bool tokenCountOnly{false};
    double temperature{1.0};
    std::string inputText;
    bool shouldExit{false};
    int exitCode{0};
};

std::string maskApiKey(const std::string &key) {
    if (key.size() <= 8) return "****";
    return key.substr(0, 5) + "..." + key.substr(key.size() - 4);
}

class Application {
public:
    Application() : logger_() {
        curl_global_init(CURL_GLOBAL_ALL);
    }

    ~Application() {
        curl_global_cleanup();
    }

    int run(int argc, char *argv[]) {
        resetSignalState();

        // Pre-pass for logging flags
        preParseLogging(argc, argv);
        logger_.configure(settings_.logLevel, settings_.debugMode, settings_.logToFile, settings_.logFilePath);

        loadEnvironment();

        ParseOutcome outcome = parseArguments(argc, argv);
        if (outcome.shouldExit) {
            return outcome.exitCode;
        }

        if (outcome.contextLast && outcome.continueMode) {
            std::fprintf(stderr, "Error: --context last and --continue are mutually exclusive.\n");
            return 1;
        }

        if (!curlHandle_.valid()) {
            logger_.log(LogLevel::Error, "Failed to initialize curl");
            return 1;
        }

        if (!installSignalHandlers()) {
            logger_.log(LogLevel::Error, "Failed to install signal handlers");
            std::fprintf(stderr, "Error: failed to install signal handlers.\n");
            return 1;
        }

        // Stdin pipe support
        if (!isatty(STDIN_FILENO)) {
            std::string stdinData;
            std::ostringstream ss;
            ss << std::cin.rdbuf();
            stdinData = ss.str();
            // Trim trailing newline
            while (!stdinData.empty() && (stdinData.back() == '\n' || stdinData.back() == '\r')) {
                stdinData.pop_back();
            }
            if (!stdinData.empty()) {
                if (outcome.inputText.empty()) {
                    outcome.inputText = stdinData;
                } else {
                    outcome.inputText = stdinData + "\n\n" + outcome.inputText;
                }
            }
            if (!outcome.rawMode) {
                outcome.rawMode = true; // auto-enable raw mode for piped input
            }
        }

        if (shutdownRequested()) {
            printInterruptNotice();
            return interruptedExitCode();
        }

        if (outcome.tokenCountOnly) {
            if (outcome.inputText.empty()) {
                std::fprintf(stderr, "Error: --tokenCount requires input text or piped stdin.\n");
                return 1;
            }

            std::vector<Message> messages;
            messages.push_back({"user", outcome.inputText});
            int tokenCount = ApiClient::countTokens(messages, settings_.model);
            std::cout << tokenCount << "\n";
            logger_.log(LogLevel::Info, "Token count: %d", tokenCount);
            return 0;
        }

        if (!outcome.continueMode && outcome.inputText.empty()) {
            logger_.log(LogLevel::Info, "No input text provided, showing usage hint");
            printUsageHint();
            return 0;
        }

        // Validate model
        if (!client().validateModel(curlHandle_.get(), settings_.model)) {
            if (shutdownRequested()) {
                printInterruptNotice();
                return interruptedExitCode();
            }
            logger_.log(LogLevel::Error, "Invalid model: %s", settings_.model.c_str());
            std::printf("Error: '%s' is not a valid model.\n", settings_.model.c_str());
            return 1;
        }

        if (outcome.continueMode) {
            runConversation(outcome);
        } else {
            runSingle(outcome);
        }

        if (shutdownRequested()) {
            return interruptedExitCode();
        }

        logger_.log(LogLevel::Info, "Exiting normally");
        return 0;
    }

private:
    void loadEnvironment() {
        if (const char *envApiKey = std::getenv("OPENAI_API_KEY")) {
            settings_.apiKey = envApiKey;
            logger_.log(LogLevel::Debug, "Loaded API key from environment");
        }
        if (const char *envModel = std::getenv("ASK_GLOBAL_MODEL")) {
            settings_.model = envModel;
            logger_.log(LogLevel::Debug, "Loaded model from environment: %s", settings_.model.c_str());
        }

        if ((settings_.apiKey.empty() || settings_.model.empty()) && access(".env", F_OK) == 0) {
            loadDotenv(".env");
        }

        if (settings_.model.empty()) {
            settings_.model = DEFAULT_MODEL;
            logger_.log(LogLevel::Info, "Using default model: %s", DEFAULT_MODEL);
        }
    }

    void loadDotenv(const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            logger_.log(LogLevel::Warn, "Could not open .env file: %s", filename.c_str());
            return;
        }

        logger_.log(LogLevel::Info, "Loading environment from %s", filename.c_str());
        std::string line;
        while (std::getline(file, line)) {
            auto equalsPos = line.find('=');
            if (equalsPos == std::string::npos) continue;

            std::string key = line.substr(0, equalsPos);
            std::string value = line.substr(equalsPos + 1);

            if (key == "OPENAI_API_KEY" && settings_.apiKey.empty()) {
                settings_.apiKey = value;
                logger_.log(LogLevel::Debug, "Loaded API key from .env");
            } else if (key == "ASK_GLOBAL_MODEL" && settings_.model.empty()) {
                settings_.model = value;
                logger_.log(LogLevel::Debug, "Loaded model from .env: %s", settings_.model.c_str());
            }
        }
    }

    void saveEnvFile() {
        std::string content = "OPENAI_API_KEY=" + settings_.apiKey + "\n"
            + "ASK_GLOBAL_MODEL=" + settings_.model + "\n";
        if (!FileUtil::writeSecureFile(".env", content, logger_)) {
            logger_.log(LogLevel::Error, "Failed to save .env file");
            return;
        }
        logger_.log(LogLevel::Info, "Saved API key and model settings to .env file");
    }

    static bool parseLogLevel(const char *level, LogLevel &outLevel) {
        if (std::strcmp(level, "none") == 0) outLevel = LogLevel::None;
        else if (std::strcmp(level, "error") == 0) outLevel = LogLevel::Error;
        else if (std::strcmp(level, "warn") == 0) outLevel = LogLevel::Warn;
        else if (std::strcmp(level, "info") == 0) outLevel = LogLevel::Info;
        else if (std::strcmp(level, "debug") == 0) outLevel = LogLevel::Debug;
        else return false;

        return true;
    }

    void preParseLogging(int argc, char *argv[]) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--log") == 0) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Error: --log requires a value\n");
                    std::exit(1);
                }
                const char *level = argv[++i];
                if (!parseLogLevel(level, settings_.logLevel)) {
                    std::fprintf(stderr, "Error: invalid log level '%s' (expected: none, error, warn, info, debug)\n", level);
                    std::exit(1);
                }
            } else if (std::strcmp(argv[i], "--logfile") == 0) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Error: --logfile requires a file path\n");
                    std::exit(1);
                }
                settings_.logFilePath = argv[++i];
                settings_.logToFile = true;
            } else if (std::strcmp(argv[i], "--debug") == 0) {
                settings_.debugMode = true;
                settings_.logLevel = LogLevel::Debug;
            }
        }
    }

    ParseOutcome parseArguments(int argc, char *argv[]) {
        ParseOutcome outcome;
        bool showVersion = false;
        bool showTokenCount = false;
        bool setApiKey = false;
        bool setModel = false;
        std::string newApiKey;
        std::string newModel;
        bool showHelp = false;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                showHelp = true;
            }
        }

        if (showHelp) {
            printHelp();
            outcome.shouldExit = true;
            outcome.exitCode = 0;
            return outcome;
        }

        logger_.log(LogLevel::Debug, "Parsing %d command line arguments", argc - 1);

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--version") == 0 || std::strcmp(argv[i], "-v") == 0) {
                showVersion = true;
                logger_.log(LogLevel::Debug, "Flag: show version");
            } else if (std::strcmp(argv[i], "--tokenCount") == 0) {
                showTokenCount = true;
                logger_.log(LogLevel::Debug, "Flag: show token count");
            } else if (std::strcmp(argv[i], "--debug") == 0) {
                settings_.debugMode = true;
                settings_.logLevel = LogLevel::Debug;
                logger_.log(LogLevel::Debug, "Flag: debug mode enabled");
            } else if (std::strcmp(argv[i], "--log") == 0) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Error: --log requires a value\n");
                    outcome.shouldExit = true;
                    outcome.exitCode = 1;
                    return outcome;
                }

                const char *level = argv[++i];
                if (!parseLogLevel(level, settings_.logLevel)) {
                    std::fprintf(stderr, "Error: invalid log level '%s' (expected: none, error, warn, info, debug)\n", level);
                    outcome.shouldExit = true;
                    outcome.exitCode = 1;
                    return outcome;
                }
                logger_.log(LogLevel::Debug, "Set log level from command line");
            } else if (std::strcmp(argv[i], "--logfile") == 0) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Error: --logfile requires a file path\n");
                    outcome.shouldExit = true;
                    outcome.exitCode = 1;
                    return outcome;
                }

                settings_.logFilePath = argv[++i];
                settings_.logToFile = true;
                logger_.log(LogLevel::Debug, "Set logfile path to %s", settings_.logFilePath.c_str());
            } else if (std::strcmp(argv[i], "--continue") == 0 || std::strcmp(argv[i], "-c") == 0) {
                outcome.continueMode = true;
                logger_.log(LogLevel::Debug, "Flag: continue mode enabled");
            } else if (std::strcmp(argv[i], "--no-stream") == 0) {
                outcome.noStream = true;
                logger_.log(LogLevel::Debug, "Flag: streaming disabled");
            } else if (std::strcmp(argv[i], "--temperature") == 0 || std::strcmp(argv[i], "-T") == 0) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Error: --temperature requires a value\n");
                    outcome.shouldExit = true;
                    outcome.exitCode = 1;
                    return outcome;
                }
                {
                    const char *tempArg = argv[++i];
                    char *endptr = nullptr;
                    outcome.temperature = std::strtod(tempArg, &endptr);
                    if (endptr == tempArg || *endptr != '\0') {
                        std::fprintf(stderr, "Error: invalid temperature value '%s'\n", tempArg);
                        outcome.shouldExit = true;
                        outcome.exitCode = 1;
                        return outcome;
                    }
                    if (outcome.temperature < 0.0 || outcome.temperature > 2.0) {
                        std::fprintf(stderr, "Error: temperature must be between 0.0 and 2.0 (got %.2f)\n", outcome.temperature);
                        outcome.shouldExit = true;
                        outcome.exitCode = 1;
                        return outcome;
                    }
                    logger_.log(LogLevel::Debug, "Set temperature to %.2f", outcome.temperature);
                }
            } else if (std::strcmp(argv[i], "--system") == 0 || std::strcmp(argv[i], "-s") == 0) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Error: --system requires a prompt string\n");
                    outcome.shouldExit = true;
                    outcome.exitCode = 1;
                    return outcome;
                }
                settings_.systemPrompt = argv[++i];
                logger_.log(LogLevel::Debug, "Set system prompt");
            } else if (std::strcmp(argv[i], "--raw") == 0) {
                outcome.rawMode = true;
                logger_.log(LogLevel::Debug, "Flag: raw mode enabled");
            } else if (std::strcmp(argv[i], "--context") == 0) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Error: --context requires a value (supported: last)\n");
                    outcome.shouldExit = true;
                    outcome.exitCode = 1;
                    return outcome;
                }
                std::string val = argv[++i];
                if (val == "last") {
                    outcome.contextLast = true;
                    logger_.log(LogLevel::Debug, "Flag: context last enabled");
                } else {
                    std::fprintf(stderr, "Error: unknown --context value '%s' (supported: last)\n", val.c_str());
                    outcome.shouldExit = true;
                    outcome.exitCode = 1;
                    return outcome;
                }
            } else if (std::strcmp(argv[i], "--tokenLimit") == 0 || std::strcmp(argv[i], "-l") == 0) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Error: --tokenLimit requires a numeric value\n");
                    outcome.shouldExit = true;
                    outcome.exitCode = 1;
                    return outcome;
                }
                {
                    const char *limitArg = argv[++i];
                    char *endptr = nullptr;
                    long val = std::strtol(limitArg, &endptr, 10);
                    if (endptr == limitArg || *endptr != '\0') {
                        std::fprintf(stderr, "Error: invalid token limit value '%s'\n", limitArg);
                        outcome.shouldExit = true;
                        outcome.exitCode = 1;
                        return outcome;
                    }
                    if (val <= 0) {
                        std::fprintf(stderr, "Error: token limit must be positive\n");
                        outcome.shouldExit = true;
                        outcome.exitCode = 1;
                        return outcome;
                    }
                    settings_.tokenLimit = static_cast<int>(val);
                    logger_.log(LogLevel::Debug, "Set token limit to %d", settings_.tokenLimit);
                }
            } else if (std::strcmp(argv[i], "--fileLimit") == 0 || std::strcmp(argv[i], "-F") == 0) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Error: --fileLimit requires a numeric value\n");
                    outcome.shouldExit = true;
                    outcome.exitCode = 1;
                    return outcome;
                }
                {
                    const char *limitArg = argv[++i];
                    char *endptr = nullptr;
                    long val = std::strtol(limitArg, &endptr, 10);
                    if (endptr == limitArg || *endptr != '\0') {
                        std::fprintf(stderr, "Error: invalid file size limit value '%s'\n", limitArg);
                        outcome.shouldExit = true;
                        outcome.exitCode = 1;
                        return outcome;
                    }
                    if (val <= 0) {
                        std::fprintf(stderr, "Error: file size limit must be positive\n");
                        outcome.shouldExit = true;
                        outcome.exitCode = 1;
                        return outcome;
                    }
                    settings_.fileSizeLimit = static_cast<int>(val);
                    logger_.log(LogLevel::Debug, "Set file size limit to %d", settings_.fileSizeLimit);
                }
            } else if (std::strcmp(argv[i], "--token") == 0 || std::strcmp(argv[i], "-t") == 0) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Error: --token requires an API key value\n");
                    outcome.shouldExit = true;
                    outcome.exitCode = 1;
                    return outcome;
                }
                settings_.apiKey = argv[++i];
                std::memset(argv[i], '*', std::strlen(argv[i]));
                std::fprintf(stderr, "Warning: passing API key via --token exposes it in the process list.\n"
                    "  Consider using OPENAI_API_KEY environment variable or --setAPIKey instead.\n");
                logger_.log(LogLevel::Debug, "Set API key from command line");
            } else if (std::strcmp(argv[i], "--model") == 0 || std::strcmp(argv[i], "-m") == 0) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Error: --model requires a model name\n");
                    outcome.shouldExit = true;
                    outcome.exitCode = 1;
                    return outcome;
                }
                settings_.model = argv[++i];
                logger_.log(LogLevel::Debug, "Set model to %s", settings_.model.c_str());
            } else if (std::strcmp(argv[i], "--setAPIKey") == 0) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Error: --setAPIKey requires an API key value\n");
                    outcome.shouldExit = true;
                    outcome.exitCode = 1;
                    return outcome;
                }
                newApiKey = argv[++i];
                setApiKey = true;
                logger_.log(LogLevel::Debug, "Will save new API key");
            } else if (std::strcmp(argv[i], "--setModel") == 0) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "Error: --setModel requires a model name\n");
                    outcome.shouldExit = true;
                    outcome.exitCode = 1;
                    return outcome;
                }
                newModel = argv[++i];
                setModel = true;
                logger_.log(LogLevel::Debug, "Will save new model: %s", newModel.c_str());
            } else {
                std::ostringstream builder;
                for (int j = i; j < argc; ++j) {
                    if (j > i) builder << ' ';
                    builder << argv[j];
                }
                outcome.inputText = builder.str();
                logger_.log(LogLevel::Debug, "Input text: \"%s\" (%zu chars)", outcome.inputText.c_str(), outcome.inputText.size());
                break;
            }
        }

        if (setApiKey || setModel) {
            if (setModel) settings_.model = newModel;
            if (setApiKey) settings_.apiKey = newApiKey;
            saveEnvFile();
            logger_.log(LogLevel::Info, "Updated configuration saved to .env file");
            if (setApiKey) std::cout << "API key saved to .env\n";
            if (setModel) std::cout << "Model set to " << settings_.model << " and saved to .env\n";
            outcome.shouldExit = true;
            outcome.exitCode = 0;
            return outcome;
        }

        if (showVersion || settings_.debugMode) {
            std::cout << "OpenAI Chatbot\n";
            std::cout << "Model: " << settings_.model << "\n";
            std::cout << "API Key: " << maskApiKey(settings_.apiKey) << "\n";
            std::cout << "Token Limit: " << settings_.tokenLimit << "\n";

            const char *levelStr = "UNKNOWN";
            switch (settings_.logLevel) {
                case LogLevel::None: levelStr = "NONE"; break;
                case LogLevel::Error: levelStr = "ERROR"; break;
                case LogLevel::Warn: levelStr = "WARN"; break;
                case LogLevel::Info: levelStr = "INFO"; break;
                case LogLevel::Debug: levelStr = "DEBUG"; break;
            }
            std::cout << "Log Level: " << levelStr << "\n";
            if (settings_.logToFile) std::cout << "Logging to file: " << settings_.logFilePath << "\n";
            if (!settings_.debugMode) {
                outcome.shouldExit = true;
                outcome.exitCode = 0;
                return outcome;
            }
        }

        outcome.tokenCountOnly = showTokenCount;

        if (settings_.apiKey.empty() && !outcome.tokenCountOnly) {
            std::fprintf(stderr, "API Key not found. Set OPENAI_API_KEY or use --setAPIKey.\n");
            outcome.shouldExit = true;
            outcome.exitCode = 1;
        }

        return outcome;
    }

    void printHelp() {
        std::cout << "OpenAI CLI Chatbot - Command Line Interface for OpenAI API\n\n";
        std::cout << "Usage: ask [OPTIONS] [query]\n\n";
        std::cout << "Options:\n";
        std::cout << "  -h, --help             Display this help message\n";
        std::cout << "  -v, --version          Display version information\n";
        std::cout << "  -c, --continue         Enable conversation mode (supports multiple exchanges)\n";
        std::cout << "      --context last     Prepend previous Q&A for lightweight follow-ups\n";
        std::cout << "      --no-stream        Disable streaming output (wait for complete response)\n";
        std::cout << "      --raw              Raw output mode (no spinner, minimal formatting)\n";
        std::cout << "  -s, --system PROMPT    Set custom system prompt\n";
        std::cout << "  -t, --token TOKEN      Set OpenAI API token\n";
        std::cout << "  -m, --model MODEL      Set model to use (default: " << DEFAULT_MODEL << ")\n";
        std::cout << "  -T, --temperature VAL  Set temperature (0.0-2.0, default: 1.0)\n";
        std::cout << "  -l, --tokenLimit NUM   Set token limit (default: " << DEFAULT_TOKEN_LIMIT << ")\n";
        std::cout << "  -F, --fileLimit NUM    Set @file size limit in bytes (default: 10000)\n";
        std::cout << "      --tokenCount       Count tokens in input text and exit\n";
        std::cout << "      --debug            Enable debug mode\n";
        std::cout << "      --log LEVEL        Set log level (none, error, warn, info, debug)\n";
        std::cout << "      --logfile FILE     Log output to specified file\n";
        std::cout << "      --setAPIKey KEY    Save API key to .env file\n";
        std::cout << "      --setModel MODEL   Save model to .env file\n\n";
        std::cout << "Examples:\n";
        std::cout << "  ask \"What is the capital of France?\"\n";
        std::cout << "  ask -c \"Let's have a conversation\"\n";
        std::cout << "  ask -s \"You are a pirate\" \"Hello there\"\n";
        std::cout << "  echo \"some code\" | ask \"explain this\"\n";
        std::cout << "  ask --model gpt-4 --temperature 0.8 \"Write a poem about AI\"\n";
    }

    void printUsageHint() {
        std::cout << "No input provided. Usage examples:\n";
        std::cout << "  ask \"What is the capital of France?\"\n";
        std::cout << "  ask -c \"Let's have a conversation\"\n";
        std::cout << "  ask --help\n";
    }

    void runConversation(const ParseOutcome &outcome) {
        logger_.log(LogLevel::Info, "Starting interactive mode");

        if (!isatty(STDIN_FILENO)) {
            std::fprintf(stderr, "Error: conversation mode is not supported with piped input.\n");
            return;
        }

        std::vector<Message> messages;
        messages.push_back({"system", effectiveSystemPrompt()});

        if (!outcome.inputText.empty()) {
            bool interactiveTty = isatty(STDIN_FILENO);
            std::string processedInput = FileUtil::processFileReferences(outcome.inputText, logger_, settings_.fileSizeLimit, interactiveTty);
            if (shutdownRequested()) {
                printInterruptNotice();
                return;
            }
            std::vector<Message> requestMessages = messages;
            requestMessages.push_back({"user", processedInput.empty() ? outcome.inputText : processedInput});
            std::string response = client().sendChat(curlHandle_.get(), requestMessages, outcome.temperature, outcome.noStream, settings_.tokenLimit, outcome.rawMode);
            if (!response.empty()) {
                requestMessages.push_back({"assistant", response});
                messages = std::move(requestMessages);
            }
            if (shutdownRequested()) {
                return;
            }
        } else {
            std::cout << "Starting conversation mode...\n";
        }

        std::cout << "Type 'exit' to quit, 'status' for conversation info, or 'help' for commands.\n";

        std::string userInput;
        while (true) {
            if (shutdownRequested()) {
                printInterruptNotice();
                break;
            }

            std::cout << "> ";
            std::cout.flush();

            if (!std::getline(std::cin, userInput)) {
                if (shutdownRequested()) {
                    printInterruptNotice();
                } else {
                    logger_.log(LogLevel::Warn, "Failed to read user input, exiting");
                }
                break;
            }

            if (userInput == "exit") {
                logger_.log(LogLevel::Info, "User requested exit");
                break;
            } else if (userInput == "status") {
                int approxTokens = ApiClient::countTokens(messages, settings_.model);
                std::cout << "Conversation Status:\n";
                std::cout << "  Messages: " << messages.size() << "\n";
                std::cout << "  Approximate tokens: " << approxTokens << " / " << settings_.tokenLimit << "\n";
                std::cout << "  Model: " << settings_.model << "\n";
                std::cout << "  Temperature: " << outcome.temperature << "\n";
                std::cout << "  Streaming: " << (outcome.noStream ? "disabled" : "enabled") << "\n";
                continue;
            } else if (userInput == "help") {
                std::cout << "Conversation Mode Commands:\n";
                std::cout << "  exit    - Exit conversation mode\n";
                std::cout << "  status  - Show conversation information\n";
                std::cout << "  help    - Show this help message\n";
                std::cout << "  Any other text will be sent to the AI assistant.\n";
                continue;
            }

            std::string processedInput = FileUtil::processFileReferences(userInput, logger_, settings_.fileSizeLimit, true);
            if (shutdownRequested()) {
                printInterruptNotice();
                break;
            }
            std::vector<Message> requestMessages = messages;
            requestMessages.push_back({"user", processedInput.empty() ? userInput : processedInput});
            std::string response = client().sendChat(curlHandle_.get(), requestMessages, outcome.temperature, outcome.noStream, settings_.tokenLimit, outcome.rawMode);
            if (!response.empty()) {
                requestMessages.push_back({"assistant", response});
                messages = std::move(requestMessages);
            }
            if (shutdownRequested()) {
                break;
            }
        }

        // Save last exchange for --context last
        auto lastExchange = findLastCompleteExchange(messages);
        if (lastExchange.has_value()) {
            contextManager_.save(*lastExchange, logger_);
        }
    }

    void runSingle(const ParseOutcome &outcome) {
        logger_.log(LogLevel::Info, "Single response mode");

        std::vector<Message> messages;
        messages.push_back({"system", effectiveSystemPrompt()});

        if (outcome.contextLast) {
            auto prev = contextManager_.load(logger_);
            if (prev.empty()) {
                std::fprintf(stderr, "Warning: no previous context found. Running without context.\n");
            } else {
                for (auto &msg : prev) {
                    if (msg.role != "system") {
                        messages.push_back(std::move(msg));
                    }
                }
            }
        }

        bool interactiveTty = isatty(STDIN_FILENO);
        std::string processedInput = FileUtil::processFileReferences(outcome.inputText, logger_, settings_.fileSizeLimit, interactiveTty);
        if (shutdownRequested()) {
            printInterruptNotice();
            return;
        }
        std::vector<Message> requestMessages = messages;
        requestMessages.push_back({"user", processedInput.empty() ? outcome.inputText : processedInput});
        std::string response = client().sendChat(curlHandle_.get(), requestMessages, outcome.temperature, outcome.noStream, settings_.tokenLimit, outcome.rawMode);

        if (!response.empty() && !shutdownRequested()) {
            std::vector<Message> toSave;
            toSave.push_back(requestMessages.front()); // system
            toSave.push_back(requestMessages.back());  // user
            toSave.push_back({"assistant", response});
            contextManager_.save(toSave, logger_);
        }
    }

    std::optional<std::vector<Message>> findLastCompleteExchange(const std::vector<Message> &messages) const {
        if (messages.size() < 3) {
            return std::nullopt;
        }

        for (size_t i = messages.size() - 1; i > 0; --i) {
            if (messages[i].role == "assistant" && messages[i - 1].role == "user") {
                return std::vector<Message>{messages.front(), messages[i - 1], messages[i]};
            }
        }

        return std::nullopt;
    }

    std::string effectiveSystemPrompt() const {
        std::string base = settings_.systemPrompt.empty()
            ? std::string(DEFAULT_SYSTEM_PROMPT)
            : settings_.systemPrompt;
        return base + " OS: " + detectOS() + ".";
    }

    ApiClient &client() {
        if (!apiClient_) {
            apiClient_.emplace(logger_, settings_, cacheManager_);
        }
        return *apiClient_;
    }

    Settings settings_{};
    Logger logger_;
    ModelsCacheManager cacheManager_{};
    ContextManager contextManager_{};
    CurlHandle curlHandle_{};
    std::optional<ApiClient> apiClient_{};
};

} // namespace

int main(int argc, char *argv[]) {
    Application app;
    return app.run(argc, argv);
}
