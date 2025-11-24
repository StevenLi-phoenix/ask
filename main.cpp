// C++ rewrite of the ask CLI
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <pwd.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <curl/curl.h>
#include <cjson/cJSON.h>

constexpr const char *DEFAULT_MODEL = "gpt-5-nano";
constexpr size_t MAX_BUFFER_SIZE = 8192;
constexpr int DEFAULT_TOKEN_LIMIT = 128000;
constexpr const char *MODELS_CACHE_FILE = "~/.cache/ask_models_cache.json";
constexpr time_t MODELS_CACHE_EXPIRY = 86400; // 24 hours

enum LogLevel {
    LOG_NONE = 0,
    LOG_ERROR = 1,
    LOG_WARN = 2,
    LOG_INFO = 3,
    LOG_DEBUG = 4
};

struct ResponseBuffer {
    std::string data;
    bool stream_enabled{false};
    bool saw_stream_data{false};
    std::atomic_bool *first_token_flag{nullptr};
};

struct Message {
    std::string role;
    std::string content;
};

struct ModelInfo {
    std::string id;
    time_t created{};
};

struct ModelsCache {
    std::vector<ModelInfo> models;
    time_t last_updated{};
};

// Globals
std::string global_api_key;
std::string global_model;
int token_limit = DEFAULT_TOKEN_LIMIT;
bool debug_mode = false;
LogLevel log_level = LOG_INFO;
FILE *log_file = nullptr;
bool log_to_file = false;
std::string log_file_path = "ask.log";
ModelsCache models_cache;

// Forward declarations
void init_logging();
void close_logging();
void log_message(LogLevel level, const char *format, ...);
void load_dotenv(const std::string &filename);
size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp);
int count_tokens_from_messages(const std::vector<Message> &messages, const std::string &model);
void ask(CURL *curl, std::vector<Message> &messages, double temperature, bool no_stream);
void parse_arguments(int argc, char *argv[], bool &continue_mode, bool &no_stream, double &temperature, std::string &input_text);
void save_env_file();
void print_help();
bool validate_model(CURL *curl, const std::string &model);
bool load_models_cache();
bool save_models_cache();
bool fetch_models_list(CURL *curl);
bool is_valid_model(const std::string &model);
void suggest_similar_model(const std::string &invalid_model);
std::string expand_home_path(const std::string &path);
bool is_plain_text_file(const std::string &filename);
std::optional<std::string> read_file_content(const std::string &filename);
std::string process_file_references(const std::string &input);
int levenshtein_distance(const std::string &s1, const std::string &s2);
void spinner_loop(std::atomic_bool &stop_flag, std::atomic_bool &first_token_flag);

void init_logging() {
    if (log_to_file) {
        log_file = std::fopen(log_file_path.c_str(), "a");
        if (!log_file) {
            std::fprintf(stderr, "Failed to open log file %s\n", log_file_path.c_str());
            log_to_file = false;
        } else {
            std::fprintf(stdout, "Logging to file: %s\n", log_file_path.c_str());
        }
    }
}

void close_logging() {
    if (log_file) {
        std::fclose(log_file);
        log_file = nullptr;
    }
}

void log_message(LogLevel level, const char *format, ...) {
    if (level > log_level) return;

    char level_str[10];
    switch (level) {
        case LOG_ERROR: std::strcpy(level_str, "ERROR"); break;
        case LOG_WARN: std::strcpy(level_str, "WARN"); break;
        case LOG_INFO: std::strcpy(level_str, "INFO"); break;
        case LOG_DEBUG: std::strcpy(level_str, "DEBUG"); break;
        default: std::strcpy(level_str, "NONE"); break;
    }

    std::time_t now = std::time(nullptr);
    std::tm *local_time = std::localtime(&now);
    char time_str[20];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", local_time);

    va_list args;
    va_start(args, format);

    char full_message[MAX_BUFFER_SIZE];
    std::vsnprintf(full_message, sizeof(full_message), format, args);

    if (level <= LOG_WARN || debug_mode) {
        std::fprintf(stdout, "[%s] %s: %s\n", time_str, level_str, full_message);
    }

    if (log_to_file && log_file) {
        std::fprintf(log_file, "[%s] %s: %s\n", time_str, level_str, full_message);
        std::fflush(log_file);
    }

    va_end(args);
}

void load_dotenv(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        log_message(LOG_WARN, "Could not open .env file: %s", filename.c_str());
        return;
    }

    log_message(LOG_INFO, "Loading environment from %s", filename.c_str());
    std::string line;
    while (std::getline(file, line)) {
        auto equals_pos = line.find('=');
        if (equals_pos == std::string::npos) continue;

        std::string key = line.substr(0, equals_pos);
        std::string value = line.substr(equals_pos + 1);

        if (key == "OPENAI_API_KEY" && global_api_key.empty()) {
            global_api_key = value;
            log_message(LOG_DEBUG, "Loaded API key from .env");
        } else if (key == "ASK_GLOBAL_MODEL" && global_model.empty()) {
            global_model = value;
            log_message(LOG_DEBUG, "Loaded model from .env: %s", global_model.c_str());
        }
    }
}

size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    auto *buffer = static_cast<ResponseBuffer *>(userp);
    if (buffer->first_token_flag) {
        buffer->first_token_flag->store(true);
    }
    buffer->data.append(static_cast<char *>(contents), realsize);

    if (buffer->stream_enabled) {
        // Process complete SSE lines separated by blank lines
        while (true) {
            size_t sep_pos = buffer->data.find("\n\n");
            if (sep_pos == std::string::npos) break;

            std::string line = buffer->data.substr(0, sep_pos);
            buffer->data.erase(0, sep_pos + 2);

            if (line.rfind("data: ", 0) == 0 && line.compare(6, std::string::npos, "[DONE]") != 0) {
                std::string json_payload = line.substr(6);
                cJSON *json = cJSON_Parse(json_payload.c_str());
                if (json) {
                    cJSON *choices = cJSON_GetObjectItem(json, "choices");
                    if (choices && cJSON_IsArray(choices) && cJSON_GetArraySize(choices) > 0) {
                        cJSON *choice = cJSON_GetArrayItem(choices, 0);
                        cJSON *delta = cJSON_GetObjectItem(choice, "delta");
                        if (delta) {
                            cJSON *content = cJSON_GetObjectItem(delta, "content");
                            if (content && cJSON_IsString(content) && content->valuestring) {
                                std::cout << content->valuestring;
                                std::cout.flush();
                                buffer->saw_stream_data = true;
                                if (buffer->first_token_flag) {
                                    buffer->first_token_flag->store(true);
                                }
                            }
                        }
                    }
                    cJSON_Delete(json);
                }
            }
        }
    }

    log_message(LOG_DEBUG, "Received %zu bytes from API", realsize);
    return realsize;
}

int count_tokens_from_messages(const std::vector<Message> &messages, const std::string &model) {
    (void)model;
    int tokens = 3;
    for (const auto &message : messages) {
        tokens += 3;
        tokens += static_cast<int>(message.content.size()) / 4;
        if (!message.role.empty()) tokens += 1;
    }
    return tokens;
}

void add_message(std::vector<Message> &messages, const std::string &role, const std::string &content) {
    messages.push_back({role, content});
    log_message(LOG_DEBUG, "Added message with role '%s' (length: %zu)", role.c_str(), content.size());
}

void save_env_file() {
    std::ofstream file(".env");
    if (!file.is_open()) {
        log_message(LOG_ERROR, "Failed to open .env file for writing");
        return;
    }
    file << "OPENAI_API_KEY=" << global_api_key << "\n";
    file << "ASK_GLOBAL_MODEL=" << global_model << "\n";
    log_message(LOG_INFO, "Saved API key and model settings to .env file");
}

void print_help() {
    std::cout << "OpenAI CLI Chatbot - Command Line Interface for OpenAI API\n\n";
    std::cout << "Usage: ask [OPTIONS] [query]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help             Display this help message\n";
    std::cout << "  -v, --version          Display version information\n";
    std::cout << "  -c, --continue         Enable conversation mode (supports multiple exchanges)\n";
    std::cout << "      --no-stream        Disable streaming output (wait for complete response)\n";
    std::cout << "  -t, --token TOKEN      Set OpenAI API token\n";
    std::cout << "  -m, --model MODEL      Set model to use (default: " << DEFAULT_MODEL << ")\n";
    std::cout << "  -T, --temperature VAL  Set temperature (0.0-1.0, default: 1.0)\n";
    std::cout << "  -l, --tokenLimit NUM   Set token limit (default: " << DEFAULT_TOKEN_LIMIT << ")\n";
    std::cout << "      --tokenCount       Count tokens in input text and exit\n";
    std::cout << "      --debug            Enable debug mode\n";
    std::cout << "      --log LEVEL        Set log level (none, error, warn, info, debug)\n";
    std::cout << "      --logfile FILE     Log output to specified file\n";
    std::cout << "      --setAPIKey KEY    Save API key to .env file\n";
    std::cout << "      --setModel MODEL   Save model to .env file\n\n";
    std::cout << "Examples:\n";
    std::cout << "  ask \"What is the capital of France?\"\n";
    std::cout << "  ask -c \"Let's have a conversation\"\n";
    std::cout << "  ask --model gpt-4 --temperature 0.8 \"Write a poem about AI\"\n";
}

void parse_arguments(int argc, char *argv[], bool &continue_mode, bool &no_stream, double &temperature, std::string &input_text) {
    continue_mode = false;
    no_stream = false;
    temperature = 1.0;
    input_text.clear();

    bool show_version = false;
    bool show_token_count = false;
    bool set_api_key = false;
    bool set_model = false;
    bool show_help = false;
    std::string new_api_key;
    std::string new_model;

    // First pass for logging flags
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--log") == 0 && i + 1 < argc) {
            const char *level = argv[++i];
            if (std::strcmp(level, "none") == 0) log_level = LOG_NONE;
            else if (std::strcmp(level, "error") == 0) log_level = LOG_ERROR;
            else if (std::strcmp(level, "warn") == 0) log_level = LOG_WARN;
            else if (std::strcmp(level, "info") == 0) log_level = LOG_INFO;
            else if (std::strcmp(level, "debug") == 0) log_level = LOG_DEBUG;
        } else if (std::strcmp(argv[i], "--logfile") == 0 && i + 1 < argc) {
            log_file_path = argv[++i];
            log_to_file = true;
        } else if (std::strcmp(argv[i], "--debug") == 0) {
            debug_mode = true;
            log_level = LOG_DEBUG;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        }
    }

    init_logging();

    if (show_help) {
        print_help();
        std::exit(0);
    }

    log_message(LOG_DEBUG, "Parsing %d command line arguments", argc - 1);

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0 || std::strcmp(argv[i], "-v") == 0) {
            show_version = true;
            log_message(LOG_DEBUG, "Flag: show version");
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            // already handled
        } else if (std::strcmp(argv[i], "--tokenCount") == 0) {
            show_token_count = true;
            log_message(LOG_DEBUG, "Flag: show token count");
        } else if (std::strcmp(argv[i], "--continue") == 0 || std::strcmp(argv[i], "-c") == 0) {
            continue_mode = true;
            log_message(LOG_DEBUG, "Flag: continue mode enabled");
        } else if (std::strcmp(argv[i], "--no-stream") == 0) {
            no_stream = true;
            log_message(LOG_DEBUG, "Flag: streaming disabled");
        } else if (std::strcmp(argv[i], "--temperature") == 0 || std::strcmp(argv[i], "-T") == 0) {
            if (i + 1 < argc) {
                temperature = std::atof(argv[++i]);
                log_message(LOG_DEBUG, "Set temperature to %.2f", temperature);
            }
        } else if (std::strcmp(argv[i], "--tokenLimit") == 0 || std::strcmp(argv[i], "-l") == 0) {
            if (i + 1 < argc) {
                token_limit = std::atoi(argv[++i]);
                log_message(LOG_DEBUG, "Set token limit to %d", token_limit);
            }
        } else if (std::strcmp(argv[i], "--token") == 0 || std::strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) {
                global_api_key = argv[++i];
                log_message(LOG_DEBUG, "Set API key from command line");
            }
        } else if (std::strcmp(argv[i], "--model") == 0 || std::strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) {
                global_model = argv[++i];
                log_message(LOG_DEBUG, "Set model to %s", global_model.c_str());
            }
        } else if (std::strcmp(argv[i], "--setAPIKey") == 0 && i + 1 < argc) {
            new_api_key = argv[++i];
            set_api_key = true;
            log_message(LOG_DEBUG, "Will save new API key");
        } else if (std::strcmp(argv[i], "--setModel") == 0 && i + 1 < argc) {
            new_model = argv[++i];
            set_model = true;
            log_message(LOG_DEBUG, "Will save new model: %s", new_model.c_str());
        } else {
            std::ostringstream builder;
            for (int j = i; j < argc; ++j) {
                if (j > i) builder << ' ';
                builder << argv[j];
            }
            input_text = builder.str();
            log_message(LOG_DEBUG, "Input text: \"%s\" (%zu chars)", input_text.c_str(), input_text.size());
            break;
        }
    }

    if (set_api_key || set_model) {
        if (set_model) global_model = new_model;
        if (set_api_key) global_api_key = new_api_key;
        save_env_file();
        log_message(LOG_INFO, "Updated configuration saved to .env file");
        std::cout << "Remember to update to make sure your curl library can handle streaming\n";
        std::exit(0);
    }

    if (show_version || debug_mode) {
        std::cout << "OpenAI Chatbot\n";
        std::cout << "Model: " << global_model << "\n";
        std::cout << "API Key: " << global_api_key << "\n";
        std::cout << "Token Limit: " << token_limit << "\n";

        const char *level_str = "UNKNOWN";
        switch (log_level) {
            case LOG_NONE: level_str = "NONE"; break;
            case LOG_ERROR: level_str = "ERROR"; break;
            case LOG_WARN: level_str = "WARN"; break;
            case LOG_INFO: level_str = "INFO"; break;
            case LOG_DEBUG: level_str = "DEBUG"; break;
        }
        std::cout << "Log Level: " << level_str << "\n";
        if (log_to_file) std::cout << "Logging to file: " << log_file_path << "\n";
        if (!debug_mode) std::exit(0);
    }

    if (show_token_count && !input_text.empty()) {
        std::vector<Message> messages;
        add_message(messages, "user", input_text);
        int token_count = count_tokens_from_messages(messages, global_model);
        std::cout << token_count << "\n";
        log_message(LOG_INFO, "Token count: %d", token_count);
        std::exit(0);
    }
}

bool is_plain_text_file(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        log_message(LOG_WARN, "Cannot open file: %s", filename.c_str());
        return false;
    }

    std::vector<unsigned char> buffer(1024);
    file.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
    size_t bytes_read = static_cast<size_t>(file.gcount());

    if (bytes_read == 0) {
        return true;
    }

    size_t null_count = 0;
    size_t control_count = 0;
    for (size_t i = 0; i < bytes_read; ++i) {
        if (buffer[i] == 0) {
            ++null_count;
        } else if (buffer[i] < 32 && buffer[i] != '\n' && buffer[i] != '\r' && buffer[i] != '\t') {
            ++control_count;
        }
    }

    if (null_count > 0 || control_count > bytes_read / 20) {
        return false;
    }
    return true;
}

std::optional<std::string> read_file_content(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        log_message(LOG_ERROR, "Failed to open file: %s", filename.c_str());
        return std::nullopt;
    }

    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size <= 0) {
        return std::string();
    }

    if (file_size > 10000) { // 10KB limit
        log_message(LOG_WARN, "File too large (>10KB): %s", filename.c_str());
        return std::nullopt;
    }

    std::string content(static_cast<size_t>(file_size), '\0');
    file.read(content.data(), file_size);
    content.resize(static_cast<size_t>(file.gcount()));
    log_message(LOG_DEBUG, "Read %zu bytes from file: %s", content.size(), filename.c_str());
    return content;
}

std::string process_file_references(const std::string &input) {
    if (input.empty()) return {};

    std::string result = input;
    size_t search_pos = 0;

    while ((search_pos = result.find('@', search_pos)) != std::string::npos) {
        size_t filename_start = search_pos + 1;
        if (filename_start >= result.size()) break;

        bool quoted = false;
        char quote_char = '\0';
        if (result[filename_start] == '"' || result[filename_start] == '\'') {
            quoted = true;
            quote_char = result[filename_start];
            ++filename_start;
        }

        size_t filename_end = filename_start;
        if (quoted) {
            while (filename_end < result.size() && result[filename_end] != quote_char) {
                ++filename_end;
            }
        } else {
            while (filename_end < result.size()) {
                char c = result[filename_end];
                if (std::isspace(static_cast<unsigned char>(c)) ||
                    c == '?' || c == '!' || c == ';' ||
                    (c == '.' && (filename_end + 1 >= result.size() || std::isspace(static_cast<unsigned char>(result[filename_end + 1])))) ||
                    c == ',' || c == ')' || c == '}') {
                    break;
                }
                ++filename_end;
            }
        }

        if (filename_end == filename_start) {
            search_pos = filename_start;
            continue;
        }

        std::string filename = result.substr(filename_start, filename_end - filename_start);
        size_t suffix_start = filename_end;
        while (!filename.empty() && (filename.back() == '"' || filename.back() == '\'' || filename.back() == '`')) {
            filename.pop_back();
        }
        if (filename.empty()) {
            search_pos = suffix_start;
            continue;
        }
        if (quoted && suffix_start < result.size() && result[suffix_start] == quote_char) {
            ++suffix_start;
        }

        std::string prefix = result.substr(0, search_pos);
        std::string suffix = result.substr(suffix_start);
        std::string replacement;

        if (access(filename.c_str(), F_OK) == 0 && is_plain_text_file(filename)) {
            auto content_opt = read_file_content(filename);
            if (content_opt.has_value()) {
                replacement = prefix + "\nFile: " + filename + "\n```\n" + *content_opt + "\n```" + suffix;
                log_message(LOG_INFO, "Attached file content: %s (%zu bytes)", filename.c_str(), content_opt->size());
            } else {
                replacement = prefix + "[Error: Could not read " + filename + "]" + suffix;
            }
        } else {
            log_message(LOG_WARN, "File not found or not plain text: %s", filename.c_str());
            replacement = prefix + "[File not found: " + filename + "]" + suffix;
        }

        result.swap(replacement);
        search_pos = prefix.size();
    }

    return result;
}

std::string expand_home_path(const std::string &path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }

    struct passwd *pw = getpwuid(getuid());
    if (pw == nullptr) {
        log_message(LOG_ERROR, "Could not determine home directory");
        return path;
    }

    std::string expanded = pw->pw_dir;
    expanded += path.substr(1);

    auto last_slash = expanded.find_last_of('/');
    if (last_slash != std::string::npos) {
        std::string dir = expanded.substr(0, last_slash);
        struct stat st{};
        if (stat(dir.c_str(), &st) == -1) {
            log_message(LOG_INFO, "Creating directory: %s", dir.c_str());
            if (mkdir(dir.c_str(), 0700) == -1) {
                log_message(LOG_ERROR, "Failed to create directory: %s", dir.c_str());
            }
        }
    }

    log_message(LOG_DEBUG, "Expanded path: %s", expanded.c_str());
    return expanded;
}

bool load_models_cache() {
    std::string cache_path = expand_home_path(MODELS_CACHE_FILE);
    std::ifstream cache_file(cache_path);
    if (!cache_file.is_open()) {
        log_message(LOG_DEBUG, "No models cache file found");
        return false;
    }

    std::stringstream buffer;
    buffer << cache_file.rdbuf();
    std::string content = buffer.str();

    if (content.empty()) {
        log_message(LOG_WARN, "Empty models cache file");
        return false;
    }

    cJSON *root = cJSON_Parse(content.c_str());
    if (!root) {
        log_message(LOG_ERROR, "Failed to parse models cache file: %s", cJSON_GetErrorPtr());
        return false;
    }

    cJSON *timestamp = cJSON_GetObjectItem(root, "timestamp");
    if (!timestamp || !cJSON_IsNumber(timestamp)) {
        log_message(LOG_ERROR, "Invalid timestamp in models cache");
        cJSON_Delete(root);
        return false;
    }
    models_cache.last_updated = static_cast<time_t>(timestamp->valuedouble);

    time_t now = time(nullptr);
    if (now - models_cache.last_updated > MODELS_CACHE_EXPIRY) {
        log_message(LOG_INFO, "Models cache is expired (older than 24 hours)");
        cJSON_Delete(root);
        return false;
    }

    cJSON *models_array = cJSON_GetObjectItem(root, "models");
    if (!models_array || !cJSON_IsArray(models_array)) {
        log_message(LOG_ERROR, "Invalid models array in cache");
        cJSON_Delete(root);
        return false;
    }

    models_cache.models.clear();
    int model_count = cJSON_GetArraySize(models_array);
    for (int i = 0; i < model_count; ++i) {
        cJSON *model = cJSON_GetArrayItem(models_array, i);
        if (!model || !cJSON_IsObject(model)) continue;

        cJSON *id = cJSON_GetObjectItem(model, "id");
        cJSON *created = cJSON_GetObjectItem(model, "created");
        if (!id || !cJSON_IsString(id) || !created || !cJSON_IsNumber(created)) continue;

        models_cache.models.push_back({id->valuestring, static_cast<time_t>(created->valuedouble)});
    }

    cJSON_Delete(root);
    log_message(LOG_INFO, "Loaded %zu models from cache (last updated: %s)", models_cache.models.size(), ctime(&models_cache.last_updated));
    return true;
}

bool save_models_cache() {
    if (models_cache.models.empty()) {
        log_message(LOG_WARN, "No models to save to cache");
        return false;
    }

    cJSON *root = cJSON_CreateObject();
    if (!root) {
        log_message(LOG_ERROR, "Failed to create JSON object for cache");
        return false;
    }

    cJSON_AddNumberToObject(root, "timestamp", static_cast<double>(models_cache.last_updated));
    cJSON *models_array = cJSON_CreateArray();
    if (!models_array) {
        log_message(LOG_ERROR, "Failed to create models array for cache");
        cJSON_Delete(root);
        return false;
    }
    cJSON_AddItemToObject(root, "models", models_array);

    for (const auto &model : models_cache.models) {
        cJSON *node = cJSON_CreateObject();
        if (!node) continue;

        cJSON_AddStringToObject(node, "id", model.id.c_str());
        cJSON_AddNumberToObject(node, "created", static_cast<double>(model.created));
        cJSON_AddItemToArray(models_array, node);
    }

    char *json_str = cJSON_Print(root);
    if (!json_str) {
        log_message(LOG_ERROR, "Failed to print JSON for cache");
        cJSON_Delete(root);
        return false;
    }

    std::string cache_path = expand_home_path(MODELS_CACHE_FILE);
    std::ofstream cache_file(cache_path);
    if (!cache_file.is_open()) {
        log_message(LOG_ERROR, "Failed to open cache file for writing");
        std::free(json_str);
        cJSON_Delete(root);
        return false;
    }

    cache_file << json_str;
    cache_file.close();

    std::free(json_str);
    cJSON_Delete(root);
    log_message(LOG_INFO, "Saved %zu models to cache", models_cache.models.size());
    return true;
}

bool fetch_models_list(CURL *curl) {
    log_message(LOG_INFO, "Fetching available models from OpenAI API");

    ResponseBuffer buffer;
    buffer.stream_enabled = false;
    buffer.saw_stream_data = false;

    curl_easy_reset(curl);

    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth_header = "Authorization: Bearer " + global_api_key;
    headers = curl_slist_append(headers, auth_header.c_str());

    curl_easy_setopt(curl, CURLOPT_URL, "https://api.openai.com/v1/models");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        log_message(LOG_ERROR, "Failed to fetch models: %s", curl_easy_strerror(res));
        curl_slist_free_all(headers);
        return false;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code != 200) {
        log_message(LOG_ERROR, "API returned HTTP %ld when fetching models", http_code);
        curl_slist_free_all(headers);
        return false;
    }

    cJSON *root = cJSON_Parse(buffer.data.c_str());
    if (!root) {
        log_message(LOG_ERROR, "Failed to parse API response: %s", cJSON_GetErrorPtr());
        curl_slist_free_all(headers);
        return false;
    }

    cJSON *data = cJSON_GetObjectItem(root, "data");
    if (!data || !cJSON_IsArray(data)) {
        log_message(LOG_ERROR, "Invalid response format: 'data' array not found");
        cJSON_Delete(root);
        curl_slist_free_all(headers);
        return false;
    }

    models_cache.models.clear();
    models_cache.last_updated = time(nullptr);

    int model_count = cJSON_GetArraySize(data);
    for (int i = 0; i < model_count; ++i) {
        cJSON *model = cJSON_GetArrayItem(data, i);
        if (!model || !cJSON_IsObject(model)) continue;

        cJSON *id = cJSON_GetObjectItem(model, "id");
        cJSON *created = cJSON_GetObjectItem(model, "created");
        if (!id || !cJSON_IsString(id)) continue;

        ModelInfo info;
        info.id = id->valuestring;
        info.created = created && cJSON_IsNumber(created) ? static_cast<time_t>(created->valuedouble) : time(nullptr);
        models_cache.models.push_back(info);
    }

    cJSON_Delete(root);
    curl_slist_free_all(headers);

    log_message(LOG_INFO, "Fetched %zu models from API", models_cache.models.size());
    if (!models_cache.models.empty()) {
        save_models_cache();
    }

    return !models_cache.models.empty();
}

bool is_valid_model(const std::string &model) {
    if (models_cache.models.empty()) {
        log_message(LOG_WARN, "No models in cache to validate against");
        return true;
    }

    for (const auto &info : models_cache.models) {
        if (info.id == model) {
            log_message(LOG_DEBUG, "Model '%s' is valid", model.c_str());
            return true;
        }
    }

    log_message(LOG_WARN, "Model '%s' not found in available models", model.c_str());
    return false;
}

int levenshtein_distance(const std::string &s1, const std::string &s2) {
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

void spinner_loop(std::atomic_bool &stop_flag, std::atomic_bool &first_token_flag) {
    const char frames[] = {'|', '/', '-', '\\'};
    size_t idx = 0;
    std::cout << "thinking... " << std::flush;
    while (!stop_flag.load()) {
        if (first_token_flag.load()) {
            std::cout << "\r                \r" << std::flush;
            return;
        }
        std::cout << "\rthinking... " << frames[idx % 4] << std::flush;
        idx++;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    std::cout << "\r                \r" << std::flush;
}

void suggest_similar_model(const std::string &invalid_model) {
    if (models_cache.models.empty()) return;

    int min_distance = std::numeric_limits<int>::max();
    std::string closest_model;

    for (const auto &info : models_cache.models) {
        int distance = levenshtein_distance(invalid_model, info.id);
        if (distance < min_distance) {
            min_distance = distance;
            closest_model = info.id;
        }
    }

    if (min_distance <= 5) {
        std::printf("Model '%s' not found. Did you mean '%s'?\n", invalid_model.c_str(), closest_model.c_str());
        log_message(LOG_INFO, "Suggested alternative model: %s (distance: %d)", closest_model.c_str(), min_distance);
    } else {
        std::printf("Model '%s' not found. Available models include: gpt-4o, gpt-4o-mini, gpt-3.5-turbo\n", invalid_model.c_str());
    }
}

bool validate_model(CURL *curl, const std::string &model) {
    bool cache_loaded = load_models_cache();
    if (!cache_loaded || models_cache.models.empty()) {
        if (!fetch_models_list(curl)) {
            log_message(LOG_WARN, "Failed to fetch models list, will continue without validation");
            return true;
        }
    }

    if (!is_valid_model(model)) {
        suggest_similar_model(model);
        return false;
    }
    return true;
}

void ask(CURL *curl, std::vector<Message> &messages, double temperature, bool no_stream) {
    if (messages.empty()) {
        log_message(LOG_WARN, "No messages to send to API");
        return;
    }

    log_message(LOG_INFO, "Sending request to OpenAI API (model: %s, temp: %.2f, stream: %s)", global_model.c_str(), temperature, no_stream ? "disabled" : "enabled");

    // keep within token limit by removing oldest non-system message
    while (messages.size() > 1 && count_tokens_from_messages(messages, global_model) + 100 > token_limit) {
        messages.erase(messages.begin() + 1);
    }

    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "model", global_model.c_str());
    cJSON_AddNumberToObject(root, "temperature", temperature);
    cJSON_AddBoolToObject(root, "stream", !no_stream);

    cJSON *message_array = cJSON_CreateArray();
    for (const auto &msg : messages) {
        cJSON *message = cJSON_CreateObject();
        cJSON_AddStringToObject(message, "role", msg.role.c_str());
        cJSON_AddStringToObject(message, "content", msg.content.c_str());
        cJSON_AddItemToArray(message_array, message);
    }
    cJSON_AddItemToObject(root, "messages", message_array);

    char *json_str = cJSON_Print(root);
    log_message(LOG_DEBUG, "Request JSON: %s", json_str);

    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth_header = "Authorization: Bearer " + global_api_key;
    headers = curl_slist_append(headers, auth_header.c_str());
    if (!no_stream) {
        headers = curl_slist_append(headers, "Accept: text/event-stream");
    }

    const long CONNECT_TIMEOUT = 10L;
    const long REQUEST_TIMEOUT = 60L;
    const int max_retries = 1; // total attempts = max_retries + 1

    int attempt = 0;
    bool done = false;

    while (!done && attempt <= max_retries) {
        attempt++;
        log_message(LOG_INFO, "Attempt %d/%d", attempt, max_retries + 1);

        std::atomic_bool spinner_stop(false);
        std::atomic_bool first_token_received(false);

        ResponseBuffer buffer;
        buffer.stream_enabled = !no_stream;
        buffer.first_token_flag = &first_token_received;

        std::thread spinner_thread(spinner_loop, std::ref(spinner_stop), std::ref(first_token_received));

        curl_easy_reset(curl);
        curl_easy_setopt(curl, CURLOPT_URL, "https://api.openai.com/v1/chat/completions");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, CONNECT_TIMEOUT);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, REQUEST_TIMEOUT);

        log_message(LOG_INFO, "Sending request to API...");
        CURLcode res = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        spinner_stop.store(true);
        if (spinner_thread.joinable()) {
            spinner_thread.join();
        }

        if (res != CURLE_OK) {
            log_message(LOG_ERROR, "curl_easy_perform() failed: %s", curl_easy_strerror(res));
            bool should_retry = (res == CURLE_OPERATION_TIMEDOUT) && attempt <= max_retries;
            if (should_retry) {
                std::cout << "Request timed out, retrying (" << attempt << "/" << max_retries + 1 << ")...\n";
                continue;
            } else {
                std::fprintf(stderr, "Request failed: %s\n", curl_easy_strerror(res));
            }
        } else {
            log_message(LOG_INFO, "Request completed successfully");
            log_message(LOG_DEBUG, "Response size: %zu bytes", buffer.data.size());

            if (http_code >= 400) {
                cJSON *err = cJSON_Parse(buffer.data.c_str());
                if (err) {
                    cJSON *error_obj = cJSON_GetObjectItem(err, "error");
                    cJSON *msg = error_obj ? cJSON_GetObjectItem(error_obj, "message") : nullptr;
                    if (msg && cJSON_IsString(msg) && msg->valuestring) {
                        std::fprintf(stderr, "API error (HTTP %ld): %s\n", http_code, msg->valuestring);
                    } else {
                        std::fprintf(stderr, "API error (HTTP %ld).\n", http_code);
                    }
                    cJSON_Delete(err);
                } else {
                    std::fprintf(stderr, "API error (HTTP %ld).\n", http_code);
                }
            } else if (no_stream) {
                cJSON *json = cJSON_Parse(buffer.data.c_str());
                if (json) {
                    cJSON *choices = cJSON_GetObjectItem(json, "choices");
                    if (choices && cJSON_IsArray(choices) && cJSON_GetArraySize(choices) > 0) {
                        cJSON *choice = cJSON_GetArrayItem(choices, 0);
                        cJSON *message = cJSON_GetObjectItem(choice, "message");
                        if (message) {
                            cJSON *content = cJSON_GetObjectItem(message, "content");
                            if (content && cJSON_IsString(content) && content->valuestring) {
                                std::cout << content->valuestring << "\n";
                            }
                        }
                    }
                    cJSON_Delete(json);
                } else {
                    log_message(LOG_ERROR, "Failed to parse API response: %s", cJSON_GetErrorPtr());
                }
            } else {
                if (!buffer.saw_stream_data && !buffer.data.empty()) {
                    cJSON *json = cJSON_Parse(buffer.data.c_str());
                    if (json) {
                        cJSON *choices = cJSON_GetObjectItem(json, "choices");
                        if (choices && cJSON_IsArray(choices) && cJSON_GetArraySize(choices) > 0) {
                            cJSON *choice = cJSON_GetArrayItem(choices, 0);
                            cJSON *message = cJSON_GetObjectItem(choice, "message");
                            if (message) {
                                cJSON *content = cJSON_GetObjectItem(message, "content");
                                if (content && cJSON_IsString(content) && content->valuestring) {
                                    std::cout << content->valuestring << "\n";
                                }
                            }
                        } else {
                            cJSON *error_obj = cJSON_GetObjectItem(json, "error");
                            cJSON *msg = error_obj ? cJSON_GetObjectItem(error_obj, "message") : nullptr;
                            if (msg && cJSON_IsString(msg) && msg->valuestring) {
                                std::fprintf(stderr, "API error: %s\n", msg->valuestring);
                            }
                        }
                        cJSON_Delete(json);
                    }
                }
                std::cout << std::endl;
            }
        }

        done = true;
    }

    curl_slist_free_all(headers);
    cJSON_Delete(root);
    std::free(json_str);
}

int main(int argc, char *argv[]) {
    log_message(LOG_INFO, "Starting OpenAI chatbot");

    curl_global_init(CURL_GLOBAL_ALL);
    log_message(LOG_DEBUG, "Initialized curl");

    CURL *curl = curl_easy_init();
    if (!curl) {
        log_message(LOG_ERROR, "Failed to initialize curl");
        close_logging();
        return 1;
    }

    if (const char *env_api_key = std::getenv("OPENAI_API_KEY")) {
        global_api_key = env_api_key;
        log_message(LOG_DEBUG, "Loaded API key from environment");
    }
    if (const char *env_model = std::getenv("ASK_GLOBAL_MODEL")) {
        global_model = env_model;
        log_message(LOG_DEBUG, "Loaded model from environment: %s", global_model.c_str());
    }

    if ((global_api_key.empty() || global_model.empty()) && access(".env", F_OK) == 0) {
        load_dotenv(".env");
    }

    if (global_model.empty()) {
        global_model = DEFAULT_MODEL;
        log_message(LOG_INFO, "Using default model: %s", DEFAULT_MODEL);
    }

    if (global_api_key.empty()) {
        log_message(LOG_ERROR, "API Key not found");
        if (access(".env", F_OK) != 0) {
            std::ofstream env_file(".env");
            if (env_file.is_open()) {
                env_file << "OPENAI_API_KEY=sk-xxxxxxxxxx\nASK_GLOBAL_MODEL=" << global_model << "\n";
                log_message(LOG_INFO, "Created default .env file template");
            } else {
                log_message(LOG_ERROR, "Failed to create .env file");
            }
        }

        curl_easy_cleanup(curl);
        curl_global_cleanup();
        close_logging();
        return 1;
    }

    bool continue_mode = false;
    bool no_stream = false;
    double temperature = 1.0;
    std::string input_text;

    parse_arguments(argc, argv, continue_mode, no_stream, temperature, input_text);

    if (input_text.empty()) {
        if (continue_mode) {
            log_message(LOG_INFO, "No input text provided, starting conversation mode anyway");
        } else {
            log_message(LOG_INFO, "No input text provided, showing usage hint");
            std::cout << "No input provided. Usage examples:\n";
            std::cout << "  ask \"What is the capital of France?\"\n";
            std::cout << "  ask -c \"Let's have a conversation\"\n";
            std::cout << "  ask --help\n";
            curl_easy_cleanup(curl);
            curl_global_cleanup();
            close_logging();
            return 0;
        }
    }

    if (!validate_model(curl, global_model)) {
        log_message(LOG_ERROR, "Invalid model: %s", global_model.c_str());
        std::printf("Error: '%s' is not a valid model.\n", global_model.c_str());
        curl_easy_cleanup(curl);
        curl_global_cleanup();
        close_logging();
        return 1;
    }

    std::vector<Message> messages;

    if (continue_mode) {
        log_message(LOG_INFO, "Starting interactive mode");
        add_message(messages, "system", "You are a cute cat running in a command line interface. The user can chat with you and the conversation can be continued.");

        if (!input_text.empty()) {
            std::string processed_input = process_file_references(input_text);
            add_message(messages, "user", processed_input.empty() ? input_text : processed_input);
            ask(curl, messages, temperature, no_stream);
            add_message(messages, "assistant", "I'm a cute cat meow! (Note: In a full implementation, this would be the actual API response)");
        } else {
            std::cout << "Starting conversation mode...\n";
        }

        std::cout << "Type 'exit' to quit, 'status' for conversation info, or 'help' for commands.\n";

        std::string user_input;
        while (true) {
            std::cout << "> ";
            std::cout.flush();

            if (!std::getline(std::cin, user_input)) {
                log_message(LOG_WARN, "Failed to read user input, exiting");
                break;
            }

            if (user_input == "exit") {
                log_message(LOG_INFO, "User requested exit");
                break;
            } else if (user_input == "status") {
                int approx_tokens = count_tokens_from_messages(messages, global_model);
                std::cout << "Conversation Status:\n";
                std::cout << "  Messages: " << messages.size() << "\n";
                std::cout << "  Approximate tokens: " << approx_tokens << " / " << token_limit << "\n";
                std::cout << "  Model: " << global_model << "\n";
                std::cout << "  Temperature: " << temperature << "\n";
                std::cout << "  Streaming: " << (no_stream ? "disabled" : "enabled") << "\n";
                continue;
            } else if (user_input == "help") {
                std::cout << "Conversation Mode Commands:\n";
                std::cout << "  exit    - Exit conversation mode\n";
                std::cout << "  status  - Show conversation information\n";
                std::cout << "  help    - Show this help message\n";
                std::cout << "  Any other text will be sent to the AI assistant.\n";
                continue;
            }

            std::string processed_input = process_file_references(user_input);
            add_message(messages, "user", processed_input.empty() ? user_input : processed_input);
            ask(curl, messages, temperature, no_stream);
            add_message(messages, "assistant", "Meow response! (This would be the actual API response in a full implementation)");
        }
    } else {
        log_message(LOG_INFO, "Single response mode");
        add_message(messages, "system", "You are a cute cat runs in a command line interface and you can only respond once to the user. Do not ask any questions in your response.");

        std::string processed_input = process_file_references(input_text);
        add_message(messages, "user", processed_input.empty() ? input_text : processed_input);
        ask(curl, messages, temperature, no_stream);
    }

    log_message(LOG_INFO, "Cleaning up resources");
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    close_logging();
    log_message(LOG_INFO, "Exiting normally");
    return 0;
}
