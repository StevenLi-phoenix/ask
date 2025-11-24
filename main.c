#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>
#include <time.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <limits.h>
#include <pwd.h>

#define DEFAULT_MODEL "gpt-5-nano"
#define MAX_ENV_VALUE_SIZE 1024
#define MAX_BUFFER_SIZE 8192
#define DEFAULT_TOKEN_LIMIT 128000
#define MAX_MESSAGES 100
#define MAX_MODELS 200
#define MODELS_CACHE_FILE "~/.cache/ask_models_cache.json"
#define MODELS_CACHE_EXPIRY 86400 // 24 hours in seconds

// Log levels
typedef enum {
    LOG_NONE = 0,
    LOG_ERROR = 1,
    LOG_WARN = 2,
    LOG_INFO = 3,
    LOG_DEBUG = 4
} LogLevel;

typedef struct {
    char *data;
    size_t size;
    bool stream_enabled;
    bool saw_stream_data;
} ResponseBuffer;

typedef struct {
    char *role;
    char *content;
} Message;

typedef struct {
    Message messages[MAX_MESSAGES];
    int count;
} MessageList;

// Model validation
typedef struct {
    char id[MAX_ENV_VALUE_SIZE];
    time_t created;
} ModelInfo;

typedef struct {
    ModelInfo models[MAX_MODELS];
    int count;
    time_t last_updated;
} ModelsCache;

// Global variables
char global_api_key[MAX_ENV_VALUE_SIZE] = {0};
char global_model[MAX_ENV_VALUE_SIZE] = {0};
int token_limit = DEFAULT_TOKEN_LIMIT;
bool debug_mode = false;
LogLevel log_level = LOG_INFO;
FILE *log_file = NULL;
bool log_to_file = false;
char log_file_path[MAX_ENV_VALUE_SIZE] = "ask.log";
ModelsCache models_cache = {0};

// Function prototypes
void load_dotenv(const char *filename);
size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp);
int count_tokens_from_messages(MessageList *messages, const char *model);
void ask(CURL *curl, MessageList *messages, double temperature, bool no_stream);
void add_message(MessageList *messages, const char *role, const char *content);
void free_messages(MessageList *messages);
void parse_arguments(int argc, char *argv[], bool *continue_mode, bool *no_stream, double *temperature, 
                    char **input_text, size_t *input_text_len);
void save_env_file();
void init_logging(void);
void close_logging(void);
void log_message(LogLevel level, const char *format, ...);
void print_help(void);
bool validate_model(CURL *curl, const char *model);
bool load_models_cache(void);
bool save_models_cache(void);
bool fetch_models_list(CURL *curl);
bool is_valid_model(const char *model);
void suggest_similar_model(const char *invalid_model);
char* expand_home_path(const char *path);
bool is_plain_text_file(const char *filename);
char* read_file_content(const char *filename);
char* process_file_references(const char *input);

// Initialize logging
void init_logging(void) {
    if (log_to_file) {
        log_file = fopen(log_file_path, "a");
        if (!log_file) {
            fprintf(stderr, "Failed to open log file %s\n", log_file_path);
            log_to_file = false;
        } else {
            fprintf(stdout, "Logging to file: %s\n", log_file_path);
        }
    }
}

// Close logging
void close_logging(void) {
    if (log_file) {
        fclose(log_file);
        log_file = NULL;
    }
}

// Log a message with a specific log level
void log_message(LogLevel level, const char *format, ...) {
    if (level > log_level) return;
    
    char level_str[10];
    switch (level) {
        case LOG_ERROR: strcpy(level_str, "ERROR"); break;
        case LOG_WARN:  strcpy(level_str, "WARN");  break;
        case LOG_INFO:  strcpy(level_str, "INFO");  break;
        case LOG_DEBUG: strcpy(level_str, "DEBUG"); break;
        default:        strcpy(level_str, "NONE");  break;
    }
    
    time_t now;
    struct tm *local_time;
    char time_str[20];
    
    time(&now);
    local_time = localtime(&now);
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", local_time);
    
    va_list args;
    va_start(args, format);
    
    char full_message[MAX_BUFFER_SIZE];
    vsnprintf(full_message, sizeof(full_message), format, args);
    
    // Print to stdout if needed (for higher priority messages)
    if (level <= LOG_WARN || debug_mode) {
        fprintf(stdout, "[%s] %s: %s\n", time_str, level_str, full_message);
    }
    
    // Log to file if enabled
    if (log_to_file && log_file) {
        fprintf(log_file, "[%s] %s: %s\n", time_str, level_str, full_message);
        fflush(log_file);
    }
    
    va_end(args);
}

// Load environment variables from .env file
void load_dotenv(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        log_message(LOG_WARN, "Could not open .env file: %s", filename);
        return;
    }
    
    log_message(LOG_INFO, "Loading environment from %s", filename);
    
    char line[MAX_ENV_VALUE_SIZE];
    while (fgets(line, sizeof(line), file)) {
        char *equals = strchr(line, '=');
        if (!equals) continue;
        
        *equals = '\0';
        char *key = line;
        char *value = equals + 1;
        
        // Remove newline from value if present
        char *newline = strchr(value, '\n');
        if (newline) *newline = '\0';
        
        if (strcmp(key, "OPENAI_API_KEY") == 0 && global_api_key[0] == '\0') {
            strcpy(global_api_key, value);
            log_message(LOG_DEBUG, "Loaded API key from .env");
        } else if (strcmp(key, "ASK_GLOBAL_MODEL") == 0 && global_model[0] == '\0') {
            strcpy(global_model, value);
            log_message(LOG_DEBUG, "Loaded model from .env: %s", global_model);
        }
    }
    
    fclose(file);
}

// CURL write callback
size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    ResponseBuffer *mem = (ResponseBuffer *)userp;
    
    // Allocate/reallocate memory for the buffer
    char *ptr = realloc(mem->data, mem->size + realsize + 1);
    if (!ptr) {
        log_message(LOG_ERROR, "Out of memory when processing API response");
        return 0;
    }
    
    // Copy the received data to the buffer
    mem->data = ptr;
    memcpy(&(mem->data[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->data[mem->size] = 0;
    
    // Process streaming data if it's a streaming response
    if (mem->stream_enabled) {
        // Find complete "data:" lines in the new chunk
        char *data = mem->data;
        char *line_start = data;
        char *line_end;
        
        // Process all complete lines ending with \n\n
        while ((line_end = strstr(line_start, "\n\n")) != NULL) {
            *line_end = '\0';  // Temporarily terminate the line
            
            // Skip empty lines and [DONE] marker
            if (strncmp(line_start, "data: ", 6) == 0 && 
                strcmp(line_start + 6, "[DONE]") != 0) {
                
                // Parse the JSON payload
                cJSON *json = cJSON_Parse(line_start + 6);
                if (json) {
                    cJSON *choices = cJSON_GetObjectItem(json, "choices");
                    if (choices && cJSON_IsArray(choices) && cJSON_GetArraySize(choices) > 0) {
                        cJSON *choice = cJSON_GetArrayItem(choices, 0);
                        cJSON *delta = cJSON_GetObjectItem(choice, "delta");
                        
                        if (delta) {
                            cJSON *content = cJSON_GetObjectItem(delta, "content");
                            if (content && cJSON_IsString(content) && content->valuestring) {
                                // Output the content as it arrives
                                printf("%s", content->valuestring);
                                fflush(stdout);  // Force flush to show progress
                                mem->saw_stream_data = true;
                            }
                        }
                    }
                    cJSON_Delete(json);
                }
            }
            
            line_start = line_end + 2;  // Move to next line after the \n\n
        }
        
        // If we processed some lines, shift the buffer to remove processed data
        if (line_start > data) {
            size_t remaining = mem->size - (line_start - data);
            memmove(data, line_start, remaining);
            mem->size = remaining;
            data[mem->size] = '\0';
        }
    }
    
    log_message(LOG_DEBUG, "Received %zu bytes from API", realsize);
    return realsize;
}

// Approximate token counting function
// In a real implementation, this would need to use a proper tokenizer like tiktoken
int count_tokens_from_messages(MessageList *messages, const char *model) {
    (void)model; // currently unused
    // This is a very crude approximation
    // In reality, you'd need to implement a tokenizer similar to tiktoken
    int tokens = 3; // Initial tokens
    for (int i = 0; i < messages->count; i++) {
        tokens += 3; // tokens per message
        
        // Approximate token count based on string length
        // (this is very rough - tiktoken would be more accurate)
        tokens += strlen(messages->messages[i].content) / 4;
        
        // Add tokens for role
        if (messages->messages[i].role)
            tokens += 1;
    }
    
    return tokens;
}

// Add a message to the message list
void add_message(MessageList *messages, const char *role, const char *content) {
    if (messages->count >= MAX_MESSAGES) {
        log_message(LOG_ERROR, "Maximum message count reached");
        return;
    }
    
    int idx = messages->count;
    messages->messages[idx].role = strdup(role);
    messages->messages[idx].content = strdup(content);
    messages->count++;
    
    log_message(LOG_DEBUG, "Added message with role '%s' (length: %zu)", 
                role, strlen(content));
}

// Free all messages in the message list
void free_messages(MessageList *messages) {
    for (int i = 0; i < messages->count; i++) {
        free(messages->messages[i].role);
        free(messages->messages[i].content);
    }
    messages->count = 0;
    log_message(LOG_DEBUG, "Freed all messages");
}

// Simulate the ask function using CURL to call OpenAI API
void ask(CURL *curl, MessageList *messages, double temperature, bool no_stream) {
    if (messages->count == 0) {
        log_message(LOG_WARN, "No messages to send to API");
        return;
    }
    
    log_message(LOG_INFO, "Sending request to OpenAI API (model: %s, temp: %.2f, stream: %s)", 
                global_model, temperature, no_stream ? "disabled" : "enabled");
    
    // Ensure we're within token limit by removing oldest messages
    int original_count = messages->count;
    while (messages->count > 1 && 
           count_tokens_from_messages(messages, global_model) + 100 > token_limit) {
        // Shift all messages left, removing the oldest (non-system) message
        for (int i = 1; i < messages->count - 1; i++) {
            free(messages->messages[i].role);
            free(messages->messages[i].content);
            messages->messages[i].role = messages->messages[i+1].role;
            messages->messages[i].content = messages->messages[i+1].content;
        }
        messages->count--;
    }
    
    if (original_count > messages->count) {
        log_message(LOG_WARN, "Removed %d messages to stay within token limit", 
                    original_count - messages->count);
    }
    
    // Build JSON body for API request
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "model", global_model);
    cJSON_AddNumberToObject(root, "temperature", temperature);
    cJSON_AddBoolToObject(root, "stream", !no_stream);
    
    cJSON *message_array = cJSON_CreateArray();
    for (int i = 0; i < messages->count; i++) {
        cJSON *message = cJSON_CreateObject();
        cJSON_AddStringToObject(message, "role", messages->messages[i].role);
        cJSON_AddStringToObject(message, "content", messages->messages[i].content);
        cJSON_AddItemToArray(message_array, message);
    }
    cJSON_AddItemToObject(root, "messages", message_array);
    
    char *json_str = cJSON_Print(root);
    log_message(LOG_DEBUG, "Request JSON: %s", json_str);
    
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    char auth_header[MAX_ENV_VALUE_SIZE + 32];
    snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", global_api_key);
    headers = curl_slist_append(headers, auth_header);
    
    // Set up CURL options
    curl_easy_setopt(curl, CURLOPT_URL, "https://api.openai.com/v1/chat/completions");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    if (!no_stream) {
        headers = curl_slist_append(headers, "Accept: text/event-stream");
    }
    
    ResponseBuffer buffer;
    buffer.data = malloc(1);
    buffer.size = 0;
    buffer.stream_enabled = !no_stream;  // Set streaming flag based on mode
    buffer.saw_stream_data = false;
    
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&buffer);
    
    // Perform the request
    log_message(LOG_INFO, "Sending request to API...");
    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    
    if (res != CURLE_OK) {
        log_message(LOG_ERROR, "curl_easy_perform() failed: %s", curl_easy_strerror(res));
    } else {
        log_message(LOG_INFO, "Request completed successfully");
        log_message(LOG_DEBUG, "Response size: %zu bytes", buffer.size);
        
        if (http_code >= 400) {
            // Try to parse and display error details
            cJSON *err = cJSON_Parse(buffer.data);
            if (err) {
                cJSON *error_obj = cJSON_GetObjectItem(err, "error");
                cJSON *msg = error_obj ? cJSON_GetObjectItem(error_obj, "message") : NULL;
                if (msg && cJSON_IsString(msg) && msg->valuestring) {
                    fprintf(stderr, "API error (HTTP %ld): %s\n", http_code, msg->valuestring);
                } else {
                    fprintf(stderr, "API error (HTTP %ld).\n", http_code);
                }
                cJSON_Delete(err);
            } else {
                fprintf(stderr, "API error (HTTP %ld).\n", http_code);
            }
        } else if (no_stream) {
            // Handle non-streaming response
            cJSON *json = cJSON_Parse(buffer.data);
            if (json) {
                cJSON *choices = cJSON_GetObjectItem(json, "choices");
                if (choices && cJSON_IsArray(choices) && cJSON_GetArraySize(choices) > 0) {
                    cJSON *choice = cJSON_GetArrayItem(choices, 0);
                    cJSON *message = cJSON_GetObjectItem(choice, "message");
                    if (message) {
                        cJSON *content = cJSON_GetObjectItem(message, "content");
                        if (content && cJSON_IsString(content) && content->valuestring) {
                            printf("%s\n", content->valuestring);
                        }
                    }
                }
                cJSON_Delete(json);
            } else {
                log_message(LOG_ERROR, "Failed to parse API response: %s", cJSON_GetErrorPtr());
            }
        } else {
            // For streaming, we've already printed the tokens in the callback,
            // but if nothing was streamed, fall back to parsing full JSON
            if (!buffer.saw_stream_data && buffer.size > 0) {
                cJSON *json = cJSON_Parse(buffer.data);
                if (json) {
                    cJSON *choices = cJSON_GetObjectItem(json, "choices");
                    if (choices && cJSON_IsArray(choices) && cJSON_GetArraySize(choices) > 0) {
                        cJSON *choice = cJSON_GetArrayItem(choices, 0);
                        cJSON *message = cJSON_GetObjectItem(choice, "message");
                        if (message) {
                            cJSON *content = cJSON_GetObjectItem(message, "content");
                            if (content && cJSON_IsString(content) && content->valuestring) {
                                printf("%s\n", content->valuestring);
                            }
                        }
                    } else {
                        // If it's an error payload, surface it
                        cJSON *error_obj = cJSON_GetObjectItem(json, "error");
                        cJSON *msg = error_obj ? cJSON_GetObjectItem(error_obj, "message") : NULL;
                        if (msg && cJSON_IsString(msg) && msg->valuestring) {
                            fprintf(stderr, "API error: %s\n", msg->valuestring);
                        }
                    }
                    cJSON_Delete(json);
                }
            }
            // Always ensure we end with a newline
            printf("\n");
        }
    }
    
    free(buffer.data);
    curl_slist_free_all(headers);
    cJSON_Delete(root);
    free(json_str);
}

void save_env_file() {
    FILE *file = fopen(".env", "w");
    if (!file) {
        log_message(LOG_ERROR, "Failed to open .env file for writing");
        return;
    }
    
    fprintf(file, "OPENAI_API_KEY=%s\nASK_GLOBAL_MODEL=%s\n", global_api_key, global_model);
    fclose(file);
    
    log_message(LOG_INFO, "Saved API key and model settings to .env file");
}

// Display help information
void print_help(void) {
    printf("OpenAI CLI Chatbot - Command Line Interface for OpenAI API\n\n");
    printf("Usage: ask [OPTIONS] [query]\n\n");
    printf("Options:\n");
    printf("  -h, --help             Display this help message\n");
    printf("  -v, --version          Display version information\n");
    printf("  -c, --continue         Enable conversation mode (supports multiple exchanges)\n");
    printf("      --no-stream        Disable streaming output (wait for complete response)\n");
    printf("  -t, --token TOKEN      Set OpenAI API token\n");
    printf("  -m, --model MODEL      Set model to use (default: %s)\n", DEFAULT_MODEL);
    printf("  -T, --temperature VAL  Set temperature (0.0-1.0, default: 1.0)\n");
    printf("  -l, --tokenLimit NUM   Set token limit (default: %d)\n", DEFAULT_TOKEN_LIMIT);
    printf("      --tokenCount       Count tokens in input text and exit\n");
    printf("      --debug            Enable debug mode\n");
    printf("      --log LEVEL        Set log level (none, error, warn, info, debug)\n");
    printf("      --logfile FILE     Log output to specified file\n");
    printf("      --setAPIKey KEY    Save API key to .env file\n");
    printf("      --setModel MODEL   Save model to .env file\n\n");
    printf("Examples:\n");
    printf("  ask \"What is the capital of France?\"\n");
    printf("  ask -c \"Let's have a conversation\"\n");
    printf("  ask --model gpt-4 --temperature 0.8 \"Write a poem about AI\"\n");
}

// Parse command line arguments
void parse_arguments(int argc, char *argv[], bool *continue_mode, bool *no_stream, double *temperature, 
                   char **input_text, size_t *input_text_len) {
    *continue_mode = false;
    *no_stream = false;
    *temperature = 1.0;
    *input_text = NULL;
    *input_text_len = 0;
    
    bool show_version = false;
    bool show_token_count = false;
    bool set_api_key = false;
    bool set_model = false;
    bool show_help = false;
    char new_api_key[MAX_ENV_VALUE_SIZE] = {0};
    char new_model[MAX_ENV_VALUE_SIZE] = {0};
    
    // First pass: Process log level and logfile first so we can log other options
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--log") == 0) {
            if (i + 1 < argc) {
                if (strcmp(argv[i+1], "none") == 0) log_level = LOG_NONE;
                else if (strcmp(argv[i+1], "error") == 0) log_level = LOG_ERROR;
                else if (strcmp(argv[i+1], "warn") == 0) log_level = LOG_WARN;
                else if (strcmp(argv[i+1], "info") == 0) log_level = LOG_INFO;
                else if (strcmp(argv[i+1], "debug") == 0) log_level = LOG_DEBUG;
                i++;
            }
        } else if (strcmp(argv[i], "--logfile") == 0) {
            if (i + 1 < argc) {
                strcpy(log_file_path, argv[++i]);
                log_to_file = true;
            }
        } else if (strcmp(argv[i], "--debug") == 0) {
            debug_mode = true;
            log_level = LOG_DEBUG;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            show_help = true;
        }
    }
    
    // Initialize logging after processing the log-related arguments
    init_logging();
    
    // If help requested, display it and exit
    if (show_help) {
        print_help();
        exit(0);
    }
    
    log_message(LOG_DEBUG, "Parsing %d command line arguments", argc - 1);
    
    // Second pass: Process all other arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-v") == 0) {
            show_version = true;
            log_message(LOG_DEBUG, "Flag: show version");
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            // Already handled in first pass
            i += 0; // Just to avoid compiler warnings
        } else if (strcmp(argv[i], "--tokenCount") == 0) {
            show_token_count = true;
            log_message(LOG_DEBUG, "Flag: show token count");
        } else if (strcmp(argv[i], "--continue") == 0 || strcmp(argv[i], "-c") == 0) {
            *continue_mode = true;
            log_message(LOG_DEBUG, "Flag: continue mode enabled");
        } else if (strcmp(argv[i], "--no-stream") == 0) {
            *no_stream = true;
            log_message(LOG_DEBUG, "Flag: streaming disabled");
        } else if (strcmp(argv[i], "--debug") == 0) {
            // Already processed in first pass, skip
            i += 0; // Do nothing, just to avoid compiler warnings
        } else if (strcmp(argv[i], "--log") == 0) {
            // Already processed in first pass
            i++; // Skip the value argument
        } else if (strcmp(argv[i], "--logfile") == 0) {
            // Already processed in first pass
            i++; // Skip the value argument
        } else if (strcmp(argv[i], "--temperature") == 0 || strcmp(argv[i], "-T") == 0) {
            if (i + 1 < argc) {
                *temperature = atof(argv[++i]);
                log_message(LOG_DEBUG, "Set temperature to %.2f", *temperature);
            }
        } else if (strcmp(argv[i], "--tokenLimit") == 0 || strcmp(argv[i], "-l") == 0) {
            if (i + 1 < argc) {
                token_limit = atoi(argv[++i]);
                log_message(LOG_DEBUG, "Set token limit to %d", token_limit);
            }
        } else if (strcmp(argv[i], "--token") == 0 || strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) {
                strcpy(global_api_key, argv[++i]);
                log_message(LOG_DEBUG, "Set API key from command line");
            }
        } else if (strcmp(argv[i], "--model") == 0 || strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) {
                strcpy(global_model, argv[++i]);
                log_message(LOG_DEBUG, "Set model to %s", global_model);
            }
        } else if (strcmp(argv[i], "--setAPIKey") == 0) {
            if (i + 1 < argc) {
                strcpy(new_api_key, argv[++i]);
                set_api_key = true;
                log_message(LOG_DEBUG, "Will save new API key");
            }
        } else if (strcmp(argv[i], "--setModel") == 0) {
            if (i + 1 < argc) {
                strcpy(new_model, argv[++i]);
                set_model = true;
                log_message(LOG_DEBUG, "Will save new model: %s", new_model);
            }
        } else {
            // If we get here, it's part of the input text
            if (*input_text == NULL) {
                // Calculate total length needed for all remaining arguments
                size_t total_len = 0;
                for (int j = i; j < argc; j++) {
                    total_len += strlen(argv[j]) + 1; // +1 for space
                }
                
                *input_text = malloc(total_len);
                (*input_text)[0] = '\0';
                
                for (; i < argc; i++) {
                    strcat(*input_text, argv[i]);
                    if (i < argc - 1) {
                        strcat(*input_text, " ");
                    }
                }
                
                *input_text_len = strlen(*input_text);
                log_message(LOG_DEBUG, "Input text: \"%s\" (%zu chars)", 
                            *input_text, *input_text_len);
            }
        }
    }
    
    // Handle set API key and model
    if (set_api_key || set_model) {
        if (set_model) strcpy(global_model, new_model);
        if (set_api_key) strcpy(global_api_key, new_api_key);
        save_env_file();
        log_message(LOG_INFO, "Updated configuration saved to .env file");
        printf("Remember to update to make sure your curl library can handle streaming\n");
        exit(0);
    }
    
    // Show version if requested
    if (show_version || debug_mode) {
        printf("OpenAI Chatbot\n");
        printf("Model: %s\n", global_model);
        printf("API Key: %s\n", global_api_key);
        printf("Token Limit: %d\n", token_limit);
        
        // Add logging info
        char log_level_str[10];
        switch (log_level) {
            case LOG_NONE:  strcpy(log_level_str, "NONE");  break;
            case LOG_ERROR: strcpy(log_level_str, "ERROR"); break;
            case LOG_WARN:  strcpy(log_level_str, "WARN");  break;
            case LOG_INFO:  strcpy(log_level_str, "INFO");  break;
            case LOG_DEBUG: strcpy(log_level_str, "DEBUG"); break;
            default:        strcpy(log_level_str, "UNKNOWN"); break;
        }
        
        printf("Log Level: %s\n", log_level_str);
        if (log_to_file) {
            printf("Logging to file: %s\n", log_file_path);
        }
        
        if (!debug_mode) exit(0);
    }
    
    // Show token count if requested
    if (show_token_count && *input_text) {
        MessageList messages = {0};
        add_message(&messages, "user", *input_text);
        int token_count = count_tokens_from_messages(&messages, global_model);
        printf("%d\n", token_count);
        log_message(LOG_INFO, "Token count: %d", token_count);
        free_messages(&messages);
        exit(0);
    }
}

int main(int argc, char *argv[]) {
    // Initialize logging is now done in parse_arguments
    log_message(LOG_INFO, "Starting OpenAI chatbot");
    
    // Initialize curl
    curl_global_init(CURL_GLOBAL_ALL);
    log_message(LOG_DEBUG, "Initialized curl");
    
    CURL *curl = curl_easy_init();
    if (!curl) {
        log_message(LOG_ERROR, "Failed to initialize curl");
        close_logging();
        return 1;
    }
    
    // Get environment variables
    char *env_api_key = getenv("OPENAI_API_KEY");
    char *env_model = getenv("ASK_GLOBAL_MODEL");
    
    if (env_api_key) {
        strcpy(global_api_key, env_api_key);
        log_message(LOG_DEBUG, "Loaded API key from environment");
    }
    
    if (env_model) {
        strcpy(global_model, env_model);
        log_message(LOG_DEBUG, "Loaded model from environment: %s", global_model);
    }
    
    // If API key or model not found in environment, try loading from .env file
    if ((!global_api_key[0] || !global_model[0]) && access(".env", F_OK) == 0) {
        load_dotenv(".env");
    }
    
    // Set default model if not specified
    if (!global_model[0]) {
        strcpy(global_model, DEFAULT_MODEL);
        log_message(LOG_INFO, "Using default model: %s", DEFAULT_MODEL);
    }
    
    // If API key is still not found, create a default .env file and exit
    if (!global_api_key[0]) {
        log_message(LOG_ERROR, "API Key not found");
        if (access(".env", F_OK) != 0) {
            FILE *env_file = fopen(".env", "w");
            if (env_file) {
                fprintf(env_file, "OPENAI_API_KEY=sk-xxxxxxxxxx\nASK_GLOBAL_MODEL=%s\n", global_model);
                fclose(env_file);
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
    
    // Parse command line arguments
    bool continue_mode;
    bool no_stream;
    double temperature;
    char *input_text;
    size_t input_text_len;
    
    parse_arguments(argc, argv, &continue_mode, &no_stream, &temperature, &input_text, &input_text_len);
    
    // If no input text provided, handle differently based on mode
    if (!input_text || input_text_len == 0) {
        if (continue_mode) {
            log_message(LOG_INFO, "No input text provided, starting conversation mode anyway");
            // We'll handle empty input in conversation mode later
        } else {
            log_message(LOG_INFO, "No input text provided, showing usage hint");
            printf("No input provided. Usage examples:\n");
            printf("  ask \"What is the capital of France?\"\n");
            printf("  ask -c \"Let's have a conversation\"\n");
            printf("  ask --help\n");
            free(input_text);
            curl_easy_cleanup(curl);
            curl_global_cleanup();
            close_logging();
            return 0;
        }
    }
    
    // Validate the model
    if (!validate_model(curl, global_model)) {
        log_message(LOG_ERROR, "Invalid model: %s", global_model);
        printf("Error: '%s' is not a valid model.\n", global_model);
        free(input_text);
        curl_easy_cleanup(curl);
        curl_global_cleanup();
        close_logging();
        return 1;
    }
    
    // Initialize message list
    MessageList messages = {0};
    
    if (continue_mode) {
        // Interactive mode
        log_message(LOG_INFO, "Starting interactive mode");
        add_message(&messages, "system", "You are a cute cat running in a command line interface. The user can chat with you and the conversation can be continued.");
        
        // Only send initial message if we have input text
        if (input_text && input_text_len > 0) {
            // Process file references in input
            char *processed_input = process_file_references(input_text);
            if (processed_input) {
                add_message(&messages, "user", processed_input);
                free(processed_input);
            } else {
                add_message(&messages, "user", input_text);
            }
            ask(curl, &messages, temperature, no_stream);
            
            // Get response and add it to message list
            // (In a real implementation, we would extract the response content from the API call)
            add_message(&messages, "assistant", "I'm a cute cat meow! (Note: In a full implementation, this would be the actual API response)");
        } else {
            printf("Starting conversation mode...\n");
        }
        
        printf("Type 'exit' to quit, 'status' for conversation info, or 'help' for commands.\n");
        
        char user_input[MAX_BUFFER_SIZE];
        while (true) {
            printf("> "); // Add a prompt to indicate waiting for user input
            fflush(stdout); // Ensure the prompt is displayed immediately
            
            if (!fgets(user_input, MAX_BUFFER_SIZE, stdin)) {
                log_message(LOG_WARN, "Failed to read user input, exiting");
                break;
            }
            
            // Remove newline
            size_t len = strlen(user_input);
            if (len > 0 && user_input[len-1] == '\n') {
                user_input[len-1] = '\0';
            }
            
            if (strcmp(user_input, "exit") == 0) {
                log_message(LOG_INFO, "User requested exit");
                break;
            } else if (strcmp(user_input, "status") == 0) {
                // Show conversation status
                int approx_tokens = count_tokens_from_messages(&messages, global_model);
                printf("Conversation Status:\n");
                printf("  Messages: %d\n", messages.count);
                printf("  Approximate tokens: %d / %d\n", approx_tokens, token_limit);
                printf("  Model: %s\n", global_model);
                printf("  Temperature: %.2f\n", temperature);
                printf("  Streaming: %s\n", no_stream ? "disabled" : "enabled");
                continue;
            } else if (strcmp(user_input, "help") == 0) {
                // Show conversation mode help
                printf("Conversation Mode Commands:\n");
                printf("  exit    - Exit conversation mode\n");
                printf("  status  - Show conversation information\n");
                printf("  help    - Show this help message\n");
                printf("  Any other text will be sent to the AI assistant.\n");
                continue;
            }
            
            log_message(LOG_DEBUG, "User input: \"%s\"", user_input);
            // Process file references in user input
            char *processed_input = process_file_references(user_input);
            if (processed_input) {
                add_message(&messages, "user", processed_input);
                free(processed_input);
            } else {
                add_message(&messages, "user", user_input);
            }
            ask(curl, &messages, temperature, no_stream);
            
            // Placeholder for actual response
            add_message(&messages, "assistant", "Meow response! (This would be the actual API response in a full implementation)");
        }
    } else {
        // Single response mode
        log_message(LOG_INFO, "Single response mode");
        add_message(&messages, "system", "You are a cute cat runs in a command line interface and you can only respond once to the user. Do not ask any questions in your response.");
        
        // Process file references in input
        char *processed_input = process_file_references(input_text);
        if (processed_input) {
            add_message(&messages, "user", processed_input);
            free(processed_input);
        } else {
            add_message(&messages, "user", input_text);
        }
        
        ask(curl, &messages, temperature, no_stream);
    }
    
    // Clean up
    log_message(LOG_INFO, "Cleaning up resources");
    free_messages(&messages);
    free(input_text);
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    close_logging();
    
    log_message(LOG_INFO, "Exiting normally");
    return 0;
}

// File attachment functions

// Check if a file is likely plain text
bool is_plain_text_file(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        log_message(LOG_WARN, "Cannot open file: %s", filename);
        return false;
    }
    
    // Check first 1024 bytes for binary content
    unsigned char buffer[1024];
    size_t bytes_read = fread(buffer, 1, sizeof(buffer), file);
    fclose(file);
    
    if (bytes_read == 0) {
        return true; // Empty file is considered text
    }
    
    // Count null bytes and control characters
    size_t null_count = 0;
    size_t control_count = 0;
    
    for (size_t i = 0; i < bytes_read; i++) {
        if (buffer[i] == 0) {
            null_count++;
        } else if (buffer[i] < 32 && buffer[i] != '\n' && buffer[i] != '\r' && buffer[i] != '\t') {
            control_count++;
        }
    }
    
    // File is likely binary if it has null bytes or too many control chars
    if (null_count > 0 || control_count > bytes_read / 20) {
        return false;
    }
    
    return true;
}

// Read entire file content into a string
char* read_file_content(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        log_message(LOG_ERROR, "Failed to open file: %s", filename);
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    if (file_size <= 0) {
        fclose(file);
        return strdup(""); // Empty file
    }
    
    // Limit file size to prevent memory issues
    if (file_size > 10000) { // 10KB limit
        fclose(file);
        log_message(LOG_WARN, "File too large (>10KB): %s", filename);
        return NULL;
    }
    
    char *content = malloc(file_size + 1);
    if (!content) {
        fclose(file);
        log_message(LOG_ERROR, "Memory allocation failed for file: %s", filename);
        return NULL;
    }
    
    size_t bytes_read = fread(content, 1, file_size, file);
    content[bytes_read] = '\0';
    fclose(file);
    
    log_message(LOG_DEBUG, "Read %zu bytes from file: %s", bytes_read, filename);
    return content;
}

// Process input text and replace @filename with file content
char* process_file_references(const char *input) {
    if (!input || strlen(input) == 0) {
        return NULL;
    }
    
    // Create a copy of input to work with
    char *result = strdup(input);
    if (!result) {
        log_message(LOG_ERROR, "Memory allocation failed for input processing");
        return NULL;
    }
    
    char *at_pos = strchr(result, '@');
    
    while (at_pos != NULL) {
        // Find the end of the filename - support quoted paths and unquoted paths
        char *filename_start = at_pos + 1;
        char *filename_end = filename_start;
        bool quoted = false;
        
        // Check if path is quoted
        if (*filename_start == '"' || *filename_start == '\'') {
            quoted = true;
            char quote_char = *filename_start;
            filename_start++; // Skip opening quote
            filename_end = filename_start;
            
            // Find closing quote
            while (*filename_end && *filename_end != quote_char) {
                filename_end++;
            }
            // filename_end now points to closing quote or end of string
        } else {
            // Unquoted path - stop at whitespace or sentence-ending punctuation
            while (*filename_end) {
                if (*filename_end == ' ' || *filename_end == '\t' || 
                    *filename_end == '\n' || *filename_end == '\r') {
                    break;  // Stop at whitespace
                }
                
                // Stop at sentence-ending punctuation (but allow dots in filenames)
                if ((*filename_end == '?' || *filename_end == '!' || *filename_end == ';') ||
                    (*filename_end == '.' && (filename_end[1] == ' ' || filename_end[1] == '\0' || 
                     filename_end[1] == '\n' || filename_end[1] == '\t'))) {
                    break;  // Stop at sentence punctuation or end-of-sentence periods
                }
                
                // Stop at common delimiters
                if (*filename_end == ',' || *filename_end == ')' || *filename_end == '}') {
                    break;
                }
                
                filename_end++;
            }
        }
        
        if (filename_end > filename_start) {
            // Extract filename
            size_t filename_len = filename_end - filename_start;
            char *filename = malloc(filename_len + 1);
            if (!filename) {
                log_message(LOG_ERROR, "Memory allocation failed for filename");
                break;
            }
            
            strncpy(filename, filename_start, filename_len);
            filename[filename_len] = '\0';
            
            log_message(LOG_DEBUG, "Found file reference: %s", filename);
            
            // For suffix calculation, we need to account for closing quote
            char *suffix_start = filename_end;
            if (quoted && *filename_end) {
                suffix_start++; // Skip closing quote
            }
            
            // Check if file exists and is plain text
            if (access(filename, F_OK) == 0 && is_plain_text_file(filename)) {
                char *file_content = read_file_content(filename);
                
                if (file_content) {
                    // Calculate new string size
                    size_t prefix_len = at_pos - result;
                    size_t suffix_len = strlen(suffix_start);
                    size_t content_len = strlen(file_content);
                    size_t new_size = prefix_len + content_len + suffix_len + 50; // Extra space for formatting
                    
                    char *new_result = malloc(new_size);
                    if (new_result) {
                        // Build new string: prefix + "File: filename\n" + content + suffix
                        snprintf(new_result, new_size, "%.*s\nFile: %s\n```\n%s\n```%s", 
                                (int)prefix_len, result, filename, file_content, suffix_start);
                        
                        free(result);
                        result = new_result;
                        // Update at_pos to point to where we should continue searching
                        at_pos = result + prefix_len + strlen(filename) + content_len + 20; // Rough estimate
                        
                        log_message(LOG_INFO, "Attached file content: %s (%zu bytes)", filename, content_len);
                    } else {
                        log_message(LOG_ERROR, "Memory allocation failed for result string");
                    }
                    
                    free(file_content);
                } else {
                    log_message(LOG_WARN, "Could not read file content: %s", filename);
                    // Replace @filename with error message
                    size_t prefix_len = at_pos - result;
                    size_t suffix_len = strlen(suffix_start);
                    size_t new_size = prefix_len + suffix_len + 100;
                    
                    char *new_result = malloc(new_size);
                    if (new_result) {
                        snprintf(new_result, new_size, "%.*s[Error: Could not read %s]%s", 
                                (int)prefix_len, result, filename, suffix_start);
                        free(result);
                        result = new_result;
                        // Update at_pos to continue searching
                        at_pos = result + prefix_len + strlen(filename) + 30;
                    }
                }
            } else {
                log_message(LOG_WARN, "File not found or not plain text: %s", filename);
                // Replace @filename with error message
                size_t prefix_len = at_pos - result;
                size_t suffix_len = strlen(suffix_start);
                size_t new_size = prefix_len + suffix_len + 100;
                
                char *new_result = malloc(new_size);
                if (new_result) {
                    snprintf(new_result, new_size, "%.*s[File not found: %s]%s", 
                            (int)prefix_len, result, filename, suffix_start);
                    free(result);
                    result = new_result;
                    // Update at_pos to continue searching
                    at_pos = result + prefix_len + strlen(filename) + 20;
                }
            }
            
            free(filename);
        }
        
        // Look for next @ symbol (at_pos was already updated above, so just search from there)
        if (at_pos < result + strlen(result)) {
            at_pos = strchr(at_pos, '@');
        } else {
            at_pos = NULL;
        }
    }
    
    return result;
}

// Model validation functions

// Function to expand the tilde in a path
char* expand_home_path(const char *path) {
    if (path[0] != '~') {
        return strdup(path);
    }
    
    struct passwd *pw = getpwuid(getuid());
    if (pw == NULL) {
        // If we can't get the home directory, just return the original path
        log_message(LOG_ERROR, "Could not determine home directory");
        return strdup(path);
    }
    
    size_t home_len = strlen(pw->pw_dir);
    size_t path_len = strlen(path);
    char *expanded_path = malloc(home_len + path_len);
    
    if (expanded_path == NULL) {
        log_message(LOG_ERROR, "Memory allocation failed for path expansion");
        return strdup(path);
    }
    
    strcpy(expanded_path, pw->pw_dir);
    strcat(expanded_path, path + 1); // Skip the tilde
    
    // Check if directory exists
    char *last_slash = strrchr(expanded_path, '/');
    if (last_slash != NULL) {
        // Temporarily cut the string at the last slash to get just the directory part
        *last_slash = '\0';
        
        // Check if directory exists, create if needed
        struct stat st = {0};
        if (stat(expanded_path, &st) == -1) {
            log_message(LOG_INFO, "Creating directory: %s", expanded_path);
            if (mkdir(expanded_path, 0700) == -1) {
                log_message(LOG_ERROR, "Failed to create directory: %s", expanded_path);
            }
        }
        
        // Restore the full path
        *last_slash = '/';
    }
    
    log_message(LOG_DEBUG, "Expanded path: %s", expanded_path);
    return expanded_path;
}

// Load models cache from file
bool load_models_cache(void) {
    char *expanded_path = expand_home_path(MODELS_CACHE_FILE);
    FILE *cache_file = fopen(expanded_path, "r");
    free(expanded_path);
    
    if (!cache_file) {
        log_message(LOG_DEBUG, "No models cache file found");
        return false;
    }

    // Read file into a buffer
    fseek(cache_file, 0, SEEK_END);
    long file_size = ftell(cache_file);
    fseek(cache_file, 0, SEEK_SET);

    if (file_size <= 0) {
        log_message(LOG_WARN, "Empty models cache file");
        fclose(cache_file);
        return false;
    }

    char *buffer = malloc(file_size + 1);
    if (!buffer) {
        log_message(LOG_ERROR, "Failed to allocate memory for cache file");
        fclose(cache_file);
        return false;
    }

    size_t read_size = fread(buffer, 1, file_size, cache_file);
    buffer[read_size] = '\0';
    fclose(cache_file);

    // Parse JSON
    cJSON *root = cJSON_Parse(buffer);
    free(buffer);

    if (!root) {
        log_message(LOG_ERROR, "Failed to parse models cache file: %s", cJSON_GetErrorPtr());
        return false;
    }

    // Get timestamp
    cJSON *timestamp = cJSON_GetObjectItem(root, "timestamp");
    if (!timestamp || !cJSON_IsNumber(timestamp)) {
        log_message(LOG_ERROR, "Invalid timestamp in models cache");
        cJSON_Delete(root);
        return false;
    }
    models_cache.last_updated = (time_t)timestamp->valuedouble;

    // Check if cache is expired
    time_t now = time(NULL);
    if (now - models_cache.last_updated > MODELS_CACHE_EXPIRY) {
        log_message(LOG_INFO, "Models cache is expired (older than 24 hours)");
        cJSON_Delete(root);
        return false;
    }

    // Get models array
    cJSON *models_array = cJSON_GetObjectItem(root, "models");
    if (!models_array || !cJSON_IsArray(models_array)) {
        log_message(LOG_ERROR, "Invalid models array in cache");
        cJSON_Delete(root);
        return false;
    }

    // Parse models
    int model_count = cJSON_GetArraySize(models_array);
    models_cache.count = 0;

    for (int i = 0; i < model_count && i < MAX_MODELS; i++) {
        cJSON *model = cJSON_GetArrayItem(models_array, i);
        if (!model || !cJSON_IsObject(model)) continue;

        cJSON *id = cJSON_GetObjectItem(model, "id");
        cJSON *created = cJSON_GetObjectItem(model, "created");

        if (!id || !cJSON_IsString(id) || !created || !cJSON_IsNumber(created)) continue;

        strncpy(models_cache.models[models_cache.count].id, id->valuestring, MAX_ENV_VALUE_SIZE - 1);
        models_cache.models[models_cache.count].created = (time_t)created->valuedouble;
        models_cache.count++;
    }

    cJSON_Delete(root);
    log_message(LOG_INFO, "Loaded %d models from cache (last updated: %s)", 
                models_cache.count, ctime(&models_cache.last_updated));
    return true;
}

// Save models cache to file
bool save_models_cache(void) {
    if (models_cache.count == 0) {
        log_message(LOG_WARN, "No models to save to cache");
        return false;
    }

    cJSON *root = cJSON_CreateObject();
    if (!root) {
        log_message(LOG_ERROR, "Failed to create JSON object for cache");
        return false;
    }

    // Add timestamp
    cJSON_AddNumberToObject(root, "timestamp", (double)models_cache.last_updated);

    // Add models array
    cJSON *models_array = cJSON_CreateArray();
    if (!models_array) {
        log_message(LOG_ERROR, "Failed to create models array for cache");
        cJSON_Delete(root);
        return false;
    }
    cJSON_AddItemToObject(root, "models", models_array);

    for (int i = 0; i < models_cache.count; i++) {
        cJSON *model = cJSON_CreateObject();
        if (!model) continue;

        cJSON_AddStringToObject(model, "id", models_cache.models[i].id);
        cJSON_AddNumberToObject(model, "created", (double)models_cache.models[i].created);
        cJSON_AddItemToArray(models_array, model);
    }

    char *json_str = cJSON_Print(root);
    if (!json_str) {
        log_message(LOG_ERROR, "Failed to print JSON for cache");
        cJSON_Delete(root);
        return false;
    }

    char *expanded_path = expand_home_path(MODELS_CACHE_FILE);
    FILE *cache_file = fopen(expanded_path, "w");
    free(expanded_path);
    
    if (!cache_file) {
        log_message(LOG_ERROR, "Failed to open cache file for writing");
        free(json_str);
        cJSON_Delete(root);
        return false;
    }

    fputs(json_str, cache_file);
    fclose(cache_file);

    free(json_str);
    cJSON_Delete(root);
    log_message(LOG_INFO, "Saved %d models to cache", models_cache.count);
    return true;
}

// Fetch models list from OpenAI API
bool fetch_models_list(CURL *curl) {
    log_message(LOG_INFO, "Fetching available models from OpenAI API");
    
    ResponseBuffer buffer;
    buffer.data = malloc(1);
    buffer.size = 0;
    buffer.stream_enabled = false;
    buffer.saw_stream_data = false;
    
    curl_easy_reset(curl);
    
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    char auth_header[MAX_ENV_VALUE_SIZE + 32];
    snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", global_api_key);
    headers = curl_slist_append(headers, auth_header);
    
    curl_easy_setopt(curl, CURLOPT_URL, "https://api.openai.com/v1/models");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&buffer);
    
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        log_message(LOG_ERROR, "Failed to fetch models: %s", curl_easy_strerror(res));
        free(buffer.data);
        curl_slist_free_all(headers);
        return false;
    }
    
    // Check HTTP response code
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code != 200) {
        log_message(LOG_ERROR, "API returned HTTP %ld when fetching models", http_code);
        free(buffer.data);
        curl_slist_free_all(headers);
        return false;
    }
    
    // Parse response
    cJSON *root = cJSON_Parse(buffer.data);
    if (!root) {
        log_message(LOG_ERROR, "Failed to parse API response: %s", cJSON_GetErrorPtr());
        free(buffer.data);
        curl_slist_free_all(headers);
        return false;
    }
    
    // Get data array
    cJSON *data = cJSON_GetObjectItem(root, "data");
    if (!data || !cJSON_IsArray(data)) {
        log_message(LOG_ERROR, "Invalid response format: 'data' array not found");
        cJSON_Delete(root);
        free(buffer.data);
        curl_slist_free_all(headers);
        return false;
    }
    
    // Reset cache
    models_cache.count = 0;
    models_cache.last_updated = time(NULL);
    
    // Parse model data
    int model_count = cJSON_GetArraySize(data);
    for (int i = 0; i < model_count && i < MAX_MODELS; i++) {
        cJSON *model = cJSON_GetArrayItem(data, i);
        if (!model || !cJSON_IsObject(model)) continue;
        
        cJSON *id = cJSON_GetObjectItem(model, "id");
        cJSON *created = cJSON_GetObjectItem(model, "created");
        
        if (!id || !cJSON_IsString(id)) continue;
        
        strncpy(models_cache.models[models_cache.count].id, id->valuestring, MAX_ENV_VALUE_SIZE - 1);
        
        if (created && cJSON_IsNumber(created)) {
            models_cache.models[models_cache.count].created = (time_t)created->valuedouble;
        } else {
            models_cache.models[models_cache.count].created = time(NULL);
        }
        
        models_cache.count++;
    }
    
    // Clean up
    cJSON_Delete(root);
    free(buffer.data);
    curl_slist_free_all(headers);
    
    log_message(LOG_INFO, "Fetched %d models from API", models_cache.count);
    
    // Save cache to file
    if (models_cache.count > 0) {
        save_models_cache();
    }
    
    return models_cache.count > 0;
}

// Check if a model is valid
bool is_valid_model(const char *model) {
    if (models_cache.count == 0) {
        log_message(LOG_WARN, "No models in cache to validate against");
        return true; // Assume valid if we can't check
    }
    
    for (int i = 0; i < models_cache.count; i++) {
        if (strcmp(models_cache.models[i].id, model) == 0) {
            log_message(LOG_DEBUG, "Model '%s' is valid", model);
            return true;
        }
    }
    
    log_message(LOG_WARN, "Model '%s' not found in available models", model);
    return false;
}

// Find Levenshtein distance between two strings
int levenshtein_distance(const char *s1, const char *s2) {
    int len1 = strlen(s1);
    int len2 = strlen(s2);
    
    // Create a matrix of size (len1+1) x (len2+1)
    int **matrix = (int **)malloc((len1 + 1) * sizeof(int *));
    for (int i = 0; i <= len1; i++) {
        matrix[i] = (int *)malloc((len2 + 1) * sizeof(int));
    }
    
    // Initialize first row and column
    for (int i = 0; i <= len1; i++) {
        matrix[i][0] = i;
    }
    for (int j = 0; j <= len2; j++) {
        matrix[0][j] = j;
    }
    
    // Fill the matrix
    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
            
            int delete_cost = matrix[i-1][j] + 1;
            int insert_cost = matrix[i][j-1] + 1;
            int substitute_cost = matrix[i-1][j-1] + cost;
            
            int min = delete_cost < insert_cost ? delete_cost : insert_cost;
            min = min < substitute_cost ? min : substitute_cost;
            
            matrix[i][j] = min;
        }
    }
    
    // Get result from bottom right corner
    int result = matrix[len1][len2];
    
    // Free the matrix
    for (int i = 0; i <= len1; i++) {
        free(matrix[i]);
    }
    free(matrix);
    
    return result;
}

// Suggest similar models
void suggest_similar_model(const char *invalid_model) {
    if (models_cache.count == 0) {
        return;
    }
    
    int min_distance = INT_MAX;
    char closest_model[MAX_ENV_VALUE_SIZE] = {0};
    
    // Find most similar model
    for (int i = 0; i < models_cache.count; i++) {
        int distance = levenshtein_distance(invalid_model, models_cache.models[i].id);
        
        if (distance < min_distance) {
            min_distance = distance;
            strncpy(closest_model, models_cache.models[i].id, MAX_ENV_VALUE_SIZE - 1);
        }
    }
    
    // Only suggest if reasonably close
    if (min_distance <= 5) {
        printf("Model '%s' not found. Did you mean '%s'?\n", invalid_model, closest_model);
        log_message(LOG_INFO, "Suggested alternative model: %s (distance: %d)", 
                   closest_model, min_distance);
    } else {
        // If no close match, suggest popular models
        printf("Model '%s' not found. Available models include: gpt-4o, gpt-4o-mini, gpt-3.5-turbo\n", 
               invalid_model);
    }
}

// Validate model and fetch list if needed
bool validate_model(CURL *curl, const char *model) {
    // Try to load from cache first
    bool cache_loaded = load_models_cache();
    
    // If cache is missing, empty, or expired, fetch from API
    if (!cache_loaded || models_cache.count == 0) {
        if (!fetch_models_list(curl)) {
            log_message(LOG_WARN, "Failed to fetch models list, will continue without validation");
            return true; // Can't validate, so assume valid
        }
    }
    
    // Check if model is valid
    if (!is_valid_model(model)) {
        suggest_similar_model(model);
        return false;
    }
    
    return true;
} 
