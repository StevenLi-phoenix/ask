#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>
#include <time.h>
#include <stdarg.h>

#define DEFAULT_MODEL "gpt-4o-mini"
#define MAX_ENV_VALUE_SIZE 1024
#define MAX_BUFFER_SIZE 8192
#define DEFAULT_TOKEN_LIMIT 128000
#define MAX_MESSAGES 100

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
} ResponseBuffer;

typedef struct {
    char *role;
    char *content;
} Message;

typedef struct {
    Message messages[MAX_MESSAGES];
    int count;
} MessageList;

// Global variables
char global_api_key[MAX_ENV_VALUE_SIZE] = {0};
char global_model[MAX_ENV_VALUE_SIZE] = {0};
int token_limit = DEFAULT_TOKEN_LIMIT;
bool debug_mode = false;
LogLevel log_level = LOG_INFO;
FILE *log_file = NULL;
bool log_to_file = false;
char log_file_path[MAX_ENV_VALUE_SIZE] = "ask.log";

// Function prototypes
void load_dotenv(const char *filename);
size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp);
int count_tokens_from_messages(MessageList *messages, const char *model);
void ask(CURL *curl, MessageList *messages, double temperature);
void add_message(MessageList *messages, const char *role, const char *content);
void free_messages(MessageList *messages);
void parse_arguments(int argc, char *argv[], bool *stream_mode, double *temperature, 
                    char **input_text, size_t *input_text_len);
void save_env_file();
void init_logging(void);
void close_logging(void);
void log_message(LogLevel level, const char *format, ...);
void print_help(void);

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
    
    char *ptr = realloc(mem->data, mem->size + realsize + 1);
    if (!ptr) {
        log_message(LOG_ERROR, "Out of memory when processing API response");
        return 0;
    }
    
    mem->data = ptr;
    memcpy(&(mem->data[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->data[mem->size] = 0;
    
    log_message(LOG_DEBUG, "Received %zu bytes from API", realsize);
    
    return realsize;
}

// Approximate token counting function
// In a real implementation, this would need to use a proper tokenizer like tiktoken
int count_tokens_from_messages(MessageList *messages, const char *model) {
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
void ask(CURL *curl, MessageList *messages, double temperature) {
    if (messages->count == 0) {
        log_message(LOG_WARN, "No messages to send to API");
        return;
    }
    
    log_message(LOG_INFO, "Sending request to OpenAI API (model: %s, temp: %.2f)", 
                global_model, temperature);
    
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
    cJSON_AddBoolToObject(root, "stream", true);
    
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
    
    ResponseBuffer buffer;
    buffer.data = malloc(1);
    buffer.size = 0;
    
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&buffer);
    
    // Perform the request
    log_message(LOG_INFO, "Sending request to API...");
    CURLcode res = curl_easy_perform(curl);
    
    if (res != CURLE_OK) {
        log_message(LOG_ERROR, "curl_easy_perform() failed: %s", curl_easy_strerror(res));
    } else {
        log_message(LOG_INFO, "Request completed successfully");
        log_message(LOG_DEBUG, "Response size: %zu bytes", buffer.size);
        
        // Parse the streaming response and extract content
        char full_response[MAX_BUFFER_SIZE] = {0};
        char *line_start, *line_end;
        line_start = buffer.data;
        
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
                                // Output only the content
                                printf("%s", content->valuestring);
                                fflush(stdout);
                                strcat(full_response, content->valuestring);
                            }
                        }
                    }
                    cJSON_Delete(json);
                }
            }
            
            line_start = line_end + 2;  // Move to next line after the \n\n
        }
        printf("\n");  // End the response with a newline
    }
    
    // In a real implementation, we would return the full extracted response
    
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
    printf("  -s, --stream           Enable conversation mode (supports multiple exchanges)\n");
    printf("  -t, --token TOKEN      Set OpenAI API token\n");
    printf("  -m, --model MODEL      Set model to use (default: %s)\n", DEFAULT_MODEL);
    printf("  -T, --temperature VAL  Set temperature (0.0-1.0, default: 0.7)\n");
    printf("  -l, --tokenLimit NUM   Set token limit (default: %d)\n", DEFAULT_TOKEN_LIMIT);
    printf("      --tokenCount       Count tokens in input text and exit\n");
    printf("      --debug            Enable debug mode\n");
    printf("      --log LEVEL        Set log level (none, error, warn, info, debug)\n");
    printf("      --logfile FILE     Log output to specified file\n");
    printf("      --setAPIKey KEY    Save API key to .env file\n");
    printf("      --setModel MODEL   Save model to .env file\n\n");
    printf("Examples:\n");
    printf("  ask \"What is the capital of France?\"\n");
    printf("  ask -s \"Let's have a conversation\"\n");
    printf("  ask --model gpt-4 --temperature 0.8 \"Write a poem about AI\"\n");
}

// Parse command line arguments
void parse_arguments(int argc, char *argv[], bool *stream_mode, double *temperature, 
                   char **input_text, size_t *input_text_len) {
    *stream_mode = false;
    *temperature = 0.7;
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
        } else if (strcmp(argv[i], "--stream") == 0 || strcmp(argv[i], "-s") == 0) {
            *stream_mode = true;
            log_message(LOG_DEBUG, "Flag: stream mode enabled");
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
    bool stream_mode;
    double temperature;
    char *input_text;
    size_t input_text_len;
    
    parse_arguments(argc, argv, &stream_mode, &temperature, &input_text, &input_text_len);
    
    // If no input text provided and we're not just showing version or setting API key, exit
    if (!input_text || input_text_len == 0) {
        log_message(LOG_INFO, "No input text provided, exiting");
        free(input_text);
        curl_easy_cleanup(curl);
        curl_global_cleanup();
        close_logging();
        return 0;
    }
    
    // Initialize message list
    MessageList messages = {0};
    
    if (stream_mode) {
        // Interactive mode
        log_message(LOG_INFO, "Starting interactive mode");
        add_message(&messages, "system", "You are a cute cat running in a command line interface. The user can chat with you and the conversation can be continued.");
        add_message(&messages, "user", input_text);
        
        ask(curl, &messages, temperature);
        
        // Get response and add it to message list
        // (In a real implementation, we would extract the response content from the API call)
        add_message(&messages, "assistant", "I'm a cute cat meow! (Note: In a full implementation, this would be the actual API response)");
        
        printf("Type 'exit' to quit.\n");
        
        char user_input[MAX_BUFFER_SIZE];
        while (true) {
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
            }
            
            log_message(LOG_DEBUG, "User input: \"%s\"", user_input);
            add_message(&messages, "user", user_input);
            ask(curl, &messages, temperature);
            
            // Placeholder for actual response
            add_message(&messages, "assistant", "Meow response! (This would be the actual API response in a full implementation)");
        }
    } else {
        // Single response mode
        log_message(LOG_INFO, "Single response mode");
        add_message(&messages, "system", "You are a cute cat runs in a command line interface and you can only respond once to the user. Do not ask any questions in your response.");
        add_message(&messages, "user", input_text);
        
        ask(curl, &messages, temperature);
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