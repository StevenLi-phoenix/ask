# ASK
"ask" is a simple command line tool for asking questions and getting answers.
It is designed to be used in terminal for quick and easy access to information.
The backend is powered by OpenAI's GPT-4, which is a powerful language model that can answer a wide range of questions.
That means you need to provide an API key to use this tool.

## Python Implementation

### Installation
You can install "ask" using pip:
```bash
pip install -r requirements.txt
```
Set OPENAI_API_KEY environment variable to your API key:
```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```
Or you can set it in the .env file located in the root directory of the project:
```bash
echo "ASK_GLOBAL_MODEL=\"gpt-4-0125-preview\"
OPENAI_API_KEY=\"YOUR_OPENAI_API_KEY\"" > .env
```

### (Optional) You can also add the "ask" script to your PATH:
modify the ask.sh file to point to the main.py file in the root directory of the project:
```bash
#!/bin/bash
cd /path/to/ask/ && python3 /path/to/ask/main.py $@
```
Then make the script executable:
```bash
chmod +x /path/to/ask/ask.sh
```
Then make a symbolic link to the script in /usr/local/bin:
```bash
sudo ln -s /path/to/ask/ask.sh /usr/local/bin/ask
```

## Usage
To use "ask", you need to provide an API key. You can get one from OpenAI's website.
Once you have the API key, you can use it like this:
```bash
ask ANY_THING
```

The default model is "gpt-4" and the default temperature is 0.7. You can change these settings using command line options:
```bash
ask --model gpt-4 --temperature 0.7 ANY_THING
```

The default token limit is 128000. You can change this using the --tokens option:
```bash
ask -l 16000 ANY_THING
```

Use the --stream option to keep the chat completion open and wait for more input:
```bash
ask -s
```

for more information, you can use the --help option:
```bash
ask --help
```

# C Implementation

A C version of the program is also available.

## Requirements

- GCC or compatible C compiler
- libcurl development libraries
- cJSON library

On macOS, you can install the required libraries with:

```bash
brew install curl cjson
```

On Ubuntu/Debian:

```bash
sudo apt-get install libcurl4-openssl-dev libcjson-dev
```

## Building

To build the C version, simply run:

```bash
make
```

This will produce an executable called `ask`.

## Setup

The C version uses the same configuration as the Python version. You can set your OpenAI API key in several ways:

1. Create a `.env` file with the following content:
   ```
   OPENAI_API_KEY=your_api_key_here
   ASK_GLOBAL_MODEL=gpt-4o-mini
   ```

2. Set environment variables:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   export ASK_GLOBAL_MODEL=gpt-4o-mini
   ```

3. Use command-line options to set them:
   ```bash
   ./ask --setAPIKey your_api_key_here --setModel gpt-4o-mini
   ```

## Usage

The C version supports the same command-line options as the Python version:

```bash
./ask What is the capital of France?
```

### Interactive Mode

```bash
./ask -s Tell me a story
```

In interactive mode, type "exit" to quit.

### Other Options

- `--version` or `-v`: Show version information
- `--tokenCount`: Display approximate token count for your message
- `--temperature` or `-T`: Set temperature (default: 0.7)
- `--tokenLimit` or `-l`: Set maximum token limit (default: 128000)
- `--token` or `-t`: Set API token for just this session
- `--model` or `-m`: Set model for just this session
- `--debug`: Show debug information

## Notes

- Token counting in the C version is approximate and not as accurate as tiktoken in Python
- The C version uses libcurl for HTTP requests and cJSON for JSON parsing

# Contributing
Any


