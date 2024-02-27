# ASK
"ask" is a simple command line tool for asking questions and getting answers.
It is designed to be used in terminal for quick and easy access to information.
The backend is powered by OpenAI's GPT-4, which is a powerful language model that can answer a wide range of questions.
That means you need to provide an API key to use this tool.

## Installation
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

```text
usage: main.py [-h] [--token TOKEN] [--model MODEL] [--version] [--tokenCount]
               [--stream] [--temperture TEMPERTURE] [--tokenLimit TOKENLIMIT]
               [string ...]

OpenAI Chatbot

positional arguments:
  string                Questions

options:
  -h, --help            show this help message and exit
  --token TOKEN, -t TOKEN
                        Set the token for the chatbot
  --model MODEL, -m MODEL
                        Set the model for the chatbot
  --version, -v         Show the version of the chatbot
  --tokenCount          Set the token for the chatbot
  --stream, -s          Don't close chat completion and wait for more input
  --temperture TEMPERTURE, -T TEMPERTURE
                        Set the temperture for the chatbot
  --tokenLimit TOKENLIMIT, -l TOKENLIMIT
                        Set the token limit for the chatbot
```

# Note
The ANY_THING is the question you want to ask the chatbot and should not using period or question mark at the end of the question.
```bash
ask What is the capital of France
```
```text
The capital of France is Paris.
```
If you want to use the symbol, you have to pass the question in the double quote.
```bash
ask "What is the capital of France?"
```
```text
The capital of France is Paris.
```

The ask command will not render the question for markdown, so the display is plain text.
if you want to use the markdown, you will have to do it manually.
```bash
ask "What is the capital of France?" > README.md
open README.md
```

# Contributing
Any


