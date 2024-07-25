import os
import time
import argparse

import openai
import tiktoken
import dotenv
from sys import stdout

ASK_GLOBAL_APIKEY=os.getenv("OPENAI_API_KEY")
if not ASK_GLOBAL_APIKEY and os.path.exists(".env"):
        dotenv.load_dotenv(".env")

ASK_GLOBAL_MODEL=os.getenv("ASK_GLOBAL_MODEL")
if not ASK_GLOBAL_APIKEY:
    ASK_GLOBAL_APIKEY=os.getenv("OPENAI_API_KEY")
if not ASK_GLOBAL_MODEL:
    ASK_GLOBAL_MODEL = "gpt-4o-mini"
if not ASK_GLOBAL_APIKEY:
    if not os.path.exists(".env"):
        open(".env", "w").write(f"OPENAI_API_KEY=sk-xxxxxxxxxx\nASK_GLOBAL_MODEL={ASK_GLOBAL_MODEL}\n")
    exit("API Key not found")
tokenLimit = 128000


def num_tokens_from_messages(messages, model=ASK_GLOBAL_MODEL):
    """
    @openai-cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


# Usage
# cmd line > python main.py
# ask <text> ---- send a message to the chatbot
# ask --setToken <token> ---- set the token for the chatbot
# Example: ask --setapikey sk-xxxxxxxxxx --model gpt-4-0125-preview What is the capital of France
# Example: ask What is the capital of France

def parser():
    parser = argparse.ArgumentParser(description="OpenAI Chatbot")
    parser.add_argument("--token", "-t", help="Set the token for the chatbot", type=str, default=ASK_GLOBAL_APIKEY,
                        required=False)
    parser.add_argument("--model", "-m", help="Set the model for the chatbot", type=str, default=ASK_GLOBAL_MODEL, required=False)
    parser.add_argument("--version", "-v", help="Show the version of the chatbot", action="store_true")
    parser.add_argument("--tokenCount", help="Set the token for the chatbot", action="store_true")
    parser.add_argument("--stream", "-s", help="Don't close chat completion and wait for more input", action="store_true", required=False)
    parser.add_argument("--temperature", "-T", help="Set the temperature for the chatbot", type=float, default=0.7, required=False)
    parser.add_argument("--tokenLimit", "-l", help="Set the token limit for the chatbot", type=int, default=tokenLimit, required=False)
    parser.add_argument("--setapikey", help="Set the token of openai apikey", type=str, required=False)
    parser.add_argument("--setmodel", help="Set the model for all chats", type=str, required=False)
    parser.add_argument("string", help="Questions", type=str, nargs='*')
    return parser.parse_args()


def ask(client: openai.OpenAI, messages: list, temperature=0.7):
    if len(messages) == 0:
        return
    if len(messages) > 1:
        while num_tokens_from_messages(messages) + 100 > tokenLimit:
            messages.pop(0)
    stream = client.chat.completions.create(model=ASK_GLOBAL_MODEL, messages=messages, stream=True, temperature=temperature)
    buffer = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
            stdout.flush()
            buffer += chunk.choices[0].delta.content
    print()
    return buffer


def main():
    global ASK_GLOBAL_APIKEY, ASK_GLOBAL_MODEL
    args = parser()
    if args.setmodel or args.setapikey:
        if args.setmodel: ASK_GLOBAL_MODEL = args.setmodel
        if args.setapikey: ASK_GLOBAL_APIKEY = args.setapikey
        open(".env", "w").write(f"OPENAI_API_KEY={ASK_GLOBAL_APIKEY}\nASK_GLOBAL_MODEL={ASK_GLOBAL_MODEL}\n")
        print("Remember to upgrade tiktoken with newer model")
        return
    if args.version:
        print("OpenAI Chatbot")
        print("Model: ", ASK_GLOBAL_MODEL)
        print("API Key: ", ASK_GLOBAL_APIKEY)
        print("Token Limit: ", tokenLimit)
        return
    if args.tokenCount:
        print(num_tokens_from_messages([{"role": "user", "content": " ".join(args.string)}]))
        return
    client = openai.OpenAI(api_key=args.token)
    userchat = [
        {"role": "user", "content": " ".join(args.string)}
    ]
    response = ask(client, userchat, args.temperature)
    userchat.append({"role": "assistant", "content": response})
    if args.stream:
        while True:
            userInput = input()
            if userInput == "exit":
                break
            userchat.append({"role": "user", "content": userInput})
            response = ask(client, userchat)
            userchat.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    main()
