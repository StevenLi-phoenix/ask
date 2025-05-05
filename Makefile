CC=gcc
CFLAGS=-Wall -Wextra -I/opt/homebrew/include
LDFLAGS=-L/opt/homebrew/lib
LIBS=-lcurl -lcjson

all: ask

ask: main.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o ask main.c $(LIBS)

clean:
	rm -f ask

.PHONY: all clean 