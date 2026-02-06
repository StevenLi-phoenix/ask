CXX      = g++
CC       = gcc
CXXFLAGS = -std=c++17 -Wall -Wextra
CFLAGS   = -std=c11 -Wall -Wextra
LDFLAGS  =
LIBS     = -lcurl

BIN = ask
SRC = main.cpp
CJSON_SRC = vendor/cjson/cJSON.c
CJSON_OBJ = cJSON.o

all: $(BIN)

$(CJSON_OBJ): $(CJSON_SRC)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BIN): $(SRC) $(CJSON_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(BIN) $(SRC) $(CJSON_OBJ) $(LIBS)

clean:
	rm -f $(BIN) $(CJSON_OBJ)

.PHONY: all clean
