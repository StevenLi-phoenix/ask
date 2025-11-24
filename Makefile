CXX=g++
CXXFLAGS=-std=c++17 -Wall -Wextra -I/opt/homebrew/include
LDFLAGS=-L/opt/homebrew/lib
LIBS=-lcurl -lcjson

BIN=ask
SRC=main.cpp

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(BIN) $(SRC) $(LIBS)

clean:
	rm -f $(BIN)

.PHONY: all clean
