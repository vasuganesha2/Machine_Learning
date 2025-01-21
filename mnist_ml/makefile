# # Compiler
# CXX = g++

# # Compiler flags
# CXXFLAGS = -Wall -std=c++17 -Iinclude

# # Source and Object directories
# SRC_DIR = src
# OBJ_DIR = obj

# # Source files and Object files
# SOURCES = $(SRC_DIR)/data_handler.cpp $(SRC_DIR)/data.cpp
# OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SOURCES))

# # Output executable
# TARGET = program

# # Default target
# all: $(TARGET)

# # Rule to build the target executable
# $(TARGET): $(OBJECTS)
# 	$(CXX) $(CXXFLAGS) $^ -o $@

# # Rule to compile source files into object files
# $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

# # Create the obj directory if it doesn't exist
# $(OBJ_DIR):
# 	mkdir -p $(OBJ_DIR)

# # Clean up build artifacts
# clean:
# 	rm -rf $(OBJ_DIR) $(TARGET)

# # Phony targets
# .PHONY: all clean


# CC=g++

# SRC := $(MNIST_ML_ROOT)/KNN/src
# CFLAGS := -std=c++17 -DEUCLID

# INCLUDE_DIR := $(MNIST_ML_ROOT)/KNN/include 

# all : main

# main : $(SRC)/knn.cpp ${CC} ${CFLAGS} ${SRC}/knn.cpp -o -main -L${MNIST_ML_ROOT}/lib/ -I${INCLUDE_DIR}\
# -I${MNIST_ML_ROOT}/include -ldata

# clean :
# 	rm main



CC = g++

INCLUDE_DIR := $(MNIST_ML_ROOT)/include
SRC = $(MNIST_ML_ROOT)/src

CFLAGS := -std=c++11 -shared -g
LIB_DATA := libdata.so
EXEC := main_program  # Define the name of the executable

all: $(LIB_DATA) $(EXEC)

$(LIB_DATA): libdir objdir obj/data_handler.o obj/data.o
	$(CC) $(CFLAGS) -o $(MNIST_ML_ROOT)/lib/$(LIB_DATA) obj/*.o

$(EXEC): $(SRC)/data_handler.cpp
	$(CC) -std=c++11 -g -o $(EXEC) $(SRC)/data_handler.cpp -L$(MNIST_ML_ROOT)/lib -ldata -I$(INCLUDE_DIR)

libdir:
	mkdir -p $(MNIST_ML_ROOT)/lib

objdir:
	mkdir -p $(MNIST_ML_ROOT)/obj

obj/data_handler.o: $(SRC)/data_handler.cpp
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -c $(SRC)/data_handler.cpp -o obj/data_handler.o

obj/data.o: $(SRC)/data.cpp
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -c $(SRC)/data.cpp -o obj/data.o

clean:
	rm -rf $(MNIST_ML_ROOT)/lib
	rm -rf $(MNIST_ML_ROOT)/obj
	rm -rf $(EXEC)

