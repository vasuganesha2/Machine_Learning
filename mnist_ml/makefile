CC = g++

MNIST_ML_ROOT := $(PWD)
LD_LIBRARY_PATH := $(MNIST_ML_ROOT)/lib
INCLUDE_DIR := $(MNIST_ML_ROOT)/include
SRC = $(MNIST_ML_ROOT)/src

CFLAGS := -std=c++11 -shared -g
LIB_DATA := libdata.so
EXEC := main_program  # Define the name of the executable

DATA_TYPE := double
DATA_TYPE_LABEL := uint8_t

all: $(LIB_DATA) $(EXEC)

$(LIB_DATA): libdir objdir obj/data_handler.o obj/data.o obj/common.o
	$(CC) $(CFLAGS) -o $(MNIST_ML_ROOT)/lib/$(LIB_DATA) obj/*.o

$(EXEC): $(SRC)/data_handler.cpp
	$(CC) -std=c++11 -g -o $(EXEC) $(SRC)/data_handler.cpp -L$(MNIST_ML_ROOT)/lib -ldata -I$(INCLUDE_DIR)

libdir:
	mkdir -p $(MNIST_ML_ROOT)/lib

objdir:
	mkdir -p $(MNIST_ML_ROOT)/obj

obj/data_handler.o: $(SRC)/data_handler.tpp
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -D DATA_TYPE=$(DATA_TYPE) -c $(SRC)/data_handler.tpp -o obj/data_handler.o

obj/data.o: $(SRC)/data.tpp
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -D DATA_TYPE=$(DATA_TYPE) -c $(SRC)/data.tpp -o obj/data.o

obj/common.o: $(SRC)/common.cpp
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -D DATA_TYPE=$(DATA_TYPE) -c $(SRC)/common.cpp -o obj/common.o

clean:
	rm -rf $(LD_LIBRARY_PATH)
	rm -rf $(MNIST_ML_ROOT)/obj
	rm -rf $(EXEC)


