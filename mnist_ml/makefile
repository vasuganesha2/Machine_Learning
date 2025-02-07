CC = g++

MNIST_ML_ROOT := $(PWD)
LD_LIBRARY_PATH := $(MNIST_ML_ROOT)/lib
INCLUDE_DIR := $(MNIST_ML_ROOT)/include
SRC = $(MNIST_ML_ROOT)/src
QP_SRC = $(MNIST_ML_ROOT)/QP

CFLAGS := -std=c++20 -shared -g
LIB_DATA := libdata.so
LIB_QP := libqp.so  # Name of the new shared library for QP
EXEC := main_program  # Define the name of the executable

DATA_TYPE := uint8_t

all: $(LIB_DATA) $(LIB_QP) $(EXEC)

$(LIB_DATA): libdir objdir obj/data_handler.o obj/data.o obj/common.o
	$(CC) $(CFLAGS) -o $(MNIST_ML_ROOT)/lib/$(LIB_DATA) obj/*.o

$(LIB_QP): libdir objdir obj/Array.o obj/QuadProg++.o
	$(CC) $(CFLAGS) -o $(MNIST_ML_ROOT)/lib/$(LIB_QP) obj/Array.o obj/QuadProg++.o

$(EXEC): $(SRC)/data_handler.cpp
	$(CC) -std=c++20 -g -o $(EXEC) $(SRC)/data_handler.cpp -L$(MNIST_ML_ROOT)/lib -ldata -lqp -I$(INCLUDE_DIR)

libdir:
	mkdir -p $(MNIST_ML_ROOT)/lib

objdir:
	mkdir -p $(MNIST_ML_ROOT)/obj

obj/data_handler.o: $(SRC)/data_handler.cpp
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -D DATA_TYPE=$(DATA_TYPE) -c $(SRC)/data_handler.cpp -o obj/data_handler.o

obj/data.o: $(SRC)/data.tpp
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -D DATA_TYPE=$(DATA_TYPE) -c $(SRC)/data.tpp -o obj/data.o

obj/common.o: $(SRC)/common.cpp
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -D DATA_TYPE=$(DATA_TYPE) -c $(SRC)/common.cpp -o obj/common.o

obj/Array.o: $(QP_SRC)/Array.cc
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -c $(QP_SRC)/Array.cc -o obj/Array.o

obj/QuadProg++.o: $(QP_SRC)/QuadProg++.cc
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -c $(QP_SRC)/QuadProg++.cc -o obj/QuadProg++.o

clean:
	rm -rf $(LD_LIBRARY_PATH)
	rm -rf $(MNIST_ML_ROOT)/obj
	rm -rf $(EXEC)