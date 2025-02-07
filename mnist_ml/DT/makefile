CC = g++

SRC := $(MNIST_ML_ROOT)/DT/src
CFLAGS := -std=c++20 
INCLUDE_DIR := $(PWD)/include/*

all: main

main: $(SRC)/dt.cc
	$(CC) $(CFLAGS) -o main $(SRC)/dt.cc -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include -ldata

clean:
	rm -f main
