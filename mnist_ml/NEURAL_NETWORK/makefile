CC = g++

SRC := $(MNIST_ML_ROOT)/NEURAL_NETWORK/src
CFLAGS := -std=c++11 
INCLUDE_DIR := $(PWD)/include/*
EIGEN_DIR := /usr/include/eigen3 

all: main

main: $(SRC)/neural_network.cc
	$(CC) $(CFLAGS) -o main $(SRC)/neural_network.cc -L$(MNIST_ML_ROOT)/lib/ \
	-I$(INCLUDE_DIR) -I$(MNIST_ML_ROOT)/include -I$(EIGEN_DIR) -ldata

clean:
	rm -f main
