CC = g++

SRC := $(MNIST_ML_ROOT)/logistic/src
CFLAGS := -std=c++11 
INCLUDE_DIR := $(MNIST_ML_ROOT)/logistic/include/*

all: main

main: $(SRC)/logistic.cpp
	$(CC) $(CFLAGS) -o main $(SRC)/logistic.cpp -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include -ldata

clean:
	rm -f main
