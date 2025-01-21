CC = g++

SRC := $(MNIST_ML_ROOT)/KNN/src
CFLAGS := -std=c++11 -DEUCLID
INCLUDE_DIR := $(PWD)/include/*

all: main

main: $(SRC)/knn.cpp
	$(CC) $(CFLAGS) -o main $(SRC)/knn.cpp -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include -ldata

clean:
	rm -f main
