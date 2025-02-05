CC = g++

SRC := $(MNIST_ML_ROOT)/PCA/src
KNN_SRC := $(MNIST_ML_ROOT)/KNN/src

CFLAGS := -std=c++11 -DEUCLID  # Add the distance metric definition here
INCLUDE_DIR := $(PWD)/include/*
EIGEN_DIR := /usr/include/eigen3 

all: main

main: $(SRC)/pca.cc $(KNN_SRC)/knn.cpp
	$(CC) $(CFLAGS) -o main $(SRC)/pca.cc $(KNN_SRC)/knn.cpp -L$(MNIST_ML_ROOT)/lib/ \
	-I$(INCLUDE_DIR) -I$(MNIST_ML_ROOT)/include -I$(EIGEN_DIR) -I$(MNIST_ML_ROOT)/KNN/include -ldata

clean:
	rm -f main
