CC = g++

SRC := $(MNIST_ML_ROOT)/GMM/src
PCA_SRC := $(MNIST_ML_ROOT)/PCA/src

CFLAGS := -std=c++20
INCLUDE_DIR := $(PWD)/include/*
EIGEN_DIR := /usr/include/eigen3 

all: main

main: $(SRC)/gmm.cc $(PCA_SRC)/pca.cc
	$(CC) $(CFLAGS) -o main $(SRC)/gmm.cc $(PCA_SRC)/pca.cc -L$(MNIST_ML_ROOT)/lib/ \
	-I$(INCLUDE_DIR) -I$(MNIST_ML_ROOT)/include -I$(EIGEN_DIR) -I$(MNIST_ML_ROOT)/PCA/include -ldata

clean:
	rm -f main
