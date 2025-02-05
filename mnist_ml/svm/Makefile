CC = g++

SRC := $(MNIST_ML_ROOT)/svm/src
CFLAGS := -std=c++11
INCLUDE_DIR := $(MNIST_ML_ROOT)/svm/include
QP_INCLUDE_DIR := $(MNIST_ML_ROOT)/QP  # Path to QP headers
LIB_DIR := $(MNIST_ML_ROOT)/lib

all: main

main: $(SRC)/svm.cpp
	$(CC) $(CFLAGS) -o main $(SRC)/svm.cpp -L$(LIB_DIR) -I$(INCLUDE_DIR) \
	-I$(QP_INCLUDE_DIR) -I$(MNIST_ML_ROOT)/include -ldata -lqp

clean:
	rm -f main