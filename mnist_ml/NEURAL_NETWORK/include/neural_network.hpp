#ifndef __NEURAL_NETWORK_HPP
#define __NEURAL_NETWORK_HPP

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <random>
#include "common.hpp"
#include "data_handler.hpp"

using namespace std;
using namespace Eigen;


class Network: public common_data
{
    //Network will have the following propety.
    //1> It should firdt have how many layers do the nerual network have...
    //2> It should then have how many neuron do each layers of the network have...
    //3> Then it should have weight matrix.. 
    //4> It will have the weight matrix where the matrix will store the weight 
    //5> It will have the 

    private:
        int l;
        double learning_rate;
        vector<MatrixXd> weights;
        vector<VectorXd> bias; 

    public:
        Network(int l, vector<int> &arr, double learning_rate);
        //would requiore the traingin data and 
        vector<VectorXd> feedforward(VectorXd &input);
        vector<VectorXd> backpropagate(vector<VectorXd> &inputs, VectorXd &expected);
        VectorXd sigmoid_prime(VectorXd &z);
        VectorXd sigmoid(VectorXd &z);
        VectorXd cost_function_prime(VectorXd &expected, VectorXd &output);
        void update_mini_batch(vector<VectorXd> &minibatch, vector<VectorXd> &outputs);
        void train(int num, int class_count);
        void validate();
        void test();

};




#endif