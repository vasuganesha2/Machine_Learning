#include"common.hpp"
#include<Eigen/Dense>
#include"../include/neural_network.hpp"
#include<bits/stdc++.h>


using namespace std;
using namespace Eigen;


Network::Network(int l, vector<int>& arr, double learning_rate)
{
    // We will store weights for layers 1 to l-1.
    weights.resize(l);
    bias.resize(l);
    this->l = l;
    this->learning_rate = learning_rate;
    
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0.0, 1.0);
    
    // Initialize weights and biases for layers 1 to l-1.
    for (int i = 1; i < l; i++) {
        MatrixXd w(arr[i], arr[i - 1]);
        for (int j = 0; j < arr[i]; j++) {
            for (int k = 0; k < arr[i - 1]; k++) {
                w(j, k) = dist(gen);
            }
        }
        weights[i] = w;
        
        VectorXd b(arr[i]);
        for (int j = 0; j < arr[i]; j++) {
            b(j) = dist(gen);
        }
        bias[i] = b;
    }
}

VectorXd Network:: sigmoid_prime(VectorXd &input) 
{
    
    VectorXd output(input.size());
    auto it_out = output.data();
    for (auto it = input.data(); it != input.data() + input.size(); ++it, ++it_out) {
        double sig = 1.0 / (1.0 + exp(-(*it)));
        *it_out = sig * (1 - sig);
    }
    return output;
}


VectorXd Network:: sigmoid(VectorXd &input)
{
   
    VectorXd output(input.size());
    auto it_out = output.data();
    for (auto it = input.data(); it != input.data() + input.size(); ++it, ++it_out) {
        double sig = 1.0 / (1.0 + exp(-(*it)));
        *it_out = sig;
    }
    return output;   
}


vector<VectorXd> Network:: feedforward(VectorXd &input)
{
    //store the input in a Matrix for the back propagation stage..

    vector<VectorXd> inputs;
    inputs.push_back(input);

    for(int i = 1; i < l; i++)
    {
        //wx + b;
        //sigmoid(wx + b);
        // cout << "Weight dimensions at layer " << i << ": " << weights[i].rows() << " x " << weights[i].cols() << endl;
        // cout << "Input dimensions at layer " << i << ": " << inputs.back().size() << endl;
        
        VectorXd output = (weights[i] * inputs.back()) + bias[i];
        //On Applying the activation function (sigmoid)
        output = sigmoid(output);
        inputs.push_back(output);
    }
    return inputs;
}

VectorXd Network:: cost_function_prime(VectorXd &expected, VectorXd &output)
{
    return (output - expected);
}

vector<VectorXd> Network:: backpropagate(vector<VectorXd> &inputs, VectorXd &expected)
{
    vector<VectorXd> delta(l);
    
    delta[l - 1] = cost_function_prime(expected, inputs[l - 1]).array() * sigmoid_prime(inputs[l - 1]).array();
    for(int i = l - 2; i >= 1; i--)
    {
        //z_l = w_l * a_(l - 1) + b_l
        delta[i] = (weights[i + 1].transpose() * delta[i + 1]).array() * sigmoid_prime(inputs[i]).array();
    }
    

    return delta;
}


void Network :: update_mini_batch(vector<VectorXd> &mini_batch, vector<VectorXd> &expected)
{
    // break the data set into the batches...

    int m = mini_batch.size();

    vector<MatrixXd> delta_w(l);
    vector<VectorXd> delta_b(l);

    for(int i = 0; i < m; i++)
    {
        vector<VectorXd> inputs = feedforward(mini_batch[i]);
        vector<VectorXd> deltas = backpropagate(inputs, expected[i]);
        if(i == 0)
        {
            for(int j = 1; j < l; j++)
            {
                delta_w[j] = deltas[j] * inputs[j - 1].transpose();
                delta_b[j] = deltas[j];
            } 
            continue;
        }
        for(int j = 1; j < l; j++)
        {
            delta_w[j] += deltas[j] * inputs[j - 1].transpose();
            delta_b[j] += deltas[j];
        } 
        //cout << "No error" << endl;
    }
    
    for(int j = 1; j < l; j++)
    {
        weights[j] -= ((learning_rate / m) * delta_w[j]);
        bias[j] -= ((learning_rate / m) * delta_b[j]);
    } 
}

void Network :: train(int num, int class_count)
{
    //num represent the number of the mini batch to be created...
    //Have to create the mini batches and send them to the update mini batch
    int n = training_data -> size();

    vector<int> v(n);
    for(int i = 0; i < n; i++) 
        v[i] = i;
    //printf("DONE WITHOUT ERROR \n");   


    random_device rd;
    mt19937 g(rd()); // Mersenne Twister PRNG

    shuffle(v.begin(), v.end(), g);

    int size = n / num;

    for(int i = 0; i < num; i++)
    {


        vector<VectorXd> mini_batch, output;
        mini_batch.resize(size);
        output.resize(size);


        for(int j = 0; j < size; j++)
        {
            mini_batch[j] = Eigen::VectorXd::Map(
                training_data->at(v[i * size + j])->get_feature_vector()->data(), 
                training_data->at(v[i * size + j])->get_feature_vector()->size()
            );

            output[j] = VectorXd::Zero(class_count);
            output[j](training_data -> at(v[i * size + j]) -> get_label()) = 1.0;
        }

        
        update_mini_batch(mini_batch, output);

        cout << "Batch is updtaed " << i << endl;
        
    }

}


void Network::validate()
{
    int n = validation_data->size();
    int correct = 0;

    for (int i = 0; i < n; i++)
    {
        VectorXd input = Eigen::VectorXd::Map(
            validation_data->at(i)->get_feature_vector()->data(),
            validation_data->at(i)->get_feature_vector()->size()
        );

        VectorXd output = feedforward(input).back();

        // Get predicted class (index of max value)
        int predicted_label;
        output.maxCoeff(&predicted_label);

        // Get actual label
        int actual_label = validation_data->at(i)->get_label();
        // Count correct predictions
        if (predicted_label == actual_label)
        {
            correct++;
        }

        cout << "acctual " << actual_label << " " << "predicted " << predicted_label << endl;

    }

    // Compute accuracy
    double accuracy = static_cast<double>(correct) / n;
    cout << "Validation Accuracy: " << accuracy * 100.0 << "%" << endl;
}



void Network::test()
{
    int n = test_data->size();
    int correct = 0;

    for (int i = 0; i < n; i++)
    {
        VectorXd input = Eigen::VectorXd::Map(
            test_data->at(i)->get_feature_vector()->data(),
            test_data->at(i)->get_feature_vector()->size()
        );

        VectorXd output = feedforward(input).back();

        // Get predicted class (index of max value)
        int predicted_label;
        output.maxCoeff(&predicted_label);

        // Get actual label
        
        int actual_label = test_data->at(i)->get_label();

        // Count correct predictions
        if (predicted_label == actual_label)
        {
            correct++;
        }
    }

    // Compute accuracy
    double accuracy = static_cast<double>(correct) / n;
    cout << "Test Accuracy: " << accuracy * 100.0 << "%" << endl;
}






int main()
{
    data_handler<> *dh = new data_handler<>();
    dh -> read_feature_vector("../../train-images.idx3-ubyte");
    dh -> read_feature_label("../../train-labels.idx1-ubyte");
    dh -> split_data();
    dh -> count_classes();
    double performance = 0.0;
    double bst_performance = 0.0;
    vector<int> arr;
    arr.push_back(dh -> get_training_data() -> at(0) -> get_feature_vector() -> size());
    cout << arr[0] << endl;
    arr.push_back(256);
    arr.push_back(128);
    //arr.push_back(21);
    arr.push_back(dh -> get_class_count());
    Network *neural = new Network(4, arr, 0.1);
    neural -> set_training_data(dh -> get_training_data());
    neural -> set_test_data(dh -> get_test_data());
    neural -> set_validation_data(dh -> get_validation_data());
    neural -> train(32, dh -> get_class_count());
    neural -> validate();
    neural -> test();
}