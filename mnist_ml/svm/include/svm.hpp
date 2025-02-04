#ifndef SVM_HPP
#define SVM_HPP
#include <bits/stdc++.h>
#include "QuadProg++.hh"
#include "data.hpp"
#include "data_handler.hpp"
#include "common.hpp"

using namespace std;
#define THRESHOLD 1e-5
class SVM
{
    public:
    double C;
    double bias;
    vector<double> alphas; //ai for support vectors
    vector<vector<double>> support_vectors;
    vector<int> support_labels;
    double (*kernel)(vector<double>&, vector<double>&);
    SVM(double C, double (*kernel_func)(vector<double>&, vector<double>&));
    void fit(vector<vector<double>>& X, vector<int>& y);
    void compute_bias();
    int predict(vector<double>& x);
};


#endif 
