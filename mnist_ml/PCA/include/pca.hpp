#ifndef __PCA_HPP
#define __PCA_HPP

#include"common.hpp"
#include"bits/stdc++.h"
#include"knn.hpp"
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

class PCA:public knn
{
    int dimensions;
    MatrixXd NewAxis;
    VectorXd mean_vector;

    public:
    PCA(int d, int k);  // Constructor declaration
    void reduction_training();
    void reduction();
};

#endif