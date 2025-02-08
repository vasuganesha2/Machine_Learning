#ifndef __GMM_H
#define __GMM_H

#include<bits/stdc++.h>
#include<Eigen/Dense>
#include"pca.hpp"
#include"kmeans.hpp"

using namespace std;
using namespace Eigen;

class GMM:public PCA
{
    private:
        int num_clusters;
        vector<VectorXd>means;
        vector<MatrixXd>covariances;
        VectorXd weights;
        map<int, int> cluster_map;
        
    public:
        GMM(int num, int d);
        void expectation_maximization();
        double gaussian_pdf(const VectorXd& x, const VectorXd& mean, const MatrixXd& cov); 
        void classify();
        void validate();
        void test();
};

#endif