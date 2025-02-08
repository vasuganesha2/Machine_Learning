#ifndef __GMM_H
#define __GMM_H

#include<bits/stdc++.h>
#include<Eigen/Dense>
#include"pca.hpp"


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
        void set_num_clusters(int num);
        double gaussian_pdf(const VectorXd& x, const VectorXd& mean, const MatrixXd& cov);
        double gaussian_log_pdf(const Eigen::VectorXd& x, const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov);
        void classify();
        double validate();
        void test();
        void print();
};

#endif