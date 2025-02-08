#include<Eigen/Dense>
#include"../include/gmm.hpp"
#include<bits/stdc++.h>


using namespace std;
using namespace Eigen;



 GMM:: GMM(int num, int d) : PCA(d),kmeans(num)
 {
    this -> num_clusters = num;
    means.resize(num + 1);
    covariances.resize(num + 1);

    weights = VectorXd::Ones(num + 1) / num;  // Initialize weights uniformly

    // Random initialization of means and variances
    for (int i = 1; i <= num; i++) {
        means[i] = VectorXd::Random(d);  // Initialize mean as random d-dimensional vector
        covariances[i] = MatrixXd::Identity(d, d);  // Initialize covariance as d × d identity matrix
    }
 }


double GMM::gaussian_pdf(const VectorXd& x, const VectorXd& mean, const MatrixXd& cov) {
    int d = x.size();
    double det = cov.determinant();
    if (det <= 0) det = 1e-6;  // Avoid numerical instability

    MatrixXd cov_inv = cov.inverse();
    VectorXd diff = x - mean;
    
    double exponent = -0.5 * diff.transpose() * cov_inv * diff;
    double coeff = 1.0 / (pow(2 * M_PI, d / 2.0) * sqrt(det));

    return coeff * exp(exponent);
}


void GMM:: expectation_maximization()
{
   auto training_data = get_training_data();
   int n = training_data->size();


    double total_prob = 0.0;
    MatrixXd responsibilities(n, num_clusters + 1);  
   for(int i = 0; i < n; i++)
   {
        Map<VectorXd> x(training_data->at(i)->get_feature_vector()->data(),
                        training_data->at(i)->get_feature_vector()->size());
        for (int k = 1; k <= num_clusters; k++) 
        {
            double pdf = gaussian_pdf(x, means[k], covariances[k]);
            responsibilities(i, k) = weights[k] * pdf;
            total_prob += responsibilities(i, k);
        }

        for (int k = 1; k <= num_clusters; k++) 
        {
            double pdf = gaussian_pdf(x, means[k], covariances[k]);
            responsibilities(i, k) = weights[k] * pdf;
            total_prob += responsibilities(i, k);
        }

        // Normalize responsibilities
        for (int k = 1; k <= num_clusters; k++) 
        {
            responsibilities(i, k) /= total_prob;
        }
   }
    int d = get_dimension();

    VectorXd new_weights(num_clusters);
    vector<VectorXd> new_means(num_clusters + 1, VectorXd::Zero(d));
    vector<MatrixXd> new_covariances(num_clusters + 1, MatrixXd::Zero(d, d));
    for (int k = 1; k <= num_clusters; k++) {
        double Nk = responsibilities.col(k).sum();  // Sum of responsibilities for cluster k
        new_weights[k - 1] = Nk / n;  // Update weight

        // Compute new mean
        VectorXd mean_k = VectorXd::Zero(d);
        for (int i = 0; i < n; i++) {
            Map<VectorXd> x(training_data->at(i)->get_feature_vector()->data(),
                            training_data->at(i)->get_feature_vector()->size());
            mean_k += responsibilities(i, k) * x;
        }
        new_means[k] = mean_k / Nk;

        // Compute new covariance
        MatrixXd cov_k = MatrixXd::Zero(d, d);
        for (int i = 0; i < n; i++) 
        {
            Map<VectorXd> x(training_data->at(i)->get_feature_vector()->data(),
                            training_data->at(i)->get_feature_vector()->size());
            VectorXd diff = x - new_means[k];
            cov_k += responsibilities(i, k) * (diff * diff.transpose());
        }
        new_covariances[k] = cov_k / Nk;
    }

    double val = 0.0;
    for(int i = 1; i <= num_clusters; i++)
    {
        double v = (new_means[i] - means[i]).cwiseAbs();

        val = max(val, v);
    }

    if(val < 1e-6)
        return;

    weights = new_weights;
    means = new_means;
    covariances = new_covariances;



    expectation_maximization();

}


void GMM:: classify()
{
    auto training_data = get_training_data();
    int n = training_data->size();
    vector<map<int, int>> mp(num_clusters + 1);
    for(int i = 0; i < n; i++)
    {
        //checking to which clusters the point belong to....
        double max_value = 0, bst_cluster = -1;
        Map<VectorXd> x(training_data->at(i)->get_feature_vector()->data(),
                        training_data->at(i)->get_feature_vector()->size());
        for (int k = 1; k <= num_clusters; k++) 
        {
            double pdf = gaussian_pdf(x, means[k], covariances[k]);
            if(weights[k] * pdf > max_value)
            {
                max_value = weights[k] * pdf;
                bst_cluster = k;
            }
            mp[bst_cluster][training_data -> at(i) -> get_label()]++;
        }
        for(int i = 1; i <= num_clusters; i++)
        {
            int bst_label = -1, max_label = -1;
            for(auto it : mp[i])
            {
                if(max_label < it.second)
                {
                    max_label = it.second;
                    bst_label = it.first;
                }
            }
            cluster_map[i] = bst_label;
        }
    }
}

void GMM:: validate()
{

}