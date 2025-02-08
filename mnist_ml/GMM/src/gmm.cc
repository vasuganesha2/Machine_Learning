#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include "../include/gmm.hpp"
#include "data_handler.hpp"

using namespace std;
using namespace Eigen;

GMM::GMM(int num, int d) : PCA(d) {
    this->num_clusters = num;
    means.resize(num + 1);
    covariances.resize(num + 1);

    weights = VectorXd::Ones(num + 1) / num;  // Initialize weights uniformly

    // Random initialization of means and variances
    for (int i = 1; i <= num; i++) {
        means[i] = VectorXd::Random(d);  // Initialize mean as random d-dimensional vector
        covariances[i] = MatrixXd::Identity(d, d);  // Initialize covariance as d × d identity matrix
    }
}

double GMM::gaussian_log_pdf(const Eigen::VectorXd& x, const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov) {
    int d = x.size();
    double det = cov.determinant();

    if (det < 1e-6) {  // Check for near-zero determinant
        det = 1e-6;   // Replace with a small positive value
    }

    // Use Cholesky decomposition for numerical stability and efficiency:
    Eigen::LLT<Eigen::MatrixXd> lltOfCov(cov);
    if (lltOfCov.info() != Eigen::Success) {
        // Handle decomposition failure (e.g., covariance not positive definite)
        std::cerr << "Error: Cholesky decomposition failed. Covariance might not be positive definite.\n";
        return -1e10; // Or another appropriate very small value
    }

    double exponent = -0.5 * (x - mean).transpose() * lltOfCov.solve(x - mean); // Use solve()
    double coeff = -0.5 * d * std::log(2 * M_PI) - 0.5 * std::log(det);

    return exponent + coeff;
}

double GMM::gaussian_pdf(const Eigen::VectorXd& x, const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov) {
    return std::exp(gaussian_log_pdf(x, mean, cov));
}

void GMM::expectation_maximization() {
    auto training_data = get_training_data();
    int n = training_data->size();
    n = min(n, 200);

    int max_iterations = 100; // Maximum number of iterations
    double convergence_threshold = 1e-4; // Convergence threshold
    double val = 0.0;

    for (int iter = 0; iter < max_iterations; iter++) {
        MatrixXd responsibilities(n, num_clusters + 1);

        // Expectation step
        for (int i = 0; i < n; i++) {
            double sample_total = 0.0; // Reset for each sample
            Map<VectorXd> x(training_data->at(i)->get_feature_vector()->data(),
                            training_data->at(i)->get_feature_vector()->size());
            for (int k = 1; k <= num_clusters; k++) {
                double pdf = gaussian_pdf(x, means[k], covariances[k]);
                responsibilities(i, k) = weights[k] * pdf;
                sample_total += responsibilities(i, k);
            }
            // Check for division by zero
            sample_total = max(sample_total, 1e-6);

            // Normalize responsibilities for this sample
            for (int k = 1; k <= num_clusters; k++) {
                responsibilities(i, k) /= sample_total;
            }
        }

        // Maximization step
        int d = get_dimension();
        VectorXd new_weights(num_clusters + 1);
        vector<VectorXd> new_means(num_clusters + 1, VectorXd::Zero(d));
        vector<MatrixXd> new_covariances(num_clusters + 1, MatrixXd::Zero(d, d));

        for (int k = 1; k <= num_clusters; k++) {
            double Nk = responsibilities.col(k).sum();  // Sum of responsibilities for cluster k
            new_weights[k] = Nk / n;  // Update weight

            new_weights[k] = max(new_weights[k], 1e-6);
            Nk = max(Nk, 1e-6);

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
            for (int i = 0; i < n; i++) {
                Map<VectorXd> x(training_data->at(i)->get_feature_vector()->data(),
                                training_data->at(i)->get_feature_vector()->size());
                VectorXd diff = x - new_means[k];
                cov_k += responsibilities(i, k) * (diff * diff.transpose());
            }
            new_covariances[k] = cov_k / Nk;

            // Regularize covariance matrix to ensure positive definiteness
            new_covariances[k] += 1e-6 * MatrixXd::Identity(d, d);
        }

        // Check for convergence
        val = 0.0;
        for (int i = 1; i <= num_clusters; i++) {
            val = max(val, (new_means[i] - means[i]).norm());
        }

        weights = new_weights;
        means = new_means;
        covariances = new_covariances;

        if (val < convergence_threshold) {
            break; // Convergence reached
        }
    }
}

void GMM::classify() {
    auto training_data = get_training_data();
    int n = training_data->size();
    n = min(n, 200);
    vector<map<int, int>> mp(num_clusters + 1);

    for (int i = 0; i < n; i++) {
        // Checking to which clusters the point belong to....
        double max_value = 0;
        int bst_cluster = -1;
        Map<VectorXd> x(training_data->at(i)->get_feature_vector()->data(),
                        training_data->at(i)->get_feature_vector()->size());
        for (int k = 1; k <= num_clusters; k++) {
            double pdf = gaussian_pdf(x, means[k], covariances[k]);

            if (weights[k] * pdf > max_value) {
                max_value = weights[k] * pdf;
                bst_cluster = k;
            }
        }
        printf("%d\n", bst_cluster);
        mp[bst_cluster][training_data->at(i)->get_label()]++;
        for (int k = 1; k <= num_clusters; k++) {
            int bst_label = -1, max_label = -1;
            for (auto it : mp[k]) {
                if (max_label < it.second) {
                    max_label = it.second;
                    bst_label = it.first;
                }
            }
            cluster_map[i] = bst_label;
        }
        printf("DONE 6\n");
    }
}

double GMM::validate() {
    auto validation_data = get_validation_data();
    int correct = 0, total = 100;

    for (int i = 0; i < total; i++) {
        Map<VectorXd> x(validation_data->at(i)->get_feature_vector()->data(),
                        validation_data->at(i)->get_feature_vector()->size());

        double max_prob = 0;
        int best_cluster = -1;
        for (int k = 1; k <= num_clusters; k++) {
            double pdf = gaussian_pdf(x, means[k], covariances[k]);
            if (weights[k] * pdf > max_prob) {
                max_prob = weights[k] * pdf;
                best_cluster = k;
            }
        }

        if (best_cluster != -1 && cluster_map[best_cluster] == validation_data->at(i)->get_label()) {
            correct++;
        }
    }

    double accuracy = (double)correct / total;
    cout << "Validation Accuracy: " << accuracy * 100 << "%" << endl;
    return accuracy;
}

void GMM::test() {
    auto test_data = get_test_data();
    int correct = 0, total = 100;

    for (int i = 0; i < total; i++) {
        Map<VectorXd> x(test_data->at(i)->get_feature_vector()->data(),
                        test_data->at(i)->get_feature_vector()->size());

        double max_prob = 0;
        int best_cluster = -1;
        for (int k = 1; k <= num_clusters; k++) {
            double pdf = gaussian_pdf(x, means[k], covariances[k]);
            if (weights[k] * pdf > max_prob) {
                max_prob = weights[k] * pdf;
                best_cluster = k;
            }
        }

        if (best_cluster != -1 && cluster_map[best_cluster] == test_data->at(i)->get_label()) {
            correct++;
        }
    }

    double accuracy = (double)correct / total;
    cout << "Test Accuracy: " << accuracy * 100 << "%" << endl;
}

void GMM::set_num_clusters(int num) {
    this->num_clusters = num;
    means.resize(num + 1);
    covariances.resize(num + 1);

    weights = VectorXd::Ones(num + 1) / num;  // Initialize weights uniformly

    int d = get_dimension();
    // Random initialization of means and variances
    for (int i = 1; i <= num; i++) {
        means[i] = VectorXd::Random(d);  // Initialize mean as random d-dimensional vector
        covariances[i] = MatrixXd::Identity(d, d);  // Initialize covariance as d × d identity matrix
    }
}

void GMM::print() {
    cout << "Matrix A:" << endl;
    cout << covariances[1] << endl << endl;

    // Print the vector
    cout << "Vector v:" << endl;
    cout << means[1] << endl;
}

int main() {
    data_handler<>* dh = new data_handler<>();
    dh->read_feature_vector("../../train-images.idx3-ubyte");
    dh->read_feature_label("../../train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    GMM* gmm = new GMM(10, 20);
    gmm->set_training_data(dh->get_training_data());
    gmm->set_test_data(dh->get_test_data());
    gmm->set_validation_data(dh->get_validation_data());
    gmm->reduction_training();
    gmm->reduction();
    double bst_performance = 0.0, best_num = -1;
    printf("DONE hey\n");

    for (int i = 10; i < 20; i++) {
        gmm->set_num_clusters(i);
        gmm->expectation_maximization();

        // gmm->print();

        // gmm->classify();
        // double performance = gmm->validate();
        printf("Current performance @ K = %d: %.2f\n", i, bst_performance);
        // if (performance > bst_performance) {
        //     bst_performance = performance;
        //     best_num = i;
        // }
    }

    // printf("%.2f %.2f \n", bst_performance, best_num);
}