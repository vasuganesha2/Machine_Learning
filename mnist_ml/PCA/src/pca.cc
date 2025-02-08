#include"bits/stdc++.h"
#include"knn.hpp"
#include"../include/pca.hpp"
#include "data_handler.hpp"

#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

// PCA::PCA(int d, int k) : knn(k), dimensions(d) 
// {}

PCA::PCA(int d) : dimensions(d)
{}


void PCA::reduction_training()
{
    auto training_data = get_training_data();
    int n = training_data->size();
    if(n == 0) 
    {
        cout << "No training data!" << endl;
        return;
    }


    int dim = training_data->at(0)->get_feature_vector()->size();


    mean_vector = VectorXd::Zero(dim);

    for (int i = 0; i < n; i++) 
    {
        Map<VectorXd> x(training_data->at(i)->get_feature_vector()->data(),
                        training_data->at(i)->get_feature_vector()->size());
        mean_vector += x;
    }
    mean_vector /= n;

    //printf("DONE||||||||DONE|||||||DONE \n");

    MatrixXd cov = MatrixXd::Zero(dim, dim);
    for (int i = 0; i < n; i++) 
    {
        Map<VectorXd> x(training_data->at(i)->get_feature_vector()->data(),
                        training_data->at(i)->get_feature_vector()->size());
        VectorXd centered = x - mean_vector;
        cov += centered * centered.transpose();
        printf("DONE %d \n", i);
    }
    cov /= n; 

    



    SelfAdjointEigenSolver<MatrixXd> solver(cov);
    if (solver.info() != Success) 
    {
        cerr << "Eigen decomposition failed!" << endl;
        return;
    }

    // The eigenvalues are in increasing order.
    VectorXd eigen_values = solver.eigenvalues();
    MatrixXd eigen_vectors = solver.eigenvectors();

    int num_components = dimensions;
    NewAxis = MatrixXd(dim, num_components);
    for (int i = 0; i < num_components; i++) 
    {
        NewAxis.col(i) = eigen_vectors.col(eigen_vectors.cols() - 1 - i);
    }


    for (int i = 0; i < n; i++) 
    {
        Map<VectorXd> x(training_data->at(i)->get_feature_vector()->data(),
                        training_data->at(i)->get_feature_vector()->size());
        VectorXd centered = x - mean_vector;
        VectorXd new_x = NewAxis.transpose() * centered;
        vector<double>* new_x_vector = new vector<double>(new_x.data(), new_x.data() + new_x.size());
        training_data->at(i)->set_feature_vector(new_x_vector);
        printf("DONE %d \n", i);
    }
}



void PCA::reduction()
{
    // === Process Validation Data ===
    auto validation_data = get_validation_data();
    int n_val = validation_data->size();
    for (int i = 0; i < n_val; i++) {
        Map<VectorXd> x(validation_data->at(i)->get_feature_vector()->data(),
                        validation_data->at(i)->get_feature_vector()->size());
        // Subtract the training mean before projection
        VectorXd centered = x - mean_vector;
        VectorXd new_x = NewAxis.transpose() * centered;
        vector<double>* new_x_vector = new vector<double>(new_x.data(), new_x.data() + new_x.size());
        validation_data->at(i)->set_feature_vector(new_x_vector);
         printf("DONE %d \n", i);
    }

    // === Process Test Data ===
    auto test_data = get_test_data();
    int n_test = test_data->size();
    for (int i = 0; i < n_test; i++) {
        Map<VectorXd> x(test_data->at(i)->get_feature_vector()->data(),
                        test_data->at(i)->get_feature_vector()->size());
        VectorXd centered = x - mean_vector;
        VectorXd new_x = NewAxis.transpose() * centered;
        vector<double>* new_x_vector = new vector<double>(new_x.data(), new_x.data() + new_x.size());
        test_data->at(i)->set_feature_vector(new_x_vector);
         printf("DONE %d \n", i);
    }
}

int PCA:: get_dimension()
{
    return dimensions;
}


// int main()
// {
//     data_handler<> *dh = new data_handler<>();
//     dh -> read_feature_vector("../../train-images.idx3-ubyte");
//     dh -> read_feature_label("../../train-labels.idx1-ubyte");
//     dh -> split_data();
//     dh -> count_classes();



//     PCA *pca = new PCA(10, 10);



//     pca->set_training_data(dh->get_training_data());
//     pca->set_test_data(dh->get_test_data());
//     pca->set_validation_data(dh->get_validation_data());


//     pca -> reduction_training();

//     pca->reduction();

//     pca -> validate_performance();
//     pca -> test_performacne();

// }


