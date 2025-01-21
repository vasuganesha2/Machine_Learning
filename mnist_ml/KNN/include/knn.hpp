#ifndef __KNN_H
#define __KNN_H


#include <bits/stdc++.h>
#include "data.hpp"

using namespace std;

class knn
{
    int k;
    vector<Data*> *neighbour;
    vector<Data*> *training_data;
    vector<Data*> *test_data;
    vector<Data*> * validation_data;
    
    public:
    knn(int);
    knn();
    ~knn();

    void find_k_nearest(Data *query);
    void set_training_data(vector<Data*> *vec);
    void set_test_data(vector<Data *> *data);
    void set_validation_data(vector<Data*> *data);
    void set_k(int val);

    int predict();
    double calculate_distance(Data *query_point, Data*input);
    double validate_performance();
    double test_performacne();
    
};

#endif