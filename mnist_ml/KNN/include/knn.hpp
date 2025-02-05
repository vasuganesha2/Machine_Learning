#ifndef __KNN_H
#define __KNN_H


#include "common.hpp"

using namespace std;

class knn
{
    int k;
    vector<Data<DATA_TYPE>*> *neighbour;
    vector<Data<DATA_TYPE>*> *training_data;
    vector<Data<DATA_TYPE>*> *test_data;
    vector<Data<DATA_TYPE>*> * validation_data;
    
    public:
    knn(int);
    knn();
    ~knn();

    void find_k_nearest(Data<DATA_TYPE> *query);
    void set_training_data(vector<Data<DATA_TYPE>*> *vec);
    void set_test_data(vector<Data<DATA_TYPE>*> *data);
    void set_validation_data(vector<Data<DATA_TYPE>*> *data);



    vector<Data<DATA_TYPE>*>* get_training_data();
    vector<Data<DATA_TYPE>*>* get_test_data();
    vector<Data<DATA_TYPE>*>* get_validation_data();

    void set_k(int val);

    int predict();
    double calculate_distance(Data<DATA_TYPE> *query_point, Data<DATA_TYPE>*input);
    double validate_performance();
    double test_performacne();
    
};

#endif