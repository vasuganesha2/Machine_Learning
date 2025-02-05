#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include <bits/stdc++.h>
#include "data.hpp"

using namespace std;

#ifndef DATA_TYPE_LABEL
    #define DATA_TYPE_LABEL double
#endif

#ifndef DATA_TYPE
    #define DATA_TYPE uint8_t
#endif

template <typename T = DATA_TYPE, typename T1 = DATA_TYPE_LABEL>
class data_handler
{
private:
    vector<Data<T>*> *data_array;
    vector<Data<T>*> *training_data;
    vector<Data<T>*> *test_data;
    vector<Data<T>*> *validation_data;

    int num_classes;
    int feature_vector_size;
    map<T1, int> class_map;
    map<int, T1> class_map_inverse;

    const double TRAIN_SET_PERCENT = 0.75;
    const double TEST_SET_PERCENT = 0.20;
    const double VALIDATION_SET_PERCENT = 0.05;

public: 
    data_handler();
    ~data_handler();

    void read_csv(string path, string delimiter);
    void read_feature_vector(string path);
    void read_feature_label(string path);
    void split_data();
    void count_classes();
    
    uint32_t convert_to_little_endian(const unsigned char* bytes);

    

    vector<Data<T>*> *get_training_data();
    vector<Data<T>*> *get_test_data();
    vector<Data<T>*> *get_validation_data();
    int get_class_count();
    map<int, T1> get_class_map_inverse();
};

#include "../src/data_handler.tpp"

#endif
