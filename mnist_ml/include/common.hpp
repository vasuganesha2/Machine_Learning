#ifndef __COMMON_HPP
#define __COMMON_HPP

#include "data.hpp"




#include <vector>  // Include the necessary header for std::vector

using namespace std;



class common_data {

vector<Data<DATA_TYPE>*> *training_data;
vector<Data<DATA_TYPE>*> *test_data;
vector<Data<DATA_TYPE>*> *validation_data;

public:
    void set_training_data(vector<Data<DATA_TYPE> *> *vect);
    void set_test_data(vector<Data<DATA_TYPE> *> *vect);
    void set_validation_data(vector<Data<DATA_TYPE> *> *vect);

    vector<Data<DATA_TYPE>*>* get_training_data();
    vector<Data<DATA_TYPE>*>* get_test_data();
    vector<Data<DATA_TYPE>*>* get_validation_data();
};


#endif
