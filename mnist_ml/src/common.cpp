#include "common.hpp"
#include <bits/stdc++.h>
using namespace std;

void common_data::set_training_data(vector<Data<DATA_TYPE> *> *vect) {
    training_data = vect;
}

void common_data::set_test_data(vector<Data<DATA_TYPE> *> *vect) {
    test_data = vect;
}

void common_data::set_validation_data(vector<Data<DATA_TYPE> *> *vect) {
    validation_data = vect;
}

vector<Data<DATA_TYPE>*>* common_data::get_training_data() {
    return training_data;
}

vector<Data<DATA_TYPE>*>* common_data::get_test_data() {
    return test_data;
}

vector<Data<DATA_TYPE>*>* common_data::get_validation_data() {
    return validation_data;
}
