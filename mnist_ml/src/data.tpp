#ifndef __DATA_TPP
#define __DATA_TPP

#include "../include/data.hpp"
#include <iostream>

// Constructor
template <typename T>
Data<T>::Data() : feature_vector(new std::vector<T>), label(0), enum_label(0), distance(0.0) {}

// Destructor
template <typename T>
Data<T>::~Data() { delete feature_vector; }

// Setters
template <typename T>
void Data<T>::set_feature_vector(std::vector<T>* fv) {
    if (feature_vector) delete feature_vector;
    feature_vector = fv;
}

template <typename T>
void Data<T>::append_to_feature_vector(T value) {
    feature_vector->push_back(value);
}

template <typename T>
void Data<T>::set_label(uint8_t lbl) {
    label = lbl;
}

template <typename T>
void Data<T>::set_enumerated_label(int enumLbl) {
    enum_label = enumLbl;
}

template <typename T>
void Data<T>::set_distance(double dist) {
    distance = dist;
}

// Getters
template <typename T>
int Data<T>::get_feature_vector_size() const {
    return feature_vector->size();
}

template <typename T>
uint8_t Data<T>::get_label() const {
    return label;
}

template <typename T>
int Data<T>::get_enumerated_label() const {
    return enum_label;
}

template <typename T>
std::vector<T>* Data<T>::get_feature_vector() const {
    return feature_vector;
}

template <typename T>
double Data<T>::get_distance() const {
    return distance;
}

template <typename T>
void Data<T>::set_class_vector(int count)
{
    class_vector = new std::vector<int>();
    for(int i = 0; i < count; i++)
    {
        if(i == label)
        {
            (*class_vector)[i] = 1;
        }
        else
        {
            (*class_vector)[i] = 0;
        }
    }
}

template <typename T>
std::vector<int> *Data<T>::get_class_vector() const{
    return class_vector;
}

#endif // __DATA_TPP
