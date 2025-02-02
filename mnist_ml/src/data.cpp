#include "../include/data.hpp"

#include <bits/stdc++.h>

using namespace std;


Data::Data() {
    feature_vector = new std::vector<uint8_t>();  // Initialize the feature_vector pointer
    label = 0;
    enum_label = 0;
}

// Destructor definition
Data::~Data() {
    delete feature_vector;  // Ensure to delete the dynamically allocated memory
}

void Data::set_feature_vector(vector<uint8_t>* vect) {
    feature_vector = vect;
}

void Data::append_to_feature_vector(uint8_t val) {
    feature_vector->push_back(val);
}

void Data::set_enumerated_label(int val) {
    enum_label = val;
}

void Data::set_label(uint8_t val) {
    label = val;
}

// Getter for feature_vector
vector<uint8_t>* Data::get_feature_vector() const {
    return feature_vector;
}

// Getter for enumerated label
int Data::get_enumerated_label() const {
    return enum_label;
}

// Getter for label
uint8_t Data::get_label() const {
    return label;
}

int Data::get_feature_vector_size() const{
    return feature_vector->size();
}


void Data :: set_distance(double distance){
    this ->distance = distance;
}

double Data :: get_distance() const{
    return distance;
}