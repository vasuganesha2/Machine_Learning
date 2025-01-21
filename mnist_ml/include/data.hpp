#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include <cstdint>
using namespace std;

class Data {
private:
    std::vector<uint8_t>* feature_vector; // Pointer to a vector of uint8_t
    uint8_t label;                       // Label of the data
    int enum_label;                      // Enumerated label (e.g., A -> 1, B -> 2)
    double distance;

public:
    // Constructor and Destructor
    Data();
    ~Data();

    // Setters
    void set_feature_vector(std::vector<uint8_t>* fv);
    void append_to_feature_vector(uint8_t value);
    void set_label(uint8_t lbl);
    void set_enumerated_label(int enumLbl);

    void set_distance(double distance);

    // Getters
    int get_feature_vector_size() const;
    uint8_t get_label() const;
    int get_enumerated_label() const;
    vector<uint8_t>* get_feature_vector() const;
    double get_distance() const;
};

#endif // __DATA_H
