#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include <cstdint>


#ifndef DATA_TYPE
    #define DATA_TYPE uint8_t
#endif

template <typename T = DATA_TYPE>

class Data 
{
private:
    std::vector<T>* feature_vector; // Pointer to a vector of type T
    uint8_t label;                  // Label of the data
    int enum_label;                 // Enumerated label (e.g., A -> 1, B -> 2)
    double distance;

public:
    // Constructor and Destructor
    Data();
    ~Data();

    // Setters
    void set_feature_vector(std::vector<T>* fv);
    void append_to_feature_vector(T value);
    void set_label(uint8_t lbl);
    void set_enumerated_label(int enumLbl);
    void set_distance(double distance);

    // Getters
    int get_feature_vector_size() const;
    uint8_t get_label() const;
    int get_enumerated_label() const;
    std::vector<T>* get_feature_vector() const;
    double get_distance() const;
};

// Include implementation file
#include "../src/data.tpp"

#endif // __DATA_H
