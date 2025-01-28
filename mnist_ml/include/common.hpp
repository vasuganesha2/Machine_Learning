#ifndef __COMMON_HPP
#define __COMMON_HPP

#include "data.hpp"




#include <vector>  // Include the necessary header for std::vector





class common_data
{
    protected:
    std::vector<Data<DATA_TYPE>*>* training_data;  // Use std::vector instead of just vector
    std::vector<Data<DATA_TYPE>*> *test_data;      // Same here
    std::vector<Data<DATA_TYPE>*> *validation_data;
    
    public:
    
        void set_training_data(std::vector<Data <DATA_TYPE>*> *vect);  // Use std::vector in function signature
        void set_test_data(std::vector<Data <DATA_TYPE>*> *vect);      // Same here
        void set_validation_data(std::vector<Data<DATA_TYPE>*> *vect);  // Same here
};

#endif
