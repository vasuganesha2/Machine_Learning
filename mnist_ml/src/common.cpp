#include "common.hpp"
#include<bits/stdc++.h>
using namespace std;


// void common_data::set_training_data(vector<Data *> *vect)
// {
//     training_data = vect;
// }

void common_data::set_test_data(vector<Data *> *vect)
{
    test_data = vect;
}
void common_data::set_validation_data(vector<Data *> *vect)
{
    validation_data = vect;
}