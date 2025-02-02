#ifndef __LOGISTIC_HPP__
#define __LOGISTIC_HPP__
#include<bits/stdc++.h>
#include "data.hpp"
#include "data_handler.hpp"
#include "common.hpp"
using namespace std;

namespace logistic
{
    using ld=long double;
    ld sigmoid(ld z);

    class LogisticRegression
    {
    private:
        vector<ld> weights;
        vector<Data<DATA_TYPE>*> *training_data;
        vector<Data<DATA_TYPE>*> *test_data;
        vector<Data<DATA_TYPE>*> * validation_data;
        ld bias;
        ld learning_rate;
        int epochs;

    public:
        LogisticRegression(int num_features, ld lr, int epochs);
        ld predict_proba(Data<DATA_TYPE>*feature);
        int predict(Data<DATA_TYPE>*feature);
        void train();
        ld test_performance();
        void set_training_data(vector<Data<DATA_TYPE>*> *vec);
        void set_test_data(vector<Data<DATA_TYPE>*> *data);
        void set_validation_data(vector<Data<DATA_TYPE>*> *data);
        

    };
};

#endif
