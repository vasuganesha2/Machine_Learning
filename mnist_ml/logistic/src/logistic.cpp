#include "../include/logistic.hpp"
using namespace logistic;

LogisticRegression::LogisticRegression(int num_features, ld lr, int epochs)
{
    this->weights=vector<ld>(num_features, 0);//TODO:initialize random wt and bias (parameter in constructor)
    this->bias=0;
    this->learning_rate=lr;
    this->epochs=epochs;
}

void LogisticRegression::set_training_data(vector<Data<DATA_TYPE>*> *vec)
{
    training_data = vec;
}

void LogisticRegression::set_test_data(vector<Data<DATA_TYPE>*> *data)
{
    test_data = data;
}

void LogisticRegression::set_validation_data(vector<Data<DATA_TYPE>*> *data)
{
    validation_data = data;
}

ld logistic::sigmoid(ld z)
{
    return 1.0/(1.0+exp(-z));
}

ld LogisticRegression::predict_proba(Data<DATA_TYPE>*feature)
{
    ld z=bias;
    vector<DATA_TYPE> *x=feature->get_feature_vector();
    for (size_t i=0;i<x->size();i++)
    {
        z+=weights[i]*((*x)[i]);
    }
    return sigmoid(z);
}

int LogisticRegression::predict(Data<DATA_TYPE>*feature)
{
    return predict_proba(feature)>=0.5?1:0;
}

void LogisticRegression::train()
{
    int m=(*training_data).size();
    int n=weights.size();

    for (int epoch=0;epoch<this->epochs;epoch++)
    {
        vector<ld>dw(n, 0);
        ld db=0;

        for (auto &data:*training_data)
        {
            DATA_TYPE label=(data->get_label()==1)?1:0; //TODO fix hardcode
            ld pred=predict_proba(data);
            ld error=pred-label;
            vector<DATA_TYPE> *features=data->get_feature_vector();
            for (int j=0;j<n;j++)
            {
                dw[j]+=error*(*features)[j];
            }
            db+=error;
        }

        for (int j=0;j<n;j++)
        {
            weights[j]-=learning_rate*(dw[j] / m);
        }
        bias-=learning_rate*(db / m);
    }
}

ld LogisticRegression::test_performance()
{
    int correct = 0;
    int total = test_data->size();

    for (auto &data : *test_data)
    {
        uint8_t label = (data->get_label() == 0) ? 0 : 1;
        int prediction = predict(data);

        if (prediction == label)
            correct++;
    }

    ld accuracy = (correct * 100.0) / total;
    std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;
    return accuracy;
}


int main()
{
    data_handler*dh = new data_handler();
    dh -> read_feature_vector("../../train-images.idx3-ubyte");
    dh -> read_feature_label("../../train-labels.idx1-ubyte");
    dh -> split_data();
    dh -> count_classes();
    vector<Data<DATA_TYPE>*> *x_train=dh->get_training_data();
    int numFeatures=(x_train)[0].size();
    LogisticRegression *logistic = new LogisticRegression(numFeatures,1e-4,1);

    logistic->set_training_data(dh->get_training_data());
    logistic->set_test_data(dh->get_test_data());
    logistic->set_validation_data(dh->get_validation_data());
    logistic->train();
    logistic->test_performance();
    // ld performance = 0;
    // ld best_performance = 0;
    // int best_k = 1;

    // for(int i = 1; i <= 4; i++)
    // {
    //     logistic->set_k(i);
    //     performance = logistic -> validate_performance();
    //     if(performance > best_performance)
    //     {
    //         best_performance = performance;
    //         best_k = i;
    //     }
    // }

    // logistic->set_k(best_k);
    // logistic->test_performance();
}