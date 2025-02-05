#include "../include/knn.hpp"
#include<bits/stdc++.h>
#include "data_handler.hpp"

knn :: knn(int val)
{
    k = val;
}

knn :: knn()
{

}

knn :: ~knn()
{

}

// void knn :: find_k_nearest(Data<DATA_TYPE> *query)
// {
//     neighbour = new vector<Data<DATA_TYPE> *>;
//     double min = numeric_limits<double>::max();
//     double previos_min = min;
//     int index = 0;
//     for(int i = 0; i < k; i++)
//     {
//         if(i == 0)
//         {
//             for(int j = 0; j < training_data -> size(); j++)
//             {
//                 double distance = calculate_distance(query, training_data -> at(j));
//                 training_data -> at(j) -> set_distance(distance);
//                 if(distance < min)
//                 {
//                     min = distance;
//                     index = j;
//                 }
//             }
//         }
//         else
//         {
//             for(int j = 0; j < training_data -> size(); j++)
//             {
//                 double distance = training_data -> at(j)->get_distance();
//                 if(distance > previos_min && distance < min)
//                 {
//                     min = distance;
//                     index = j;
//                 }
//             }
//         }
//         neighbour -> push_back(training_data->at(index));
//         previos_min = min;
//         min = numeric_limits<double>:: max();
//     }
// }



struct compare {
    bool operator()(Data<DATA_TYPE>* d1, Data<DATA_TYPE>* d2) {
        return d1->get_distance() < d2->get_distance(); // Max-Heap: largest distance at top
    }
};

void knn::find_k_nearest(Data<DATA_TYPE> *query) 
{
    priority_queue<Data<DATA_TYPE>*, vector<Data<DATA_TYPE>*>, compare> pq;

    for (unsigned int j = 0; j < training_data->size(); j++) 
    {
        double distance = calculate_distance(query, training_data->at(j));
        training_data->at(j)->set_distance(distance);
        
        pq.push(training_data->at(j));
        
        if ((int)pq.size() > k) {
            pq.pop();  // Remove the farthest neighbor (now the largest in the max-heap)
        }
    }

    // Store k-nearest neighbors in neighbour vector
    neighbour = new vector<Data<DATA_TYPE>*>;
    while (!pq.empty()) {
        neighbour->push_back(pq.top());
        pq.pop();
    }

    // Reverse to order from nearest (smallest distance) to farthest (largest distance)
    reverse(neighbour->begin(), neighbour->end());
}





void knn :: set_training_data(vector<Data<DATA_TYPE>*> *vec)
{
    training_data = vec;
}

void knn :: set_test_data(vector<Data<DATA_TYPE>*> *data)
{
    test_data = data;
}

void knn :: set_validation_data(vector<Data<DATA_TYPE>*> *data)
{
    validation_data = data;
}


vector<Data<DATA_TYPE>*>* knn::get_training_data() {
    return training_data;
}

vector<Data<DATA_TYPE>*>* knn::get_test_data() {
    return test_data;
}

vector<Data<DATA_TYPE>*>* knn::get_validation_data() {
    return validation_data;
}


void knn :: set_k(int val)
{
    k = val;
}


int knn:: predict()
{
    map<int8_t, int>class_freq;
    for(unsigned int i = 0; i < neighbour->size(); i++)
        class_freq[neighbour->at(i)->get_label()]++;

    uint8_t best = -1;
    int maxi = -1;
    for(auto it : class_freq)
    {
        if(it.second > maxi)
        {
            maxi = it.second;
            best = it.first;
        }
    }
    neighbour->clear();
    return best;
}



double knn::calculate_distance(Data<DATA_TYPE> *query_point, Data<DATA_TYPE> *input)
{
    double distance = 0.0;

    if (query_point->get_feature_vector_size() != input->get_feature_vector_size())
    {
        printf("Error: Feature vector sizes do not match.\n");
        exit(1);
    }

    #ifdef EUCLID
        for (int i = 0; i < query_point->get_feature_vector_size(); i++)
        {
            distance += pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i), 2);
            //printf("%f\n", distance);
        }
        distance = sqrt(distance);
        return distance;
    #elif defined MANHATTAN
        for (unsigned int i = 0; i < query_point->get_feature_vector_size(); i++)
        {
            distance += fabs(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i));
        }
    #else
        printf("Error: No distance metric defined.\n");
        exit(1);
    #endif
    
}


double knn :: validate_performance()
{
    double current_performance = 0.0;
    int count = 0;
    int data_index = 0;
    
    for(Data<DATA_TYPE> * query_point : *validation_data)
    {
        find_k_nearest(query_point);
        int prediction = predict();
        if(prediction == query_point -> get_label()) count++;
        data_index++;

        //printf("Predicted Class: %d  Actual Class: %d Count %d\n", prediction, query_point->get_label(), count);

        current_performance = (count * 100) / data_index; 
        
        cout << fixed << setprecision(3);
        cout << "Current_performance " << current_performance << endl;
    }
    current_performance = (count * 100.0) / validation_data->size(); 
    cout << fixed << setprecision(3);
    cout << "Current_performance of the Model on Validation data " << current_performance << endl;
    return current_performance;
}

double knn :: test_performacne()
{
    double current_performance = 0.0;
    int count = 0;

    int data_index = 0;
    
    for(Data<DATA_TYPE> * query_point : *validation_data)
    {
        find_k_nearest(query_point);
        int prediction = predict();
        if(prediction == query_point -> get_label()) count++;
        data_index++;
        current_performance = (count * 100)/ data_index; 
        cout << fixed << setprecision(3);
        cout << "Current_performance " << current_performance << endl;
    }
    current_performance = (count * 100.0) / validation_data->size(); 
    cout << fixed << setprecision(3);
    cout << "Current_performance of the Model on Test data " << current_performance << endl;
    return current_performance;
}

// int main()
// {
//     data_handler<> *dh = new data_handler<>();
//     dh -> read_feature_vector("../../train-images.idx3-ubyte");
//     dh -> read_feature_label("../../train-labels.idx1-ubyte");
//     dh -> split_data();
//     dh -> count_classes();


//     knn *knearest = new knn();

//     knearest->set_training_data(dh->get_training_data());
//     knearest->set_test_data(dh->get_test_data());
//     knearest->set_validation_data(dh->get_validation_data());

//     double performance = 0;
//     double best_performance = 0;
//     int best_k = 1;

//     for(int i = 1; i <= 4; i++)
//     {
//         knearest->set_k(i);
//         performance = knearest -> validate_performance();
//         if(performance > best_performance)
//         {
//             best_performance = performance;
//             best_k = i;
//         }
//     }

//     knearest->set_k(best_k);
//     knearest->test_performacne();

// }