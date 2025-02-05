#include "../include/kmeans.hpp"
#include<common.hpp>
#include<bits/stdc++.h>

using namespace std;


kmeans :: kmeans(int k)
{
    num_cluster = k;
    clusters = new vector<cluster_t *>;
    used_indices = new set<int>;
}

void kmeans :: init_cluster()
{
    for(int i = 0; i < num_cluster; i++)
    {
        int index = rand() % training_data -> size();
        while(used_indices -> find(index) != used_indices -> end())
        {
            index = rand() % training_data -> size();
        }

        clusters -> push_back(new cluster_t(training_data -> at(index)));
        used_indices -> insert(index);
    }
}

void kmeans :: init_cluster_for_each_class()
{
    set<int> clusters_used;
    for(int i = 0; i < training_data -> size(); i++)
    {
        if(clusters_used.find(training_data -> at(i) -> get_label()) == clusters_used.end())
        {
            clusters -> push_back(new cluster_t(training_data -> at(i)));
            clusters_used.insert(training_data -> at(i) -> get_label());
            used_indices -> insert(i);
        }
    }
}
void kmeans :: train()
{
    int index = 0;
    while(used_indices -> size() < training_data -> size())
    { 
        while(used_indices -> find(index) != used_indices -> end())
        {
            index++;
        }
        double min_dist = numeric_limits<double>::max();
        int bst_cluster = 0;
        for(int j = 0; j < clusters -> size(); j++)
        {
            double current_dist = euclidean_distance(clusters -> at(j) -> centroid, training_data -> at(index));
            if(current_dist < min_dist)
            {
                min_dist = current_dist;
                bst_cluster = j;
            }
        }
        clusters -> at(bst_cluster) -> add_to_cluster(training_data -> at(index));
        used_indices -> insert(index);
    }
}

double kmeans :: euclidean_distance(vector<double> *centroid, Data<DATA_TYPE> *point)
{
    double dist = 0.0;
    for(int i = 0; i < centroid -> size(); i++)
    {
        dist += pow(centroid -> at(i) - point->get_feature_vector()-> at(i), 2);
    }
    dist = sqrt(dist);
    return dist;
}

double kmeans :: validate()
{
    double num_correct = 0.0;
    for(auto query_point : *validation_data)
    {
        double min_dist = numeric_limits<double> :: max();
        int best_cluster = 0;
        for(int j = 0; j < clusters -> size(); j++)
        {
          double current_dist = euclidean_distance(clusters -> at(j) -> centroid, query_point);
          if(current_dist < min_dist)
          {
            min_dist = current_dist;
            best_cluster = j;
          }
        }
        if(clusters -> at(best_cluster) -> most_frequent_class == query_point -> get_label()) num_correct++;
    }
    return 100.0 * (num_correct) / (double) validation_data -> size();
}

double kmeans :: test()
{
    double num_correct = 0.0;
    for(auto query_point : *test_data)
    {
        double min_dist = numeric_limits<double> :: max();
        int best_cluster = 0;
        for(int j = 0; j < clusters -> size(); j++)
        {
          double current_dist = euclidean_distance(clusters -> at(j) -> centroid, query_point);
          if(current_dist < min_dist)
          {
            min_dist = current_dist;
            best_cluster = j;
          }
        }
        if(clusters -> at(best_cluster) -> most_frequent_class == query_point -> get_label()) num_correct++;
    }
    return 100.0 * (num_correct) / (double) test_data -> size();
}


int main()
{
    data_handler<> *dh = new data_handler<>();
    dh -> read_feature_vector("../../train-images.idx3-ubyte");
    dh -> read_feature_label("../../train-labels.idx1-ubyte");
    dh -> split_data();
    dh -> count_classes();

    double performance = 0.0;
    double bst_performance = 0.0;

    int best_k = 1;

    for(int k = dh -> get_class_count(); k < min(dh->get_class_count() + 400,(int)(dh -> get_training_data() -> size() * 0.1)); k++)
    {
        kmeans *knn = new kmeans(k);
        knn -> set_training_data(dh -> get_training_data());
        knn -> set_test_data(dh -> get_test_data());
        knn -> set_validation_data(dh -> get_validation_data());
        knn -> init_cluster();
        knn -> train();
        performance = knn -> validate();

        printf("Current performance @ K = %d: %.2f\n", k, performance);
        if(performance > bst_performance)
        {
            bst_performance = performance;
            best_k = k;
        }
    }

    kmeans *knn = new kmeans(best_k);

    knn -> set_training_data(dh -> get_training_data());
    knn -> set_test_data(dh -> get_test_data());
    knn -> set_validation_data(dh -> get_validation_data());
    knn -> init_cluster();
    knn -> train();
    performance = knn -> test();

    printf("Tested performance @ k = %d: %.2f \n", best_k, performance);

}