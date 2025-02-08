#ifndef __KMEAN_HPP
#define _KMEAN_HPP

#include "common.hpp"
#include "data_handler.hpp"
#include<bits/stdc++.h>

using namespace std;


typedef struct cluster
{

    vector<double> *centroid;
    vector<Data<DATA_TYPE> *> *cluster_point;
    map<int, int> class_count;

    int most_frequent_class = 0;


    cluster(Data<DATA_TYPE> *initial_point)
    {
        centroid = new vector<double>;
        cluster_point = new vector<Data<DATA_TYPE> *>;
        for(auto value : *(initial_point -> get_feature_vector()))
        {
            centroid -> push_back(value);
        }
        cluster_point -> push_back(initial_point);
        class_count[initial_point -> get_label()] = 1;
        most_frequent_class = initial_point -> get_label();
    }


    void add_to_cluster(Data<DATA_TYPE> *point)
    {
        int previous_size = cluster_point -> size();
        cluster_point -> push_back(point);
        for(int i = 0; i < centroid -> size() - 1; i++)
        {
            double value = centroid -> at(i);
            value *= previous_size;
            value += point -> get_feature_vector() -> at(i);
            value /= (double)cluster_point->size();
            centroid -> at(i) = value;
        } 
        class_count[point -> get_label()]++;
        set_most_freq_class();
    }
    void set_most_freq_class()
    {
        int best_class = -1;
        int freq = 0;
        for(auto kv : class_count)
        {
            if(kv.second > freq)
            {
                freq = kv.second;
                best_class = kv.first;
            }
        }
        most_frequent_class = best_class;
    }


}cluster_t;

class kmeans: virtual public common_data
{
    int num_cluster;
    vector<cluster_t *> *clusters;
    set<int> *used_indices;

    public:
    kmeans(int k);
    //~kmeans();

    void init_cluster();
    void init_cluster_for_each_class();
    void train();
    double euclidean_distance(vector<double>*, Data<DATA_TYPE> *);
    double validate();
    double test();

};

#endif