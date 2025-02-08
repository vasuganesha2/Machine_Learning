#ifndef DT_HPP
#define DT_HPP
#include <bits/stdc++.h>
#include "common.hpp"
#include "data_handler.hpp"
#include "data.hpp"
#include "../../PCA/include/pca.hpp"
using namespace std;

template<typename T>
class Node
{
public:
    virtual ~Node()=default;
};

template<typename T>
class Question 
{
public:
    int column;
    T value;
    int numeric;
    Question(int col,T val,int numeric);
    bool match(vector<T>&example);
};

template<typename T>
class Leaf:public Node<T> 
{
public:
    map<T,int> predictions;
    Leaf(vector<vector<T>>&rows);
};

template<typename T>
class Decision_Node:public Node<T> 
{
public:
    Question<T>*question;
    Node<T>*true_branch;
    Node<T>*false_branch;

    Decision_Node(Question<T>*q,Node<T>*tb,Node<T>*fb);
};



//helper

template<typename T>
set<T> unq(vector<vector<T>>&rows,int col);

template<typename T>
map<T,int> cnt(vector<vector<T>>&rows);


//! numeric
//? 1->int /double,2->string but int like "123" ,3 ->string (categorical data) 


template<typename T>
pair<vector<vector<T>>,vector<vector<T>>> partition(vector<vector<T>>&rows,Question<T>*question);

template<typename T>
double gini(vector<vector<T>>&rows);

template<typename T>
double info_gain(vector<vector<T>>&left,vector<vector<T>>&right,double cur);

template<typename T>
pair<double,Question<T>*> find_best_split(vector<vector<T>>&rows,map<int,int>&numeric);

template<typename T>
Node<T>* build_tree(vector<vector<T>>& rows, map<int, int>& numeric, int depth);

template<typename T>
map<T,int> classify(vector<T>&row,Node<T>*node);

template<typename T>
map<T,double> print_leaf(map<T,int>&counts);

void prepare_data(vector<Data<DATA_TYPE>*>*vec, vector<vector<DATA_TYPE>>&res, int siz);


#endif // DT_HPP