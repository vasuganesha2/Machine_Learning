#include "../include/dt.hpp"

template<typename T>
set<T>unq(vector<vector<T>>&rows,int col) 
{
    set<T>val;
    for (auto&row:rows) 
    {
        val.insert(row[col]);
    }
    return val;
}

template<typename T>
map<T,int> cnt(vector<vector<T>>&rows) 
{
    map<T,int>counts;
    for (auto&row:rows) 
    {
        T label=row.back();
        counts[label]++;
    }
    return counts;
}



template<typename T>
Question<T>::Question(int col,T val,int numeric) 
{
    this->column=col;
    this->value=val;
    this->numeric=numeric;
}

//? numeric 1->int /double,2->string but int,3 ->string

template<typename T>
bool Question<T>::match(vector<T>&example) 
{
    T val=example[column];
    if (this->numeric==1)
    {
        return val>=value;
    }
    // else if (this->numeric==2) 
    // {
    //     return stod(val)>=stod(value);
    // } 
    else 
    {
        return val==value;
    }
}

template<typename T>
pair<vector<vector<T>>,vector<vector<T>>> partition(vector<vector<T>>&rows,Question<T>* question) 
{
    vector<vector<T>>true_rows,false_rows;
    for (auto&row:rows) 
    {
        if (question->match(row)) 
        {
            true_rows.push_back(row);
        } else 
        {
            false_rows.push_back(row);
        }
    }
    return {true_rows,false_rows};
}

template<typename T>
double gini(vector<vector<T>>&rows) 
{
    auto counts=cnt(rows);
    double impurity=1.0;
    for (auto& count:counts) 
    {
        double prob_of_lbl=(double)(count.second)/rows.size();
        impurity-=pow(prob_of_lbl,2);
    }
    return impurity;
}

template<typename T>
double info_gain(vector<vector<T>>&left,vector<vector<T>>&right, double uncertain) 
{
    double p=((double)(left.size()))/((double)(left.size()+right.size()));
    return uncertain-p*gini(left)-(1-p)*gini(right);
}

template<typename T>
pair<double,Question<T>*>find_best_split(vector<vector<T>>&rows,map<int,int>&numeric) 
{
    double best_gain=0.0;
    Question<T>* best_question=new Question<T>(0,0,0);
    double uncertain=gini(rows);
    int n_features=rows[0].size()-1;

    for (int col=0; col < n_features; ++col) 
    {
        auto values=unq(rows,col);
        for (auto&val:values) 
        {
            Question<T>* question=new Question<T>(col,val,numeric[col]);
            auto [true_rows,false_rows]=partition(rows,question);
            if (true_rows.empty() || false_rows.empty()) 
            {
                continue;
            }
            double gain=info_gain(true_rows,false_rows,uncertain);
            if (gain>=best_gain) 
            {
                best_gain=gain;
                best_question=question;
            }
        }
    }
    return {best_gain,best_question};
}

template<typename T>
Leaf<T>::Leaf(vector<vector<T>>&rows) 
{
    predictions=cnt(rows);
}

template<typename T>
Decision_Node<T>::Decision_Node(Question<T>* q,Node<T>* tb,Node<T>* fb) 
{
    question=q;
    true_branch=tb;
    false_branch=fb;
}

int a=0;
#define MAX_DEPTH 10
template <typename T>
Node<T>* build_tree(vector<vector<T>>& rows, map<int, int>& numeric, int depth) 
{
    auto [gain, question] = find_best_split(rows, numeric);
    
    // If no gain or max depth reached, return a leaf node
    if (!gain || depth >= MAX_DEPTH) 
    {
        return new Leaf<T>(rows);
    }
    
    auto [true_rows, false_rows] = partition(rows, question);
    cout << "DONE " << (a++) << endl;
    
    // Recursively build the true and false branches
    auto true_branch = build_tree(true_rows, numeric, depth + 1);
    auto false_branch = build_tree(false_rows, numeric, depth + 1);
    
    return new Decision_Node<T>(question, true_branch, false_branch);
}
template<typename T>
map<T,int>classify(vector<T>& row,Node<T>* node) 
{
    if (auto leaf=dynamic_cast<Leaf<T>*>(node)) 
    {
        return leaf->predictions;
    }
    if (auto decision_node=dynamic_cast<Decision_Node<T>*>(node)) 
    {
        if (decision_node->question->match(row)) 
        {
            return classify(row,decision_node->true_branch);
        } 
        else 
        {
            return classify(row,decision_node->false_branch);
        }
    }
    return {};
}

template<typename T>
map<T,double>print_leaf(map<T,int>& counts) 
{
    double total=0.0;
    for (auto& count:counts) 
    {
        total+=count.second;
    }
    map<T,double>probs;
    for (auto&count:counts) 
    {
        probs[count.first]=(((double)count.second / (double)total) * 100);
    }
    return probs;
}
void prepare_data(vector<Data<DATA_TYPE>*>*vec, vector<vector<DATA_TYPE>>&res, int siz)
{
    for(auto &i:*vec)
    {
        vector<DATA_TYPE>*v=i->get_feature_vector();
        vector<DATA_TYPE>temp=*v;
        // temp.push_back((i->get_label()==4)?1:0);
        temp.push_back(i->get_label());
        res.push_back(temp);
        if(res.size()>siz) break;
    }
}
int main() 
{
    // vector<vector<string>>training_data=
    // {
    //     {"Green", "3","Person a"},
    //     {"Yellow","3","Person a"},
    //     {"Red","1","Person b"},
    //     {"Red","1","Person b"},
    //     {"Yellow","4","Person c"}
    // };
    // map<int,int>numeric;
    // numeric[0]=3;
    // numeric[2]=3;
    // numeric[1]=2;
    // auto my_tree=build_tree(training_data,numeric);

    // vector<vector<string>>testing_data=
    // {
    //     {"Green","3","Person a"},
    //     {"Yellow","4","Person a"},
    //     {"Red","2","Person b"},
    //     {"Red","1","Person b"},
    //     {"Yellow","3","Person c"}
    // };

    // for (auto&row:testing_data) 
    // {
    //     auto predictions=classify(row,my_tree);
    //     cout << "Actual: " << row.back() << ". Predicted: ";
    //     auto probs=print_leaf(predictions);
    //     for (const auto&prob:probs) 
    //     {
    //         cout << prob.first << " " << prob.second << " ";
    //     }
    //     cout << endl;
    // }

    data_handler<>*dh = new data_handler();
    dh -> read_feature_vector("../../train-images.idx3-ubyte");
    dh -> read_feature_label("../../train-labels.idx1-ubyte");
    dh -> split_data();
    dh -> count_classes();
    vector<Data<DATA_TYPE> *> *train_data=dh->get_training_data();
    // vector<Data<DATA_TYPE> *> *valid_data=dh->get_validation_data();
    vector<Data<DATA_TYPE> *> *test_data=dh->get_test_data();
    vector<vector<DATA_TYPE>>train;
    vector<vector<DATA_TYPE>>test;

    prepare_data(train_data,train,200);
    prepare_data(test_data,test,100);
    
    map<int,int>numeric;
    int a=0;
    for(int i=0;i<test[0].size();i++) numeric[i]=1,a++;
    auto my_tree=build_tree(train,numeric,0);
    
    double correct=0,total=0;
    for (auto&row:test) 
    {
        auto predictions=classify(row,my_tree);
        // cout << "Actual: " << row.back() << ". Predicted: ";
        auto probs=print_leaf(predictions);
        double mx=0;

        for (const auto&prob:probs) 
        {
            cout << prob.first << " " << prob.second << " ";
            mx=max(mx,prob.second);
        }
        if(probs[row.back()]==mx) correct++;
        
        total++;
        cout << endl;
    }
    cout<<correct<<" "<<total<<endl;
    cout<<"Accuracy"<<setprecision(4)<<correct/total<<endl;


    return 0;
}