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

//numeric 1->int /double,2->string but int,3 ->string

template<typename T>
bool Question<T>::match(vector<T>&example) 
{
    T val=example[column];
    if (this->numeric==1)
    {
        return val>=value;
    }
    else if (this->numeric==2) 
    {
        return stod(val)>=stod(value);
    } 
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
    Question<T>* best_question=new Question<T>(0,"",0);
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

template<typename T>
Node<T>* build_tree(vector<vector<T>>&rows,map<int,int>&numeric) 
{
    auto [gain,question]=find_best_split(rows,numeric);
    if (!gain) 
    {
        return new Leaf<T>(rows);
    }
    auto [true_rows,false_rows]=partition(rows,question);
    auto true_branch=build_tree(true_rows,numeric);
    auto false_branch=build_tree(false_rows,numeric);
    return new Decision_Node<T>(question,true_branch,false_branch);
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
map<T,string>print_leaf(map<T,int>& counts) 
{
    double total=0.0;
    for (auto& count:counts) 
    {
        total+=count.second;
    }
    map<T,string>probs;
    for (auto&count:counts) 
    {
        probs[count.first]=to_string(static_cast<int>((count.second / total) * 100)) + "%";
    }
    return probs;
}

int main() 
{
    vector<vector<string>>training_data=
    {
        {"Green","3","Apple"},
        {"Yellow","3","Apple"},
        {"Red","1","Grape"},
        {"Red","1","Grape"},
        {"Yellow","3","Lemon"}
    };
    map<int,int>numeric;
    numeric[0]=3;
    numeric[2]=3;
    numeric[1]=2;
    auto my_tree=build_tree(training_data,numeric);

    vector<vector<string>>testing_data={
        {"Green","3","Apple"},
        {"Yellow","4","Apple"},
        {"Red","2","Grape"},
        {"Red","1","Grape"},
        {"Yellow","3","Lemon"}
    };

    for (auto&row:testing_data) 
    {
        auto predictions=classify(row,my_tree);
        cout << "Actual: " << row.back() << ". Predicted: ";
        auto probs=print_leaf(predictions);
        for (const auto&prob:probs) 
        {
            cout << prob.first << " " << prob.second << " ";
        }
        cout << endl;
    }

    return 0;
}