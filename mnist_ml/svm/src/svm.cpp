#include <iostream>
#include <vector>
#include "../include/svm.hpp"


SVM::SVM(double C,double (*kernel_func)(vector<double>&,vector<double>&))
{
    this->C=C;
    this->kernel=kernel_func;
}

void SVM::fit(vector<vector<double>>&X,vector<int>&y) 
{
    int n=X.size();//n is no. of training pts
    vector<vector<double>>K(n,vector<double>(n,0.0));
    Matrix<double>G(n,n);
    Vector<double>g0(n);
    Matrix<double>CE(n,1);
    Vector<double>ce0(n,1);
    Vector<double>alpha(n);
    Matrix<double>CI(n,2*n);
    Vector<double>ci0(2*n);

    for(int i=0;i<n;i++) 
    {
        for(int j=0;j<n;j++)
        {
            G[i][j]=kernel(X[i],X[j])*y[i]*y[j];
            CI[i][j]=(i==j)?1.0:0.0;
            CI[i][j+n]=(i==j)?-1.0:0.0;
        }
    }
    
    ce0[0]=0.0;

    for(int i=0;i<n;i++) 
    {
        // CE[0][i]=y[i];
        CE[i][0] = y[i];
        g0[i]=-1.0;
        ci0[i]=0.0;
        ci0[n+i]=-C;
    }

    solve_quadprog(G,g0,CE,ce0,CI,ci0,alpha);

    for(int i=0;i<n;i++) 
    {
        if (alpha[i]>THRESHOLD) 
        {  
            alphas.push_back(alpha[i]);
            support_vectors.push_back(X[i]);
            support_labels.push_back(y[i]);
        }
    }
    compute_bias();

}

int SVM::predict(vector<double>&x) 
{
    double sum=0.0;
    for(int i=0;i<alphas.size();i++) 
    {
        sum+=alphas[i]*support_labels[i]*kernel(support_vectors[i],x);
    }
    return (sum+bias)>0?1:-1;
}

    

void SVM::compute_bias() 
{
    double sum=0.0;
    int count=0;
    for(int i=0;i<alphas.size();i++) 
    {
        if (alphas[i]>THRESHOLD) 
        { 
            sum+=support_labels[i];
            for(int j=0;j<alphas.size();j++) 
            {
                sum-=alphas[j]*support_labels[j]*kernel(support_vectors[j],support_vectors[i]); //!shak
            }
            count++;
        }
    }
    bias=sum/count;
}

double rbf_kernel(vector<double>&x1,vector<double>&x2) 
{
    double gamma=0.5;
    double sum=0.0;
    for(int i=0;i<x1.size();i++) 
    {
        sum+=(x1[i]-x2[i])*(x1[i]-x2[i]);
    }
    return exp(-gamma*sum);
}

void standardize(vector<vector<double>> &x)
{
    int n=x.size();
    int features=x[0].size();
    vector<double> mean(features,0.0);
    vector<double> std(features,0.0);
    for(int i=0;i<features;i++)
    {
        for(int j=0;j<n;j++)
        {
            mean[i]+=x[j][i];
        }
        mean[i]/=n;
    }
    for(int i=0;i<features;i++)
    {
        for(int j=0;j<n;j++)
        {
            std[i]+=(x[j][i]-mean[i])*(x[j][i]-mean[i]);
        }
        std[i]=sqrt(std[i]/n);
    }
    for(int i=0;i<features;i++)
    {
        for(int j=0;j<n;j++)
        {
            if(std[i])
            {
                x[j][i]=(x[j][i]-mean[i])/std[i];
            }
            else x[j][i]=0.0; //?what to do if all features have same value -> feature not imp ??
        }
    }
}

int main() 
{
    data_handler*dh = new data_handler();
    dh -> read_feature_vector("../../train-images.idx3-ubyte");
    dh -> read_feature_label("../../train-labels.idx1-ubyte");
    dh -> split_data();
    dh -> count_classes();
    vector<Data<DATA_TYPE>*> *x_train=dh->get_training_data();
    vector<vector<double>> x(100);
    vector<int> y(100,0);
    int i=0;
    map<int,int>mp;
    for (Data<DATA_TYPE>* data:*x_train)
    {
        vector<DATA_TYPE>* cur=(data->get_feature_vector());
        int label=(data->get_label());
        if(label!=1) continue;
        mp[label]++;
        for (auto element:*cur) 
        {
            x[i].push_back((int32_t)element);
        }
        y[i]=(1);
        i++;
        if(mp[label]>=50) break;
    }
    for (auto data:*x_train)
    {
        vector<DATA_TYPE>* cur=(data->get_feature_vector());
        int label=(data->get_label());
        if(label==1) continue;
        mp[label]++;
        for (auto element:*cur) 
        {
            x[i].push_back((int32_t)element);
        }
        y[i]=(-1);
        i++;
        if(i==100) break;
    }
    standardize(x);
    
    vector<int>idx(100);
    iota(idx.begin(),idx.end(),0);
    random_device rd;
    mt19937 g(rd());
    shuffle(idx.begin(),idx.end(),g);
    auto x_final=x;
    auto y_final=y;
    for(int i=0;i<idx.size();i++)
    {
        x_final[i]=x[idx[i]];
        y_final[i]=y[idx[i]];
    }
    x=x_final;
    y=y_final;

    SVM svc(1.0,rbf_kernel);
    svc.fit(x,y);

    vector<Data<DATA_TYPE>*> *x_valid=dh->get_validation_data();
    vector<vector<double>> x_val(20);
    vector<int> y_val(20,0);
    i=0;
    mp.clear();
    for (auto data:*x_valid)
    {
        vector<DATA_TYPE>* cur=(data->get_feature_vector());
        int label=(data->get_label());
        if(label!=1) continue;
        mp[label]++;
        for (auto element:*cur) 
        {
            x_val[i].push_back((int32_t)element);
        }
        y_val[i]=(1);
        i++;
        if(mp[label]>=10) break;
    }
    for (auto data:*x_valid)
    {
        vector<DATA_TYPE>* cur=(data->get_feature_vector());
        int label=(data->get_label());
        if(label==1) continue;
        mp[label]++;
        for (auto element:*cur) 
        {
            x_val[i].push_back((int32_t)element);
        }
        y_val[i]=(-1);
        i++;
        if(i==20) break;
    }
    standardize(x_val);



    vector<Data<DATA_TYPE>*> *x_testing=dh->get_test_data();
    vector<vector<double>> x_test(25);
    vector<int> y_test(25,0);
    i=0;
    mp.clear();
    for (auto data:*x_testing)
    {
        vector<DATA_TYPE>* cur=(data->get_feature_vector());
        int label=(data->get_label());
        if(label!=1) continue;
        mp[label]++;
        for (auto element:*cur) 
        {
            x_test[i].push_back((int32_t)element);
        }
        y_test[i]=(1);
        i++;
        if(mp[label]>=10) break;
    }
    for (auto data:*x_testing)
    {
        vector<DATA_TYPE>* cur=(data->get_feature_vector());
        int label=(data->get_label());
        if(label==1) continue;
        mp[label]++;
        for (auto element:*cur) 
        {
            x_test[i].push_back((int32_t)element);
        }
        y_test[i]=(-1);
        i++;
        if(i==25) break;
    }
    standardize(x_test);
    int correct=0;
    int total=0;
    for(int i=0;i<x_test.size();i++)
    {
        if(y_test[i]==svc.predict(x_test[i])) correct++;
        total++;
    }
    cout<<"Done!\n";
    cout<<setprecision(4)<<(double)correct/(double)total<<'\n';

    return 0;
}
