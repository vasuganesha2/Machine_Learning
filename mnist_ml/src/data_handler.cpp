#include "data_handler.hpp"

#include <bits/stdc++.h>

using namespace std;

data_handler::data_handler()
{
    data_array = new vector<Data<DATA_TYPE>*>;
    training_data = new vector<Data<DATA_TYPE>*>;
    test_data = new vector<Data<DATA_TYPE>*>;
    validation_data = new vector<Data<DATA_TYPE>*>;
}

data_handler::~data_handler()
{
}



void data_handler:: read_feature_vector(std:: string path)
{
    uint32_t header[4];
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if(f)
    {
        for(int i = 0; i < 4; i++)
        {
            if(fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Converted to litle endian succesfully \n");

        int image_size = header[2] * header[3];

        for(uint32_t i = 0; i < header[1]; i++)
        {
            Data<DATA_TYPE> * d = new Data<DATA_TYPE>();
            uint8_t element[1];
            for(int j = 0; j < image_size; j++)
            {
                if(fread(element, sizeof(element), 1, f)) d->append_to_feature_vector(element[0]);
                else 
                {
                    printf("Error while Readint the File \n");
                    exit(1);
                }
            }
            data_array->push_back(d);
        }
        printf("Succesfully read and store the feature vector \n");
    }
    else
    {
        printf("Could Not Find The File \n");
        exit(1);
    }

}





void data_handler::read_feature_label(std:: string path)
{
    uint32_t header[2];
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if(f)
    {
        for(int i = 0; i < 2; i++)
        {
            if(fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Converted to litle endian succesfully \n");

        //cout << header[0] << " " << header[1] << endl;



        for(uint32_t i = 0; i < header[1]; i++)
        {
            uint8_t element[1];

            if(fread(element, sizeof(element), 1, f)) data_array->at(i)->set_label(element[0]);
            else 
            {
                printf("Error while Reading the File \n");
                exit(1);
            }
            

        }
        printf("Succesfully read and store the feature label \n");
    }
    else
    {
        printf("Could Not Find The File \n");
        exit(1);
    }
}




void data_handler::split_data()
{
    int train_size = TRAIN_SET_PERCENT * data_array->size();
    int validation_size = VALIDATION_SET_PERCENT * data_array->size();
    int test_size = TEST_SET_PERCENT * data_array->size();

    printf("Training data size is %d \n", train_size);
    printf("Validation data size is %d \n", validation_size);
    printf("Testing data size is %d \n", test_size);

    // Generate a list of all indices
    std::vector<int> indices(data_array->size());
    std::iota(indices.begin(), indices.end(), 0);  // Fill indices with 0, 1, 2, ..., n-1

    // Shuffle the indices to randomize their order
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Split data using shuffled indices
    for (int i = 0; i < train_size; ++i) {
        training_data->push_back(data_array->at(indices[i]));
    }

    for (int i = train_size; i < train_size + test_size; ++i) {
        test_data->push_back(data_array->at(indices[i]));
    }

    for (int i = train_size + test_size; i < train_size + test_size + validation_size; ++i) {
        validation_data->push_back(data_array->at(indices[i]));
    }

    printf("Training data size is %d \n", train_size);
    printf("Validation data size is %d \n", validation_size);
    printf("Testing data size is %d \n", test_size);
}




void  data_handler :: count_classes()
{
    int count = 0;
    for(uint32_t i = 0; i < data_array->size(); i++)
    {
        if(class_map.find(data_array->at(i)->get_label()) == class_map.end())
        {
            class_map[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enumerated_label(count);
            count++;
        }
    }
    num_classes = count;
    cout << "Successfully Extracted " << num_classes << " Unique Classes." << endl;
}

uint32_t data_handler::convert_to_little_endian(const unsigned char *bytes)
{
    return (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]));
}

vector<Data<DATA_TYPE>*> * data_handler :: get_training_data() 
{
    return training_data;
}
vector<Data<DATA_TYPE>*>* data_handler::get_test_data() 
{
    return test_data;
}

vector<Data<DATA_TYPE>*> * data_handler :: get_validation_data()
{
    return validation_data;
}


int data_handler :: get_class_count()
{
    return num_classes;
}

int main()
{
    data_handler *dh = new data_handler();
    dh -> read_feature_vector("../train-images.idx3-ubyte");
    dh -> read_feature_label("../train-labels.idx1-ubyte");
    dh -> split_data();
    dh -> count_classes();
}