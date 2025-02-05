#ifndef __DATA_HANDLER_TPP
#define __DATA_HANDLER_TPP

#include "data_handler.hpp"
#include <bits/stdc++.h>

using namespace std;

template <typename T, typename T1>
data_handler<T, T1>::data_handler()
{
    data_array = new vector<Data<T>*>;
    training_data = new vector<Data<T>*>;
    test_data = new vector<Data<T>*>;
    validation_data = new vector<Data<T>*>;
}

template <typename T, typename T1>
data_handler<T, T1>::~data_handler()
{
    for (auto d : *data_array) delete d;
    delete data_array;
    delete training_data;
    delete test_data;
    delete validation_data;
}

template <typename T, typename T1>
void data_handler<T, T1>::read_feature_vector(string path)
{
    uint32_t header[4];
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "rb"); // Open in binary mode
    if (!f)
    {
        printf("Could not find the file: %s\n", path.c_str());
        exit(1);
    }

    for (int i = 0; i < 4; i++)
    {
        if (fread(bytes, sizeof(bytes), 1, f))
        {
            header[i] = convert_to_little_endian(bytes);
        }
        else
        {
            printf("Error reading file header\n");
            fclose(f);
            exit(1);
        }
    }

    printf("Converted to little endian successfully\n");

    int image_size = header[2] * header[3];

    for (uint32_t i = 0; i < header[1]; i++)
    {
        Data<T>* d = new Data<T>();
        uint8_t element;
        for (int j = 0; j < image_size; j++)
        {
            if (fread(&element, sizeof(element), 1, f))
                d->append_to_feature_vector(element);
            else 
            {
                printf("Error while reading feature vector\n");
                fclose(f);
                exit(1);
            }
        }
        data_array->push_back(d);
    }

    fclose(f);
    printf("Successfully read and stored the feature vector\n");
}

template <typename T, typename T1>
void data_handler<T, T1>::read_feature_label(string path)
{
    uint32_t header[2];
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "rb"); // Open in binary mode
    if (!f)
    {
        printf("Could not find the file: %s\n", path.c_str());
        exit(1);
    }

    for (int i = 0; i < 2; i++)
    {
        if (fread(bytes, sizeof(bytes), 1, f))
        {
            header[i] = convert_to_little_endian(bytes);
        }
        else
        {
            printf("Error reading file header\n");
            fclose(f);
            exit(1);
        }
    }

    printf("Converted to little endian successfully\n");

    for (uint32_t i = 0; i < header[1]; i++)
    {
        uint8_t element;
        if (fread(&element, sizeof(element), 1, f))
            data_array->at(i)->set_label(element);
        else 
        {
            printf("Error while reading feature labels\n");
            fclose(f);
            exit(1);
        }
    }

    fclose(f);
    printf("Successfully read and stored the feature labels\n");
}

template <typename T, typename T1>
void data_handler<T, T1>::split_data()
{
    int train_size = TRAIN_SET_PERCENT * data_array->size();
    int validation_size = VALIDATION_SET_PERCENT * data_array->size();
    int test_size = TEST_SET_PERCENT * data_array->size();

    printf("Training data size: %d\n", train_size);
    printf("Validation data size: %d\n", validation_size);
    printf("Testing data size: %d\n", test_size);

    // Generate a list of all indices
    vector<int> indices(data_array->size());
    iota(indices.begin(), indices.end(), 0);  // Fill indices with 0, 1, 2, ..., n-1

    // Shuffle the indices to randomize their order
    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    // Split data using shuffled indices
    for (int i = 0; i < train_size; ++i) 
        training_data->push_back(data_array->at(indices[i]));

    for (int i = train_size; i < train_size + test_size; ++i) 
        test_data->push_back(data_array->at(indices[i]));

    for (int i = train_size + test_size; i < train_size + test_size + validation_size; ++i) 
        validation_data->push_back(data_array->at(indices[i]));
}

template <typename T, typename T1>
void data_handler<T, T1>::count_classes()
{
    int count = 0;
    for (uint32_t i = 0; i < data_array->size(); i++)
    {
        T1 label = data_array->at(i)->get_label();
        if (class_map.find(label) == class_map.end())
        {
            class_map[label] = count;
            class_map_inverse[count] = label;
            data_array->at(i)->set_enumerated_label(count);
            count++;
        }
    }
    num_classes = count;
    cout << "Successfully extracted " << num_classes << " unique classes." << endl;
}

template <typename T, typename T1>
uint32_t data_handler<T, T1>::convert_to_little_endian(const unsigned char *bytes)
{
    return (static_cast<uint32_t>(bytes[0]) << 24) | 
           (static_cast<uint32_t>(bytes[1]) << 16) | 
           (static_cast<uint32_t>(bytes[2]) << 8) | 
           (static_cast<uint32_t>(bytes[3]));
}

template <typename T, typename T1>
vector<Data<T>*>* data_handler<T, T1>::get_training_data() 
{
    return training_data;
}

template <typename T, typename T1>
vector<Data<T>*>* data_handler<T, T1>::get_test_data() 
{
    return test_data;
}

template <typename T, typename T1>
vector<Data<T>*>* data_handler<T, T1>::get_validation_data()
{
    return validation_data;
}

template <typename T, typename T1>
int data_handler<T, T1>::get_class_count()
{
    return num_classes;
}

template <typename T, typename T1>
map<int, T1> data_handler<T, T1>::get_class_map_inverse()
{
    return class_map_inverse;
}

template <typename T, typename T1>
void data_handler<T, T1>::read_csv(string path, string delimiter)
{
    int num_classes = 0;
    std::ifstream data_file(path.c_str());
    string line;

    if (!data_file.is_open())
    {
        cerr << "Error: Could not open file " << path << endl;
        return;
    }

    while (std::getline(data_file, line))
    {
        if (line.length() == 0) continue; 
        Data<T> *d = new Data<T>(); 
        d->feature_vector(new std::vector<T>());  

        size_t position = 0;
        string token;  // Value in between the delimiter

        vector<double> *features = d->get_double_feature_vector();  // Store features in Data object

        // Tokenize the line using the delimiter
        while ((position = line.find(delimiter)) != string::npos)
        {
            token = line.substr(0, position);  // Extract token
            features->push_back(static_cast<T>(stoll(token)));  // Convert to double and store in feature vector
            line.erase(0, position + delimiter.length());  // Remove processed part
        }

        if (!line.empty())
        {
            T1 label = static_cast<T1>(stoll(line)); // Convert line to T1

            if (class_map.find(label) == class_map.end()) 
            {
                class_map[label] = num_classes++; // Store converted label in class_map
            }

            d->set_label(class_map[label]); // Assign label to the data object
        }

        data_array->push_back(d);  // Store Data object in main dataset
    }

    data_file.close();
}


#endif // __DATA_HANDLER_TPP
