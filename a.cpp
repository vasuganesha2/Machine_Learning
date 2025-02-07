#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <memory>

using namespace std;

// Toy dataset.
// Format: each row is an example.
// The last column is the label.
// The first two columns are features.
vector<vector<string>> training_data = {
    {"Green", "3", "Apple"},
    {"Yellow", "3", "Apple"},
    {"Red", "1", "Grape"},
    {"Red", "1", "Grape"},
    {"Yellow", "3", "Lemon"}
};

// Column labels.
// These are used only to print the tree.
vector<string> header = {"color", "diameter", "label"};

// Function to find unique values for a column in a dataset.
set<string> unique_vals(const vector<vector<string>>& rows, int col) {
    set<string> unique_values;
    for (const auto& row : rows) {
        unique_values.insert(row[col]);
    }
    return unique_values;
}

// Function to count the number of each type of example in a dataset.
map<string, int> class_counts(const vector<vector<string>>& rows) {
    map<string, int> counts;
    for (const auto& row : rows) {
        string label = row.back();
        counts[label]++;
    }
    return counts;
}

// Function to check if a value is numeric.
bool is_numeric(const string& value) {
    try {
        stod(value);
        return true;
    } catch (invalid_argument&) {
        return false;
    }
}

// Class representing a question used to partition the dataset.
class Question {
public:
    int column;
    string value;

    Question(int col, string val) : column(col), value(val) {}

    bool match(const vector<string>& example) const {
        string val = example[column];
        if (is_numeric(val)) {
            return stod(val) >= stod(value);
        } else {
            return val == value;
        }
    }

    string repr() const {
        string condition = "==";
        if (is_numeric(value)) {
            condition = ">=";
        }
        return "Is " + header[column] + " " + condition + " " + value + "?";
    }
};

// Function to partition a dataset.
pair<vector<vector<string>>, vector<vector<string>>> partition(const vector<vector<string>>& rows, const Question& question) {
    vector<vector<string>> true_rows, false_rows;
    for (const auto& row : rows) {
        if (question.match(row)) {
            true_rows.push_back(row);
        } else {
            false_rows.push_back(row);
        }
    }
    return {true_rows, false_rows};
}

// Function to calculate the Gini Impurity for a list of rows.
double gini(const vector<vector<string>>& rows) {
    auto counts = class_counts(rows);
    double impurity = 1.0;
    for (const auto& count : counts) {
        double prob_of_lbl = static_cast<double>(count.second) / rows.size();
        impurity -= pow(prob_of_lbl, 2);
    }
    return impurity;
}

// Function to calculate the information gain.
double info_gain(const vector<vector<string>>& left, const vector<vector<string>>& right, double current_uncertainty) {
    double p = static_cast<double>(left.size()) / (left.size() + right.size());
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right);
}

// Function to find the best question to ask by iterating over every feature / value.
pair<double, Question> find_best_split(const vector<vector<string>>& rows) {
    double best_gain = 0.0;
    Question best_question(0, "");
    double current_uncertainty = gini(rows);
    int n_features = rows[0].size() - 1;

    for (int col = 0; col < n_features; ++col) {
        auto values = unique_vals(rows, col);
        for (const auto& val : values) {
            Question question(col, val);
            auto [true_rows, false_rows] = partition(rows, question);
            if (true_rows.empty() || false_rows.empty()) {
                continue;
            }
            double gain = info_gain(true_rows, false_rows, current_uncertainty);
            if (gain >= best_gain) {
                best_gain = gain;
                best_question = question;
            }
        }
    }
    return {best_gain, best_question};
}

// Base class for nodes in the decision tree.
class Node {
public:
    virtual ~Node() = default; // Virtual destructor for polymorphism
};

// Class representing a leaf node.
class Leaf : public Node {
public:
    map<string, int> predictions;

    Leaf(const vector<vector<string>>& rows) {
        predictions = class_counts(rows);
    }
};

// Class representing a decision node.
class Decision_Node : public Node {
public:
    Question question;
    unique_ptr<Node> true_branch;
    unique_ptr<Node> false_branch;

    Decision_Node(Question q, unique_ptr<Node> tb, unique_ptr<Node> fb)
        : question(q), true_branch(move(tb)), false_branch(move(fb)) {}
};

// Function to build the tree.
unique_ptr<Node> build_tree(const vector<vector<string>>& rows) {
    auto [gain, question] = find_best_split(rows);
    if (gain == 0) {
        return make_unique<Leaf>(rows);
    }
    auto [true_rows, false_rows] = partition(rows, question);
    auto true_branch = build_tree(true_rows);
    auto false_branch = build_tree(false_rows);
    return make_unique<Decision_Node>(question, move(true_branch), move(false_branch));
}

// Function to print the tree.
void print_tree(const unique_ptr<Node>& node, const string& spacing = "") {
    if (auto leaf = dynamic_cast<Leaf*>(node.get())) {
        cout << spacing << "Predict " << endl;
        for (const auto& pred : leaf->predictions) {
            cout << spacing << pred.first << ": " << pred.second << endl;
        }
        return;
    }
    if (auto decision_node = dynamic_cast<Decision_Node*>(node.get())) {
        cout << spacing << decision_node->question.repr() << endl;
        cout << spacing << "--> True:" << endl;
        print_tree(decision_node->true_branch, spacing + "  ");
        cout << spacing << "--> False:" << endl;
        print_tree(decision_node->false_branch, spacing + "  ");
    }
}

// Function to classify a row.
map<string, int> classify(const vector<string>& row, const unique_ptr<Node>& node) {
    if (auto leaf = dynamic_cast<Leaf*>(node.get())) {
        return leaf->predictions;
    }
    if (auto decision_node = dynamic_cast<Decision_Node*>(node.get())) {
        if (decision_node->question.match(row)) {
            return classify(row, decision_node->true_branch);
        } else {
            return classify(row, decision_node->false_branch);
        }
    }
    return {};
}

// Function to print the predictions at a leaf.
map<string, string> print_leaf(const map<string, int>& counts) {
    double total = 0.0;
    for (const auto& count : counts) {
        total += count.second;
    }
    map<string, string> probs;
    for (const auto& count : counts) {
        probs[count.first] = to_string(static_cast<int>((count.second / total) * 100)) + "%";
    }
    return probs;
}

int main() {
    auto my_tree = build_tree(training_data);
    print_tree(my_tree);

    // Evaluate
    vector<vector<string>> testing_data = {
        {"Green", "3", "Apple"},
        {"Yellow", "4", "Apple"},
        {"Red", "2", "Grape"},
        {"Red", "1", "Grape"},
        {"Yellow", "3", "Lemon"}
    };

    for (const auto& row : testing_data) {
        auto predictions = classify(row, my_tree);
        cout << "Actual: " << row.back() << ". Predicted: ";
        auto probs = print_leaf(predictions);
        for (const auto& prob : probs) {
            cout << prob.first << " " << prob.second << " ";
        }
        cout << endl;
    }

    return 0;
}