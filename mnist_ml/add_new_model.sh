
if [[ -z $MNIST_ML_ROOT ]]; then
    echo "Please define MNSIT ML ROOT"
    exit 1
fi

dir=$(echo "$@" |tr a-z A-Z)
model_name_lower=$(echo "$@" |  tr A-Z a-z)

mkdir -p $MNIST_ML_ROOT/$dir/include $MNIST_ML_ROOT/$dir/src
touch $MNIST_ML_ROOT/$dir/makefile
touch $MNIST_ML_ROOT/$dir/include/"$model_name_lower.hpp"
touch $MNIST_ML_ROOT/$dir/src/"$model_name_lower.cc"