# FunWithTensorflow


## Test interface to c++ 
TestCpp needs to be executed inside `/tensorflow/tensorflow/TestCpp` (tensorflow repo cloned from git@github.com:tensorflow/tensorflow.git ) with `bazel build :loader`. This places an executable (`loader`) in `tensorflow/bazel-bin/.../TestCpp`. Make sure to copy the graph from `tensorflow/tensorflow/TestCpp/models/graph.pb` to `tensorflow/bazel-bin/.../TestCpp/models/train.pb`. 

cf. tutorial <https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f#.6ryjxfu38>
