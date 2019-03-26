import tensorflow as tf

# 关于tf.train.Example的定义
message Example {
    Features features = 1;
};

message Features {
    map<string, Feature> feature = 1;
};

message Feature {
    oneof kind {
        BytesList bytes_list = 1;
        FloatList = float_list = 2;
        Int64List int64_list = 3;
    }
};

所以从上面可以看出有三种存放数据的类型

