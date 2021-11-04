# import json
# import os

# data_path1 = "/home/akira/Desktop/p-poteka/mlflow_pipeline_example/mlruns/0/5c25106843264812a45f4cf21f2bf3f3/artifacts/downstream_directory"
# data_path2 = "/home/akira/Desktop/p-poteka/mlflow_pipeline_example/mlruns/0/48978dff7b254dc1b77b1755e9d06958/artifacts/downstream_directory"


# data1_train_f, data1_valid_f = open(os.path.join(data_path1, "meta_train.json")), open(os.path.join(data_path1, "meta_valid.json"))
# data2_train_f, data2_valid_f = open(os.path.join(data_path2, "meta_train.json")), open(os.path.join(data_path2, "meta_valid.json"))

# train1, valid1 = json.load(data1_train_f), json.load(data1_valid_f)
# train2, valid2 = json.load(data2_train_f), json.load(data2_valid_f)

# assert train1 == train2
# assert valid1 == valid2

# from train.src.data_loader import sample_data_loader

# train_dataset, test_dataset = sample_data_loader(
#     train_size=30,
#     test_size=10,
#     batch_num=6,
#     height=50,
#     width=50,
#     vector_size=4,
# )

# X_train, y_train = train_dataset[0], train_dataset[1]
# X_test, y_test = test_dataset[0], test_dataset[1]

# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)
# print(X_test.max(), X_test.min())
# print("hello")

# root_dir = "/home/akira/Desktop/p-poteka/data"
# child_path = "data/dadfvv/dsmaple.scv"
# import os

# print(os.path.join(root_dir, child_path))

import numpy as np

l = [[np.empty([2])] * 3]
print(l)

print(l)
