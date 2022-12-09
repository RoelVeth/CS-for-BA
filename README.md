# CS-for-BA
Assignment for computer science of business analytics by Roel Veth (622593)
This project is about finding duplicate products in different web shops.
Duplicates are indicated by classification.

Data is first loaded and cleaned.
A characteristic matrix is made from binary vectors.
A signature matrix is made by minhashing the characteristic matrix.

By bootstrapping training and test data is created.
Parameters are optimized on the training data, performance is measured on the test data.
Performance over each bootstrap is averaged.

For a more indepth description of the program I refer to the data en method chapters of the paper in this repository.
