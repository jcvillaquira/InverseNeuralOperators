# Inverse HJB Problem with NeuralOperators.jl
This repository contains a 1-dimensional HJB inverse problem.
```
.
├── data
│   ├── burgers_data_R10.mat
│   └── Burgers_R10.zip
├── hjb
│   ├── 20230729
│   │   ├── data.bson
│   │   ├── forward.bson
│   │   ├── inverse.bson
│   │   └── tuning.bson
│   └── 20230730
│       ├── inverse_definitive.bson
│       ├── inverse_definitive_.bson
│       └── tuning.bson
├── main_hjb_nn.jl
├── main_hjb_no.jl
├── Manifest.toml
├── models
│   ├── data.bson
│   ├── forward.bson
│   ├── forward_hjb.bson
│   ├── inverse.bson
│   ├── test.bson
│   └── xdata.bson
├── notes
│   └── compacity.md
├── old
│   ├── main_hjb_legacy.jl
│   └── main.jl
├── Project.toml
├── README.md
├── src
│   ├── DataGenerator.jl
│   └── HJBDataGenerator.jl
└── tests.jl
9 directories, 26 files
```
