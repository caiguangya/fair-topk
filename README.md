## Overview
This branch contains codes for preprocessing 6-D COMPAS and 3-D IIT-JEE datasets for different $k$ and $n$ values.

## Build
Same as the `main` branch.

## Preprocessing
1. [Download raw datasets](https://www.dropbox.com/scl/fo/of387p4m1lpgh05q2x75j/AJy3Sn5r97WBRI3Vi4VRb_A?rlkey=hvqpbr6qv3xe0gl5h7teez2tk&st=f6se30uq&dl=0)
2. Launch the container
3. Run preprocess_data inside the container
    ```
    preprocess_data <RAW DATASET PATH> k n <PREPROCESSED DATASET PATH>
    ```

    e.g., ```preprocess_data compas-scores-two-years.csv 50 1.0 compas-50.csv``` produces the compas-50.csv of the preprocessed datasets.

Again, you might need to apply a new [Gurobi license](https://www.gurobi.com/features/web-license-service/) for preprocessing the 6-D COMPAS datasets.

## Note
Since the preprocessing time is not the focus of this work, preprocessing algorithms and their implemenations have not been optimized for speed.