# Finding a Fair Scoring Function for Top-k Selection: Hardness, Algorithms and Experiments

## Overview
This repository hosts codes for *Finding a Fair Scoring Function for Top-k Selection: Hardness, Algorithms and Experiments* (in submission, [full version](https://arxiv.org/abs/2503.11575) available).

The `main` branch is all you need for reproducing experimental results and the `preprocessing` branch contains codes for data preprocessing.

## Build
1. [Install Apptainer](https://apptainer.org/docs/admin/main/installation.html)
2. Build container image
    ```
    apptainer build ./container/fair_topk_container.sif ./container/fair_topk_container.def
    ```
3. Launch the container
    ```
    apptainer run ./container/fair_topk_container.sif
    ```
4. Compilation
    ```
    cmake . && make -j
    ```

Output programs: **klevel_based_method**, **klevel_based_method_2d**, **mip_based_method**, **baseline** and **baseline_2d**

## Reproducibility
1. [Download preprocessed datasets](https://www.dropbox.com/scl/fo/of387p4m1lpgh05q2x75j/AJy3Sn5r97WBRI3Vi4VRb_A?rlkey=hvqpbr6qv3xe0gl5h7teez2tk&st=f6se30uq&dl=0)
2. Launch the container
3. Run programs inside the container
    ```
    program <PREPROCESSED DATASET PATH> k lower_bound upper_bound epsilon sample_count [thread_count]
    ```
    * program: klevel_based_method, klevel_based_method_2d, mip_based_method, baseline or baseline_2d
    * lower_bound: Proportional lower bound for the protected group
    * upper_bound: Proportional upper bound for the protected group
    * sample_count: Number of unfair weight vectors
    * thread_count: Number of threads (only needed for klevel_based_method and mip_based_method)

    e.g., ``` klevel_based_method compas-50.csv 50 0.4 0.6 0.05 20 64 ``` is the default experimental setting for the k-level-based method on the 6-D COMPAS dataset.

For executing mip_based_method inside the container, you might need to apply a new [Gurobi license](https://www.gurobi.com/features/web-license-service/). Before executing mip_based_method, run the following command:
```
export GRB_LICENSE_FILE=\path\to\gurobi\license
```
Make sure that the license file is accessible inside the container.
