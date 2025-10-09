# Finding a Fair Scoring Function for Top-k Selection: From Hardness to Practice

## Overview
This repository hosts codes for *Finding a Fair Scoring Function for Top-k Selection: From Hardness to Practice* ([arXiv](https://arxiv.org/abs/2503.11575)).

The `main` branch is all you need for reproducing experimental results and the `preprocessing` branch contains codes for data preprocessing.

## Build
### Containerization (recommended)
This project can be built using either Apptainer or Docker containerization. 

#### Apptainer
1. [Install Apptainer](https://apptainer.org/docs/admin/main/installation.html) (see [this](https://github.com/apptainer/apptainer/blob/main/INSTALL.md#apparmor-profile-ubuntu-2310) for installation on Ubuntu 23.10+)
2. Build Apptainer image
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
#### Docker
1. [Install Docker](https://docs.docker.com/engine/install)
2. Pull Docker image
   ```
   docker pull caiguangya/fair-topk:latest
   ```
3. Launch the container
   ```
   docker run -it -v $(pwd):/fair-topk -w /fair-topk caiguangya/fair-topk:latest
   ```
4. Compilation
   ```
   cmake . && make -j
   ```
### Local
1. Install dependencies
   - g++ (>= 13.3), [CMake](https://cmake.org/) (>= 3.12)
   - [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (>= 3.4)
   - [Gurobi](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer) (>= 11.0.3)
   - [SCIP](https://github.com/scipopt/scip) (>= 9.2.2), [PaPILO](https://github.com/scipopt/papilo) (>= 2.4.2), [SoPlex](https://github.com/scipopt/soplex) (>= 7.1.4)
   - [CLI11](https://github.com/CLIUtils/CLI11) (>= 2.5.0)
   - [libcds](https://github.com/khizmax/libcds)
   - [Boost](https://www.boost.org/)
   - [xtensor](https://github.com/xtensor-stack/xtensor)
  
   See ```container/fair_topk_container.def``` for installation commands on Debian or Ubuntu systems.
  
2. Compilation
   ```
   cmake . && make -j
   ```
   
Output programs: **klevel_based_method**, **klevel_based_method_2d**, **mip_based_method**, **baseline** and **baseline_2d**

## Reproducibility
1. [Download preprocessed datasets](https://www.dropbox.com/scl/fo/of387p4m1lpgh05q2x75j/AJy3Sn5r97WBRI3Vi4VRb_A?rlkey=hvqpbr6qv3xe0gl5h7teez2tk&st=f6se30uq&dl=0)
2. Launch the container (skip this step if locally built)
3. Run programs (inside the container)
    ```
    program [-t] [-q] [-f <PREPROCESSED DATASET PATH>] [-k k_value] \
        [-plb lower_bound] [-pub upper_bound] [-eps epsilon] \ 
        [-ns num_samples [-us]] [-nt num_threads] [-uo] [-sol milp_solver]
    ```
    * program: klevel_based_method, klevel_based_method_2d, mip_based_method, baseline or baseline_2d
    * -t: Runtime experiment
    * -q: Quality evaluation experiment
    * -plb: Proportional lower bound of the protected group
    * -pub: Proportional upper bound of the protected group
    * -ns: Number of weight vectors
    * -us: Uniform weight vector sampling method
    * -nt: Number of threads (optional for klevel_based_method and mip_based_method with Gurobi solver)
    * -uo: Unoptimized (only works for klevel_based_method_2d)
    * -sol: gurobi or scip (default: gurobi)

    See below for examples of commands and their outputs.

For executing mip_based_method with Gurobi solver inside the container, you might need to apply a new [Gurobi license](https://www.gurobi.com/features/web-license-service/). Before executing mip_based_method, run the following command
```
export GRB_LICENSE_FILE=\path\to\gurobi\license
```
Make sure that the license file is accessible inside the container.

## Output samples
### Runtime experiment
Command: ``` klevel_based_method -t -f compas-50.csv -k 50 -plb 0.4 -pub 0.6 -eps 0.05 -ns 20 -nt 64 ```

Output:
``` 
k: 50 | Protected Group Proportion Lower Bound: 0.4 | Protected Group Proportion Upper Bound: 0.6 | Epsilon: 0.05 | Number of Threads: 64
2/20 fair weight vectors are found
Average run time: 4.945373e+02
``` 

### Quality evaluation experiment
Command:  ``` klevel_based_method -q -f compas-50.csv -k 50 -plb 0.4 -pub 0.6 -eps 0.05 -ns 50 -us -nt 128 ```

Output:
``` 
k: 50 | Protected Group Proportion Lower Bound: 0.4 | Protected Group Proportion Upper Bound: 0.6 | Epsilon: 0.05 | Number of Threads: 128
6/50 input weight vectors are fair
8/44 fair weight vectors are found
Average weight vector difference: 1.438719e-01
Average protected group proportion: 5.975000e-01
Average utility loss: 4.489584e-03
``` 