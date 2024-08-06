# TargAD
Implementation of ["A Robust Prioritized Anomaly Detection when Not All Anomalies are of Primary Interest"]. (Accepted by ICDE 2024)

## Paper abstract
Anomaly detection has emerged as a prominent research area with extensive exploration across various applications. Existing methods predominantly focus on detecting all anomalies exhibiting unusual patterns, however, they overlook the critical need to prioritize the detection of target anomaly categories (anomalies of primary interest) that could pose significant threats to various systems. This oversight results in the excessive involvement of valuable human labor and resources in dealing with non-target anomalies (that are of lower interest).

This work is focused on target-class anomaly detection, which entails overcoming several challenges: (1) deficient prior information regarding non-target anomalies and (2) an elevated false positive rate caused by the presence of non-target anomalies. Thus, we introduce a novel semi-supervised model, called TargAD, which leverages a few labeled target anomalies, along with potential non-target anomaly candidates and normal candidates selected from unlabeled data. By introducing a novel loss function, TargAD effectively maximizes the distributional differences among normal candidates, target anomalies, and non-target anomaly candidates, leading to a significant improvement in detecting target anomalies. Furthermore, when confronted with novel nontarget anomaly scenarios, TargAD maintains its accuracy in detecting target anomalies.

We conducted extensive experiments, the results of which demonstrate that TargAD outperforms eleven state-of-the-art baselines on a real-world dataset and three publicly available datasets, with average AUPRC improvements of 5.9%-24.8%, 9.2%-57.8%, 2.7%-71.3%, and 2.0%-70.3%, respectively.

## Usage
* TargAD.py is all the codes of the TargAD model.
* The data folder is used to store experimental data.

## Running environment
Python version 3.9.7

Create suitable conda environment:
```
conda env create -f environment.yml
```

## Full paper source:
https://ieeexplore.ieee.org/document/10597675

## Citation
>Lu G., Zhou F., Pavlovski M., Zhou C., Jin C., “A Robust Prioritized Anomaly Detection when Not All Anomalies are of Primary Interest”, Proc. 40th International Conference on Data Engineering (ICDE), 2024, 775-788.
