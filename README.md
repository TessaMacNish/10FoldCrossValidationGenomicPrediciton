# Genomic Prediction Models
This repository contains examples of code used in single nucleotide polymorphisms (SNP)-, haplotype-based, and multi-modal SNP and haplotype genomic prediction (GP). Each R script includes the model code as well as the data preperation steps for the genotype, covariate, and phenotype data that is specific to each model. 

The models shown are:

1) Genomic Best Linear Unbiased Prediction (GBLUP) in the file GBLUP.R, using the R package rrBLUP version 4.6.3 (Endelman, 2011)
2) Random Forest (RF) in the file RF.R, using the R package ranger version 0.18.0 (Wright & Ziegler, 2017)
3) eXtreme Gradient Boosting (XGBoost) in the file XGBoost.R, using the R package xgboost version 3.2.0.1 (Chen & Guestrin, 2016)

In addition there are R scripts for these three model where feature selection (FS) was used to reduce the dimentions of the SNP and haplotype data prior to genomic prediction. As FS was applied to the training data only, the training and test datasets were split, in the same manner as the above models, before the GP R scripts were run. The three FS GP R scripts provided are as below:

1) GBLUP in the file FS_GBLUP.R
2) RF in the file FS_RF.R
3) XGBoost in the file FS_XGBoost.R

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 13-17-, 785–794. https://doi.org/10.1145/2939672.2939785

Endelman, J. B. (2011). Ridge regression and other kernels for genomic selection with R package rrBLUP. Plant Breeding, 4(3), 250–255. https://doi.org/10.3835/plantgenome2011.08.0024

Wright, M. N., & Ziegler, A. (2017). ranger: A fast implementation of random forests for high dimensional data in C++ and R. Journal of Statistical Software, 77(1), 1–17. https://doi.org/10.18637/jss.v077.i01
