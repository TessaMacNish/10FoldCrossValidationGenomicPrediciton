###################
####   GBLUP   ####
###################
# Load packages
library(rrBLUP)
library(Metrics)
library(caret)
library(doParallel)

# Load genotype
GBLUP_SNP <- read.csv("SNPdata.csv", header = TRUE, row.names = 1)

# Load phenotype
Pheno_Freezing <- read.csv("Freezing_ID.csv", header = TRUE, row.names = 1)

# Load covariate
Covar <-read.csv("covariate.csv", header = TRUE, row.names = 1)

# Filter out the individuals that are not in the phenotype File
GBLUP_SNP_Freezing <- GBLUP_SNP [rownames(GBLUP_SNP) %in% rownames(Pheno_Freezing), ]
Covar_Freezing <- Covar [rownames(Covar) %in% rownames(Pheno_Freezing), ]

# Combine phenotype and covariates
pheno_gblup_Freezing <- data.frame(gid = rownames(Pheno_Freezing), trait = Pheno_Freezing[[1]], Covar_Freezing)

# Compute additive relationship matrix (VanRaden method)
K <- A.mat(GBLUP_SNP_Freezing)

# 10 fold cross validation setup
set.seed(222)
folds <- createFolds(pheno_gblup_Freezing$trait, k = 10, list = TRUE, returnTrain = FALSE)

# Storage for results
Train_MAE_list <- c()
Test_MAE_list <- c()
Train_RMSE_list <- c()
Test_RMSE_list <- c()
Train_cor_list <- c()
Test_cor_list <- c()

ncores <- 10
cl <- makeCluster(ncores)
registerDoParallel(cl)

results <- foreach(i = 1:10,
                   .packages = c("rrBLUP", "Metrics"),
                   .export = c("pheno_gblup_Freezing", "K", "folds")) %dopar% {
                     
                     test_idx <- folds[[i]]
                     train_idx <- setdiff(1:nrow(pheno_gblup_Freezing), test_idx)
                     
                     pheno_temp <- pheno_gblup_Freezing
                     pheno_temp$trait[test_idx] <- NA
                     
                     # Train model
                     rr_model <- kin.blup(
                       data = pheno_temp,
                       geno = "gid",
                       pheno = "trait",
                       K = K,
                       fixed = c("Ecotype", "Country")
                     )
                     
                     # Extract predictions
                     preds_test  <- rr_model$pred[test_idx]
                     preds_train <- rr_model$pred[train_idx]
                     
                     obs_test  <- pheno_gblup_Freezing$trait[test_idx]
                     obs_train <- pheno_gblup_Freezing$trait[train_idx]
                     
                     # Save predictions
                     write.csv(data.frame(gid = pheno_gblup_Freezing$gid[test_idx],
                                          observed = obs_test,
                                          predicted = preds_test),
                               paste0("Chapter3_seed222_SNP_GBLUP_Freezing_Fold", i, "_predicted_test.csv"),
                               row.names = FALSE)
                     
                     write.csv(data.frame(gid = pheno_gblup_Freezing$gid[train_idx],
                                          observed = obs_train,
                                          predicted = preds_train),
                               paste0("Chapter3_seed222_SNP_GBLUP_Freezing_Fold", i, "_predicted_train.csv"),
                               row.names = FALSE)
                     
                     # Save model
                     saveRDS(rr_model, paste0("Chapter3_seed222_SNP_GBLUP_Freezing_Fold", i, "_model.rds"))
                     
                     # Return metrics for this fold
                     list(
                       Train_MAE  = mae(obs_train, preds_train),
                       Test_MAE   = mae(obs_test,  preds_test),
                       Train_RMSE = rmse(obs_train, preds_train),
                       Test_RMSE  = rmse(obs_test,  preds_test),
                       Train_cor  = cor(preds_train, obs_train),
                       Test_cor   = cor(preds_test,  obs_test)
                     )
                   }

stopCluster(cl)

# Convert results list → dataframe
results_summary <- data.frame(
  Fold = 1:10,
  Train_MAE = sapply(results, `[[`, "Train_MAE"),
  Test_MAE  = sapply(results, `[[`, "Test_MAE"),
  Train_RMSE = sapply(results, `[[`, "Train_RMSE"),
  Test_RMSE  = sapply(results, `[[`, "Test_RMSE"),
  Train_Pearson = sapply(results, `[[`, "Train_cor"),
  Test_Pearson  = sapply(results, `[[`, "Test_cor")
)
