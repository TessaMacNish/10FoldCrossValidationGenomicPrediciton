######################
### Random Forest ###
#####################
# Load libaries
library(ranger)
library(Metrics)
library(doParallel)

set.seed(111)

# Parallel setup
ncores <- 10
cl <- makeCluster(ncores)
registerDoParallel(cl)

#Run random forest
results <- foreach(i = 1:10,
                   .packages = c("ranger", "Metrics")) %dopar% {
                     
                     ## Load phenotype
                     pheno_train <- read.csv(paste0("C16.0_Pheno_seed111_fold", i, "_train_GBLUP.csv"), row.names = 1)
                     pheno_test  <- read.csv(paste0("C16.0_Pheno_seed111_fold", i, "_test_GBLUP.csv"), row.names = 1)
                     
                     ## Load covariate
                     covar_train <- read.csv(paste0("C16.0_Covar_seed111_fold", i, "_train_GBLUP.csv"), row.names = 1)
                     covar_test  <- read.csv(paste0("C16.0_Covar_seed111_fold", i, "_test_GBLUP.csv"), row.names = 1)
                     
                     ## Load genotype (SNP + haplotype)
                     geno_train <- read.csv(paste0("C16.0_RF_XGBoost_SNP_haplotype_seed111_fold", i, "_train_feature_selected.csv"), row.names = 1)
                     geno_test  <- read.csv(paste0("C16.0_RF_XGBoost_SNP_haplotype_seed111_fold", i, "_test_RF_XGBoost.csv"), row.names = 1)
                     
                     # Align features
                     common_markers <- intersect(colnames(geno_train), colnames(geno_test))
                     geno_train <- geno_train[, common_markers, drop = FALSE]
                     geno_test  <- geno_test[,  common_markers, drop = FALSE]
                     
                     
                     # Combine genotype + covariates
                     x_train <- cbind(geno_train, covar_train)
                     x_test  <- cbind(geno_test,  covar_test)
                     
                     # Convert to matrix
                     x_train <- data.matrix(x_train)
                     x_test  <- data.matrix(x_test)
                     
                     # Response
                     y_train <- pheno_train[[1]]
                     y_test  <- pheno_test[[1]]
                     
                     train_ids <- rownames(x_train)
                     test_ids  <- rownames(x_test)
                     
                     # Train model
                     rf_model <- ranger(
                       x = x_train,
                       y = y_train,
                       importance = "permutation",
                       seed = 111
                     )
                     
                     # Predictions
                     preds_test <- predict(rf_model, data = x_test)$predictions
                     preds_train <- rf_model$predictions
                     
                     # Save outputs
                     saveRDS(rf_model, paste0("C16.0_ensemble_RF_seed111_fold", i, "_model.rds"))
                     write.csv(data.frame(gid = train_ids, observed = y_train, predicted = preds_train), paste0("C16.0_RF_Feature_selection_XGBoost_SNP_haplotype_seed111_fold", i, "_train_predictions.csv"), row.names = FALSE)
                     write.csv(data.frame(gid = test_ids, observed = y_test, predicted = preds_test), paste0("C16.0_RF_Feature_selection_XGBoost_SNP_haplotype_seed111_fold", i, "_test_predictions.csv"), row.names = FALSE)
                     
                     # Metrics
                     list(
                       Train_MAE  = mae(y_train, preds_train),
                       Test_MAE   = mae(y_test, preds_test),
                       Train_RMSE = rmse(y_train, preds_train),
                       Test_RMSE  = rmse(y_test, preds_test),
                       Train_cor  = cor(preds_train, y_train),
                       Test_cor   = cor(preds_test, y_test)
                     )
                   }

stopCluster(cl)

# Combine results
results_summary <- data.frame(
  Fold = 1:10,
  Train_MAE = sapply(results, `[[`, "Train_MAE"),
  Test_MAE  = sapply(results, `[[`, "Test_MAE"),
  Train_RMSE = sapply(results, `[[`, "Train_RMSE"),
  Test_RMSE  = sapply(results, `[[`, "Test_RMSE"),
  Train_Pearson = sapply(results, `[[`, "Train_cor"),
  Test_Pearson  = sapply(results, `[[`, "Test_cor")
)
