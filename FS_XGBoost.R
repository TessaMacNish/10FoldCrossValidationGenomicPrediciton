###############
### XGBoost ###
###############
# Load libaries
library(xgboost)
library(Metrics)
library(caret)
library(doParallel)

set.seed(111)

# Parallel setup
ncores <- 10
cl <- makeCluster(ncores)
registerDoParallel(cl)

params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eval_metric = "rmse",
  nthread = 1
)

#Run feature selecetd XGBoost
results <- foreach(i = 1:10,
                   .packages = c("xgboost", "Metrics")) %dopar% {
                     
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
                     
                     # One hot encode
                     covar_all <- rbind(covar_train, covar_test)
                     covar_all_onehot <- model.matrix(~ . - 1, data = covar_all)
                     
                     covar_train_onehot <- covar_all_onehot[rownames(covar_train), ]
                     covar_test_onehot  <- covar_all_onehot[rownames(covar_test), ]
                     
                     # Combine genotype + covariates
                     x_train <- cbind(geno_train, covar_train_onehot)
                     x_test  <- cbind(geno_test,  covar_test_onehot)
                     
                     # Convert to matrix
                     x_train <- as.matrix(x_train)
                     x_test  <- as.matrix(x_test)
                     
                     # Response
                     y_train <- pheno_train[[1]]
                     y_test  <- pheno_test[[1]]
                     
                     # DMatrix
                     dtrain <- xgb.DMatrix(data = x_train, label = y_train)
                     dtest  <- xgb.DMatrix(data = x_test,  label = y_test)
                     
                     # Train model
                     xgb_model <- xgb.train(
                       params = params,
                       data = dtrain,
                       nrounds = 500,
                       watchlist = list(train = dtrain, eval = dtest),
                       early_stopping_rounds = 10,
                       print_every_n = 10
                     )
                     
                     # Predictions
                     preds_train <- predict(xgb_model, dtrain)
                     preds_test  <- predict(xgb_model, dtest)
                     
                     train_ids <- rownames(x_train)
                     test_ids  <- rownames(x_test)
                     
                     # Save outputs
                     write.csv(data.frame(gid = train_ids, observed = y_train, predicted = preds_train), paste0("C16.0_XGBoost_Feature_selection_XGBoost_SNP_haplotype_seed111_fold", i, "_train_predictions.csv"), row.names = FALSE)
                     write.csv(data.frame(gid = test_ids, observed = y_test, predicted = preds_test), paste0("C16.0_XGBoost_Feature_selection_XGBoost_SNP_haplotype_seed111_fold", i, "_test_predictions.csv"), row.names = FALSE)
                     xgb.save(xgb_model, paste0("C16.0_ensemble_XGBoost_seed111_fold", i, ".model"))
                     
                     # Variable importance
                     importance <- xgb.importance(model = xgb_model)
                     write.csv(importance, paste0("C16.0_ensemble_XGBoost_seed111_fold", i, "_variable_importance.csv"))
                     
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

## Combine results
results_summary <- data.frame(
  Fold = 1:10,
  Train_MAE = sapply(results, `[[`, "Train_MAE"),
  Test_MAE  = sapply(results, `[[`, "Test_MAE"),
  Train_RMSE = sapply(results, `[[`, "Train_RMSE"),
  Test_RMSE  = sapply(results, `[[`, "Test_RMSE"),
  Train_Pearson = sapply(results, `[[`, "Train_cor"),
  Test_Pearson  = sapply(results, `[[`, "Test_cor")
)
