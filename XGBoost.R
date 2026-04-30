###############
### XGBoost ###
###############
# Load libaries
library(xgboost)
library(Metrics)
library(caret)
library(doParallel)

# Load genotype
XGBoost_SNP <- read.csv("SNPdata.csv", header = TRUE, row.names = 1)

# Load phenotype
Pheno_Freezing <- read.csv("Freezing_ID.csv", header = TRUE, row.names = 1)

# Load covariate
Covar <-read.csv("covariate.csv", header = TRUE, row.names = 1)

# Filter out the individuals that are not in the phenotype File
XGBoost_SNP_Freezing <- XGBoost_SNP [rownames(XGBoost_SNP) %in% rownames(Pheno_Freezing), ]
Covar_Freezing <- Covar [rownames(Covar) %in% rownames(Pheno_Freezing), ]

# One hot encode
Covar_Freezing_onehot <- model.matrix(~ . - 1, data = Covar_Freezing)
XGBoost_SNP_Freezing <- cbind(XGBoost_SNP_Freezing, Covar_Freezing_onehot)

# 10-Fold Cross Validation
set.seed(222)
folds <- createFolds(Pheno_Freezing[[1]], k = 10, list = TRUE)

# Parallel setup
ncores <- 10
cl <- makeCluster(ncores)
registerDoParallel(cl)

# Run XGBoost
params <- list(booster = "gbtree", objective = "reg:squarederror", eval_metric = "rmse", nthread = 1)

results <- foreach(i = 1:10, .packages = c("xgboost", "Metrics", "caret")) %dopar% {
  
  test_idx <- folds[[i]]
  train_idx <- setdiff(1:nrow(XGBoost_SNP_Freezing), test_idx)
  
  XGB_train <- XGBoost_SNP_Freezing[train_idx, ]
  XGB_test  <- XGBoost_SNP_Freezing[test_idx, ]
  
  y_train <- Pheno_Freezing[rownames(XGB_train), , drop = FALSE][[1]]
  y_test  <- Pheno_Freezing[rownames(XGB_test), , drop = FALSE][[1]]
  
  dtrain <- xgb.DMatrix(data = as.matrix(XGB_train), label = y_train)
  dtest  <- xgb.DMatrix(data = as.matrix(XGB_test),  label = y_test)
  
  ## Train model
  xgb_model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 500,
    watchlist = list(train = dtrain, eval = dtest),
    early_stopping_rounds = 10,
    print_every_n = 10
  )
  
  # Save model
  xgb.save(xgb_model, paste0("Chapter3_seed222_XGBoost_SNP_Freezing_Fold", i, ".model"))
  
  # Predictions
  preds_test <- predict(xgb_model, dtest)
  preds_train <- predict(xgb_model, dtrain)
  
  # Save predictions and observed
  write.csv(preds_test, paste0("Chapter3_seed222_XGBoost_SNP_Freezing_Fold", i, "_predicted_test.csv"), row.names = FALSE)
  write.csv(preds_train, paste0("Chapter3_seed222_XGBoost_SNP_Freezing_Fold", i, "_predicted_train.csv"), row.names = FALSE)
  write.csv(y_test, paste0("Chapter3_seed222_XGBoost_SNP_Freezing_Fold", i, "_observed_test.csv"), row.names = FALSE)
  write.csv(y_train, paste0("Chapter3_seed222_XGBoost_SNP_Freezing_Fold", i, "_observed_train.csv"), row.names = FALSE)
  
  # Variable importance
  importance <- xgb.importance(model = xgb_model)
  write.csv(importance, paste0("Chapter3_seed222_XGBoost_SNP_Freezing_Fold", i, "_variable_importance.csv"))
  
  # Metrics
  list(
    Train_MAE = mae(y_train, preds_train),
    Test_MAE  = mae(y_test, preds_test),
    Train_RMSE = rmse(y_train, preds_train),
    Test_RMSE  = rmse(y_test, preds_test),
    Train_Pearson = cor(preds_train, y_train),
    Test_Pearson  = cor(preds_test, y_test)
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
  Train_Pearson = sapply(results, `[[`, "Train_Pearson"),
  Test_Pearson  = sapply(results, `[[`, "Test_Pearson")
)


