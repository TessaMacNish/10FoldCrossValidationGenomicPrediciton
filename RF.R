######################
### Random Forest ###
#####################
# Load libaries
library(ranger)
library(Metrics)
library(caret)
library(doParallel)

# Load genotype
RF_SNP <- read.csv("SNPdata.csv", header = TRUE, row.names = 1)

# Load phenotype
Pheno_Freezing <- read.csv("Freezing_ID.csv", header = TRUE, row.names = 1)

# Load covariate
Covar <-read.csv("covariate.csv", header = TRUE, row.names = 1)

# Filter out the individuals that are not in the phenotype File
RF_SNP_Freezing <- RF_SNP [rownames(RF_SNP) %in% rownames(Pheno_Freezing), ]
Covar_Freezing <- Covar [rownames(Covar) %in% rownames(Pheno_Freezing), ]

# Combine genotype and covariate
RF_SNP_Freezing <- cbind(RF_SNP_Freezing, Covar_Freezing)

# 10-fold setup
set.seed(222)
folds <- createFolds(Pheno_Freezing[[1]], k = 10, list = TRUE)

# Parallel setup
ncores <- 10
cl <- makeCluster(ncores)
registerDoParallel(cl)

## Run Random Forest in parallel
results <- foreach(i = 1:10, .packages = c("ranger", "Metrics")) %dopar% {
  
  test_idx <- folds[[i]]
  train_idx <- setdiff(1:nrow(Pheno_Freezing), test_idx)
  
  x_train <- data.matrix(RF_SNP_Freezing[train_idx, ])
  x_test  <- data.matrix(RF_SNP_Freezing[test_idx, ])
  y_train <- Pheno_Freezing[rownames(x_train), , drop = FALSE][[1]]
  y_test  <- Pheno_Freezing[rownames(x_test), , drop = FALSE][[1]]
  
  # Train model
  rf_model <- ranger(
    x = x_train,
    y = y_train,
    importance = "permutation",
    seed = 222
  )
  
  # Save model
  saveRDS(rf_model, paste0("Chapter3_seed222_RF_SNP_Freezing_Fold", i, "_model.rds"))
  
  # Predictions
  preds_test <- predict(rf_model, data = x_test)$predictions
  preds_train <- rf_model$predictions
  
  # Save predictions and observed
  write.csv(preds_test, paste0("Chapter3_seed222_RF_SNP_Freezing_Fold", i, "_predicted_test.csv"), row.names = FALSE)
  write.csv(preds_train, paste0("Chapter3_seed222_RF_SNP_Freezing_Fold", i, "_predicted_train.csv"), row.names = FALSE)
  write.csv(y_test, paste0("Chapter3_seed222_RF_SNP_Freezing_Fold", i, "_observed_test.csv"), row.names = FALSE)
  write.csv(y_train, paste0("Chapter3_seed222_RF_SNP_Freezing_Fold", i, "_observed_train.csv"), row.names = FALSE)
  
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

# Combine results
results_summary <- data.frame(
  Fold = 1:10,
  Train_MAE = sapply(results, `[[`, "Train_MAE"),
  Test_MAE  = sapply(results, `[[`, "Test_MAE"),
  Train_RMSE = sapply(results, `[[`, "Train_RMSE"),
  Test_RMSE  = sapply(results, `[[`, "Test_RMSE"),
  Train_Pearson = sapply(results, `[[`, "Train_Pearson"),
  Test_Pearson  = sapply(results, `[[`, "Test_Pearson")
)
