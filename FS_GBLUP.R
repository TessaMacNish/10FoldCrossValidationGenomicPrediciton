###################
####   GBLUP   ####
###################
# Load packages
library(rrBLUP)
library(Metrics)
library(caret)
library(doParallel)

set.seed(111)

# Parallel setup
ncores <- 10
cl <- makeCluster(ncores)
registerDoParallel(cl)

# Run feature selected GBLUP
results <- foreach(i = 1:10,
                   .packages = c("rrBLUP", "Metrics")) %dopar% {
                     # Load data
                     # Phenotype
                     pheno_train <- read.csv(paste0("C16.0_Pheno_seed111_fold", i, "_train_GBLUP.csv"), row.names = 1)
                     pheno_test  <- read.csv(paste0("C16.0_Pheno_seed111_fold", i, "_test_GBLUP.csv"), row.names = 1)
                     
                     # Covariate
                     covar_train <- read.csv(paste0("C16.0_Covar_seed111_fold", i, "_train_GBLUP.csv"), row.names = 1)
                     covar_test  <- read.csv(paste0("C16.0_Covar_seed111_fold", i, "_test_GBLUP.csv"), row.names = 1)
                     
                     # Genotype (haplotype & SNP)
                     geno_train <- read.csv(paste0("C16.0_GBLUP_SNP_haplotype_seed111_fold", i, "_train_feature_selected.csv"), row.names = 1)
                     geno_test  <- read.csv(paste0("C16.0_GBLUP_SNP_haplotype_seed111_fold", i, "_test_GBLUP.csv"), row.names = 1)
                     
                     # Align features
                     common_markers <- intersect(colnames(geno_train), colnames(geno_test))
                     geno_train <- geno_train[, common_markers, drop = FALSE]
                     geno_test  <- geno_test[,  common_markers, drop = FALSE]
                     
                     # Compute additive relationship matrix (VanRaden method)
                     geno_all <- rbind(geno_train, geno_test)
                     K <- A.mat(geno_all)
                     
                     # Combine phenotype and covariate data
                     pheno_all <- rbind(pheno_train, pheno_test)
                     covar_all <- rbind(covar_train, covar_test)
                     pheno_gblup <- data.frame(gid = rownames(pheno_all), trait = pheno_all[[1]], covar_all)
                     
                     # Mask test phenotype
                     test_ids <- rownames(pheno_test)
                     pheno_gblup$trait[pheno_gblup$gid %in% test_ids] <- NA
                     
                     # Run model
                     rr_model <- kin.blup(
                       data = pheno_gblup,
                       geno = "gid",
                       pheno = "trait",
                       K = K,
                       fixed = c("Ecotype", "Country")
                     )
                     
                     #predicitons
                     preds <- rr_model$pred
                     train_ids <- rownames(pheno_train)
                     preds_train <- preds[match(train_ids, pheno_gblup$gid)]
                     preds_test  <- preds[match(test_ids,  pheno_gblup$gid)]
                     obs_train <- pheno_train[[1]]
                     obs_test  <- pheno_test[[1]]
                     
                     #save outputs
                     write.csv(data.frame(gid = train_ids, observed = obs_train, predicted = preds_train), paste0("C16.0_GBLUP_Feature_selection_XGBoost_SNP_haplotype_seed111_fold", i, "_train_predictions.csv"),row.names = FALSE)
                     write.csv(data.frame(gid = test_ids, observed = obs_test, predicted = preds_test), paste0("C16.0_GBLUP_Feature_selection_XGBoost_SNP_haplotype_seed111_fold", i, "_test_predictions.csv"), row.names = FALSE)
                     saveRDS(rr_model, paste0("C16.0_ensemble_GBLUP_seed111_fold", i, "_model.rds"))
                     
                     #metrics
                     list(
                       Train_MAE  = mae(obs_train, preds_train),
                       Test_MAE   = mae(obs_test, preds_test),
                       Train_RMSE = rmse(obs_train, preds_train),
                       Test_RMSE  = rmse(obs_test, preds_test),
                       Train_cor  = cor(preds_train, obs_train),
                       Test_cor   = cor(preds_test, obs_test)
                     )
                   }

stopCluster(cl)

results_summary <- data.frame(
  Fold = 1:10,
  Train_MAE = sapply(results, `[[`, "Train_MAE"),
  Test_MAE  = sapply(results, `[[`, "Test_MAE"),
  Train_RMSE = sapply(results, `[[`, "Train_RMSE"),
  Test_RMSE  = sapply(results, `[[`, "Test_RMSE"),
  Train_Pearson = sapply(results, `[[`, "Train_cor"),
  Test_Pearson  = sapply(results, `[[`, "Test_cor")
)
