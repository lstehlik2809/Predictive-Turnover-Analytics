#=================================================================
# Predictive Turnoever Analytics 
# Tutorial using sample data from IBM (https://www.ibm.com/communities/analytics/watson-analytics-blog/hr-employee-attrition/)
#=================================================================

# Clearing Workspace
rm(list = ls())

# Uploading libraries
library(pROC)
library(ROCR)
library(MLmetrics)
library(dplyr)
library(ggplot2)
library(tidyr)
library(readxl)
library(caTools)
library(caret)
library(LiblineaR)
library(glmnet)
library(caretEnsemble)
library(rpart)
library(Amelia)
library(RANN)
library(gbm)
library(randomForest)

#=================================================================
# Uploadign data
#=================================================================

myData <- read_excel("attritionData.xlsx")

# Checking structure of the data
dim(myData)
names(myData)
glimpse(myData)

# Checking missing data
missmap(myData)
any(is.na(myData))
sum(is.na(myData))

# Basic summary of data  
summary(myData)

#=================================================================
# Data wrangling
#=================================================================

# Changing all chr variables to factor variables
myData %>%
  mutate_if(is.character, as.factor) -> myData

# Removing near zero variance variables
remove_cols <- nearZeroVar(myData[,-2], names = TRUE, freqCut = 19, uniqueCut = 10)
all_cols <- names(myData)
myDataSmall <- myData[ , setdiff(all_cols, remove_cols)]

# Removing EmployeeNumber variable
myDataSmall <- myDataSmall[ , setdiff(names(myDataSmall), "EmployeeNumber")]

# Converting every categorical variable to numerical using dummy variables
# Converting outcome variable to numeric
myDataSmall$Attrition <- ifelse(myDataSmall$Attrition == "Yes",1,0)

dmy <- dummyVars(" ~ .", data = myDataSmall, fullRank = T)
myDataTransformed <- data.frame(predict(dmy, newdata = myDataSmall))

# Removing one of the dummy variables made from binary categorical variables  
myDataTransformed$Gender.Male <- NULL
myDataTransformed$OverTime.No <- NULL

# Converting the dependent variable back to categorical
myDataTransformed$Attrition <- factor(myDataTransformed$Attrition, levels = c(1,0), labels = c("Yes", "No"))

# Centering and scaling data
preProcValues <- preProcess(myDataTransformed, method = c("center", "scale")) 
myDataTransformed <- predict(preProcValues, myDataTransformed)

#=================================================================
# Splitting data to training and testing data set
#=================================================================

set.seed(1234) 
Sample <- createDataPartition(myDataTransformed$Attrition, p=0.7, list=FALSE)
train <- myDataTransformed[ Sample, ]
test <- myDataTransformed[-Sample, ]

# Controling distribution of criterion in training/test set
prop.table(table(train$Attrition))
prop.table(table(test$Attrition))

#=================================================================
# Feature selection using Recursive Feature Elimination
#=================================================================

control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)

outcomeName <- 'Attrition'
predictors <- names(train)[!names(train) %in% outcomeName]
Churn_Pred_Profile <- rfe(train[,predictors], train[,outcomeName],
                         rfeControl = control)

print(Churn_Pred_Profile)
Churn_Pred_Profile$optVariables[1:20]

# Taking top 20 predictors
predictors <- Churn_Pred_Profile$optVariables[1:20]

#=================================================================
# Training model(s)
#=================================================================

# Listing possible models
names(getModelInfo()) # For more details see http://topepo.github.io/caret/available-models.html

# Creating reusable trainControl object
set.seed(1234)
fitControl <- trainControl(
  method="boot",
  number=25,
  savePredictions="final",
  classProbs=TRUE,
  index=createResample(train$Attrition, 25),
  summaryFunction=twoClassSummary,
  verboseIter = TRUE
)

# Training models
model_list <- caretList(
  train[,predictors],train[,outcomeName], 
  trControl=fitControl,
  metric = "ROC",
  tuneList=list(
    gbm = caretModelSpec(method="gbm", tuneLength = 3), 
    rf = caretModelSpec(method="rf", tuneLength = 3),
    nnet = caretModelSpec(method="nnet", tuneLength = 3),
    glmnet = caretModelSpec(method="glmnet", tuneLength = 3)
    )
)

#=================================================================
# Parameter tuning
#=================================================================

# If the search space for parameters is not defined, Caret will use 3 random values 
# of each tunable parameter and use the cross-validation results to find the best set 
# of parameters for that algorithm.

# Parameters of a model that can be tuned
# modelLookup(model='rpart')

# Parameter tuning using tuneGrid
# Creating grid
# grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))
# Training the model
# model_gbm <- train(train[,predictors],train[,outcomeName],method='gbm', metric = "ROC", trControl = fitControl, tuneGrid = grid)
# print(model_gbm)
# plot(model_gbm)

# Parameter tuning using tuneLength
# model_gbm <- train(train[,predictors],train[,outcomeName],method='gbm', metric = "ROC", trControl = fitControl, tuneLength = 10)
# print(model_glmnet)
# plot(model_glmnet)

#=================================================================
# Comparing models
#=================================================================

# Collecting resamples from CV folds
resamps <- resamples(model_list)
summary(resamps)

# Ploting performance metrics of individual models
bwplot(resamps, metric = "ROC")
dotplot(resamps, metric = "ROC")
dotplot(resamps, metric = "Spec")
dotplot(resamps, metric = "Sens")
densityplot(resamps, metric = "ROC")
xyplot(resamps, metric = "ROC")

# Checking intrecorrelations between outputs of individual models 
modelCor(resamples(model_list))

#=================================================================
# Variable importance estimation
#=================================================================

varImp <- lapply(model_list, varImp, scale = T)
print(varImp)
plot(varImp$rf, main="")
plot(varImp$gbm, main="")
plot(varImp$glmnet, main="")
plot(varImp$nnet, main="")

# Predictors that are important for the majority of models represents genuinely important 
# predictors. Foe ensembling, we should use predictions from models that have significantly 
# different variable importance as their predictions are also expected to be different. 
# Although, one thing that must be make sure is that all of them are sufficiently accurate.

#=================================================================
# Creating ensemble
#=================================================================

# Models make a good candidate for an ensemble when their predicitons are fairly 
# un-correlated, but their overall accuaracy is similar. 

# Creating "meta-models" (moving beyond simple blends of models)
glm_ensemble <- caretStack(
  model_list,
  method="glm",
  metric="ROC",
  trControl=trainControl(
    method="boot",
    number=10,
    savePredictions="final",
    classProbs=TRUE,
    summaryFunction=twoClassSummary
  )
)

summary(glm_ensemble)

#=================================================================
# Predictions
#=================================================================

# Comparing ROCs/AUCs of individual models and ensemble on testing dataset
model_preds <- lapply(model_list, predict, newdata=test[,predictors], type="prob")
model_preds <- lapply(model_preds, function(x) x[,"Yes"])
model_preds <- data.frame(model_preds)
ens_preds <- predict(glm_ensemble, newdata= test[, predictors], type="prob")
model_preds$ensemble <- ens_preds
caTools::colAUC(model_preds, test$Attrition)

# Confusion matrix for ensemble model (on testing dataset)
threshold <- 0.5
pred <- factor(ifelse(model_preds$ensemble > threshold, "Yes", "No") )
pred <- relevel(pred, "Yes")   # you may or may not need this; I did
confusionMatrix(pred, reference = test$Attrition, positive = "Yes")

# Plotting distribution of prediction scores grouped by known outcome for testing data
testPredData <- data.frame(atRisk = test$Attrition, pred = model_preds$ensemble)  
ggplot(testPredData, aes(x = pred, color = atRisk, linetype = atRisk))+
  geom_density()

# Computing AUC for testing data
ROC <- performance(prediction(model_preds$ensemble, test$Attrition),"auc")
AUC <- round(ROC@y.values[[1]],2)
AUC # alternatively (gini+1/2) or roc(test$Attrition, model_preds$ensemble)

# Computing Gini for testing data
GINI <- (AUC-0.5)/0.5
GINI # alternatively Gini(y_pred = model_preds$ensemble, y_true = as.numeric(test$Attrition))

# Plotting ROC curve with various threshold values for testing data
predTest <- prediction(model_preds$ensemble, test$Attrition)
# Performance function
ROCRperfTest = performance(predTest, "tpr", "fpr")
performance(predTest, "auc")
# Plotting ROC curve
plot(ROCRperfTest)
# Adding colors
plot(ROCRperfTest, colorize=TRUE)
# Adding threshold labels 
plot(ROCRperfTest, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

# Plotting lift chart with various threshold values for testing data
LIFTperfTest <- performance(predTest,"lift","rpp")
plot(LIFTperfTest, main="lift curve", colorize=T)
plot(LIFTperfTest, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

# Computing lift for 1st decile for testing data
lift.x = unlist(slot(LIFTperfTest, 'x.values'))
lift.y = unlist(slot(LIFTperfTest, 'y.values'))
liftDfTest <- as.data.frame(cbind(lift.x, lift.y))
liftDfTest$lift.x <- round(liftDfTest$lift.x,2)
liftDfTest$lift.y <- round(liftDfTest$lift.y,2)
mean(liftDfTest[which(liftDfTest$lift.x == 0.10),2])

# Plotting gain chart with various threshold values for testing data
GAINperfTest <- performance(predTest, "tpr", "rpp")
plot(GAINperfTest, colorize=T, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7), ylab="Proportion of detected events", 
     xlab="Proportion of population")

#=================================================================
# Saving predictive model
#=================================================================

saveRDS(glm_ensemble, "model_ensemble.rds")

