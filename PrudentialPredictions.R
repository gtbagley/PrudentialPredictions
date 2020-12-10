#### Prudential

## Libraries I Need
library(tidyverse)
library(DataExplorer)
library(caret)
library(RANN)

train <- read.csv("/Users/griffinbagley/Documents/STAT 495R/prudential/prudential-life-insurance-assessment/train.csv")
test <- read.csv("/Users/griffinbagley/Documents/STAT 495R/prudential/prudential-life-insurance-assessment/test.csv")

## Combine sets to clean both simultaneously
combined <- bind_rows(train=train, test=test, .id="Set")
summary(combined)

################################################
## Setting Variables to the Correct Data Type ##
################################################

## first I set all variables to factors, since there aren't many numeric
col_names <- names(combined)
combined[,col_names] <- lapply(combined[,col_names] , factor)

## Set dependent variable, Response, to numeric. It is ordered, so we'll treat it as numeric for ease and then round values in the end
combined$Response <- as.numeric(combined$Response)

## change those that shouldn't be factors to numeric, according to info on kaggle
names <- c('Product_Info_4','Ins_Age','Ht','Wt','BMI','Employment_Info_1','Employment_Info_4','Employment_Info_6','Insurance_History_5','Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5','Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32')

## I'm suspicious of product info 3, employment info 2, insured info 3, 'Medical_History_2', as it seems they could be numeric

## I've decided to treat med history 2 as numeric, contrary to the information provided. It has high values and a lot of them.
combined$Medical_History_2 <- as.numeric(combined$Medical_History_2)

#This is in case I decide to assume all of the suspiciously numeric seeming variables are in fact numeric
#names <- c('Product_Info_4','Ins_Age','Ht','Wt','BMI','Employment_Info_1','Employment_Info_4','Employment_Info_6','Insurance_History_5','Medical_History_1', 'Product_Info_3', 'Employment_Info_2', 'InsuredInfo_3', 'Medical_History_2')

## Change specified variables to numeric
combined[,names] <- lapply(combined[,names] , as.numeric)

################
## Imputation ##
################

## See which variables have missing data
na_columns <- colnames(combined)[colSums(is.na.data.frame(combined)) > 0]
plot_missing(combined %>% select(na_columns))

## Here I select the columns the imputation will involve
noresponse <- combined %>% select(-Response, -Id, -Set)

## Impute using medians
preprocess <- preProcess(noresponse, 
                         method = "medianImpute",
                         pcaComp = 10,
                         na.remove = TRUE
)

## create new dataframe with imputed values
transformed <- predict(preprocess, newdata = combined)

## Check to ensure imputed data has replaced missing values
na_columns <- colnames(transformed)[colSums(is.na.data.frame(transformed)) > 0]
plot_missing(transformed %>% select(na_columns))

##############
## Modeling ##
##############

## separate combined to get cleaned training and test data
traindata <- transformed %>% filter(Set == "train") %>% select(-Set, -Id)

testdata <- transformed %>% filter(Set == "test") %>% select(-Set, -Id)


## Use xgboost to generate a model
id <- combined %>% filter(Set == "test") %>% select(Id)

xgb <- train(form=Response~.,
             data=traindata,
             method="xgbTree",
             trControl=trainControl(method="repeatedcv",
                                    number=2, #Number of pieces of your data
                                    repeats=1) #repeats=1 = "cv"
)
plot(xgb)
xgb$results

## Use model to generate predictions
preds <- predict(xgb, newdata=testdata)

## Since I treated the dependent variable as numeric, we need to round the predictions
## and replace any values that are > 8 or < 1 with 8 and 1, respectively
preds <- round(preds)

preds <- replace(preds, preds>8, 8)
preds <- replace(preds, preds<1, 1)

## Create submission dataframe and write it to a csv
xgb.preds <- data.frame(Id=id, Response=preds)
write_csv(x=xgb.preds, path="./xgb3.csv")