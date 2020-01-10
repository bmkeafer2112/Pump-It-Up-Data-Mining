#####################################Pump it Up: Data Mining the Water Table#######################################
###Initial LLibrary Load
library(tidyverse)
library(tidyverse) 
library(lubridate)
library(ggplot2)
library(forcats)
library(caret)
library(glmnet)
library(magrittr)
library(MASS)
library(dplyr)
library(randomForest)
library(rpart)
library(naivebayes)
library(car)
library(readr)
library(magrittr)
library(reshape2)

###Read Training and Test Data sets
Training_Set_Values <- read_csv("Training Set Values.csv")


Training_Set_Labels <- read_csv("Training Set Labels.csv")


Test_Set_Values <- read_csv("Test Set Values.csv")


###Combine Training and Test Values
training_df <- Training_Set_Values %>% inner_join(Training_Set_Labels, by = "id")


#######################################################Structure EDA################################################
###Explore Dateframe Structure
str(training_df)

###Explore Missingness
training_df %>% dplyr::select(-c(id)) %>% mutate_all(is.na) %>% summarise_all(mean) %>% glimpse()

###Check Duplicate ids
dup <- duplicated(training_df$id)
table(dup)

###Convert Characters, Logical, and some numeric classes to Factors
factor_cols <- c("wpt_name", "installer", "funder", "basin", "subvillage", "region", "region_code", 
                           "district_code", "lga", "ward", "public_meeting", "recorded_by", "scheme_management", 
                           "scheme_name", "permit", "extraction_type", "extraction_type_group", "extraction_type_class",
                           "management", "management_group", "payment", "payment_type", "water_quality", "quality_group",
                           "quantity", "quantity_group", "source", "source_type", "source_class", "waterpoint_type",
                           "waterpoint_type_group", "status_group")
head(factor_cols)

training_df %<>% mutate_each_(funs(factor(.)),factor_cols)
str(training_df)

####################################################Visualized EDA#############################################################

###Latitide versus Longitude with colored categories for status group
ggplot(data = training_df)+
  geom_point(mapping = aes(x = longitude, y = latitude, color = status_group), position = "jitter", alpha = 0.5)+
  xlim(25,45)

###Correlation Matrix & Heatmap of Numeric Variables
num_varaibles <- as.data.frame(dplyr::select(training_df, amount_tsh, gps_height, longitude,
                            longitude, num_private, population, 
                            construction_year))
str(num_varaibles)
corMat<-cor(num_varaibles)
corMat
heatmap(corMat, margins = c(10,10))

###Construction Year versus gps height with colored categories for status group
ggplot(data = training_df)+
  geom_point(mapping = aes(x = construction_year, y = gps_height, color = status_group), position = "jitter", alpha = 0.5)+
  xlim(1950,2020)


###############################################Data Transformations/Feature Engineering################################### 
###Create Unique IDs for Test Data (Training IDs are in "train_values")
testIds <- unique(Test_Set_Values$id)
trainIds <- unique(Training_Set_Labels$id)

###combine Train and Test Data Sets to perform Transformations/Feature Engineering
combined_DF <- Training_Set_Values %>% bind_rows(Test_Set_Values)
count(combined_DF)

###View Structure
str(combined_DF) 

###Remove underscores from column names
names(combined_DF)
names(combined_DF) <- gsub("\\_","",names(combined_DF))
names(combined_DF)

names(Training_Set_Labels)
names(Training_Set_Labels) <- gsub("\\_","",names(Training_Set_Labels))
names(Training_Set_Labels)

###Explore Missingness
combined_DF %>% dplyr::select(-c(id)) %>% mutate_all(is.na) %>% summarise_all(mean) %>% glimpse()


###Remove columns with missingness & high factor levels
combined_DF <- combined_DF %>% dplyr::select(-c(funder, installer, subvillage, 
                               publicmeeting, schememanagement, schemename, permit, wptname,daterecorded, 
                               regioncode, districtcode))

###Check Missingness
combined_DF %>% dplyr::select(-c(id)) %>% mutate_all(is.na) %>% summarise_all(mean) %>% glimpse()

###Convert Characters, Logical, and some numeric classes to Factors
factor_cols <- c("basin", "region", 
                "lga", "ward", "recordedby", 
                 "extractiontype", "extractiontypegroup", "extractiontypeclass",
                 "management", "managementgroup", "payment", "paymenttype", "waterquality", "qualitygroup",
                 "quantity", "quantitygroup", "source", "sourcetype", "sourceclass", "waterpointtype",
                 "waterpointtypegroup")
head(factor_cols)

combined_DF %<>% mutate_each_(funs(factor(.)),factor_cols)
str(combined_DF)

###Check Duplicate ids
dup <- duplicated(training_df$id)
table(dup)

###Factor Lump to reduce level and complexity
combined_DF <- combined_DF %>% 
  mutate(lga = fct_lump(fct_explicit_na(lga), prop = 0.01)) %>%
  mutate(ward = fct_lump(fct_explicit_na(ward), prop = 0.01)) %>%
  mutate(ward = fct_lump(fct_explicit_na(ward), prop = 0.01))

###Create Factor Group for gpsheight
combined_DF$gpsheight[combined_DF$gpsheight>-100 & combined_DF$gpsheight < 300] <- 1
combined_DF$gpsheight[combined_DF$gpsheight>=300 & combined_DF$gpsheight < 600] <- 2
combined_DF$gpsheight[combined_DF$gpsheight>=600 & combined_DF$gpsheight < 900] <- 3
combined_DF$gpsheight[combined_DF$gpsheight>=900 & combined_DF$gpsheight < 1200] <- 4
combined_DF$gpsheight[combined_DF$gpsheight>=1200 & combined_DF$gpsheight < 1500] <- 5
combined_DF$gpsheight[combined_DF$gpsheight>=1500 & combined_DF$gpsheight < 1800] <- 6
combined_DF$gpsheight[combined_DF$gpsheight>=1800 & combined_DF$gpsheight < 2100] <- 7
combined_DF$gpsheight[combined_DF$gpsheight>=2100 & combined_DF$gpsheight < 2400] <- 8
combined_DF$gpsheight[combined_DF$gpsheight>=2400 & combined_DF$gpsheight < 2700] <- 9
combined_DF$gpsheight[combined_DF$gpsheight>=2700 & combined_DF$gpsheight < 3000] <- 10

factor_cols <- c("gpsheight")
head(factor_cols)

combined_DF %<>% mutate_each_(funs(factor(.)),factor_cols)
str(combined_DF)

###Remove Variables without variance
check <-nearZeroVar(combined_DF)
check
combined_DF <- combined_DF %>% dplyr::select(-c(nearZeroVar(combined_DF)))

###Final Selected Variables
selected_DF <- dplyr::select(combined_DF, id, amounttsh, gpsheight, longitude, latitude, basin, population, constructionyear, 
                             extractiontypeclass, managementgroup, paymenttype, qualitygroup, quantitygroup,
                             sourceclass, waterpointtypegroup)
str(selected_DF)

###Retrieve Transformed Train Data
train_trans_DF <- data.frame(selected_DF %>% filter(id %in% trainIds))

###Join statusgroup variable to transformed train data
train_trans_DF <- train_trans_DF %>% inner_join(Training_Set_Labels, by = "id")

###Remove spaces in factor levels of statusgroup variable
train_trans_DF <- train_trans_DF %>%
  mutate(statusgroup = replace(statusgroup, statusgroup ==  "non functional", 'nonfunctional'))

train_trans_DF <- train_trans_DF %>%
  mutate(statusgroup = replace(statusgroup, statusgroup ==  "functional needs repair", 'repair'))

str(train_trans_DF)


train_trans_DF$statusgroup <- as.factor(train_trans_DF$statusgroup)

###Retrieve Transformed Test Data
test_trans_DF <- selected_DF %>% filter(id %in% testIds)
count(test_trans_DF)

###Check Structure
str(train_trans_DF)
str(test_trans_DF)

###Check Missingness
train_trans_DF %>% dplyr::select(-c(id)) %>% mutate_all(is.na) %>% summarise_all(mean) %>% glimpse()
test_trans_DF %>% dplyr::select(-c(id)) %>% mutate_all(is.na) %>% summarise_all(mean) %>% glimpse()

#############################################Prepare Dataframes for Evaluation##########################################
###Provide Train Values Evaluation (FULL DATASET)
full_eval_train_DF <- dplyr::select(train_trans_DF, -statusgroup)
str(eval_train_DF)

full_expected_train_DF <- dplyr::select(train_trans_DF, statusgroup)
str(expected_train_DF)

###Split Training Dataframe for Evaluation (confusion matrix)
split_train <- train_trans_DF %>% sample_frac(0.7)
str(split_train)

split_test <- subset(train_trans_DF, !(id %in% split_train$id))
str(split_test)

###Provide Train Values Evaluation (SPLIT DATASET)
eval_train_DF <- dplyr::select(split_test, -statusgroup)
str(eval_train_DF)

expected_train_DF <- dplyr::select(split_test, statusgroup)
str(expected_train_DF)

###Uniform Status Random Sampling to ensure proper training of the model
Func <- split_train %>% filter(statusgroup == 'functional')
SampleFunc <- sample_n(Func, 4000)

FuncR <- split_train %>% filter(statusgroup == 'repair')
SampleFunc <- sample_n(FuncR, 3000) %>% bind_rows(SampleFunc)

FuncNF <- split_train %>% filter(statusgroup == 'nonfunctional')
SampleFunc <- sample_n(FuncNF, 5000) %>% bind_rows(SampleFunc)

str(SampleFunc)

################################################Penalized Logistic Regression Model################################################

###Train Control
penlogreg_control1 = trainControl(method = "cv", number = 5, classProbs = TRUE) 

###Optimize Hyperparameters
penlogreg_grid <- expand.grid(alpha=1, lambda=0)

###Build Formula
penlogreg_form <- statusgroup ~.

###Model using caret
set.seed(21)
penlogreg_fit1 <- train(penlogreg_form,data = train_trans_DF %>% dplyr::select(-c(id)),
                     method = "glmnet", metric = "Accuracy", trControl = penlogreg_control1, 
                     tuneGrid=penlogreg_grid)

help("train")

###Summary of Model
print(penlogreg_fit1)


###Feature Selection
varImp(penlogreg_fit1)

###Predict Probablities
penlogreg_probabilities <- penlogreg_fit1 %>% predict(test_trans_DF) 
view(penlogreg_probabilities)

summary(penlogreg_probabilities)
#############################################Decision Tree Model########################################################
###Train Control
rpart_control1 = trainControl(method = "cv", number = 5) 

###Optimize Hyperparameters
rpart_grid <- expand.grid(cp=0)

###Build Formula
rpart_form <- statusgroup ~.

###Model using caret
set.seed(21)
rpart_fit1 <- train(rpart_form,data = train_trans_DF %>% dplyr::select(-c(id)),
                    method = "rpart", metric = "Accuracy", trControl = rpart_control1,
                    tuneGrid=rpart_grid)

###Summary of Model
print(rpart_fit1)

###Feature Selection
varImp(rpart_fit1)


###Predict Probablities
rpart_probabilities <- rpart_fit1 %>% predict(test_trans_DF, type = "prob")
view(rpart_probabilities)


###Confusion Matrix
confusionMatrix(rpart_fit1, test_trans_DF)

#############################################Random Forrest Model########################################################
###############################################Full Dataset########################################################
###Train Control
rf_control1 = trainControl(method = "cv", number = 5, classProbs = TRUE) 

###Optimize Hyperparameters
rf_grid <- expand.grid(mtry = 8)

###Build Formula
rf_form <- statusgroup ~.

###Model using caret
rf_fit1 <- train(rf_form, data = train_trans_DF %>% dplyr::select(-c(id)),
                 method = "rf", metric = "Accuracy", trControl = rf_control1,
                 tuneGrid=rf_grid)

###Summary of Model
print(rf_fit1)

###Feature Selection
varImp(rf_fit1)

###Predictions for Competition
rf_probabilities <- rf_fit1 %>% predict(test_trans_DF)
summary(rf_probabilities)
str(rf_probabilities)

ID <- dplyr::select(test_trans_DF, id)
RF_results <- cbind(ID, rf_probabilities)
RF_results <- RF_results %>% dplyr::rename("status_group" = "rf_probabilities")

#write out results
write.csv(RF_results, 'RF_model.csv', row.names = F)
  
###Predictions converted to dataframe
rf_probabilities <- rf_fit1 %>% predict(full_eval_train_DF)
summary(rf_probabilities)
str(rf_probabilities)

predicted_train_DF<- as.data.frame(rf_probabilities)
names(predicted_train_DF) <- c("statusgroup")
str(predicted_train_DF)

###Confusion Matrix
confusionMatrix(predicted_train_DF$statusgroup, full_expected_train_DF$statusgroup, positive = "2")

##############################################Split DataSet#############################################################
###Train Control
rf_control1 = trainControl(method = "cv", number = 5, classProbs = TRUE) 

###Optimize Hyperparameters
rf_grid <- expand.grid(mtry = 8)

###Build Formula
rf_form <- statusgroup ~.

###Model using caret
rf_fit2 <- train(rf_form, data = split_train %>% dplyr::select(-c(id)),
                 method = "rf", metric = "Accuracy", trControl = rf_control1,
                 tuneGrid=rf_grid)

###Summary of Model
print(rf_fit2)

###Feature Selection
varImp(rf_fit2)

###Predictions converted to dataframe
rf_probabilities <- rf_fit2 %>% predict(eval_train_DF)
summary(rf_probabilities)
str(rf_probabilities)

predicted_train_DF<- as.data.frame(rf_probabilities)
names(predicted_train_DF) <- c("statusgroup")
str(predicted_train_DF)

###Confusion Matrix
confusionMatrix(predicted_train_DF$statusgroup, expected_train_DF$statusgroup, positive = "2")

######################################################Split-Sampled Dataset##############################################
###Train Control
rf_control1 = trainControl(method = "cv", number = 5, classProbs = TRUE) 

###Optimize Hyperparameters
rf_grid <- expand.grid(mtry = 8)

###Build Formula
rf_form <- statusgroup ~.

###Model using caret
rf_fit3 <- train(rf_form, data = SampleFunc %>% dplyr::select(-c(id)),
                 method = "rf", metric = "Accuracy", trControl = rf_control1,
                 tuneGrid=rf_grid)

###Summary of Model
print(rf_fit1)

###Feature Selection
varImp(rf_fit3)

###Predictions converted to dataframe
rf_probabilities <- rf_fit3 %>% predict(eval_train_DF)
summary(rf_probabilities)
str(rf_probabilities)

predicted_train_DF<- as.data.frame(rf_probabilities)
names(predicted_train_DF) <- c("statusgroup")
str(predicted_train_DF)

###Confusion Matrix
confusionMatrix(predicted_train_DF$statusgroup, expected_train_DF$statusgroup, positive = "2")


















