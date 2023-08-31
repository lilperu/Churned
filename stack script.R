
install.packages("tidymodels")
library(tidymodels)
install.packages("rpart")
install.packages("rsample")
install.packages("lubridate")
install.packages("magrittr")
library(magrittr)
install.packages("dplyr")
library(dplyr)
install.packages("baguette")
library(baguette)
install.packages("ranger")
install.packages("vip")

#######################################################
bank<- read.csv("bank_churners.csv")
str(bank)

bank$still_customer<- as.factor(bank$still_customer)
head(bank)
colSums(is.na(bank))

set.seed(1998)
bank_split<- initial_split(bank, strata=still_customer)
bank

bank_train<- training(bank_split)
bank_test<- testing(bank_split)
nrow(bank_train)/nrow(bank)

#######################################################
# single decision tree
# Specify, fit, predict and combine with training data
predictions <- decision_tree() %>%
  set_mode("classification") %>%
  set_engine("rpart") %>% 
  fit(still_customer ~ ., data = bank_train) %>%
  predict(new_data = bank_train, type = "prob") %>% 
  bind_cols(bank_train)

preds_tree<- decision_tree() %>% set_mode("classification") %>% set_engine("rpart") %>% fit(still_customer ~ ., data = bank_test) %>% predict(bank_test, type="prob") %>% mutate(bank_test$still_customer)
preds_tree

##########################################################################################
# bagged
# create the specification
spec_bagged<- bag_tree() %>% set_mode("classification") %>% set_engine("rpart", times= 100)

# fit to the training data
model_bagged<- fit(spec_bagged, formula= still_customer~total_trans_amt+ customer_age+ education_level, data= bank_train)

# predict on training set and add to training set
predictions<- predict(model_bagged, bank_train, type= "prob") %>% bind_cols(bank_train)
predictions

# create and plot the ROC curve
roc_curve(predictions, truth= still_customer, .pred_no) %>% autoplot()

# calculate the AUC
roc_auc(predictions, truth= still_customer, .pred_no)

# predicted probabilities of test data
preds_bagging<- predict(model_bagged, bank_test, type="prob") %>% bind_cols(bank_test) %>% mutate(bank_test$still_customer)
preds_bagging

##################################################################################################
# specify random forest
spec<- rand_forest(mtry= 4, trees= 500, min_n= 10) %>% set_mode("classification") %>% 
# specify algorithm that controls node split
  set_engine("ranger", importance= "impurity")

# train random forest
forest_model<- spec %>% fit(still_customer~., data= bank_train)

preds_forest<- forest_model %>% predict(bank_test, type="prob") %>% mutate(bank_test$still_customer)
preds_forest

######################################################################################################

# specify the model class
boost_spec<- boost_tree() %>% set_mode("classification") %>% set_engine("xgboost")

boost_model <- fit(boost_spec, still_customer~., bank_train)

set.seed(1998)
folds<- vfold_cv(bank_train, v=5)

cv_results<- fit_resamples(boost_spec, still_customer~., resamples= folds, metric_set(roc_auc))
collect_metrics(cv_results)

#create tuning spec with placeholders
boost_spec<- boost_tree(trees= 500, learn_rate= tune(), tree_depth= tune(), sample_size= tune()) %>% set_mode("classification") %>% set_engine("xgboost")

# create tuning grid
tunegrid_boost<- grid_regular(parameters(boost_spec), levels= 3)
# or, for 8 random combinations
grid_random(parameters(boost_spec), size= 8)

folds= vfold_cv(bank_train, v=6)

# tune along the grid
tune_results<- tune_grid(boost_spec, still_customer~., folds, tunegrid_boost, metric_set(roc_auc))

# visualize the result
autoplot(tune_results)

best_params<- select_best(tune_results)
best_params

#finalize the model
final_spec<- finalize_model(boost_spec, best_params)
final_spec

# train the final model
final_model<- final_spec %>% fit(still_customer~., data= bank_train)
final_model

preds_boosting<- final_spec %>% fit(formula= still_customer~., data= bank_train) %>% predict(bank_test, type= "prob") %>% mutate(bank_test$still_customer)
preds_boosting

bind_cols(preds_tree, preds_bagging, preds_forest, preds_boosting, bank_test %>% select(still_customer))
as_tibble(preds_combined)

preds_combined<- bind_cols(preds_tree, preds_bagging, preds_forest, preds_boosting, bank_test %>% select(still_customer))
as_tibble(preds_combined)

roc_auc(preds_combined, still_customer, estimate= preds_tree)
#Error in `roc_auc()`:
#! Can't subset columns that don't exist.
#✖ Column `still_customer` doesn't exist.

# model AUC comparison tibble
bind_rows(decision_tree= roc_auc(preds_combined, truth= still_customer, preds_tree), bagged_trees= roc_auc(preds_combined, truth= still_customer, preds_bagging), random_forest= roc_auc(preds_combined, truth= still_customer, preds_forest), boosted_trees= roc_auc(preds_combined, truth= still_customer, preds_boosting), .id="model")
#Error in `roc_auc()`:
#! Can't subset columns that don't exist.
#✖ Column `still_customer` doesn't exist.
