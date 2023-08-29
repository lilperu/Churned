
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

# create the specification
spec_bagged<- bag_tree() %>% set_mode("classification") %>% set_engine("rpart", times= 100)

# fit to the training data
model_bagged<- fit(spec_bagged, formula= still_customer~total_trans_amt+ customer_age+ education_level, data= bank_train)

# predict on training set and add to training set
predictions<- predict(model_bagged, bank_train, type= "prob") %>% bind_cols(bank_train)
predictions

# Create and plot the ROC curve
roc_curve(predictions,
          estimate = .pred_yes,
          truth = still_customer) %>% autoplot()
# Error in 'roc_curve()': ! Can't rename variables in this context