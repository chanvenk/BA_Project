

#Setting working directory ----
working_dir <- "/Users/chandrasekarve/Desktop/SMU/Term3/BA/GroupProject/BA_Project"
setwd(working_dir)

#Installing packages ----
# importing packages
install.packages("tidyverse","pacman","dplyr","ggplot2")
pacman::p_load(tidyverse, lubridate, # Tidy data science
               tidymodels, # Tidy Machine Learning
               skimr, GGally, ggstatsplot, Hmisc, broom, # EDA
               plotly, DT, doParallel, parsnip, themis # Interactive Data Display
)

doParallel::registerDoParallel()
#dataset_fileid <- "1xRtAU4csPPQCfu7TsbCvSxXxMm_xj69lemUeMJFVRso"
#churn <- read_csv(sprintf("https://docs.google.com/uc?id=%s&export=download", dataset_fileid))
churn_df <- read_csv("churn_dataset_train.csv")
curr_date <- as.Date("2022-07-01")

churn_cleaned <- churn_df %>% 
  filter(days_since_last_login > 0 & avg_time_spent > 0 & avg_frequency_login_days > 0
         & avg_frequency_login_days != "Error" & gender != "Unknown") %>% 
  drop_na() %>% 
  slice(1:10000) %>% 
  mutate(age_with_company = difftime(curr_date,joining_date, units = "days"),
         across(c(age_with_company,last_visit_time),as.numeric)) %>% 
  select(-medium_of_operation, -internet_option, 
         -offer_application_preference, -feedback,
         -referral_id, -customer_id, -security_no, -Name) %>% 
  mutate(churn = ifelse(churn_risk_score >=3, "Yes", "No")) %>% 
  complete() %>% 
  dplyr::mutate_all(as.factor) %>% 
  mutate(churn1 = fct_relevel(churn, "Yes"))

churn_cleaned <- churn_cleaned %>% 
  select(churn, age, avg_time_spent, points_in_wallet,
         gender, age_with_company, avg_transaction_value, region_category,
         membership_category, used_special_discount, avg_frequency_login_days)

# EDA ((NEED TO WORK ON THIS)) ####
## Planning
is.na(churn_df)
EDA_recipe <- 
  recipe(formula = churn ~ .,data = churn_cleaned) %>% 
  step_normalize(all_numeric_predictors()) %>%  # setting Ms at 0; SDs at 1 %>% 
  step_dummy(all_nominal_predictors())

## Execution

EDA_baked <- 
  EDA_recipe %>% # plan 
  prep() %>% # for calculation
  bake(new_data = churn_cleaned) 
EDA_baked
skim(EDA_baked)



## 1. Correlation matrix using Hmisc:: & DT::
EDA_baked
EDA_baked %>% 
  as.matrix(.) %>% 
  rcorr(.) %>% 
  tidy(.) %>% 
  rename(var1 = column1, 
         var2 = column2,
         CORR = estimate) %>% 
  mutate(absCORR = abs(CORR)) %>% 
  filter(var1 == "churn" | 
           var2 == "churn") %>% 
  DT::datatable()

# Splitting the data ----
set.seed(100)
churn_split <- churn_cleaned %>% 
  initial_split(prop = 0.8)

## Executing

churn_train <- 
  churn_split %>% 
  training() 

churn_test <- 
  churn_split %>% 
  testing() # 20%
# Random Forreset Workflowwithout any up/down sampling ----
recipe_RF <- 
  recipe(churn ~ age + avg_time_spent + points_in_wallet + gender + 
           age_with_company + avg_transaction_value + region_category +
           membership_category + used_special_discount + 
           avg_frequency_login_days,
         data = churn_train) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())
library(ranger)
model_RF <- 
  rand_forest() %>% 
  set_args(mtry = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

workflow_RF <- 
  workflow() %>% 
  add_recipe(recipe_RF) %>% 
  add_model(model_RF)
workflow_RF

grid_RF <- expand.grid(mtry = c(3,4,5))

set.seed(100)
CV_10 <- churn_train %>% 
  vfold_cv(v = 10)
CV_10

tuned_RF <- workflow_RF %>% 
  tune::tune_grid(resamples = CV_10,
                  grid = grid_RF,
                  metrics = metric_set(accuracy, roc_auc, f_meas))

tuned_RF_results <- tuned_RF %>% 
  collect_metrics()
tuned_RF_results

parameters_tuned_RF <- tuned_RF %>% 
  select_best(metric = "roc_auc")
parameters_tuned_RF

finalized_workflow_RF <- workflow_RF %>% 
  finalize_workflow(parameters_tuned_RF)
finalized_workflow_RF

fit_RF <- finalized_workflow_RF %>% 
  last_fit(churn_split)
fit_RF

performance_RF <- fit_RF %>% 
  collect_metrics()
performance_RF

predictions_RF <- fit_RF %>% 
  collect_predictions()
predictions_RF

# Random Forrest Workflow with upsampling ----
recipe_RF_up<- 
  recipe(churn ~ age + avg_time_spent + points_in_wallet + gender + 
           age_with_company + avg_transaction_value + region_category +
           membership_category + used_special_discount + 
           avg_frequency_login_days,
         data = churn_train) %>% 
  step_upsample(churn)
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())
library(ranger)
model_RF_up <- 
  rand_forest() %>% 
  set_args(mtry = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

workflow_RF_up <- 
  workflow() %>% 
  add_recipe(recipe_RF_up) %>% 
  add_model(model_RF_up)

workflow_RF_up

grid_RF <- expand.grid(mtry = c(3,4,5))

set.seed(100)
CV_10_up <- churn_train %>% 
  vfold_cv(v = 10)
CV_10_up

tuned_RF_up <- workflow_RF_up %>% 
  tune::tune_grid(resamples = CV_10_up,
                  grid = grid_RF,
                  metrics = metric_set(accuracy, roc_auc, f_meas))

tuned_RF_results_up <- tuned_RF_up %>% 
  collect_metrics()
tuned_RF_results_up

parameters_tuned_RF_up <- tuned_RF_up %>% 
  select_best(metric = "roc_auc")
parameters_tuned_RF

finalized_workflow_RF_up <- workflow_RF_up %>% 
  finalize_workflow(parameters_tuned_RF_up)
finalized_workflow_RF_up

fit_RF_up <- finalized_workflow_RF_up %>% 
  last_fit(churn_split)
fit_RF_up

performance_RF_up <- fit_RF_up %>% 
  collect_metrics()
performance_RF

predictions_RF_up <- fit_RF_up %>% 
  collect_predictions()
predictions_RF_up
 
# Random Forrest Workflow with downsampling ----
recipe_RF_down<- 
  recipe(churn ~ age + avg_time_spent + points_in_wallet + gender + 
           age_with_company + avg_transaction_value + region_category +
           membership_category + used_special_discount + 
           avg_frequency_login_days,
         data = churn_train) %>% 
  step_downsample(churn) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())
library(ranger)
model_RF_down <- 
  rand_forest() %>% 
  set_args(mtry = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

workflow_RF_down <- 
  workflow() %>% 
  add_recipe(recipe_RF_down) %>% 
  add_model(model_RF_down)

workflow_RF_down

grid_RF <- expand.grid(mtry = c(3,4,5))

set.seed(100)
CV_10_down <- churn_train %>% 
  vfold_cv(v = 10)
CV_10_down

tuned_RF_down <- workflow_RF_down %>% 
  tune::tune_grid(resamples = CV_10_down,
                  grid = grid_RF,
                  metrics = metric_set(accuracy, roc_auc, f_meas))

tuned_RF_results_down <- tuned_RF_down %>% 
  collect_metrics()
tuned_RF_results_down

parameters_tuned_RF <- tuned_RF %>% 
  select_best(metric = "roc_auc")
parameters_tuned_RF

finalized_workflow_RF <- workflow_RF %>% 
  finalize_workflow(parameters_tuned_RF)
finalized_workflow_RF

fit_RF <- finalized_workflow_RF %>% 
  last_fit(churn_split)
fit_RF

performance_RF <- fit_RF %>% 
  collect_metrics()
performance_RF

predictions_RF <- fit_RF %>% 
  collect_predictions()
predictions_RF


# Combined Results from all random forrest iterations
tuned_RF_results
tuned_RF_results_up
tuned_RF_results_down