

#1. Setting working directory ----
working_dir <- "/Users/chandrasekarve/Desktop/SMU/Term3/BA/GroupProject/BA_Project"
setwd(working_dir)

#2. Installing packages ----
# importing packages
install.packages("tidyverse","pacman","dplyr","ggplot2")
pacman::p_load(tidyverse, lubridate, # Tidy data science
               tidymodels, # Tidy Machine Learning
               skimr, GGally, ggstatsplot, Hmisc, broom, # EDA
               plotly, DT, doParallel, parsnip # Interactive Data Display
)

doParallel::registerDoParallel()
#dataset_fileid <- "1xRtAU4csPPQCfu7TsbCvSxXxMm_xj69lemUeMJFVRso"
#churn <- read_csv(sprintf("https://docs.google.com/uc?id=%s&export=download", dataset_fileid))

#3. Import, Transform ----
churn_df <- read_csv("churn_dataset_train.csv")
curr_date <- as.Date("2022-07-01")
skim(churn_df)
names(churn_df)
churn_df %>% count(customer_id)
c#Taking first 10000 rows for simplicity

## Clean/Transform
churn_cleaned <- churn_df %>% 
  filter(days_since_last_login > 0 & avg_time_spent > 0) %>% 
  drop_na() %>% 
  slice(1:3000) %>% 
  mutate(age_with_company = difftime(curr_date,joining_date, units = "days"),
         across(c(age_with_company,last_visit_time),as.numeric)) %>% 
  select(-medium_of_operation, -internet_option, 
         -offer_application_preference, -feedback,
         -referral_id, -customer_id, -security_no, -Name) %>% 
  mutate(churn = ifelse(churn_risk_score >=3, "Yes", "No"))
names(churn_cleaned)
skim(churn_cleaned)
unique(churn_cleaned$churn)
churn_cleaned <- churn_cleaned %>% 
  complete() %>% 
  dplyr::mutate_all(as.factor) %>% 
  mutate(churn = fct_relevel(churn, "Yes"))
names(churn_cleaned)
churn_cleaned <- churn_cleaned %>% 
  select(age,churn, avg_time_spent, points_in_wallet,
         gender, age_with_company, avg_transaction_value)

## Summarize
names(churn_cleaned)
skim(churn_cleaned)
summary(churn_cleaned$churn)

# 4. EDA ----

## Planning
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

## 4.1 Correlation matrix ----
## using Hmisc:: & DT::
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

# 5. Predictive Model ----

## 5.1 Split ----
## Planning
set.seed(12345678)
churn_split <- 
  churn_cleaned %>% 
  initial_split(prop = 0.8)

## Executing
churn_train <- 
  churn_split %>% 
  training() 

churn_test <- 
  churn_split %>% 
  testing() # 20%

## 5.2 Pre-processing (Feature Engineering) ----
recipe_RF <- 
  recipe(churn ~ age + avg_time_spent + points_in_wallet + gender + 
           age_with_company + avg_transaction_value,
         data = churn_train) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

## 5.3 Fitting ----
library(ranger)
### 5.3.1 Random Forest ----
model_RF <- 
  rand_forest() %>% 
  set_args(mtry = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

## 5.4 Tuning ----
workflow_RF <- 
  workflow() %>% 
  add_recipe(recipe_RF) %>% 
  add_model(model_RF)
workflow_RF

grid_RF <- expand.grid(mtry = c(3,4,5))

## 5.5 Cross Validation ----
set.seed(12345678)
CV_10 <- churn_train %>% 
  vfold_cv(v = 10)
CV_10

## 5.6 Parallel Processing ----
tuned_RF <- workflow_RF %>% 
  tune::tune_grid(resamples = CV_10,
                  grid = grid_RF,
                  metrics = metric_set(accuracy, roc_auc, f_meas))

tuned_RF_results <- tuned_RF %>% 
  collect_metrics()
tuned_RF_results

## 5.7 Select Best ----
parameters_tuned_RF <- tuned_RF %>% 
  select_best(metric = "roc_auc")
parameters_tuned_RF

## 5.8 Finalize Workflow ----
finalized_workflow_RF <- workflow_RF %>% 
  finalize_workflow(parameters_tuned_RF)
finalized_workflow_RF

## 5.9 Last Fit ----
### 5.9.1 Random Forest ----
fit_RF <- finalized_workflow_RF %>% 
  last_fit(churn_split)
fit_RF

# 6. Assessing Model Performance ----
performance_RF <- fit_RF %>% 
  collect_metrics()
performance_RF

## 6.1 Prediction ----
predictions_RF <- fit_RF %>% 
  collect_predictions()
