

#Setting working directory ----
working_dir <- "/Users/chandrasekarve/Desktop/SMU/Term3/BA/BA_Project"
setwd(working_dir)

#Installing packages ----
# importing packages
#install.packages("tidyverse","pacman","dplyr","ggplot2")
pacman::p_load(tidyverse, lubridate, # Tidy data science
               tidymodels, # Tidy Machine Learning
               skimr, GGally, ggstatsplot, Hmisc, broom, # EDA
               plotly, DT, doParallel, parsnip, themis, ranger,
               ggpubr, vip# Interactive Data Display
)

doParallel::registerDoParallel()
churn_df <- read_csv("churn_dataset_train.csv")
curr_date <- as.Date("2022-07-01")

churn_cleaned <- churn_df %>%
  filter(days_since_last_login > 0 & avg_time_spent > 0 & avg_frequency_login_days > 0
         & avg_frequency_login_days != "Error" & gender != "Unknown"
         & joined_through_referral != "?" & churn_risk_score > 0) %>%
  drop_na() %>%
  mutate(age_with_company = difftime(Sys.Date(),joining_date, units = "days")) %>%
  mutate(avg_frequency_login_days_interval =
           ifelse(as.numeric(avg_frequency_login_days) > 0 & as.numeric(avg_frequency_login_days) <= 10, 1,
                  ifelse(as.numeric(avg_frequency_login_days) > 10 & as.numeric(avg_frequency_login_days) <= 20, 2,
                         ifelse(as.numeric(avg_frequency_login_days) > 20 & as.numeric(avg_frequency_login_days) <= 30, 3,
                                ifelse(as.numeric(avg_frequency_login_days) > 30, 4, 0)
                         )))) %>%
  select(-medium_of_operation,
         -referral_id, -customer_id, -security_no, -Name) %>%
  mutate(churn = ifelse(churn_risk_score >=5, "Yes", "No")) %>%
  complete() %>%
  dplyr::mutate_all(as.factor) %>%
  dplyr::mutate(across(c(
    age, avg_time_spent, points_in_wallet,
    age_with_company, avg_transaction_value,
  avg_frequency_login_days
  ),as.numeric)) %>% 
  mutate(churn1 = fct_relevel(churn, "Yes"))
churn_cleaned <- churn_cleaned %>% 
  select(churn, points_in_wallet, membership_category, avg_transaction_value,
         avg_time_spent, age_with_company, age, avg_frequency_login_days_interval)

# EDA using ggapirs ----
library(GGally)
churn_cleaned %>% 
  select(churn, where(is.numeric)) %>% 
  ggpairs(.,aes(color = churn),
          lower = list(continuous = wrap("smooth",
                                         alpha = 0.25,
                                         size = 0.2)
                  )
  ) + 
  theme_bw()
## Correlation values 
recipe_eda <- 
  recipe(formula = churn ~ ., data = churn_cleaned) %>% 
  step_normalize(points_in_wallet, avg_transaction_value, avg_time_spent, age_with_company, age, 
) %>% 
  step_dummy(membership_category, avg_frequency_login_days_interval, region_category, 
             internet_option, past_complaint, used_special_discount, past_complaint, feedback,
             complaint_status)

baked_eda <- recipe_eda %>%
  prep(retain = TRUE) %>%
  bake(new_data = NULL)

baked_eda %>% 
  as.matrix(.) %>% 
  rcorr(.) %>% 
  tidy(.) %>% 
  mutate(absolute_corr = abs(estimate)
  ) %>% 
  rename(variable1 = column1,
         variable2 = column2,
         corr = estimate) %>% 
  filter(variable1 == "churn" | variable2 == "churn") %>% 
  datatable()


churn_cleaned %>% 
  select(churn, where(is.numeric)
  ) %>% 
  mutate(churn = as.numeric(churn),
         churn = ifelse(churn == 1, 1, 0)
  ) %>% 
  as.matrix(.) %>% 
  Hmisc::rcorr(.) %>% 
  broom::tidy(.) %>% 
  mutate(strength = abs(estimate)
  ) %>% 
  datatable()

churn_cleaned %>% 
  ggplot(aes(x = ))

# Function to run the models ----
workflow_generator <- function(var_recipe, var_model)
{
    workflow() %>% 
    add_recipe({var_recipe}) %>% 
    add_model({var_model})
}

perf_and_pred_generator <- function(var_workflow, var_tune, var_split)
{
  var_parameters <- {var_tune} %>% 
    select_best(metric = "roc_auc")
  
  var_finalized_workflow <- {var_workflow} %>% 
    finalize_workflow(var_parameters)
  
  var_fit <- var_finalized_workflow %>% 
    last_fit(var_split)
  
  var_performance <- var_fit %>% 
    collect_metrics()
  
  var_predictions <- var_fit %>% 
    collect_predictions()
  
  return(list(perf = var_performance, pred = var_predictions))
  
}

CM_builder <- function(data, outcome)
{ 
  {data} %>% 
    conf_mat({outcome}, .pred_class) %>% 
    pluck(1) %>% 
    as_tibble() %>% 
    mutate(cm_colors = ifelse(Truth == "Yes" & Prediction == "Yes", "True Positive",
                              ifelse(Truth == "Yes" & Prediction == "No", "False Negative",
                                     ifelse(Truth == "No" & Prediction == "Yes", 
                                            "False Positive", 
                                            "True Negative")
                              )
    )
    ) %>% 
    ggplot(aes(x = Prediction, y = Truth)) + 
    geom_tile(aes(fill = cm_colors), show.legend = F) +
    scale_fill_manual(values = c("True Positive" = "green",
                                 "False Negative" = "red",
                                 "False Positive" = "red",
                                 "True Negative" = "green")
    ) + 
    geom_text(aes(label = n), color = "white", size = 10) + 
    geom_label(aes(label = cm_colors), vjust = 2
    ) + 
    theme_fivethirtyeight() + 
    theme(axis.title = element_text()
    ) # + 
  # labs(title = "Your First Confusion Matrix")
}

feature_importance_extractor <- function(workflow_data, full_dataset)
{
  finalized_model <- {workflow_data} %>% fit({full_dataset})
  
  model_summary <- extract_fit_parsnip(finalized_model)$fit
  
  feature_importance <- data.frame(importance = model_summary$variable.importance) %>% 
    rownames_to_column("feature") %>% 
    as_tibble() %>% 
    mutate(feature = as.factor(feature)) 
  
  feature_importance %>% 
    ggplot(aes(x = importance, y = reorder(feature, importance), fill = importance)) +
    geom_col(show.legend = F) +
    scale_fill_gradient(low = "deepskyblue1", high = "deepskyblue4") +
    scale_x_continuous(expand = c(0, 0)) +
    labs(
      y = NULL,
      title = "Feature (Variable) Importance for Churn Prediction") + 
    ggthemes::theme_fivethirtyeight()
}


# Splitting the data ----
set.seed(100)

churn_split <- churn_cleaned %>% 
  initial_split(prop = 0.8)

churn_train <- 
  churn_split %>% 
  training() 

churn_test <- 
  churn_split %>% 
  testing() # 20%

## Splitting the data into cross validation sets ----
set.seed(100)
CV_10 <- churn_train %>% 
  vfold_cv(v = 10, strata = churn)


# Recipes ----
recipe_common <- recipe(churn ~ age + avg_time_spent + points_in_wallet + 
                          age_with_company + avg_transaction_value +
                          membership_category + 
                          avg_frequency_login_days_interval,
                        data = churn_train)

recipe_all <- 
  recipe(churn ~ age + avg_time_spent + points_in_wallet + 
           age_with_company + avg_transaction_value +
           membership_category + 
           avg_frequency_login_days_interval + preferred_offer_types + internet_option + 
         days_since_last_login + used_special_discount+ past_complaint+
         feedback + complaint_status,
         data = churn_train)
  


recipe_norm_dummy <- 
  recipe_common %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

recipe_up <-   
  recipe_all %>% 
  step_upsample(churn) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact(terms = ~ avg_transaction_value:starts_with("membership_category")) %>% 
  step_poly(avg_transaction_value, degree = 2, role = "predictor") %>% 
  step_poly(points_in_wallet, degree = 2, role = "predictor")

recipe_down <- 
  recipe_common %>% 
  step_downsample(churn) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

recipe_boxcox <- 
  recipe_common %>% 
  step_BoxCox(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact(terms = ~ avg_transaction_value:starts_with("membership_category")) #%>% 
  step_poly(avg_transaction_value, degree = 2, role = "predictor") %>% 
  step_poly(points_in_wallet, degree = 2, role = "predictor")


recipe_rose <- 
  recipe_common %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_rose(churn)

recipe_yeojohnson <- 
  recipe_common %>% 
  step_YeoJohnson(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact(terms = ~ avg_transaction_value:starts_with("membership_category")) 

recipe_log <- 
  recipe_common %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_log(points_in_wallet)



# Models ----

##Logistic Regression ----
## Logistic Regression without any up/down sampling
model_glm <- 
  logistic_reg(mode = "classification") %>% 
  set_engine("glm") 

workflow_glm <- workflow_generator(recipe_norm_dummy, model_glm)

tuned_glm <- 
  workflow_glm %>% 
  tune::tune_grid(resamples = CV_10,
                  metrics = metric_set(accuracy, roc_auc, f_meas)
  )

perf_and_pred <- perf_and_pred_generator(workflow_glm, tuned_glm, churn_split)
performance_glm <- perf_and_pred$perf
predictions_glm <- perf_and_pred$pred



##NULL Model
model_null <- null_model() %>% 
  set_engine("parsnip") %>% 
  set_mode("classification")

workflow_null <- workflow_generator(recipe_norm_dummy,model_null)

fit_null <- 
  workflow_null %>% 
  fit_resamples(CV_10,
                control = control_resamples(save_pred = T))


performance_null <- fit_null %>% collect_metrics()
predictions_null <- fit_null %>% collect_predictions()


## Random Forrest ----
### Random Forreset Workflowwithout any up/down sampling ----
model_RF <- 
  rand_forest() %>% 
  set_args(mtry = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

workflow_RF <- workflow_generator(recipe_norm_dummy, model_RF)

grid_RF <- expand.grid(mtry = c(3,4,5))

tuned_RF <- workflow_RF %>% 
  tune::tune_grid(resamples = CV_10,
                  grid = grid_RF,
                  metrics = metric_set(accuracy, roc_auc, f_meas))
perf_and_pred <- perf_and_pred_generator(workflow_RF,tuned_RF, churn_split)
performance_RF <- perf_and_pred$perf
predictions_RF <- perf_and_pred$pred

### Random Forrest Workflow with upsampling ----
workflow_RF_up <- workflow_generator(recipe_up, model_RF)

tuned_RF_up <- workflow_RF_up %>% 
  tune::tune_grid(resamples = CV_10,
                  grid = grid_RF,
                  metrics = metric_set(accuracy, roc_auc, f_meas))

perf_and_pred <- perf_and_pred_generator(workflow_RF_up, tuned_RF_up, churn_split)
performance_RF_up <- perf_and_pred$perf
predictions_RF_up <- perf_and_pred$pred
performance_RF_up
predictions_RF_up %>% f_meas(churn, .pred_class)
### Random Forrest Workflow with downsampling ----
workflow_RF_down <- workflow_generator(recipe_down, model_RF)

tuned_RF_down <- workflow_RF_down %>% 
  tune::tune_grid(resamples = CV_10,
                  grid = grid_RF,
                  metrics = metric_set(accuracy, roc_auc, f_meas))
perf_and_pred <- perf_and_pred_generator(workflow_RF_down, tuned_RF_down, churn_split)
performance_RF_down <- perf_and_pred$perf
predictions_RF_down <- perf_and_pred$pred
performance_RF_down 
### Random Forrest Workflow with boxcox ----
workflow_RF_boxcox <- workflow_generator(recipe_boxcox, model_RF)

tuned_RF_boxcox <- workflow_RF_boxcox %>% 
  tune::tune_grid(resamples = CV_10,
                  grid = grid_RF,
                  metrics = metric_set(accuracy, roc_auc, f_meas))

perf_and_pred <- perf_and_pred_generator(workflow_RF_boxcox, tuned_RF_boxcox, churn_split)
performance_RF_boxcox <- perf_and_pred$perf
predictions_RF_boxcox <- perf_and_pred$pred
performance_RF_boxcox
predictions_RF_boxcox %>% f_meas(churn, .pred_class)
### Random Forreest Workflow with YeoJohnson ----
workflow_RF_yeojohnson <- workflow_generator(recipe_yeojohnson, model_RF)

tuned_RF_yeojohnson <- workflow_RF_yeojohnson %>% 
  tune::tune_grid(resamples = CV_10,
                  grid = grid_RF,
                  metrics = metric_set(accuracy, roc_auc, f_meas))
perf_and_pred <- perf_and_pred_generator(workflow_RF_yeojohnson, tuned_RF_yeojohnson, churn_split)
performance_RF_yeojohnson <- perf_and_pred$perf
predictions_RF_yeojohnson <- perf_and_pred$pred

### Random Forrest Workflow with log
workflow_RF_log <- workflow_generator(recipe_log, model_RF)

tuned_RF_log <- workflow_RF_log %>% 
  tune::tune_grid(resamples = CV_10,
                  grid = grid_RF,
                  metrics = metric_set(accuracy, roc_auc, f_meas))
perf_and_pred <- perf_and_pred_generator(workflow_RF_log, tuned_RF_log, churn_split)
performance_RF_log <- perf_and_pred$perf
predictions_RF_log <- perf_and_pred$pred


### Random Forrest Workflow with rose ----
workflow_RF_rose <- workflow_generator(recipe_rose, model_RF)

tuned_RF_rose <- workflow_RF_rose %>% 
  tune::tune_grid(resamples = CV_10,
                  grid = grid_RF,
                  metrics = metric_set(accuracy, roc_auc, f_meas))
perf_and_pred <- perf_and_pred_generator(workflow_RF_rose, tuned_RF_rose, churn_split)
performance_RF_rose <- perf_and_pred$perf
predictions_RF_rose <- perf_and_pred$pred




##XGBoost ----
### XGBoost w/o up or down sampling----
model_xgb <- 
  boost_tree(trees = 1000,
             mtry = tune(),
             min_n = tune(),
             tree_depth = tune(),
             sample_size = tune(),
             learn_rate = tune()
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

workflow_xg <- workflow_generator(recipe_norm_dummy, model_xgb)

set.seed(100)
grid_XG <-
  grid_max_entropy(
    mtry(c(5L, 10L),
    ),
    min_n(c(10L, 40L)
    ),
    tree_depth(c(5L, 10L)
    ),
    sample_prop(c(0.5, 1.0)
    ),
    learn_rate(c(-2, -1)
    ),
    size = 20
  )
tuned_xg <- 
  workflow_xg %>% 
  tune_grid(resamples = CV_10,
            grid = grid_XG,
            control = control_grid(save_pred = T),
            metrics = metric_set(accuracy, roc_auc, f_meas)
  )

perf_and_pred <- perf_and_pred_generator(workflow_xg, tuned_xg, churn_split)
performance_xg <- perf_and_pred$perf
predictions_xg <- perf_and_pred$pred

### XGBoost with upsampling ----

workflow_xg_up <- workflow_generator(recipe_up, model_xgb)
tuned_xg_up <- 
  workflow_xg_up %>% 
  tune_grid(resamples = CV_10,
            grid = grid_XG,
            control = control_grid(save_pred = T),
            metrics = metric_set(accuracy, roc_auc, f_meas)
  )
perf_and_pred <- perf_and_pred_generator(workflow_xg_up, tuned_xg_up, churn_split)
performance_xg_up <- perf_and_pred$perf
predictions_xg_up <- perf_and_pred$pred

### XGBoost with downsampling ----

workflow_xg_down <- workflow_generator(recipe_down, model_xgb)
tuned_xg_down <- 
  workflow_xg_down %>% 
  tune_grid(resamples = CV_10,
            grid = grid_XG,
            control = control_grid(save_pred = T),
            metrics = metric_set(accuracy, roc_auc, f_meas)
  )
perf_and_pred <- perf_and_pred_generator(workflow_xg_down, tuned_xg_down, churn_split)
performance_xg_down <- perf_and_pred$perf
predictions_xg_down <- perf_and_pred$pred

### XGBoost with boxcox  ----

workflow_xg_boxcox <- workflow_generator(recipe_boxcox, model_xgb)
tuned_xg_boxcox <- 
  workflow_xg_boxcox %>% 
  tune_grid(resamples = CV_10,
            grid = grid_XG,
            control = control_grid(save_pred = T),
            metrics = metric_set(accuracy, roc_auc, f_meas)
  )
perf_and_pred <- perf_and_pred_generator(workflow_xg_boxcox, tuned_xg_boxcox, churn_split)
performance_xg_boxcox <- perf_and_pred$perf
predictions_xg_boxcox <- perf_and_pred$pred
performance_xg_boxcox
predictions_xg_boxcox %>% f_meas(churn, .pred_class)
### XGBoost with rose  ----

workflow_xg_rose <- workflow_generator(recipe_rose, model_xgb)
tuned_xg_rose <- 
  workflow_xg_rose %>% 
  tune_grid(resamples = CV_10,
            grid = grid_XG,
            control = control_grid(save_pred = T),
            metrics = metric_set(accuracy, roc_auc, f_meas)
  )
perf_and_pred <- perf_and_pred_generator(workflow_xg_rose, tuned_xg_rose, churn_split)
performance_xg_rose <- perf_and_pred$perf
predictions_xg_rose <- perf_and_pred$pred


  
# Combined Results from all models ----
performance_RF
performance_RF_up
performance_RF_down
perf_table <- performance_RF %>% 
  mutate(performance_RF_up$.estimate) %>% 
  mutate(performance_RF_down$.estimate) %>% 
  mutate(performance_RF_rose$.estimate) %>% 
  mutate(performance_RF_boxcox$.estimate) %>% 
  mutate(performance_xg$.estimate) %>% 
  mutate(performance_xg_up$.estimate) %>%
  mutate(performance_xg_down$.estimate) %>%
  mutate(performance_xg_rose$.estimate) %>%
  mutate(performance_xg_boxcox$.estimate) %>%
  select(-.estimator,-.config) 
column_names <- c("Metric","RF","RF_up","RF_down","RF_rose","RF_boxcox",
                  "XG","XG_up","XG_down","XG_rose","XG_boxcox")
colnames(perf_table) <- column_names
perf_table
### Output from Chandra's computer
#Metric   `W/o Sampling` UpSampling Downsampling
#<chr>             <dbl>      <dbl>        <dbl>
#  1 accuracy          0.871      0.874        0.876
#2 roc_auc           0.954      0.952        0.951
predictions_RF_down <- predictions_RF_down %>% 
  mutate(algo = "RF_down")
predictions_RF_up <- predictions_RF_up %>% 
  mutate(algo = "RF_up")
predictions_RF <- predictions_RF %>% 
  mutate(algo = "RF")
predictions_RF_rose <- predictions_RF_rose %>% 
  mutate(algo = "RF_rose")
predictions_RF_boxcox <- predictions_RF_boxcox %>% 
  mutate(algo = "RF_boxcox")

predictions_xg_down <- predictions_xg_down %>% 
  mutate(algo = "xg_down")
predictions_xg_up <- predictions_xg_up %>% 
  mutate(algo = "xg_up")
predictions_xg <- predictions_xg %>% 
  mutate(algo = "xg")
predictions_xg_rose <- predictions_xg_rose %>% 
  mutate(algo = "xg_rose")
predictions_xg_boxcox <- predictions_xg_boxcox %>% 
  mutate(algo = "xg_boxcox")


# Drawing the ROC-AUC curve between multiple models
comparing_predictions <- bind_rows(predictions_RF, 
                                   predictions_RF_down,
                                   predictions_RF_up,
                                   predictions_RF_boxcox,
                                   predictions_xg,
                                   predictions_xg_up,
                                   predictions_xg_rose,
                                   predictions_xg_boxcox)
comparing_predictions %>%
  group_by(algo) %>% # Say hello to group_by()
  roc_curve(truth = churn, 
            .pred_Yes) %>%
  autoplot() +
#  ggthemes::scale_color_fivethirtyeight() +
  labs(title = "Comparing different models",
       color = "Prediction Tools")
# Confusion Matrix ----
CM_RF <- CM_builder(predictions_RF, "churn")
CM_RF_Up <- CM_builder(predictions_RF_up, "churn")
CM_RF_Down <- CM_builder(predictions_RF_down, "churn")
CM_RF_Rose <- CM_builder(predictions_RF_rose,"churn")
CM_RF_boxcox <- CM_builder(predictions_RF_boxcox,"churn")

CM_xg <- CM_builder(predictions_xg, "churn")
CM_xg_Up <- CM_builder(predictions_xg_up, "churn")
CM_xg_Down <- CM_builder(predictions_xg_down, "churn")
CM_xg_Rose <- CM_builder(predictions_xg_rose,"churn")
CM_xg_boxcox <- CM_builder(predictions_xg_boxcox,"churn")




# To see each of the plots, run the commands below :

ggarrange(CM_RF, CM_RF_Up, CM_RF_Down, CM_RF_Rose, CM_RF_boxcox,
          CM_xg, CM_xg_Up, CM_xg_Down, CM_xg_Rose, CM_xg_boxcox,
          ncol = 5, nrow = 2, 
          labels = c("RF","RF_up","RF_down","RF_rose","RF_boxcox",
                  "XG","XG_up","XG_down","XG_rose","XG_boxcox"))
# Extracting feature importance ----

finalized_model <- workflow_RF %>% fit(churn_cleaned)

model_summary <- pull_workflow_fit(finalized_model)$fit

feature_importance <- data.frame(importance = model_summary$variable.importance) %>% 
  rownames_to_column("feature") %>% 
  as_tibble() %>% 
  mutate(feature = as.factor(feature)) %>% 
  feature_importance %>% 
  ggplot(aes(x = importance, y = reorder(feature, importance), fill = importance)) +
  geom_col(show.legend = F) +
  scale_fill_gradient(low = "deepskyblue1", high = "deepskyblue4") +
  scale_x_continuous(expand = c(0, 0)) +
  labs(
    y = NULL,
    title = "Feature (Variable) Importance for Churn Prediction") + 
  ggthemes::theme_fivethirtyeight()


feature_imp_RF <- feature_importance_extractor(workflow_RF, churn_cleaned)
feature_imp_RF_up <- feature_importance_extractor(workflow_RF_up, churn_cleaned)
feature_imp_RF_down <- feature_importance_extractor(workflow_RF_down, churn_cleaned)
feature_imp_RF_Rose <- feature_importance_extractor(workflow_RF_rose, churn_cleaned)
feature_imp_RF_boxcox <- feature_importance_extractor(workflow_RF_boxcox, churn_cleaned)

feature_imp_xg <- feature_importance_extractor(workflow_xg, churn_cleaned)
feature_imp_xg_up <- feature_importance_extractor(workflow_xg_up, churn_cleaned)
feature_imp_xg_down <- feature_importance_extractor(workflow_xg_down, churn_cleaned)
feature_imp_xg_Rose <- feature_importance_extractor(workflow_xg_rose, churn_cleaned)
feature_imp_xg_boxcox <- feature_importance_extractor(workflow_xg_boxcox, churn_cleaned)

# To see feature importance for each workflow run the commands below 
feature_imp_RF
feature_imp_RF_up
feature_imp_RF_down
feature_imp_RF_boxcox

ggarrange(feature_imp_RF, feature_imp_RF_up, feature_imp_RF_down, feature_imp_RF_Rose,
          feature_imp_RF_boxcox, feature_imp_xg, feature_imp_xg_up, feature_imp_xg_down, 
          feature_imp_xg_Rose, feature_imp_xg_boxcox,
          ncol = 2, nrow = 5, 
          labels = c("RF","RF_up","RF_down","RF_rose","RF_boxcox",
                     "XG","XG_up","XG_down","XG_rose","XG_boxcox"))
# Finalized model for shiny ----
finalized_parameters_tuned <- 
  tuned_RF_boxcox %>% 
  select_best(metric = "roc_auc")

finalized_workflow <- 
  workflow_RF_boxcox %>% 
  finalize_workflow(finalized_parameters_tuned)
finalized_model <-
  finalized_workflow %>% 
  fit(churn_cleaned)

finalized_model %>% 
  saveRDS("finalized_model_BAProject.rds")

# Shiny App ----
library(shiny)
library(shinydashboard)

##UI ----

MODEL <- readRDS("finalized_model_BAProject.rds")

ui <- 
  dashboardPage(
    
    dashboardHeader(title = "Churn Prediction App"),
    
    dashboardSidebar(
      menuItem(
        "Churn Prediction",
        tabName = "prices_tab",
        icon = icon("snowflake")
      )
    ),
    
    dashboardBody(
      
      tabItem(
        tabName = "attribute_tab",
        box(sliderInput(inputId = "age", label = "Age of the customer",
                        min = 1, max = 55, value = 8)
        ),
        box(sliderInput(inputId = "avg_transaction_value", 
                        label = "Avg Transaction Value",
                        min = 1, max = 20000, value = 10000)
        ),
        box(sliderInput(inputId = "avg_time_spent", 
                        label = "Avg Time Spent",
                        min = 1, max = 15000, value = 7500)
        ),
        box(sliderInput(inputId = "points_in_wallet", 
                        label = "Points in Wallet",
                        min = 1, max = 18000, value = 9000)
        ),
        box(sliderInput(inputId = "age_with_company", 
                        label = "Duration with Company",
                        min = 1, max = 1500, value = 750)
        ),
        box(selectInput(inputId = "avg_frequency_login_days_interval", 
                        label = "Avg Frequency Login",
                        c("0-10" = 1, "10-20" = 2,
                          "20-30" = 2, ">30" = 4))
        ),
        box(selectInput(inputId = "membership_category", 
                        label = "Membership Category",
                        c("No Membership" = "No Membership",
                          "Basic" = "Basic Membership",
                          "Silver" = "Silver Membership",
                          "Gold" = "Gold Membership",
                          "Platinum" = "Platinum Membership"))
        ),
      ),
      tabItem(tabName = "churn_risk",
              gaugeOutput("churn_risk")
      ),
      tabItem(
        tabName = "prediction_tab",
        box(valueBoxOutput("churn_prediction")
        )
      )

      
    )
  )

## Server ----

server <- function(input, output) 
{ 
  output$churn_prediction <- 
    renderValueBox(
      {
        
        prediction <- 
          predict(MODEL,
                  tibble("age" = input$age,
                         "age_with_company" = input$age_with_company,
                         "points_in_wallet" = input$points_in_wallet,
                         "avg_transaction_value" = input$avg_transaction_value,
                         "avg_time_spent" = input$avg_time_spent,
                         "membership_category" = input$membership_category,
                         "avg_frequency_login_days_interval" = input$avg_frequency_login_days_interval
                  )
                  
          )
        
        predicted_values <- 
          predict(MODEL,
                  tibble("age" = input$age,
                         "age_with_company" = input$age_with_company,
                         "points_in_wallet" = input$points_in_wallet,
                         "avg_transaction_value" = input$avg_transaction_value,
                         "avg_time_spent" = input$avg_time_spent,
                         "membership_category" = input$membership_category,
                         "avg_frequency_login_days_interval" = input$avg_frequency_login_days_interval
                  ))
        predicted_probabilities <- 
          predict(MODEL,
                  tibble("age" = input$age,
                         "age_with_company" = input$age_with_company,
                         "points_in_wallet" = input$points_in_wallet,
                         "avg_transaction_value" = input$avg_transaction_value,
                         "avg_time_spent" = input$avg_time_spent,
                         "membership_category" = input$membership_category,
                         "avg_frequency_login_days_interval" = input$avg_frequency_login_days_interval
                  ),type="prob")
        
        for_prediction_statement <- prediction$.pred_class

        shinydashboard::valueBox(
          value = predicted_values,
          subtitle = paste0("Will this customer churn?")
        )
        gauge(predicted_probabilities$.pred_Yes * 100, 
              min = 0, 
              max = 100, 
              sectors = gaugeSectors(success = c(0, 33), 
                                     warning = c(33, 66),
                                     danger = c(66, 100)))

        
      }
    )
  
}

## Run Shiny App ----

shinyApp(ui, server)


# Results ----
## With all possible predictors
## churn ~ age + avg_time_spent + points_in_wallet + gender + 
##age_with_company + avg_transaction_value + region_category +
##  membership_category + used_special_discount + 
##  avg_frequency_login_days + internet_option + 
##  offer_application_preference
##Metric      RF RF_up RF_down RF_rose RF_boxcox    XG XG_up XG_down XG_rose XG_boxcox
##<chr>    <dbl> <dbl>   <dbl>   <dbl>     <dbl> <dbl> <dbl>   <dbl>   <dbl>     <dbl>
##  1 accuracy 0.869 0.872   0.873   0.610     0.870 0.872 0.872   0.871   0.576     0.876
##2 roc_auc  0.954 0.955   0.952   0.908     0.954 0.954 0.954   0.952   0.932     0.954

#> perf_table
## A tibble: 2 × 11
#Metric      RF RF_up RF_down RF_rose RF_boxcox    XG XG_up XG_down XG_rose XG_boxcox
#<chr>    <dbl> <dbl>   <dbl>   <dbl>     <dbl> <dbl> <dbl>   <dbl>   <dbl>     <dbl>
#  1 accuracy 0.871 0.872   0.873   0.428     0.871 0.876 0.871   0.870   0.413     0.874
#2 roc_auc  0.955 0.954   0.950   0.905     0.955 0.955 0.956   0.952   0.930     0.955
#churn ~ age + avg_time_spent + points_in_wallet + gender + 
#  age_with_company + avg_transaction_value + region_category +
#  membership_category + used_special_discount + 
#  avg_frequency_login_days + internet_option + 
#  offer_application_preference + feedback + past_complaint +
#  complaint_status + preferred_offer_types + joined_through_referral#

# Clustering ----
churn_clustering <- churn_cleaned %>% 
  select(churn, gender, membership_category, avg_frequency_login_days_interval)

churn_clustering <- churn_clustering[sample(nrow(churn_cleaned),1000),]
reciped_EDA <- recipe(formula = churn ~ .,
                      data = churn_clustering) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(churn, all_nominal_predictors())

baked_EDA <- reciped_EDA %>%
  prep(retain = TRUE) %>%
  bake(new_data = NULL)


baked_EDA <- baked_EDA %>% select(-starts_with("churn"))
fviz_nbclust(baked_EDA,
             kmeans,
             method = "wss")

# A novel approach to specify the number of clusters 

# Gap statistic ####

install.packages("cluster")
library(cluster)

gap_statistic <- 
  clusGap(baked_EDA,
          FUN = kmeans,
          nstart = 50,
          K.max = 10,
          B = 1000)

bfactoextra::fviz_gap_stat(gap_statistic)

# Visualization for Reporting ----

cluster3 <- 
  baked_EDA %>% 
  kmeans(3, nstart = 30)

# You might want to provide lables for each observation.

fviz_cluster(cluster3,
             data = baked_EDA,
             geom = "text",
             repel = T) + 
  theme_bw()

# Interpretable ML: Tidy ML advanced ----

devtools::install_github("EmilHvitfeldt/tidyclust")
library(tidyclust)

Model_kMeans <- 
  k_means(k = 3) %>% 
  set_engine_tidyclust("stats")

Model_kMeans

kMeans_Algorithm <- 
  Model_kMeans %>% 
  fit(formula =  ~., 
      data = as.data.frame(baked_EDA)
  )

kMeans_Algorithm %>% 
  extract_cluster_assignment() %>% 
  print(n = nrow(.)
  )

# Persona Analysis ----

# customer_id
# item1: Shopping is fun                              
# item2: Shopping is bad for your budget               
# item3: I combine shopping with eating out            
# item4: I try to get the best buys while shopping     
# item5: I don't care about shopping                   
# item6: You can save lot of money by comparing prices 
# income: The household income of the respondent      
# visits: How often they visit the mall   


for_persona_analysis <-
  extract_centroids(kMeans_Algorithm) 

persona <- 
  for_persona_analysis %>% 
  ggradar(grid.min = -1.5,
          grid.max = 1.3,
          legend.position = "top")


persona
# Thank you for working with the R script :)

