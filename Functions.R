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

CM_builder <- function(data, outcome, title_name)
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
    )  + 
    labs(title = {title_name})
}

feature_importance_extractor <- function(workflow_data, full_dataset)
{
  finalized_model <- {workflow_data} %>% fit({full_dataset})
  
  model_summary <- pull_workflow_fit(finalized_model)$fit
  
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
