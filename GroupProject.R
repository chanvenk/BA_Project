

#Setting working directory ####
working_dir <- "/Users/chandrasekarve/Desktop/SMU/Term3/BA/GroupProject"
setwd(working_dir)

#Installing packages ####
# importing packages
install.packages("tidyverse","pacman","dplyr","ggplot2")
pacman::p_load(tidyverse, lubridate, # Tidy data science
               tidymodels, # Tidy Machine Learning
               skimr, GGally, ggstatsplot, Hmisc, broom, # EDA
               plotly, DT, doParallel # Interactive Data Display
)

doParallel::registerDoParallel()
#dataset_fileid <- "1xRtAU4csPPQCfu7TsbCvSxXxMm_xj69lemUeMJFVRso"
#churn <- read_csv(sprintf("https://docs.google.com/uc?id=%s&export=download", dataset_fileid))
churn <- read_csv("churn_dataset_train.csv")
curr_date <- as.Date("2022-07-01")
skim(churn)
names(churn)
churn %>% count(customer_id)
c#Taking first 10000 rows for simplicity

churn_cleaned <- churn %>% 
  filter(days_since_last_login > 0 & avg_time_spent > 0) %>% 
  drop_na() %>% 
  slice(1:3000) %>% 
  mutate(age_with_company = difftime(curr_date,joining_date, units = "days"),
         across(c(age_with_company,last_visit_time),as.numeric)) %>% 
  select(-medium_of_operation, -internet_option, 
         -offer_application_preference, -feedback,
         -referral_id, -customer_id, -security_no, -Name)
names(churn_cleaned)
skim(churn_cleaned)
churn_cleaned <- churn_cleaned %>% 
  complete() %>% 
  dplyr::mutate_all(as.factor)
names(churn_cleaned)
churn_cleaned <- churn_cleaned %>% 
  select(age,churn_risk_score, avg_time_spent, points_in_wallet,
         gender, age_with_company, avg_transaction_value)

names(churn_cleaned)
skim(churn_cleaned)
### EDA ####
## Planning

EDA_recipe <- 
  recipe(formula = churn_risk_score ~ .,data = churn_cleaned) %>% 
  step_normalize(all_numeric_predictors()) %>%  # setting Ms at 0; SDs at 1 %>% 
  step_dummy(all_nominal_predictors())

## Execution

EDA_baked <- 
  EDA_recipe %>% # plan 
  prep() %>% # for calculation
  bake(new_data = churn_cleaned) 

skim(EDA_baked)

## Three way of running EDA ----

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
  filter(var1 == "churn_risk_score" | 
           var2 == "churn_risk_score") %>% 
  DT::datatable()

####  

EDA_baked %>% 
  names(.) %>% 
  as_tibble()

rent_cleaned  %>% 
  ggplot(aes(x = dist_to_mrt,
             y = log10_Price)
  )+
  geom_point(color = "dodgerblue", 
             alpha = 0.3)+
  geom_smooth(method = "loess",
              formula = y ~ x,
              se = F,
              color = "purple") +
  geom_smooth(method = "lm",
              formula = y ~ x,
              se = F,
              color = "green") +
  geom_smooth(method = "lm",
              formula = y ~ poly (x, degree = 2),
              se = F,
              color = "tomato3")+
  theme_bw()
####
set.seed(12345678)
churn_split <- churn_cleaned %>% 
  initial_split(prop = 0.8)

### Pre-processing
recipe_linear <- 
  recipe(formula = log10_Price ~ dist_to_mrt + age,
         data = rent_train) %>% 
  step_normalize(all_numeric_predictors()
  ) %>% 
  step_poly(dist_to_mrt,
            degree = 2,
            role = "predictor")


