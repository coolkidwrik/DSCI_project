library(tidyverse)
library(repr)
library(tidymodels)
options(repr.matrix.max.rows = 6)
set.seed(2022)

columns = c("age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "maxheartrate", "exang", "oldpeak", "slope", "ca", "thal", "num")
heart_data1 <- read_delim("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data", delim = ",", col_names = columns)
heart_data2 <- read_delim("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", delim = ",", col_names = columns)
heart_data3 <- read_delim("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data", delim = ",", col_names = columns)
heart_data4 <- read_delim("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data", delim = ",", col_names = columns)

heart_data1

heart_data2

heart_data3

heart_data4


heart_disease <- rbind(heart_data1, heart_data2, heart_data3, heart_data4) %>%
  select(-slope, -ca, -thal, -sex, -cp, -fbs, -restecg, -exang, -oldpeak) %>%
  filter(trestbps != "?", chol != "?", num != "?", num %in% c(0,1), chol != "0")%>%
  mutate(num = as_factor(num), trestbps = as.numeric(trestbps), 
         chol = as.numeric(chol), maxheartrate = as.numeric(maxheartrate)) %>%
  rename(disease_diagnosis = num)
heart_disease



heart_split <- heart_disease %>%
  initial_split(prop = 0.75, strata = disease_diagnosis)
training_data <- training(heart_split)
testing_data <- testing(heart_split)


testing_data


observation_per_class <- training_data %>%
  group_by(disease_diagnosis) %>%
  summarize(n = n())
observation_per_class


mean_data <- training_data %>%
  select(chol, age, trestbps, maxheartrate) %>%
  summarize(chol_avg = mean(chol),
            age_avg = mean(age),
            trestbps_avg = mean(trestbps),
            maxheartrate_avg = mean(maxheartrate))
mean_data



options (repr.plot.width = 20, repr.plot.height = 8)
disease_chol_hist <- ggplot(training_data, aes(x = chol, fill = as_factor(disease_diagnosis)))+
  geom_histogram(binwidth = 50) +
  labs(x = "Cholestrol(mg/dl)", y = "#measurements",
       fill = "disease diagnosis")+
  theme(text = element_text(size = 30))+
  facet_grid(cols = vars(disease_diagnosis))+
  ggtitle("Distribution of cholestrol")

disease_age_hist <- ggplot(training_data, aes(x = age, fill = as_factor(disease_diagnosis)))+
  geom_histogram(binwidth = 2) +
  labs(x = "age (years)", y = "#measurements",
       fill = "disease diagnosis")+
  theme(text = element_text(size = 30))+
  facet_grid(cols = vars(disease_diagnosis))+
  ggtitle("Distribution of age")

disease_trestbps_hist <- ggplot(training_data, aes(x = trestbps, fill = as_factor(disease_diagnosis)))+
  geom_histogram(binwidth = 10) +
  labs(x = "rest bps(mm Hg)", y = "#measurements",
       fill = "disease diagnosis")+
  theme(text = element_text(size = 30))+
  facet_grid(cols = vars(disease_diagnosis))+
  ggtitle("Distribution of resting bps")

disease_maxheartrate_hist <- ggplot(training_data, aes(x = maxheartrate, fill = as_factor(disease_diagnosis)))+
  geom_histogram(binwidth = 5) +
  labs(x = "max heart rate", y = "#measurements",
       fill = "disease diagnosis")+
  theme(text = element_text(size = 30))+
  facet_grid(cols = vars(disease_diagnosis))+
  ggtitle("Distribution of max heart rate")

disease_age_hist

disease_trestbps_hist

disease_maxheartrate_hist




options (repr.plot.width = 10, repr.plot.height = 8)

chol_vs_maxheartrate <- ggplot(training_data, aes(x = maxheartrate, y = chol))+
  geom_point(aes(colour = disease_diagnosis)) +
  labs(x = "max heart rate", y = "cholestrol(mg/dl)",
       colour = "disease diagnosis")+
  scale_color_manual(labels = c("< 50% narrowing ", "> 50% narrowing"), 
                     values = c("orange2", "steelblue2")) +
  theme(text = element_text(size = 20))+
  ggtitle("Classification of disease based on cholestrol and max heart rate")

chol_vs_maxheartrate




trestbps_vs_maxheartrate <- ggplot(training_data, aes(x = maxheartrate, y = trestbps))+
  geom_point(aes(colour = disease_diagnosis)) +
  labs(x = "max heart rate", y = "resting bps(mm Hg)",
       colour = "disease diagnosis")+
  scale_color_manual(labels = c("< 50% narrowing ", "> 50% narrowing"), 
                     values = c("orange2", "steelblue2")) +
  theme(text = element_text(size = 20))+
  ggtitle("Classification of disease based on trestbps and max heart rate")
trestbps_vs_maxheartrate




training_with_missing <- initial_split(rbind(heart_data1, heart_data2, heart_data3, heart_data4),
                                       prop = 0.75, strata = num) %>%
  training()
missing_data <- training_with_missing %>%
  filter(chol == "?" | chol == 0 | maxheartrate == "?") %>%
  summarize(missing = n())
missing_data




heart_recipe <- recipe(disease_diagnosis ~  chol + maxheartrate, data = training_data) %>%
  step_scale(all_predictors()) %>%
  step_center(all_predictors())





heart_vfold <- vfold_cv(training_data, v = 5, strata = disease_diagnosis)

gridvals <- tibble(neighbors = seq(1,40))

tune_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = tune()) %>%
  set_engine("kknn") %>%
  set_mode("classification")

heart_accuracy <- workflow() %>%
  add_recipe(heart_recipe) %>%
  add_model(tune_spec) %>%
  tune_grid(resamples = heart_vfold, grid = gridvals) %>% 
  collect_metrics() %>%
  
  
  

options (repr.plot.width = 10, repr.plot.height = 5)
cross_val_plot <- ggplot(heart_accuracy, aes(x = neighbors, y = mean))+
  geom_point() +
  geom_line() +
  labs(x = "Neighbors", y = "Accuracy Estimate") +
  theme(text = element_text(size = 15)) +
  ggtitle("accuracy vs neighbours")

cross_val_plot
  filter(.metric == "accuracy")
  
  
  
best_k_stats <- heart_accuracy %>%
  filter(mean == max(mean))
best_k <- best_k_stats %>%
  slice(1) %>%
  select(neighbors) %>%
  pull()
best_k_stats



model_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = best_k) %>%
  set_engine("kknn") %>%
  set_mode("classification")
model_fit <- workflow() %>%
  add_recipe(heart_recipe) %>%
  add_model(model_spec) %>%
  fit(data = training_data)



model_predictions <- predict(model_fit, testing_data) %>%
  bind_cols(testing_data)

model_metrics <- model_predictions %>%
  metrics(truth = disease_diagnosis, estimate = .pred_class) %>%
  filter(.metric == "accuracy")

model_conf_mat <- model_predictions %>% 
  conf_mat(truth = disease_diagnosis, estimate = .pred_class)

model_metrics



model_conf_mat
  