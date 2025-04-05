df <- read.csv("C:/Users/anime/Downloads/model metrics.csv")

names(df)

library(ggplot2)

ggplot(df, aes(x = Logistic.Regression.Accuracy, y = after_stat(density))) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "lightblue", 
                 color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(Logistic.Regression.Accuracy, na.rm = TRUE)), color = "red", 
             linetype = "dashed", size = 1.5) +
  geom_density(color = "black", size = 1.5, alpha = 0.5) +
  
  # Customize labels and theme
  ggtitle("Probability Distribution of Accuracy Scores for Logistic Regression") +
  xlab("Accuracy Score") +
  ylab("Probability") +
  theme_minimal()


ggplot(df, aes(x = Random.Forest.Accuracy, y = after_stat(density))) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "lightblue", 
                 color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(Random.Forest.Accuracy, na.rm = TRUE)), color = "red", 
             linetype = "dashed", size = 1.5) +
  geom_density(color = "black", size = 1.5, alpha = 0.5) +
  
  # Customize labels and theme
  ggtitle("Probability Distribution of Accuracy Scores for Random Forest") +
  xlab("Accuracy Score") +
  ylab("Probability") +
  theme_minimal()


ggplot(df, aes(x = Decision.Tree.Accuracy, y = after_stat(density))) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "lightblue", 
                 color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(Decision.Tree.Accuracy, na.rm = TRUE)), color = "red", 
             linetype = "dashed", size = 1.5) +
  geom_density(color = "black", size = 1.5, alpha = 0.5) +
  
  # Customize labels and theme
  ggtitle("Probability Distribution of Accuracy Scores for Decision Tree") +
  xlab("Accuracy Score") +
  ylab("Probability") +
  theme_minimal()


ggplot(df, aes(x = LDA.Accuracy, y = after_stat(density))) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "lightblue", 
                 color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(LDA.Accuracy, na.rm = TRUE)), color = "red", 
             linetype = "dashed", size = 1.5) +
  geom_density(color = "black", size = 1.5, alpha = 0.5) +
  
  # Customize labels and theme
  ggtitle("Probability Distribution of Accuracy Scores for Linear Discriminant Analysis") +
  xlab("Accuracy Score") +
  ylab("Probability") +
  theme_minimal()
names(df)

ggplot(df, aes(x = KNN.Accuracy, y = after_stat(density))) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "lightblue", 
                 color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(KNN.Accuracy, na.rm = TRUE)), color = "red", 
             linetype = "dashed", size = 1.5) +
  geom_density(color = "black", size = 1.5, alpha = 0.5) +
  
  # Customize labels and theme
  ggtitle("Probability Distribution of Accuracy Scores for Extreme Gradient Boosting") +
  xlab("Accuracy Score") +
  ylab("Probability") +
  theme_minimal()

equal_odds <- read.csv("C:/Users/anime/Downloads/equalized odds.csv")

names(equal_odds)


ggplot(equal_odds, aes(x = Logistic.Regression.Priviledged.Rate, y = after_stat(density))) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "lightblue", 
                 color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(Logistic.Regression.Priviledged.Rate, na.rm = TRUE)), color = "red", 
             linetype = "dashed", size = 1.5) +
  geom_density(color = "black", size = 1.5, alpha = 0.5) +
  
  # Customize labels and theme
  ggtitle("Probability Distribution of Positive Prediction Rates among Whites for Logistic Regression") +
  xlab("Accuracy Score") +
  ylab("Probability") +
  theme_minimal()


ggplot(equal_odds, aes(x = Logistic.Regression.Discriminated.Rate, y = after_stat(density))) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "lightblue", 
                 color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(Logistic.Regression.Discriminated.Rate, na.rm = TRUE)), color = "red", 
             linetype = "dashed", size = 1.5) +
  geom_density(color = "black", size = 1.5, alpha = 0.5) +
  
  # Customize labels and theme
  ggtitle("Probability Distribution of Positive Prediction Rates among Blacks for Logistic Regression") +
  xlab("Accuracy Score") +
  ylab("Probability") +
  theme_minimal()





names(df)

mean(df$Logistic.Regression.Accuracy)

mean(df$LDA.Accuracy)

mean(df$Decision.Tree.Accuracy)

mean(df$Random.Forest.Accuracy)

mean(df$KNN.Accuracy)

names(equal_odds)

summary(equal_odds)


sd(equal_odds$Logistic.Regression.Discriminated.Rate)**2/sd(equal_odds$Logistic.Regression.Priviledged.Rate)**2

qt(0.975, 498)

s_pool <- 249*sd(equal_odds$Logistic.Regression.Discriminated.Rate)**2 + 249*sd(equal_odds$Logistic.Regression.Priviledged.Rate)**2

s_pool <- s_pool/498

pt(ts, 498, lower.tail = FALSE)

ts <- (mean(equal_odds$Logistic.Regression.Priviledged.Rate) - mean(equal_odds$Logistic.Regression.Discriminated.Rate))/(((sd(equal_odds$Logistic.Regression.Discriminated.Rate)**2) + (sd(equal_odds$Logistic.Regression.Priviledged.Rate)**2))/250)**0.5

ts
pnorm(ts, lower.tail = FALSE)



sd(equal_odds$LDA.Discriminated.Rate)**2/sd(equal_odds$LDA.Priviledged.Rate)**2

qt(0.975, 498)

s_pool <- 249*sd(equal_odds$LDA.Discriminated.Rate)**2 + 249*sd(equal_odds$LDA.Priviledged.Rate)**2

s_pool <- s_pool/498



ts <- (mean(equal_odds$LDA.Priviledged.Rate) - mean(equal_odds$LDA.Discriminated.Rate))/(((sd(equal_odds$LDA.Discriminated.Rate)**2) + (sd(equal_odds$LDA.Priviledged.Rate)**2))/250)**0.5


pnorm(ts, lower.tail = FALSE)



sd(equal_odds$RF.Discriminated.Rate)**2/sd(equal_odds$RF.Priviledged.Rate)**2

qt(0.975, 498)

s_pool <- 249*sd(equal_odds$RF.Discriminated.Rate)**2 + 249*sd(equal_odds$RF.Priviledged.Rate)**2

s_pool <- s_pool/498



ts <- (mean(equal_odds$RF.Priviledged.Rate) - mean(equal_odds$RF.Discriminated.Rate))/(((sd(equal_odds$RF.Discriminated.Rate)**2) + (sd(equal_odds$RF.Priviledged.Rate)**2))/250)**0.5


pnorm(ts, lower.tail = FALSE)


sd(equal_odds$XG.Discriminated.Rate)**2/sd(equal_odds$XG.Priviledged.Rate)**2

qt(0.975, 498)

s_pool <- 249*sd(equal_odds$XG.Discriminated.Rate)**2 + 249*sd(equal_odds$XG.Priviledged.Rate)**2

s_pool <- s_pool/498



ts <- (mean(equal_odds$XG.Priviledged.Rate) - mean(equal_odds$XG.Discriminated.Rate))/(((sd(equal_odds$XG.Discriminated.Rate)**2) + (sd(equal_odds$XG.Priviledged.Rate)**2))/250)**0.5


pnorm(ts, lower.tail = FALSE)


confusion_test <- read.csv("C:/Users/anime/Downloads/confusion metrics.csv")


confusion_test

names(confusion_test)

mean(confusion_test$RFB)

summary(confusion_test)

sd(confusion_test$RFB)**2
mean(confusion_test$LDA)

null_likelihood <- (1/(((2*pi)**0.5)*sd(confusion_test$RFA)))**250

names(confusion_test)


