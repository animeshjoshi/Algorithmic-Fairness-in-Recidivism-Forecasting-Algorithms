confusion_test <- read.csv("C:/Users/anime/Downloads/confusion metrics.csv")

df <- read.csv("C:/Users/anime/Downloads/model metrics.csv")
equal_odds <- read.csv("C:/Users/anime/Downloads/equalized odds.csv")
confusion_test

names(confusion_test)

mean(confusion_test$RFB)

summary(confusion_test)

two_sample_test_upper <- function(group1, group2) {
  
  print((sd(group1)**2)/(sd(group2)**2))
  
  pooled_est <- 249*(sd(group1)**2) + 249*(sd(group2)**2)
  pooled_est <- pooled_est/498
  
  print(mean(group1))
  print(mean(group2))
  
  ts <- (mean(group1) - mean(group2))/(pooled_est**0.5*((2/250)**0.5))
  
  print(ts)
  
  print(pt(df = 498, ts, lower.tail = FALSE))
 
  
}

two_sample_test_lower <- function(group1, group2) {
  
  print((sd(group1)**2)/(sd(group2)**2))
  
  pooled_est <- 249*(sd(group1)**2) + 249*(sd(group2)**2)
  pooled_est <- pooled_est/498
  
  print(mean(group1))
  print(mean(group2))
  
  ts <- (mean(group1) - mean(group2))/(pooled_est**0.5*((2/250)**0.5))
  
  print(ts)
  
  print(pt(df = 498, ts))
  
  
}
names(equal_odds)

two_sample_test_upper(equal_odds$Logistic.Regression.Priviledged.Rate,equal_odds$Logistic.Regression.Discriminated.Rate)
two_sample_test_upper(equal_odds$LDA.Priviledged.Rate,equal_odds$LDA.Discriminated.Rate)
two_sample_test_upper(equal_odds$XG.Priviledged.Rate,equal_odds$XG.Discriminated.Rate)
two_sample_test_upper(equal_odds$Decision.Tree.Priviledged.Rate,equal_odds$Decision.Tree.Discriminated.Rate)
two_sample_test_upper(equal_odds$RF.Priviledged.Rate,equal_odds$RF.Discriminated.Rate)

two_sample_test_lower(confusion_test$LRG, confusion_test$LRH)

two_sample_test_lower(confusion_test$LDG, confusion_test$LDH)

two_sample_test_lower(confusion_test$RFG, confusion_test$RFH)

two_sample_test_lower(confusion_test$XGG, confusion_test$XGH)


two_sample_test_upper(confusion_test$LRC, confusion_test$LRD)

two_sample_test_upper(confusion_test$LDC, confusion_test$LDD)

two_sample_test_upper(confusion_test$RFC, confusion_test$RFD)

two_sample_test_upper(confusion_test$XGC, confusion_test$XGD)

confusion_test

confusion_test$LRPPR1 <- (confusion_test$LRA + confusion_test$LRC)/(confusion_test$LRA + confusion_test$LRC + confusion_test$LRE + confusion_test$LRG)
confusion_test$LRPPR2 <- (confusion_test$LRB + confusion_test$LRD)/(confusion_test$LRB + confusion_test$LRD + confusion_test$LRF + confusion_test$LRH)
confusion_test$LRNPR2 <- (confusion_test$LRF + confusion_test$LRH)/(confusion_test$LRB + confusion_test$LRD + confusion_test$LRF + confusion_test$LRH)
confusion_test$LRNPR1 <- (confusion_test$LRE + confusion_test$LRG)/(confusion_test$LRA + confusion_test$LRC + confusion_test$LRE + confusion_test$LRG)

ppr <- read.csv('C:/Users/anime/Downloads/NPR and PPR Distribution.csv')

ppr
names(ppr)
ppr$LRPrivPPR
ppr$LRPrivNPR

mean(ppr$LRPrivPPR)
mean(ppr$LRProtPPR)
two_sample_test_upper(ppr$LRPrivPPR,ppr$LRProtPPR)
two_sample_test_upper(ppr$LDAPrivPPR,ppr$LDAProtPPR)

two_sample_test_upper(ppr$RFPrivPPR,ppr$RFProtPPR)

two_sample_test_upper(ppr$XGPrivPPR,ppr$XGProtPPR)
two_sample_test_lower(ppr$LRPrivNPR,ppr$LRProtNPR)
two_sample_test_upper(ppr$RFPrivPPR,ppr$RFProtPPR)
two_sample_test_lower(ppr$RFPrivNPR,ppr$RFProtNPR)

plot(ppr$LRPrivPPR, ppr$LRPrivNPR)


summary(equal_odds)

total <- confusion_test$RFA + confusion_test$RFC + confusion_test$RFE + confusion_test$RFG
total

confusion_test

two_sample_test_lower(confusion_test$LRG, confusion_test$LRH)
two_sample_test_lower(confusion_test$LDG, confusion_test$LDH)

two_sample_test_lower(confusion_test$RFG, confusion_test$RFH)
two_sample_test_lower(confusion_test$XGG, confusion_test$XGH)

two_sample_test_upper(confusion_test$LRC, confusion_test$LRD)
two_sample_test_upper(confusion_test$LDC, confusion_test$LDD)
two_sample_test_upper(confusion_test$RFC, confusion_test$RFD)
two_sample_test_upper(confusion_test$XGC, confusion_test$XGD)


names(ppr)

two_sample_test_upper(ppr$LRPrivPPR, ppr$LRProtPPR)
two_sample_test_lower(ppr$LRPrivNPR, ppr$LRProtNPR)


contingencies <- read.csv("C:/Users/anime/Downloads/contingency tables.csv")

head(contingencies)

mean(contingencies$LR.White.Recicidivism)

summary(contingencies)

