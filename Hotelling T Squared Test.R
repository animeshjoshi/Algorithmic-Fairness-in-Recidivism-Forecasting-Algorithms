library(MASS)
data <- read.csv("C:/Users/anime/Downloads/confusion metrics.csv")

head(data)

mean(data$LRA)

log_priv_meanvector <- colMeans(cbind(data$LRA, data$LRC, data$LRE, data$LRG))
log_prot_meanvector <- colMeans(cbind(data$LRB, data$LRD, data$LRF, data$LRH))
diff_vector <- log_priv_meanvector - log_prot_meanvector

prot_matrix <- cbind(data$LRB, data$LRD, data$LRF, data$LRH)

priv_matrix <- cbind(data$LRA, data$LRC, data$LRE, data$LRG)

covprot <- cov(prot_matrix)

covpriv <- cov(priv_matrix)

S <- (249*covprot + 249*covpriv)/498

md <- t(diff_vector)%*%ginv(S)%*%diff_vector

t_squared = (250*250)/500 * md

f = 495/(4*498) * t_squared

pf(f, 4, 495, lower.tail = FALSE)



log_priv_meanvector

log_prot_meanvector

names(data)

data

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

two_sample_test_twotail <- function(group1, group2) {
  
  print((sd(group1)**2)/(sd(group2)**2))
  
  pooled_est <- 249*(sd(group1)**2) + 249*(sd(group2)**2)
  pooled_est <- pooled_est/498
  
  print(mean(group1))
  print(mean(group2))
  
  ts <- (mean(group1) - mean(group2))/(pooled_est**0.5*((2/250)**0.5))
  
  print(ts)
  
  print(2*pt(df = 498, abs(ts), lower.tail = FALSE))
  
  
}

two_sample_test_upper(data$LRC, data$LRD)
two_sample_test_lower(data$LRG, data$LRH)

mean(data$LRC)
mean(data$LRG)

data$LRPrivErrorRatio <- data$LRC/data$LRG
data$LRProtErrorRatio <- data$LRD/data$LRH

data$LDAPrivErrorRatio <- data$LDC/data$LDG
data$LDAProtErrorRatio <- data$LDD/data$LDH

data$RFPrivErrorRatio <- data$RFC/data$RFG
data$RFProtErrorRatio <- data$RFD/data$RFH

data$XGPrivErrorRatio <- data$XGC/data$XGG
data$XGProtErrorRatio <- data$XGD/data$XGH


two_sample_test_twotail(data$LRPrivErrorRatio, data$LRProtErrorRatio)
two_sample_test_twotail(data$LDAPrivErrorRatio, data$LDAProtErrorRatio)
two_sample_test_twotail(data$RFPrivErrorRatio, data$RFProtErrorRatio)
two_sample_test_twotail(data$XGPrivErrorRatio, data$XGProtErrorRatio)

data$LRPrivAccuracy <- data$LRA + data$LRE
data$LRProtAccuracy <- data$LRB + data$LRF

data$RFPrivAccuracy <- data$RFA + data$RFE
data$RFProtAccuracy <- data$RFB + data$RFF

data$LDAPrivAccuracy <- data$LDA + data$LDE
data$LDAProtAccuracy <- data$LDB + data$LDF

data$XGPrivAccuracy <- data$XGA + data$XGE
data$XGProtAccuracy <- data$XGB + data$XGF

two_sample_test_twotail(data$LRPrivAccuracy, data$LRProtAccuracy)
two_sample_test_twotail(data$LDAPrivAccuracy, data$LDAProtAccuracy)
two_sample_test_twotail(data$RFPrivAccuracy, data$RFProtAccuracy)
two_sample_test_twotail(data$XGPrivAccuracy, data$XGProtAccuracy)