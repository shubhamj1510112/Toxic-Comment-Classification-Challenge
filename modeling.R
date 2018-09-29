# Set the path
setwd("F:/edwisors/assignments/project-2/all/training")
getwd()
rm(list=ls())


###### construct training data for selected variables #################
###########################################################################

# construct new data for training based on selected variables from each toxic type
train = read.table(file="train_95743.csv", header=T, stringsAsFactors = F, sep="\t", na.strings = c("NA", ""), fill = T)
id = train[,1]
target = train[,3:8]
varsel = read.table(file="final_selected_variables.csv", header=F, stringsAsFactors = F, sep="\t", na.strings = c("NA", ""), fill = T)

# construct comment vector to generate the training data
require("stringi")
require("stringr")
require("tm")
require("wordcloud")
require("slam")
require("RSentiment")
# create corpus so that text preprocessing methods can be applied
train[,2] = gsub("[^a-zA-Z0-9 ]", "", train[,2])
postCorpus = Corpus(VectorSource(train[,2]))
# convert complete text to lowercase
postCorpus = tm_map(postCorpus, tolower)
writeLines(as.character(postCorpus[[2]]))
# remove stop words
postCorpus = tm_map(postCorpus, removeWords, stopwords('english'))
writeLines(as.character(postCorpus[[2]]))
# remove punctuation marks
postCorpus = tm_map(postCorpus, removePunctuation)
writeLines(as.character(postCorpus[[2]]))
# remove unnecesary whitespaces
postCorpus = tm_map(postCorpus, stripWhitespace)
writeLines(as.character(postCorpus[[2]]))
#remove numbers as they can not lead to any type of toxicity
postCorpus = tm_map(postCorpus, removeNumbers)
writeLines(as.character(postCorpus[[2]]))
# text stemming: convert to base word and remove suffix and prefix
postCorpus  = tm_map(postCorpus, stemDocument, language="english")
writeLines(as.character(postCorpus[[2]]))
# convert to string vector again
comment = data.frame(text=sapply(postCorpus, identity))
# comments lemmatization
require(textstem)
comment = lemmatize_words(comment)

fea = data.frame(id = train[,1])
for (i in c(1:nrow(varsel))) {
  vec = as.integer()
  for (j in c(1:nrow(comment))) {
    c = as.integer(str_count(as.character(comment[j,1]), pattern=as.character(varsel[i,1])))
    vec = c(vec, c)
  } 
  fea = cbind(fea, vec)
  print(paste0("iteration:", i))
}
dim(fea) 
colnames(fea) = c("id", varsel[,1]) 
predictors = fea[,2:(nrow(varsel)+1)] 

train_for_modeling = cbind(id, predictors, target)
write.table(train_for_modeling, file="train_for_modeling.csv", row.names=F, col.names=T, sep="\t", quote=F)
# this selected data will be used for further modeling 
# and traning of machine learning algorithms


# our problem is a multilabel calssification problem
# in which every observation can have more than one label
# so we try different methods used for performing multilabel calssification


#########################################################################
######## data preparation for 5-fole cross validation ###################
# as this a algorithm comparison step and this can be done with a sample data
# moreover running this comparison on complete data is computationally very expensive
# so we take randomly sampled 5000 observations and use this for making this comparison 
# randomly shuffle the data so that any bias in data spliting is gone
df<-train_for_modeling[sample(nrow(train_for_modeling)),]
# create sample data
samp = df[sample(nrow(df),  5000, replace=F),]
# this comparison will be done with 5-fold cross validation 
# manually create data partitions for 5-fold cross validation
# this is done so that our customized performance metric (RMSLE)can be evaluated at each fold
# therefore can have clear idea about the bias and variance of different algorithms
#Create 5 equally size folds
folds <- cut(seq(1,nrow(samp)),breaks=5,labels=FALSE)
# create 5 different train test dataset pairs
for(i in 1:5){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  test <- samp[testIndexes, ]
  train <- samp[-testIndexes, ]
  write.table(train, file=paste0("samp_train_cv_", i, ".csv"), row.names=F, col.names=T, sep="\t", quote=F)
  write.table(test, file=paste0("samp_test_cv_", i, ".csv"), row.names=F, col.names=T, sep="\t", quote=F)
}


###########################################################################
###### comparison among five types of problem atransformation methods ####### 

# we will use problem transformation methods as algorithm transformation 
# is very limited due to the lack of multilabel capable algorithms
# comparison of most powerful and most famous multilabel approaches
# binary relevance, classifer chains. netsted stacking, DBR wrapper, stacking wrapper
# this is done only for cv1 (one-fold) data as bias and variace is not
# a problem for comparsion among these four algorithms

train = read.table(file=paste0("samp_train_cv_", 1, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
train_ip = train[,-1]
test = read.table(file=paste0("samp_test_cv_", 1, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
test_ip = test[,-1]

# convert data to multilabel accpted format (target variables into logical)
for (i in c(111:116)) {
  train_ip[,i] <- as.character(train_ip[,i])
  train_ip[,i][train_ip[,i] == "0"] <- "FALSE"
  train_ip[,i][train_ip[,i] == "1"] <- "TRUE"
}
# comsider only complete cases
train_ip = train_ip[complete.cases(train_ip), ] # there was no missing values so missing value treatment not required
# change class of target varaibles 
train_ip[,111] = as.logical(train_ip[,111])
train_ip[,112] = as.logical(train_ip[,112])
train_ip[,113] = as.logical(train_ip[,113])
train_ip[,114] = as.logical(train_ip[,114])
train_ip[,115] = as.logical(train_ip[,115])
train_ip[,116] = as.logical(train_ip[,116])

# convert data to multilabel accpted format (target variables into logical)
for (i in c(111:116)) {
  test_ip[,i] <- as.character(test_ip[,i])
  test_ip[,i][test_ip[,i] == "0"] <- "FALSE"
  test_ip[,i][test_ip[,i] == "1"] <- "TRUE"
}
# comsider only complete cases
test_ip = test_ip[complete.cases(test_ip), ] # there was no missing values so missing value treatment not required
# change class of target varaibles 
test_ip[,111] = as.logical(test_ip[,111])
test_ip[,112] = as.logical(test_ip[,112])
test_ip[,113] = as.logical(test_ip[,113])
test_ip[,114] = as.logical(test_ip[,114])
test_ip[,115] = as.logical(test_ip[,115])
test_ip[,116] = as.logical(test_ip[,116])

library(mlr)
library(OpenML)
# define varable with target variable names
labels = c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
# define task
train_ip.task = makeMultilabelTask(id = "multi", data = train_ip, target = labels)

# comparison of different methods available for multilabel classification
#Core learner
lrn.core= makeLearner("classif.rpart", predict.type = "prob", fix.factors.prediction = TRUE)

#construct different types of Wrapped learners
lrn.binrel=makeMultilabelBinaryRelevanceWrapper(lrn.core)
lrn.chain=makeMultilabelClassifierChainsWrapper(lrn.core)
lrn.nest=makeMultilabelNestedStackingWrapper(lrn.core)
lrn.dbr= makeMultilabelDBRWrapper(lrn.core)
lrn.stack=makeMultilabelStackingWrapper(lrn.core)

#Training 6 models on train subset
mod.binrel=mlr::train(lrn.binrel,train_ip.task)
mod.chain=mlr::train(lrn.chain,train_ip.task)
mod.nest=mlr::train(lrn.nest,train_ip.task)
mod.dbr=mlr::train(lrn.dbr,train_ip.task)
mod.stack=mlr::train(lrn.stack,train_ip.task)

#Prediction on test subset
pred.binrel=predict(mod.binrel,newdata=test_ip)
pred.chain=predict(mod.chain,newdata=test_ip)
pred.nest=predict(mod.nest,newdata=test_ip)
pred.dbr=predict(mod.dbr,newdata=test_ip)
pred.stack=predict(mod.stack,newdata=test_ip)

# Performance analysis
measures=list(multilabel.acc,multilabel.f1,multilabel.hamloss,multilabel.subset01,multilabel.ppv,multilabel.tpr)
p2=performance(pred.binrel,measures)
p3=performance(pred.chain,measures)
p4=performance(pred.nest,measures)
p5=performance(pred.dbr,measures)
p6=performance(pred.stack,measures)
performance=as.data.frame(rbind(p2,p3,p4,p5,p6))

# creat visual plot for performance comparison
performance$model=c("Binaryrelevance","Chains","Nested","DBR","Stacking")
jpeg(file="prob_trans_comparison_dotplot.jpeg", height = 3000, width = 6500, res=300)
library(RColorBrewer)
library(tidyr)
plong=gather(performance,metrics,value,multilabel.acc:multilabel.tpr, factor_key=TRUE)
library(ggplot2)
mycolors=c("#db0229","#026bdb","#48039e","#0d7502","#c97c02","#c40c09")
ggplot(plong)+geom_point(aes(x=model,y=value,color=metrics),size=7,alpha=0.7)+facet_grid(metrics~.)+coord_flip()+theme_bw()+scale_color_manual(values=mycolors) + theme(text=element_text(size=25), axis.text.x = element_text(angle=65, hjust = 1))
dev.off()

# create actual value matrix
jpeg(file="prob_trans_comparison_matrix.jpeg", height = 3000, width = 6500, res=300)
ggplot(plong)+geom_tile(aes(x=model,y=metrics,fill=value),color="black")+geom_text(aes(x=model,y=metrics,label=round(value,3)),color="black", size=15)+scale_fill_distiller(palette = "Spectral")+theme(text=element_text(size=25), axis.text.x = element_text(angle=65, hjust = 1))
dev.off()

# based on above metrices with special focus of tpr and accuracy, 
# classifer chains was selected for further analysis  


###########################################################################
########### 5-fold cv comparison among different core learners #########

# 5-fold cross-validation with xgboost
acc_vec_xgboost = as.numeric()
for (j in c(1:5)) { 
    train = read.table(file=paste0("samp_train_cv_", j, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
    train_ip = train[,-1]
    test = read.table(file=paste0("samp_test_cv_", j, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
    test_ip = test[,-1]
    # processing of training data
    # convert data to multilabel accpted format (target variables into logical)
    for (i in c(1:110)) {
      train_ip[,i] <- as.numeric(train_ip[,i])
    }
    for (i in c(111:116)) {
      train_ip[,i] <- as.character(train_ip[,i])
      train_ip[,i][train_ip[,i] == "0"] <- "FALSE"
      train_ip[,i][train_ip[,i] == "1"] <- "TRUE"
    }
    # comsider only complete cases
    train_ip = train_ip[complete.cases(train_ip), ] # there was no missing values so missing value treatment not required
    # change class of target varaibles 
    train_ip[,111] = as.logical(train_ip[,111])
    train_ip[,112] = as.logical(train_ip[,112])
    train_ip[,113] = as.logical(train_ip[,113])
    train_ip[,114] = as.logical(train_ip[,114])
    train_ip[,115] = as.logical(train_ip[,115])
    train_ip[,116] = as.logical(train_ip[,116])
    # processing of test data
    # convert data to multilabel accpted format (target variables into logical)
    for (i in c(1:110)) {
      test_ip[,i] <- as.numeric(test_ip[,i])
    }
    for (i in c(111:116)) {
      test_ip[,i] <- as.character(test_ip[,i])
      test_ip[,i][test_ip[,i] == "0"] <- "FALSE"
      test_ip[,i][test_ip[,i] == "1"] <- "TRUE"
    }
    # comsider only complete cases
    test_ip = test_ip[complete.cases(test_ip), ] # there was no missing values so missing value treatment not required
    # change class of target varaibles 
    test_ip[,111] = as.logical(test_ip[,111])
    test_ip[,112] = as.logical(test_ip[,112])
    test_ip[,113] = as.logical(test_ip[,113])
    test_ip[,114] = as.logical(test_ip[,114])
    test_ip[,115] = as.logical(test_ip[,115])
    test_ip[,116] = as.logical(test_ip[,116])
    # load the libraries
    library(mlr)
    library(OpenML)
    # define varable with target variable names
    labels = c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
    # define task
    train_ip.task = makeMultilabelTask(id = "multi", data = train_ip, target = labels)
    # comparison of different methods available for multilabel classification
    #define Core learner here xgboost
    lrn.core= makeLearner("classif.xgboost", predict.type = "prob", fix.factors.prediction = TRUE)
    # as optimized construct classifier chains
    lrn.chain=makeMultilabelClassifierChainsWrapper(lrn.core)
    #Training 
    mod.chain=mlr::train(lrn.chain,train_ip.task)
    #Prediction on test subset
    pred.chain=predict(mod.chain,newdata=test_ip)
    # Performance evaluation
    measures=list(multilabel.acc)
    p3=performance(pred.chain,measures)
    acc_vec_xgboost = c(acc_vec_xgboost, p3)
}
print(acc_vec_xgboost)

# 5-fold cross-validation with C50
acc_vec_c50 = as.numeric()
for (j in c(1:5)) { 
  train = read.table(file=paste0("samp_train_cv_", j, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
  train_ip = train[,-1]
  test = read.table(file=paste0("samp_test_cv_", j, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
  test_ip = test[,-1]
  # processing of training data
  # convert data to multilabel accpted format (target variables into logical)
  for (i in c(1:110)) {
    train_ip[,i] <- as.numeric(train_ip[,i])
  }
  for (i in c(111:116)) {
    train_ip[,i] <- as.character(train_ip[,i])
    train_ip[,i][train_ip[,i] == "0"] <- "FALSE"
    train_ip[,i][train_ip[,i] == "1"] <- "TRUE"
  }
  # comsider only complete cases
  train_ip = train_ip[complete.cases(train_ip), ] # there was no missing values so missing value treatment not required
  # change class of target varaibles 
  train_ip[,111] = as.logical(train_ip[,111])
  train_ip[,112] = as.logical(train_ip[,112])
  train_ip[,113] = as.logical(train_ip[,113])
  train_ip[,114] = as.logical(train_ip[,114])
  train_ip[,115] = as.logical(train_ip[,115])
  train_ip[,116] = as.logical(train_ip[,116])
  # processing of test data
  # convert data to multilabel accpted format (target variables into logical)
  for (i in c(1:110)) {
    test_ip[,i] <- as.numeric(test_ip[,i])
  }
  for (i in c(111:116)) {
    test_ip[,i] <- as.character(test_ip[,i])
    test_ip[,i][test_ip[,i] == "0"] <- "FALSE"
    test_ip[,i][test_ip[,i] == "1"] <- "TRUE"
  }
  # comsider only complete cases
  test_ip = test_ip[complete.cases(test_ip), ] # there was no missing values so missing value treatment not required
  # change class of target varaibles 
  test_ip[,111] = as.logical(test_ip[,111])
  test_ip[,112] = as.logical(test_ip[,112])
  test_ip[,113] = as.logical(test_ip[,113])
  test_ip[,114] = as.logical(test_ip[,114])
  test_ip[,115] = as.logical(test_ip[,115])
  test_ip[,116] = as.logical(test_ip[,116])
  # load the libraries
  library(mlr)
  library(OpenML)
  # define varable with target variable names
  labels = c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
  # define task
  train_ip.task = makeMultilabelTask(id = "multi", data = train_ip, target = labels)
  # comparison of different methods available for multilabel classification
  #define Core learner here C50
  lrn.core= makeLearner("classif.C50", predict.type = "prob", fix.factors.prediction = TRUE)
  # as optimized construct classifier chains
  lrn.chain=makeMultilabelClassifierChainsWrapper(lrn.core)
  #Training 
  mod.chain=mlr::train(lrn.chain,train_ip.task)
  #Prediction on test subset
  pred.chain=predict(mod.chain,newdata=test_ip)
  # Performance evaluation
  measures=list(multilabel.acc)
  p3=performance(pred.chain,measures)
  acc_vec_c50 = c(acc_vec_c50, p3)
}
print(acc_vec_c50)


# 5-fold cross-validation with cforest
acc_vec_cforest = as.numeric()
for (j in c(1:5)) { 
  train = read.table(file=paste0("samp_train_cv_", j, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
  train_ip = train[,-1]
  test = read.table(file=paste0("samp_test_cv_", j, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
  test_ip = test[,-1]
  # processing of training data
  # convert data to multilabel accpted format (target variables into logical)
  for (i in c(1:110)) {
    train_ip[,i] <- as.numeric(train_ip[,i])
  }
  for (i in c(111:116)) {
    train_ip[,i] <- as.character(train_ip[,i])
    train_ip[,i][train_ip[,i] == "0"] <- "FALSE"
    train_ip[,i][train_ip[,i] == "1"] <- "TRUE"
  }
  # comsider only complete cases
  train_ip = train_ip[complete.cases(train_ip), ] # there was no missing values so missing value treatment not required
  # change class of target varaibles 
  train_ip[,111] = as.logical(train_ip[,111])
  train_ip[,112] = as.logical(train_ip[,112])
  train_ip[,113] = as.logical(train_ip[,113])
  train_ip[,114] = as.logical(train_ip[,114])
  train_ip[,115] = as.logical(train_ip[,115])
  train_ip[,116] = as.logical(train_ip[,116])
  # processing of test data
  # convert data to multilabel accpted format (target variables into logical)
  for (i in c(1:110)) {
    test_ip[,i] <- as.numeric(test_ip[,i])
  }
  for (i in c(111:116)) {
    test_ip[,i] <- as.character(test_ip[,i])
    test_ip[,i][test_ip[,i] == "0"] <- "FALSE"
    test_ip[,i][test_ip[,i] == "1"] <- "TRUE"
  }
  # comsider only complete cases
  test_ip = test_ip[complete.cases(test_ip), ] # there was no missing values so missing value treatment not required
  # change class of target varaibles 
  test_ip[,111] = as.logical(test_ip[,111])
  test_ip[,112] = as.logical(test_ip[,112])
  test_ip[,113] = as.logical(test_ip[,113])
  test_ip[,114] = as.logical(test_ip[,114])
  test_ip[,115] = as.logical(test_ip[,115])
  test_ip[,116] = as.logical(test_ip[,116])
  # load the libraries
  library(mlr)
  library(OpenML)
  # define varable with target variable names
  labels = c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
  # define task
  train_ip.task = makeMultilabelTask(id = "multi", data = train_ip, target = labels)
  # comparison of different methods available for multilabel classification
  #define Core learner here cforest
  lrn.core= makeLearner("classif.cforest", predict.type = "prob", fix.factors.prediction = TRUE)
  # as optimized construct classifier chains
  lrn.chain=makeMultilabelClassifierChainsWrapper(lrn.core)
  #Training
  mod.chain=mlr::train(lrn.chain,train_ip.task)
  #Prediction on test subset
  pred.chain=predict(mod.chain,newdata=test_ip)
  # Performance evaluation
  measures=list(multilabel.acc)
  p3=performance(pred.chain,measures)
  acc_vec_cforest = c(acc_vec_cforest, p3)
}
print(acc_vec_cforest)


# 5-fold cross-validation with neural network
acc_vec_nnet = as.numeric()
for (j in c(1:5)) { 
  train = read.table(file=paste0("samp_train_cv_", j, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
  train_ip = train[,-1]
  test = read.table(file=paste0("samp_test_cv_", j, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
  test_ip = test[,-1]
  # processing of training data
  # convert data to multilabel accpted format (target variables into logical)
  for (i in c(1:110)) {
    train_ip[,i] <- as.numeric(train_ip[,i])
  }
  for (i in c(111:116)) {
    train_ip[,i] <- as.character(train_ip[,i])
    train_ip[,i][train_ip[,i] == "0"] <- "FALSE"
    train_ip[,i][train_ip[,i] == "1"] <- "TRUE"
  }
  # comsider only complete cases
  train_ip = train_ip[complete.cases(train_ip), ] # there was no missing values so missing value treatment not required
  # change class of target varaibles 
  train_ip[,111] = as.logical(train_ip[,111])
  train_ip[,112] = as.logical(train_ip[,112])
  train_ip[,113] = as.logical(train_ip[,113])
  train_ip[,114] = as.logical(train_ip[,114])
  train_ip[,115] = as.logical(train_ip[,115])
  train_ip[,116] = as.logical(train_ip[,116])
  # processing of test data
  # convert data to multilabel accpted format (target variables into logical)
  for (i in c(1:110)) {
    test_ip[,i] <- as.numeric(test_ip[,i])
  }
  for (i in c(111:116)) {
    test_ip[,i] <- as.character(test_ip[,i])
    test_ip[,i][test_ip[,i] == "0"] <- "FALSE"
    test_ip[,i][test_ip[,i] == "1"] <- "TRUE"
  }
  # comsider only complete cases
  test_ip = test_ip[complete.cases(test_ip), ] # there was no missing values so missing value treatment not required
  # change class of target varaibles 
  test_ip[,111] = as.logical(test_ip[,111])
  test_ip[,112] = as.logical(test_ip[,112])
  test_ip[,113] = as.logical(test_ip[,113])
  test_ip[,114] = as.logical(test_ip[,114])
  test_ip[,115] = as.logical(test_ip[,115])
  test_ip[,116] = as.logical(test_ip[,116])
  # load the libraries
  library(mlr)
  library(OpenML)
  # define varable with target variable names
  labels = c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
  # define task
  train_ip.task = makeMultilabelTask(id = "multi", data = train_ip, target = labels)
  # comparison of different methods available for multilabel classification
  #define Core learner here nnet
  lrn.core= makeLearner("classif.nnet", predict.type = "prob", fix.factors.prediction = TRUE)
  # as optimized construct classifier chains
  lrn.chain=makeMultilabelClassifierChainsWrapper(lrn.core)
  #Training
  mod.chain=mlr::train(lrn.chain,train_ip.task)
  #Prediction on test subset
  pred.chain=predict(mod.chain,newdata=test_ip)
  # Performance evaluation
  measures=list(multilabel.acc)
  p3=performance(pred.chain,measures)
  acc_vec_nnet = c(acc_vec_nnet, p3)
}
print(acc_vec_nnet)


# 5-fold cross-validation with random forest
acc_vec_rf = as.numeric()
for (j in c(1:5)) { 
  train = read.table(file=paste0("samp_train_cv_", j, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
  train_ip = train[,-1]
  test = read.table(file=paste0("samp_test_cv_", j, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
  test_ip = test[,-1]
  # processing of training data
  # convert data to multilabel accpted format (target variables into logical)
  for (i in c(1:110)) {
    train_ip[,i] <- as.numeric(train_ip[,i])
  }
  for (i in c(111:116)) {
    train_ip[,i] <- as.character(train_ip[,i])
    train_ip[,i][train_ip[,i] == "0"] <- "FALSE"
    train_ip[,i][train_ip[,i] == "1"] <- "TRUE"
  }
  # comsider only complete cases
  train_ip = train_ip[complete.cases(train_ip), ] # there was no missing values so missing value treatment not required
  # change class of target varaibles 
  train_ip[,111] = as.logical(train_ip[,111])
  train_ip[,112] = as.logical(train_ip[,112])
  train_ip[,113] = as.logical(train_ip[,113])
  train_ip[,114] = as.logical(train_ip[,114])
  train_ip[,115] = as.logical(train_ip[,115])
  train_ip[,116] = as.logical(train_ip[,116])
  # processing of test data
  # convert data to multilabel accpted format (target variables into logical)
  for (i in c(1:110)) {
    test_ip[,i] <- as.numeric(test_ip[,i])
  }
  for (i in c(111:116)) {
    test_ip[,i] <- as.character(test_ip[,i])
    test_ip[,i][test_ip[,i] == "0"] <- "FALSE"
    test_ip[,i][test_ip[,i] == "1"] <- "TRUE"
  }
  # comsider only complete cases
  test_ip = test_ip[complete.cases(test_ip), ] # there was no missing values so missing value treatment not required
  # change class of target varaibles 
  test_ip[,111] = as.logical(test_ip[,111])
  test_ip[,112] = as.logical(test_ip[,112])
  test_ip[,113] = as.logical(test_ip[,113])
  test_ip[,114] = as.logical(test_ip[,114])
  test_ip[,115] = as.logical(test_ip[,115])
  test_ip[,116] = as.logical(test_ip[,116])
  # load the libraries
  library(mlr)
  library(OpenML)
  # define varable with target variable names
  labels = c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
  # define task
  train_ip.task = makeMultilabelTask(id = "multi", data = train_ip, target = labels)
  # comparison of different methods available for multilabel classification
  #define Core learner here rf
  lrn.core= makeLearner("classif.randomForest", predict.type = "prob", fix.factors.prediction = TRUE)
  # as optimized construct classifier chains
  lrn.chain=makeMultilabelClassifierChainsWrapper(lrn.core)
  #Training
  mod.chain=mlr::train(lrn.chain,train_ip.task)
  #Prediction on test subset
  pred.chain=predict(mod.chain,newdata=test_ip)
  # Performance evaluation
  measures=list(multilabel.acc)
  p3=performance(pred.chain,measures)
  acc_vec_rf = c(acc_vec_rf, p3)
}
print(acc_vec_rf)


# 5-fold cross-validation with rpart
acc_vec_rpart = as.numeric()
for (j in c(1:5)) { 
  train = read.table(file=paste0("samp_train_cv_", j, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
  train_ip = train[,-1]
  test = read.table(file=paste0("samp_test_cv_", j, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
  test_ip = test[,-1]
  # processing of training data
  # convert data to multilabel accpted format (target variables into logical)
  for (i in c(1:110)) {
    train_ip[,i] <- as.numeric(train_ip[,i])
  }
  for (i in c(111:116)) {
    train_ip[,i] <- as.character(train_ip[,i])
    train_ip[,i][train_ip[,i] == "0"] <- "FALSE"
    train_ip[,i][train_ip[,i] == "1"] <- "TRUE"
  }
  # comsider only complete cases
  train_ip = train_ip[complete.cases(train_ip), ] # there was no missing values so missing value treatment not required
  # change class of target varaibles 
  train_ip[,111] = as.logical(train_ip[,111])
  train_ip[,112] = as.logical(train_ip[,112])
  train_ip[,113] = as.logical(train_ip[,113])
  train_ip[,114] = as.logical(train_ip[,114])
  train_ip[,115] = as.logical(train_ip[,115])
  train_ip[,116] = as.logical(train_ip[,116])
  # processing of test data
  # convert data to multilabel accpted format (target variables into logical)
  for (i in c(1:110)) {
    test_ip[,i] <- as.numeric(test_ip[,i])
  }
  for (i in c(111:116)) {
    test_ip[,i] <- as.character(test_ip[,i])
    test_ip[,i][test_ip[,i] == "0"] <- "FALSE"
    test_ip[,i][test_ip[,i] == "1"] <- "TRUE"
  }
  # comsider only complete cases
  test_ip = test_ip[complete.cases(test_ip), ] # there was no missing values so missing value treatment not required
  # change class of target varaibles 
  test_ip[,111] = as.logical(test_ip[,111])
  test_ip[,112] = as.logical(test_ip[,112])
  test_ip[,113] = as.logical(test_ip[,113])
  test_ip[,114] = as.logical(test_ip[,114])
  test_ip[,115] = as.logical(test_ip[,115])
  test_ip[,116] = as.logical(test_ip[,116])
  # load the libraries
  library(mlr)
  library(OpenML)
  # define varable with target variable names
  labels = c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
  # define task
  train_ip.task = makeMultilabelTask(id = "multi", data = train_ip, target = labels)
  # comparison of different methods available for multilabel classification
  #define Core learner here rpart
  lrn.core= makeLearner("classif.rpart", predict.type = "prob", fix.factors.prediction = TRUE)
  # as optimized construct classifier chains
  lrn.chain=makeMultilabelClassifierChainsWrapper(lrn.core)
  #Training
  mod.chain=mlr::train(lrn.chain,train_ip.task)
  #Prediction on test subset
  pred.chain=predict(mod.chain,newdata=test_ip)
  # Performance evaluation
  measures=list(multilabel.acc)
  p3=performance(pred.chain,measures)
  acc_vec_rpart = c(acc_vec_rpart, p3)
}
print(acc_vec_rpart)


# 5-fold cross-validation with ksvm
acc_vec_ksvm = as.numeric()
for (j in c(1:5)) { 
  train = read.table(file=paste0("samp_train_cv_", j, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
  train_ip = train[,-1]
  test = read.table(file=paste0("samp_test_cv_", j, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
  test_ip = test[,-1]
  # processing of training data
  # convert data to multilabel accpted format (target variables into logical)
  for (i in c(1:110)) {
    train_ip[,i] <- as.numeric(train_ip[,i])
  }
  for (i in c(111:116)) {
    train_ip[,i] <- as.character(train_ip[,i])
    train_ip[,i][train_ip[,i] == "0"] <- "FALSE"
    train_ip[,i][train_ip[,i] == "1"] <- "TRUE"
  }
  # comsider only complete cases
  train_ip = train_ip[complete.cases(train_ip), ] # there was no missing values so missing value treatment not required
  # change class of target varaibles 
  train_ip[,111] = as.logical(train_ip[,111])
  train_ip[,112] = as.logical(train_ip[,112])
  train_ip[,113] = as.logical(train_ip[,113])
  train_ip[,114] = as.logical(train_ip[,114])
  train_ip[,115] = as.logical(train_ip[,115])
  train_ip[,116] = as.logical(train_ip[,116])
  # processing of test data
  # convert data to multilabel accpted format (target variables into logical)
  for (i in c(1:110)) {
    test_ip[,i] <- as.numeric(test_ip[,i])
  }
  for (i in c(111:116)) {
    test_ip[,i] <- as.character(test_ip[,i])
    test_ip[,i][test_ip[,i] == "0"] <- "FALSE"
    test_ip[,i][test_ip[,i] == "1"] <- "TRUE"
  }
  # comsider only complete cases
  test_ip = test_ip[complete.cases(test_ip), ] # there was no missing values so missing value treatment not required
  # change class of target varaibles 
  test_ip[,111] = as.logical(test_ip[,111])
  test_ip[,112] = as.logical(test_ip[,112])
  test_ip[,113] = as.logical(test_ip[,113])
  test_ip[,114] = as.logical(test_ip[,114])
  test_ip[,115] = as.logical(test_ip[,115])
  test_ip[,116] = as.logical(test_ip[,116])
  # load the libraries
  library(mlr)
  library(OpenML)
  # define varable with target variable names
  labels = c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
  # define task
  train_ip.task = makeMultilabelTask(id = "multi", data = train_ip, target = labels)
  # comparison of different methods available for multilabel classification
  #define Core learner here ksvm
  lrn.core= makeLearner("classif.ksvm", predict.type = "prob", fix.factors.prediction = TRUE)
  # as optimized construct classifier chains
  lrn.chain=makeMultilabelClassifierChainsWrapper(lrn.core)
  #Training
  mod.chain=mlr::train(lrn.chain,train_ip.task)
  #Prediction on test subset
  pred.chain=predict(mod.chain,newdata=test_ip)
  # Performance evaluation
  measures=list(multilabel.acc)
  p3=performance(pred.chain,measures)
  acc_vec_ksvm = c(acc_vec_ksvm, p3)
}
print(acc_vec_ksvm)

algo_comp = data.frame(cbind(acc_vec_xgboost, acc_vec_c50, acc_vec_cforest, acc_vec_nnet, acc_vec_rf, acc_vec_rpart, acc_vec_ksvm))
row.names(algo_comp) = c("cv1", "cv2", "cv3", "cv4", "cv5")
colnames(algo_comp) = c("xgBoost","C50","rf_conditional","Neural_Network","Random_Forest","Decision_Tree","SVM")
algo_comp_stack = data.frame(stack(algo_comp[,1:7]))
colnames(algo_comp_stack) = c("value", "algorithm")
write.table(algo_comp, file="algo_comp.csv", row.names=T, col.names=T, sep="\t", quote=F)

# create figure of algorithm selection
jpeg(file="algo_comp.jpeg", height = 3000, width = 6500, res=300)
par(mgp=c(10,1,0))
ggplot(data=algo_comp_stack, aes_string(x = algo_comp_stack[,2], y=algo_comp_stack[,1])) +
  geom_boxplot(color="red", outlier.size=5) +
  theme_bw() + 
  xlab("Algorithms") + ylab("Multilabel_Accuracy") +
  scale_y_continuous() +
  ggtitle("") + 
  theme(text=element_text(size=30), axis.text.x = element_text(angle=65, hjust = 1))
dev.off()

# based on the comparison xgboost outperfomed all the other algorithms
# in terms of accuracy with minimum variance in the performace


###########################################################################
############### algorithm tuning to obatin best performance ##########

# 5-fold cross-validation with random forest
acc_tun_xgboost = as.numeric()
for (i in c(25, 100, 150)) {
  for (j in c(10,15,25)) {
    for (k in c(1:5)) { 
      train = read.table(file=paste0("samp_train_cv_", k, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
      train_ip = train[,-1]
      test = read.table(file=paste0("samp_test_cv_", k, ".csv"), header=T, sep="\t", quote="", na.strings = "NA", fill = T, comment.char = "")
      test_ip = test[,-1]
      # processing of training data
      # convert data to multilabel accpted format (target variables into logical)
      for (i in c(1:110)) {
        train_ip[,i] <- as.numeric(train_ip[,i])
      }
      for (i in c(111:116)) {
        train_ip[,i] <- as.character(train_ip[,i])
        train_ip[,i][train_ip[,i] == "0"] <- "FALSE"
        train_ip[,i][train_ip[,i] == "1"] <- "TRUE"
      }
      # comsider only complete cases
      train_ip = train_ip[complete.cases(train_ip), ] # there was no missing values so missing value treatment not required
      # change class of target varaibles 
      train_ip[,111] = as.logical(train_ip[,111])
      train_ip[,112] = as.logical(train_ip[,112])
      train_ip[,113] = as.logical(train_ip[,113])
      train_ip[,114] = as.logical(train_ip[,114])
      train_ip[,115] = as.logical(train_ip[,115])
      train_ip[,116] = as.logical(train_ip[,116])
      # processing of test data
      # convert data to multilabel accpted format (target variables into logical)
      for (i in c(1:110)) {
        test_ip[,i] <- as.numeric(test_ip[,i])
      }
      for (i in c(111:116)) {
        test_ip[,i] <- as.character(test_ip[,i])
        test_ip[,i][test_ip[,i] == "0"] <- "FALSE"
        test_ip[,i][test_ip[,i] == "1"] <- "TRUE"
      }
      # comsider only complete cases
      test_ip = test_ip[complete.cases(test_ip), ] # there was no missing values so missing value treatment not required
      # change class of target varaibles 
      test_ip[,111] = as.logical(test_ip[,111])
      test_ip[,112] = as.logical(test_ip[,112])
      test_ip[,113] = as.logical(test_ip[,113])
      test_ip[,114] = as.logical(test_ip[,114])
      test_ip[,115] = as.logical(test_ip[,115])
      test_ip[,116] = as.logical(test_ip[,116])
      # load the libraries
      library(mlr)
      library(OpenML)
      # define varable with target variable names
      labels = c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
      # define task
      train_ip.task = makeMultilabelTask(id = "multi", data = train_ip, target = labels)
      # comparison of different methods available for multilabel classification
      #define Core learner here xgboost
      lrn.core= makeLearner("classif.xgboost", predict.type = "prob", fix.factors.prediction = TRUE)
      lrn.core$par.vals <- list(nrounds = i, max_depth=j)
      # as optimized construct classifier chains
      lrn.chain=makeMultilabelClassifierChainsWrapper(lrn.core, )
      #Training
      mod.chain=mlr::train(lrn.chain,train_ip.task)
      #Prediction on test subset
      pred.chain=predict(mod.chain,newdata=test_ip)
      # Performance evaluation
      measures=list(multilabel.acc)
      p3=performance(pred.chain,measures)
      acc_tun_xgboost = c(acc_tun_xgboost, p3)
    }
  }
}
print(acc_tun_xgboost)
name_vec = c(rep("10_25", 5), rep("15_25", 5), rep("25_25", 5), rep("10_100", 5), rep("15_100", 5), rep("25_100", 5), rep("10_150", 5), rep("15_150", 5), rep("25_150", 5))
name_vec = as.character(name_vec)
acc_tun_xgboost = as.numeric(as.character(acc_tun_xgboost))
tune_comp = data.frame(value = acc_tun_xgboost, Parameter = name_vec)
#write.table(tune_comp, file="tune_comp.csv", row.names=T, col.names=T, sep="\t", quote=F)
# create figure of algorithm selection
jpeg(file="tune_comp.jpeg", height = 3000, width = 6500, res=300)
par(mgp=c(10,1,0))
ggplot(data=tune_comp, aes_string(x = tune_comp$Parameter, y=tune_comp$value)) +
  geom_boxplot(color="blue", outlier.size=5) +
  theme_bw() + 
  xlab("Parameters") + ylab("Multilabel_Accuracy") +
  ggtitle("") + 
  theme(text=element_text(size=30), axis.text.x = element_text(angle=65, hjust = 1))
dev.off()
# from the comparison it is apparent that max_depth value of 25 and nrounds of 150 gave the best results
# in terms of least variance in the performance on 5-fold cross validation
# therefore max_depth of 25 and nrounds of 150 were selected for further analysis

##################################################################
################## Final model construction #####################

# final model will be constructed using the most optimum algorithm 
# with best selected parameters
# the complete training dataset (60% of original data) will be used for training the final model
train_for_modeling = read.table(file="train_for_modeling.csv", header=T, stringsAsFactors = F, sep="\t", na.strings = c("NA", ""), fill = T)
dim(train_for_modeling)
names(train_for_modeling)
train = train_for_modeling[,-1]
# processing of training data
# convert data to multilabel accpted format (target variables into logical)
for (i in c(1:110)) {
  train[,i] <- as.numeric(train[,i])
}
for (i in c(111:116)) {
  train[,i] <- as.character(train[,i])
  train[,i][train[,i] == "0"] <- "FALSE"
  train[,i][train[,i] == "1"] <- "TRUE"
}
# comsider only complete cases
train = train[complete.cases(train), ] # there was no missing values so missing value treatment not required
# change class of target varaibles 
train[,111] = as.logical(train[,111])
train[,112] = as.logical(train[,112])
train[,113] = as.logical(train[,113])
train[,114] = as.logical(train[,114])
train[,115] = as.logical(train[,115])
train[,116] = as.logical(train[,116])
# load the libraries
library(mlr)
library(OpenML)
# define varable with target variable names
labels = c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
# define task
train.task = makeMultilabelTask(id = "multi", data = train, target = labels)
# comparison of different methods available for multilabel classification
#define Core learner here xgboost
lrn.core= makeLearner("classif.xgboost", predict.type = "prob", fix.factors.prediction = TRUE)
lrn.core$par.vals <- list(nrounds = 150, max_depth=25)
# as optimized construct classifier chains
lrn.chain=makeMultilabelClassifierChainsWrapper(lrn.core)
#Training the final model
Final_mod_chain=mlr::train(lrn.chain,train.task)
# save the final model for future use to make predictions
save(Final_mod_chain, file="Final_model.RData")
