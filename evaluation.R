# Set the path
setwd("F:/edwisors/assignments/project-2/all/evaluation")
getwd()
rm(list=ls())


############### evaluation on the blind-set of 40% data #################
###########################################################################

# This blind-set the algorithm has never seen and 
# since we have real target values for this data we can truely evaluate
# our model. This step is needed as the competition is already over
# and now new ranking can not be assigned

# construct new data for training based on selected variables from each toxic type
test = read.table(file="test_63828.csv", header=T, stringsAsFactors = F, sep="\t", na.strings = c("NA", ""), fill = T)
dim(test)
names(test)
id = test[,1]
target = test[,3:8]
varsel = read.table(file="final_selected_variables.csv", header=F, stringsAsFactors = F, sep="\t", na.strings = c("NA", ""), fill = T)

# construct comment vector to generate the training data
require("stringi")
require("stringr")
require("tm")
require("wordcloud")
require("slam")
require("RSentiment")
# create corpus so that text preprocessing methods can be applied
test[,2] = gsub("[^a-zA-Z0-9 ]", "", test[,2])
postCorpus = Corpus(VectorSource(test[,2]))
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

fea = data.frame(id = test[,1])
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

blindset_for_prediction = cbind(id, predictors, target)
# save the blindset data for future reference
write.table(blindset_for_prediction, file="blindset_for_prediction.csv", row.names=F, col.names=T, sep="\t", quote=F)
# these calculated variables will be used for the 

# preprocessing of test data
test = blindset_for_prediction[,-1]
for (i in c(1:110)) {
  test[,i] <- as.numeric(test[,i])
}
for (i in c(111:116)) {
  test[,i] <- as.character(test[,i])
  test[,i][test[,i] == "0"] <- "FALSE"
  test[,i][test[,i] == "1"] <- "TRUE"
}
# comsider only complete cases
test = test[complete.cases(test), ] # there was no missing values so missing value treatment not required
# change class of target varaibles 
test[,111] = as.logical(test[,111])
test[,112] = as.logical(test[,112])
test[,113] = as.logical(test[,113])
test[,114] = as.logical(test[,114])
test[,115] = as.logical(test[,115])
test[,116] = as.logical(test[,116])

# predictions with the pre-trained model
require(mlr)
xgbTree = get(load("Final_model.RData"))
prediction_blindset = predict(xgbTree,newdata=test)
write.table(prediction_blindset, file="predictions_for_blindset_tab.csv", row.names=F, col.names=T, sep="\t", quote=F)
# Performance analysis combined for multilabel
measures=list(multilabel.acc,multilabel.f1,multilabel.hamloss,multilabel.subset01,multilabel.ppv,multilabel.tpr)
performance=performance(prediction,measures)
require(tidyr)

# binary performance analysis for each type of toxicity
bin_performance=getMultilabelBinaryPerformances(prediction_blindset,list(bac,auc,mmce,fnr,fpr))%>%as.data.frame()
bin_performance$Labels=c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
bin_performance=plyr::rename(bin_performance, c("bac.test.mean"="BAC","auc.test.mean"="AUC","mmce.test.mean"="MMCE","fnr.test.mean"="FNR","fpr.test.mean"="FPR"))
pblong=gather(bin_performance,metrics,value,BAC:FPR, factor_key=TRUE)
require(ggplot2)
jpeg(file="binary_performace_matrix.jpeg", height = 3000, width = 6500, res=300)
ggplot(pblong)+geom_tile(aes(x=metrics,y=Labels,fill=value),color="black")+scale_fill_distiller(palette = "Spectral")+geom_text(aes(x=metrics,y=Labels,label=round(value,3)),color="black",size=12)+ggtitle("")+theme(text=element_text(size=30), axis.text.x = element_text(angle=65, hjust = 1))
dev.off()
jpeg(file="binary_performace_scatterplot.jpeg", height = 3000, width = 6500, res=300)
ggplot(pblong)+geom_point(aes(x=Labels,y=value,color=Labels),size=7)+scale_y_continuous("Value",breaks=c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))+coord_flip()+facet_grid(metrics~.)+theme_bw()+ggtitle("")+theme(text=element_text(size=30), axis.text.x = element_text(angle=65, hjust = 1))
dev.off()

# As mentioned on the kaggle competition website the evaluation parameter 
# was mean coulumn-wise AUC of ROC curve
# which is mean of AUC for each toxic class
# here we obtained the mean AUC ROC of 89.58 

# the coulmn-wise AUC of ROC  is mentioned in the figure
# attached in the report (please see Figure-11&12 of the report)
# In this result only 60% of given training data with 110 terms was used 
# for the evaluation purpose 
# as the competition is over and no more rankings are provided

# however when this same model was run on the complete training data 
# which is the actual kaggle training file with 1000 terms 
# (this was run on a cluster as my laptop is lower end)
# the mean AUC ROC obtained was 98.60 
# according to the already assigned ranking I would have obtained 
# a rank of around 1000 
# This is not so great but 
# given some more time I can improve upon the results


####################### predictions on kaggle test-set #################
###########################################################################

# this is the actual test set provided on kaggle
# based on performace on this test set the ranking was assigned
# the file for upload to kaggle is constructed in this section
# as competition is already over the file is provided with
# this project for your kind reference and evaluation

# construct new data for training based on selected variables from each toxic type
rm(list=ls())
test = read.table(file="kaggle_test.csv", header=T, sep=",", na.strings = "", fill = T)
dim(test)
names(test)
id = test[,1]
varsel = read.table(file="final_selected_variables.csv", header=F, stringsAsFactors = F, sep="\t", na.strings = c("NA", ""), fill = T)

# construct comment vector to generate the training data
require("stringi")
require("stringr")
require("tm")
require("wordcloud")
require("slam")
require("RSentiment")
# create corpus so that text preprocessing methods can be applied
# remove all special chracters
test[,2] = gsub("[^a-zA-Z0-9 ]", "", test[,2])
postCorpus = Corpus(VectorSource(test[,2]))
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
# check dimensions if all okay and match the initial values
dim(comment)
dim(test) # all okay so can move ahead

fea = data.frame(id = test[,1])
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

kaggle_test_formatted = cbind(id, predictors)
write.table(kaggle_test_formatted, file="kaggle_test_formatted.csv", row.names=F, col.names=T, sep="\t", quote=F)

# preprocessing of kaggle test data
kaggle_test = kaggle_test_formatted[,-1]
for (i in c(1:110)) {
  kaggle_test[,i] <- as.numeric(kaggle_test[,i])
}
# comsider only complete cases
kaggle_test = kaggle_test[complete.cases(kaggle_test), ] # there was no missing values so missing value treatment not required

# predictions on kaggle test data
require(mlr)
xgbTree = get(load("Final_model.RData"))
prediction_kaggle_test = data.frame(predict(xgbTree,newdata=kaggle_test))
kaggle_final_upload = cbind(test_id = id, prediction_kaggle_test)

write.table(kaggle_final_upload, file="kaggle_test_results_upload.csv", row.names=F, col.names=T, sep="\t", quote=F)

# this file=kaggle_test_results_upload.csv needs to be uploaded on kaggle
# to obtain the kaggle ranking in this competition 
# However, the competition is closed now so this file is uploaded
# with this submission for your kind reference and evaluation

