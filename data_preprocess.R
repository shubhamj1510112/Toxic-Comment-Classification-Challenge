# Set the path
setwd("F:/edwisors/assignments/project-2/all/variable_encoding")
getwd()
rm(list=ls())

################### reading and exploring raw data #######################
###########################################################################

# to view few lines of training file to understand the data 
cat(readLines("train.csv", n=10), sep = "\n")
# read training file
t = read.table(file="train.csv", header=T, sep=",", na.strings = "", fill = T)
# remove special characters from the data
t = as.data.frame(lapply(t, function(x) gsub("[^a-zA-Z0-9 ]", "", x)))
# save the new formatted training data file
write.table(t, file="train_new.csv", row.names=F, col.names=T, sep=",", quote=F) 
# nearly equal file size of original train.csv and train_new.csv shows that
# all the observations were read properly
# in total approximatly 160K observations
# check dimension
dim(t)
# count of different variable types in training data
table(sapply(1:ncol(t), function(x) class(t[,x])))
t$comment_text = as.character(t$comment_text)
t$id = as.character(t$id)
# check column names
colnames(t)

######################### identify missing values #############################
###############################################################################

# count missing values in each column
missing_val = data.frame(apply(t, 2, function(x) {sum(is.na(x))}))
# none of the columns had any missing value 
# so missing value treatment was not required  

################### divide into train and test #############################
###############################################################################

# divide data into train and test dataset
# using random sampling without replacemnt method
# we use 60-40 ratio; 60% for training and 40% for testing
set.seed(1234)
train = t[sample(nrow(t),  95743, replace=F),] # 95,743 is 60% of total observations
# here this test dataset can be considered as blind or real dataset
# as this data will never be seen by the algorithm at the time of training
# not even at the cross validation stage
test  = t[!(1:nrow(t)) %in% as.numeric(row.names(train)), ]
# save this train and test data for future reference
write.table(train, file="train_95743.csv", row.names=F, col.names=T, sep="\t", quote=F)
write.table(test, file="test_63828.csv", row.names=F, col.names=T, sep="\t", quote=F)


######## derive the term variables from the comment_text variable ############
##############################################################################

# for this text processing tools will be used
# comment_text variable has a lot of text 
# so we will identify most useful terms 
# which have specific correlations with the specific types of toxicity
# later frequency of these useful terms will be used as variables 
# for model training and predictions

# load libraries ; try with R version 3.4.4 or above
require("stringi")
require("stringr")
require("tm")
require("wordcloud")
require("slam")
require("RSentiment")
# create corpus so that text preprocessing methods can be applied
postCorpus = Corpus(VectorSource(train$comment_text))
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
# create term document matrix to select useful terms
# here term frequency will be used as metric to find most useful word
tdm = TermDocumentMatrix(postCorpus, control=list(bounds=list(global=c(250, 10000)), minWordLength=2, weighting=weightTf))
dim(tdm)
terms = as.character(rownames(tdm))
# terms lemmatization
require(textstem)
terms = lemmatize_words(terms)
# convert to string vector again
comment = data.frame(text=sapply(postCorpus, identity))
# comments lemmatization
require(textstem)
comment = lemmatize_words(comment)
# save the comments and terms variables for furture reference
write.table(comment, file="processed_comments_95743.csv", row.names=F, col.names=F, sep="\t", quote=F)
write.table(terms, file="all_terms_1433.csv", row.names=F, col.names=F, sep="\t", quote=F)
# A total of 1,433 useful terms were extracted
# we need to calculate the frequecy of these terms in each comment
# so that these terms frquency can be used as predictor variables

#### construct new dataset with terms frequency as target variable ##################
##############################################################################

# use below code for creating dichotomous dummy variaables for selected terms
# but running this code will take more than one day so it was run on a 
# multicore cluster and results are stored in a file whiich is directly loaded here
############ run on a cluster #################
# this code can be tested on laptop with fewer iterations
# for example here shown for only 5 iterations
fea = data.frame(id = train[,1])
for (i in 1:5) {
  vec = as.integer()
  for (j in c(1:length(comment$text))) {
    c = as.integer(str_count(as.character(comment$text[j]), pattern=as.character(terms[i])))
    vec = c(vec, c)
  } 
  fea = cbind(fea, vec)
  print(paste0("iteration:", i))
}
dim(fea)
rm(c,vec,fea)
#################################################
# load the file made from cluster
var1 = read.table(file="comment_coded.csv", header=F, stringsAsFactors = F, sep="\t", na.strings = c("NA", ""), fill = T)
var1= var1[,2:1434]
dim(var1)
# check and count variable types
table(sapply(1:ncol(var1), function(x) class(var1[,x]))) # all of them were integer so fine
# redefine the column names for easy identification
colnames(var1) = paste("var", 1:1433, sep="")
# construct training data with selected terms frquency as the variables
# create six datasets one for each type of toxicity
# this is done for individual features selection process

train_all_tox = cbind(id = train[,1], var1, toxic = train[,3], severe_toxic = train[,4],
obscene = train[,5], threat = train[,6], insult = train[,7], identity_hate = train[,8]) 

# check variable types
table(sapply(1:ncol(train_all_tox), function(x) class(train_all_tox[,x])))

# remove observations with missing price value as they can not be used for training
train_all_tox = train_all_tox[complete.cases(train_all_tox), ]

# out of 95,743 observations only 95,710 were complete and hence taken forward



#####################################################################################
################### remove near-zero / useless predictors ############################

# load library for parallel processing
require(foreach)
# identify useless or not varying variables
require(caret)
nzv <- nearZeroVar(train_all_tox, freqCut=97/3, uniqueCut=5, allowParallel=T)

# check nzv and modify so that id and target variables are kept safely
remove = c(1,1435, 1436, 1437, 1438, 1439, 1440)
nzv = nzv[! nzv %in% remove]
# remove the useless variables from the training dataset
train_all_tox_nzv <- train_all_tox[, -nzv]
# cleaning
rm(nzv, remove)


#####################################################################################
################### remove multicollinearity ########################################

# in this step we will filter the highly correlated variables
# calculate the correlation
descrCor <- cor(train_all_tox_nzv[,c(2:207)])
# replace missing correlation values with zero
# where correlation could not be calculated due to zero variance
descrCor[is.na(descrCor)] = 0
highlyCorDescr <- findCorrelation(descrCor, cutoff = .85)
# we got two predictor var14 (position=8+1) and var113 highly correlated 
# >0.85 cut-off for correlation coefficient 
# please note change in position after the previous step of remove useless
# for example var14 has the 9th palce in the data
# so we remove one "var14" of them at this stage

train_all_tox_nzv_cor <- train_all_tox_nzv[,-c(9)]

# check for succesful removal

names(train_all_tox_nzv_cor)

dim(train_all_tox_nzv_cor)

# cleaning

rm(descrCor, highlyCorDescr)

# remove linear dependencies as this is also a special type of multicollinearity
# in this step we will filter the linear combinations among the predictor variables
comboInfo <- findLinearCombos(train_all_tox_nzv_cor[,2:206])
# list of variables where one is linear combination of other
comboInfo$remove
# we see that column var412 (position 134+1) is linear combination of other preidctors
# please note change in position after the previous step of remove useless
# for example var412 has the 135th palce in the data  
# so we removed this predictor variable at this stage
train_all_tox_nzv_cor_lc <- train_all_tox_nzv_cor[, -c(135)]

# write the preprocessed file for future reference
#write.table(dataset_new_nzv_cor_lc, file="dataset_new_nzv_cor_lc.csv", row.names=F, col.names=T, sep="\t", quote=F)


#####################################################################################
####################### Outlier Treatment ##########################################

# function to detect outliers
outdet = function(data) {
  outvec = numeric()
  for (i in 1:ncol(data)) {
    var = as.numeric(data[,i])
    cutoff_up = quantile(var, 0.75) + 3*(IQR(var))
    cutoff_low = quantile(var, 0.25) - 3*(IQR(var))
    l_up = length(var[var>cutoff_up])
    l_low = length(var[var<cutoff_low])
    if(l_up > 0 || l_low > 0) {
      colna = colnames(data)
      colname = colna[i]
      print(paste0("Iteration:", i, "There are outliers in: ", colname))
      outvec = c(outvec, i)
    }
    else {
      print (paste0("Iteration:", i, "There are no outliers"))
    }
  }
  return(outvec)
}
outvec1 = outdet(train_all_tox_nzv_cor_lc[,c(2:205)])    
# due to spurious comments all the predictor variables had outliers so we need to treat them

# for treatment the outlier values are replaced with outer fence boundary values 
# example upper fence outlier with upper fence boundary values
# separate the predictor and target variables

train_all_tox_nzv_cor_lc_id = train_all_tox_nzv_cor_lc[,c(1)]
train_all_tox_nzv_cor_lc_target = train_all_tox_nzv_cor_lc[,c(206:211)]
train_all_tox_nzv_cor_lc_predictors = train_all_tox_nzv_cor_lc[,2:205]

# function for outlier treatment
outtre = function(data, i) {
    colname_out = numeric()
    mx = quantile(data[,i], 0.75) + 3*(IQR(data[,i]))
    mn = quantile(data[,i], 0.25) - 3*(IQR(data[,i]))
    for (j in 1:length(data[,i])) {
      if ( data[,i][j] < mn) {
        colname_out = c(colname_out, mn)
      } else if ( data[,i][j] > mx ) {
        colname_out = c(colname_out, mx)
      } else {
        colname_out = c(colname_out, data[,i][j])
      }
    }
    data[,i] = colname_out
    return(data)		
}
# run function over the predictor variables with outliers
# be patient this step may take upto four hours
for (i in c(1:length(outvec1))) {
train_all_tox_nzv_cor_lc_predictors = outtre(train_all_tox_nzv_cor_lc_predictors, outvec1[i])
print (paste0("iteration:", i))
}

# combine the predictor and target variables again
train_all_tox_nzv_cor_lc_out = cbind(id = train_all_tox_nzv_cor_lc_id, train_all_tox_nzv_cor_lc_predictors, train_all_tox_nzv_cor_lc_target)

# save the file future reference
write.table(train_all_tox_nzv_cor_lc_out, file="train_all_tox_nzv_cor_lc_out.csv", row.names=F, col.names=T, sep="\t", quote=F)

#################################################################################
####### variable importance specific to different types of  toxicities ########## 

# we will perform individual variable importance calculation separatly for each toxicity type

# this step will also take some time upto half an hour
# also please make sure that you have 10 threads in your systme to define cluster variable
# else redefine this number

# variable importance calculation for "toxic" with random forest

# create separate dataset 

train_tox1 = cbind(id = train_all_tox_nzv_cor_lc_out[,1], train_all_tox_nzv_cor_lc_out[,2:205], toxic = train_all_tox_nzv_cor_lc_out[,206]) 
library(parallel)
library(doParallel)
require(caret)
cluster=makeCluster(10)
registerDoParallel(cluster)
fitControl = trainControl(allowParallel=TRUE, method="none")

mtry = floor(sqrt((ncol(train_tox1)-2)))

tunegrid = expand.grid(.mtry=mtry)

metric="Accuracy"
rf1 = train(toxic ~ ., method="rf", ntree=100, data=train_tox1[,-1], importance=T, trControl = fitControl, tuneGrid=tunegrid, metric=metric)
stopCluster(cluster)

# save created model for future reference
save(rf1, file="rf_varimp_tox1.RData")
# load the saved rf model 
forest1 = get(load("rf_varimp_tox1.RData"))
# create table of variable importance
require(caret)
var_imp1 = data.frame(varImp(forest1)$importance)
var_imp1$variable = row.names(var_imp1)
# variable sorted based on Gini Index
var_imp1 = data.frame(var_imp1[order(-var_imp1$X1),]) 
var_imp1 = var_imp1[,c(3,2)] 
# cleaning
rm(rf1, forest1)
# create figure of variable importance
jpeg(file="var_imp1.jpeg", height = 3000, width = 6500, res=300)
par(mgp=c(10,1,0))

var_imp1 = var_imp1[1:20,]
ggplot(var_imp1, aes_string(x = reorder(as.factor(var_imp1[,1]), -var_imp1[,2]), y=var_imp1[,2], group=1)) +
  geom_line(color="red", size=2) + geom_point(size=4) +
  theme_bw() + 
  xlab("Variable Name") + ylab("Overall Importance") +
  scale_y_continuous() +
  ggtitle("") + 
  theme(text=element_text(size=30), axis.text.x = element_text(angle=65, hjust = 1))
dev.off()



#variable importance calculation for "severe_toxic" with random forest

# create separate dataset 

train_tox2 = cbind(id = train_all_tox_nzv_cor_lc_out[,1], train_all_tox_nzv_cor_lc_out[,2:205], severe_toxic = train_all_tox_nzv_cor_lc_out[,207]) 
library(parallel)
library(doParallel)
require(caret)
cluster=makeCluster(10)
registerDoParallel(cluster)
fitControl = trainControl(allowParallel=TRUE, method="none")

mtry = floor(sqrt((ncol(train_tox2)-2)))

tunegrid = expand.grid(.mtry=mtry)

metric="Accuracy"
rf2 = train(severe_toxic ~ ., method="rf", ntree=100, data=train_tox2[,-1], importance=T, trControl = fitControl, tuneGrid=tunegrid, metric=metric)
stopCluster(cluster)

# save created model for future reference
save(rf2, file="rf_varimp_tox2.RData")
# load the saved rf model 
forest2 = get(load("rf_varimp_tox2.RData"))
# create table of variable importance
require(caret)
var_imp2 = data.frame(varImp(forest2)$importance)
var_imp2$variable = row.names(var_imp2)
# variable sorted based on Gini Index
var_imp2 = data.frame(var_imp2[order(-var_imp2$X1),]) 
var_imp2 = var_imp2[,c(3,2)] 

# cleaning

rm(rf2, forest2)
# create figure of variable importance
jpeg(file="var_imp2.jpeg", height = 3000, width = 6500, res=300)
par(mgp=c(10,1,0))

var_imp2 = var_imp2[1:20,]
ggplot(var_imp2, aes_string(x = reorder(as.factor(var_imp2[,1]), -var_imp2[,2]), y=var_imp2[,2], group=1)) +
  geom_line(color="red", size=2) + geom_point(size=4) +
  theme_bw() + 
  xlab("Variable Name") + ylab("Overall Importance") +
  scale_y_continuous() +
  ggtitle("") + 
  theme(text=element_text(size=30), axis.text.x = element_text(angle=65, hjust = 1))
dev.off()




#variable importance calculation for "obscene" with random forest

# create separate dataset 

train_tox3 = cbind(id = train_all_tox_nzv_cor_lc_out[,1], train_all_tox_nzv_cor_lc_out[,2:205], obscene = train_all_tox_nzv_cor_lc_out[,208]) 
library(parallel)

library(doParallel)
require(caret)
cluster=makeCluster(10)

registerDoParallel(cluster)
fitControl = trainControl(allowParallel=TRUE, method="none")

mtry = floor(sqrt((ncol(train_tox3)-2)))

tunegrid = expand.grid(.mtry=mtry)

metric="Accuracy"
rf3 = train(obscene ~ ., method="rf", ntree=100, data=train_tox3[,-1], importance=T, trControl = fitControl, tuneGrid=tunegrid, metric=metric)
stopCluster(cluster)

# save created model for future reference
save(rf3, file="rf_varimp_tox3.RData")
# load the saved rf model 
forest3 = get(load("rf_varimp_tox3.RData"))
# create table of variable importance
require(caret)
var_imp3 = data.frame(varImp(forest3)$importance)
# variable sorted based on Gini Index
var_imp3$variable = row.names(var_imp3)

var_imp3 = data.frame(var_imp3[order(-var_imp3$X1),])
var_imp3 = var_imp3[,c(3,2)] 
# cleaning

rm(rf3, forest3)
# create figure of variable importance
jpeg(file="var_imp3.jpeg", height = 3000, width = 6500, res=300)
par(mgp=c(10,1,0))

var_imp3 = var_imp3[1:20,]
ggplot(var_imp3, aes_string(x = reorder(as.factor(var_imp3[,1]), -var_imp3[,2]), y=var_imp3[,2], group=1)) +
  geom_line(color="red", size=2) + geom_point(size=4) +
  theme_bw() + 
  xlab("Variable Name") + ylab("Overall Importance") +
  scale_y_continuous() +
  ggtitle("") + 
  theme(text=element_text(size=30), axis.text.x = element_text(angle=65, hjust = 1))

dev.off()




#variable importance calculation for "threat" with random forest

# create separate dataset 

train_tox4 = cbind(id = train_all_tox_nzv_cor_lc_out[,1], train_all_tox_nzv_cor_lc_out[,2:205], threat = train_all_tox_nzv_cor_lc_out[,209]) 
library(parallel)

library(doParallel)
require(caret)
cluster=makeCluster(10)
registerDoParallel(cluster)
fitControl = trainControl(allowParallel=TRUE, method="none")

mtry = floor(sqrt((ncol(train_tox4)-2)))

tunegrid = expand.grid(.mtry=mtry)

metric="Accuracy"

rf4 = train(threat ~ ., method="rf", ntree=100, data=train_tox4[,-1], importance=T, trControl = fitControl, tuneGrid=tunegrid, metric=metric)
stopCluster(cluster)

# save created model for future reference
save(rf4, file="rf_varimp_tox4.RData")
# load the saved rf model 
forest4 = get(load("rf_varimp_tox4.RData"))
# create table of variable importance
require(caret)

var_imp4 = data.frame(varImp(forest4)$importance)
var_imp4$variable = row.names(var_imp4)

# variable sorted based on Gini Index
var_imp4 = data.frame(var_imp4[order(-var_imp4$X1),]) 

var_imp4 = var_imp4[,c(3,2)] 

# cleaning
rm(rf4, forest4)
# create figure of variable importance

jpeg(file="var_imp4.jpeg", height = 3000, width = 6500, res=300)
par(mgp=c(10,1,0))

var_imp4 = var_imp4[1:20,]

ggplot(var_imp4, aes_string(x = reorder(as.factor(var_imp4[,1]), -var_imp4[,2]), y=var_imp4[,2], group=1)) +
  geom_line(color="red", size=2) + geom_point(size=4) +
  theme_bw() + 
  xlab("Variable Name") + ylab("Overall Importance") +
  scale_y_continuous() +
  ggtitle("") + 
  theme(text=element_text(size=30), axis.text.x = element_text(angle=65, hjust = 1))

dev.off()




#variable importance calculation for "insult" with random forest

# create separate dataset 

train_tox5 = cbind(id = train_all_tox_nzv_cor_lc_out[,1], train_all_tox_nzv_cor_lc_out[,2:205], insult = train_all_tox_nzv_cor_lc_out[,210]) 
library(parallel)

library(doParallel)

require(caret)
cluster=makeCluster(10)

registerDoParallel(cluster)
fitControl = trainControl(allowParallel=TRUE, method="none")

mtry = floor(sqrt((ncol(train_tox5)-2)))

tunegrid = expand.grid(.mtry=mtry)

metric="Accuracy"

rf5 = train(insult ~ ., method="rf", ntree=100, data=train_tox5[,-1], importance=T, trControl = fitControl, tuneGrid=tunegrid, metric=metric)
stopCluster(cluster)

# save created model for future reference
save(rf5, file="rf_varimp_tox5.RData")
# load the saved rf model 
forest5 = get(load("rf_varimp_tox5.RData"))
# create table of variable importance
require(caret)

var_imp5 = data.frame(varImp(forest5)$importance)

var_imp5$variable = row.names(var_imp5)

# variable sorted based on Gini Index
var_imp5 = data.frame(var_imp5[order(-var_imp5$X1),]) 

var_imp5 = var_imp5[,c(3,2)] 

# cleaning

rm(rf5, forest5)
# create figure of variable importance

jpeg(file="var_imp5.jpeg", height = 3000, width = 6500, res=300)
par(mgp=c(10,1,0))

var_imp5 = var_imp5[1:20,]

ggplot(var_imp5, aes_string(x = reorder(as.factor(var_imp5[,1]), -var_imp5[,2]), y=var_imp5[,2], group=1)) +
  geom_line(color="red", size=2) + geom_point(size=4) +
  theme_bw() + 
  xlab("Variable Name") + ylab("Overall Importance") +
  scale_y_continuous() +
  ggtitle("") + 
  theme(text=element_text(size=30), axis.text.x = element_text(angle=65, hjust = 1))

dev.off()



#variable importance calculation for "identity_hate" with random forest

# create separate dataset 

train_tox6 = cbind(id = train_all_tox_nzv_cor_lc_out[,1], train_all_tox_nzv_cor_lc_out[,2:205], identity_hate = train_all_tox_nzv_cor_lc_out[,211]) 
library(parallel)

library(doParallel)

require(caret)
cluster=makeCluster(10)

registerDoParallel(cluster)

fitControl = trainControl(allowParallel=TRUE, method="none")

mtry = floor(sqrt((ncol(train_tox6)-2)))

tunegrid = expand.grid(.mtry=mtry)

metric="Accuracy"

rf6 = train(identity_hate ~ ., method="rf", ntree=100, data=train_tox6[,-1], importance=T, trControl = fitControl, tuneGrid=tunegrid, metric=metric)
stopCluster(cluster)

# save created model for future reference

save(rf6, file="rf_varimp_tox6.RData")
# load the saved rf model 
forest6 = get(load("rf_varimp_tox6.RData"))
# create table of variable importance

require(caret)

var_imp6 = data.frame(varImp(forest6)$importance)

var_imp6$variable = row.names(var_imp6)

# variable sorted based on Gini Index
var_imp6 = data.frame(var_imp6[order(-var_imp6$X1),]) 
var_imp6 = var_imp6[,c(3,2)] 
# cleaning

rm(rf6, forest6)
# create figure of variable importance

jpeg(file="var_imp6.jpeg", height = 3000, width = 6500, res=300)
par(mgp=c(10,1,0))

var_imp6 = var_imp6[1:20,]

ggplot(var_imp6, aes_string(x = reorder(as.factor(var_imp6[,1]), -var_imp6[,2]), y=var_imp6[,2], group=1)) +
  geom_line(color="red", size=2) + geom_point(size=4) +
  theme_bw() + 
  xlab("Variable Name") + ylab("Overall Importance") +
  scale_y_continuous() +
  ggtitle("") + 
  theme(text=element_text(size=30), axis.text.x = element_text(angle=65, hjust = 1))

dev.off()


# combine importent variables from each toxicity type
all_var_imp = as.character(c(var_imp1$variable, var_imp2$variable, var_imp3$variable, var_imp4$variable, var_imp5$variable, var_imp6$variable)) 
# identify unique among all combined
all_var_imp_unq = as.character(unique(all_var_imp))

# in total 49 most useful variables identified


# associate the variablres with their respective words or terms
terms = read.table(file="all_terms_1433.csv", header=F, stringsAsFactors = F, sep="\t", na.strings = c("NA", ""), fill = T)

word = as.character()

for (i in c(1:length(all_var_imp_unq))) {
	
	c = as.numeric(gsub("var", "", all_var_imp_unq[i]))
	
	a = terms[c,1]
	
	word = c(word, a)	

}

all_var_imp_comb = data.frame(variable = all_var_imp_unq, Term = word)


# save in file for future reference
write.table(all_var_imp_comb, file="var_imp.csv", row.names=F, col.names=F, sep="\t", quote=F)

# this final selected term variable list had 49 most important term avriables

# also more toxic words were added from the encycloDB dirty words database
# Link: https://github.com/turalus/encycloDB/blob/master/Dirty%20Words/DirtyWords.csv
# Combining these two a total of 110 term variables were selected for further analysis
# these 110 term variables are stored as file=final_selected_variables.csv 

