Sys.setenv(lang = "en")
Sys.setlocale(category = "LC_ALL", locale = "English_United States.1252")

library(RTextTools)
library(e1071)
library(config)
source("logic/reader.R")

trainIMDBData <- shuffleDataframe(readFirstDataset())
trainIMDBData$sentiment <- as.factor(trainIMDBData$sentiment)
trainIMDBData$review <- as.factor(trainIMDBData$review)
trainIMDBData$id <- NULL

matrix = create_matrix(trainIMDBData[1:500, 2],language = "english", removeStopwords = FALSE, 
                       removeNumbers = TRUE, stemWords = FALSE, tm::weightTfIdf)
matrix
mat = as.matrix(matrix)

classifier = naiveBayes(mat[1:400, ], as.factor(trainIMDBData[1:400, 1]))
predicted = predict(classifier, mat[401:500, ])
recall_accuracy(trainIMDBData[401:500, 1], predicted)

# container = create_container(matrix, as.numeric(as.factor(trainIMDBData[1:1000, 1])), trainSize = 1:800, 
#                              testSize = 801:1000, virgin = FALSE)  #removeSparseTerms
# 
# models = train_models(container, algorithms = c("SVM", "RF", 
#                                                 "TREE"))
# 
# results = classify_models(container, models)
# 
# # accuracy
# recall_accuracy(as.numeric(as.factor(trainIMDBData[801:1000, 1])), results[, "FORESTS_LABEL"])
# 
# recall_accuracy(as.numeric(as.factor(trainIMDBData[801:1000, 1])), results[, "MAXENTROPY_LABEL"])
# 
# recall_accuracy(as.numeric(as.factor(trainIMDBData[801:1000, 1])), results[, "TREE_LABEL"])
# 
# recall_accuracy(as.numeric(as.factor(trainIMDBData[801:1000, 1])), results[, "BAGGING_LABEL"])
# 
# recall_accuracy(as.numeric(as.factor(tweets[11:15, 2])), results[, "SVM_LABEL"])

test1 <- function(){
  pos_tweets = rbind(c("I love this car", "positive"), c("This view is amazing", 
                                                         "positive"), c("I feel great this morning", "positive"), c("I am so excited about the concert", 
                                                                                                                    "positive"), c("He is my best friend", "positive"))
  
  
  neg_tweets = rbind(c("I do not like this car", "negative"), c("This view is horrible", 
                                                                "negative"), c("I feel tired this morning", "negative"), c("I am not looking forward to the concert", 
                                                                                                                           "negative"), c("He is my enemy", "negative"))
  
  
  test_tweets = rbind(c("feel happy this morning", "positive"), c("larry friend", 
                                                                  "positive"), c("not like that man", "negative"), c("house not great", "negative"), 
                      c("your song annoying", "negative"))
  
  tweets = rbind(pos_tweets, neg_tweets, test_tweets)
  tweets <- as.data.frame(tweets)
  str(tweets)
  head(tweets[, 1])
  # native bayes
  matrix = create_matrix(tweets[, 1], language = "english", removeStopwords = FALSE, 
                         removeNumbers = TRUE, stemWords = FALSE, tm::weightTfIdf)
  mat = as.matrix(matrix)
  classifier = naiveBayes(mat[1:10, ], as.factor(tweets[1:10, 2]))
  predicted = predict(classifier, mat[11:15, ])
  predicted
  recall_accuracy(tweets[11:15, 2], predicted)
  
  container = create_container(matrix, as.numeric(as.factor(tweets[, 2])), trainSize = 1:10, 
                               testSize = 11:15, virgin = FALSE)  #removeSparseTerms
  
  models = train_models(container, algorithms = c("MAXENT", "SVM", "RF", "BAGGING", 
                                                  "TREE"))
  
  results = classify_models(container, models)
  
  recall_accuracy(as.numeric(as.factor(tweets[11:15, 2])), results[, "FORESTS_LABEL"])

  recall_accuracy(as.numeric(as.factor(tweets[11:15, 2])), results[, "MAXENTROPY_LABEL"])

  recall_accuracy(as.numeric(as.factor(tweets[11:15, 2])), results[, "TREE_LABEL"])

  recall_accuracy(as.numeric(as.factor(tweets[11:15, 2])), results[, "BAGGING_LABEL"])

  recall_accuracy(as.numeric(as.factor(tweets[11:15, 2])), results[, "SVM_LABEL"])
}
