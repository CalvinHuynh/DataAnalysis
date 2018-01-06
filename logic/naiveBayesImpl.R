Sys.setenv(lang = "en")
Sys.setlocale(category = "LC_ALL", locale = "English_United States.1252")

library(config)
library(qdap)
library(tm)
library(readtext)
library(RTextTools)
library(e1071)
library(caret)
source("logic/reader.R")

trainIMDBData <- readFirstDataset()
combinedTrainData2 <- readSecondDataset(trainIMDBData)

# trainIMDBData$review <- convertToUtf8Enc(trainIMDBData$review)
# trainIMDBData$review <- commonCleaning(trainIMDBData$review)
# trainIMDBData$review <- removeCommonStopWords(trainIMDBData$review)

# combinedTrainData2$review <-
#   convertToUtf8Enc(combinedTrainData2$review)
# combinedTrainData2$review <-
#   commonCleaning(combinedTrainData2$review)
# combinedTrainData2$review <-
#   removeCommonStopWords(combinedTrainData2$review)

# TDM & DTM creation ---------------------------------
temporaryStashed <- function() {
  trainIMDBDataCorpus <-
    convertTextToCorpus(shuffleDataframe(trainIMDBData)$review)
  
  train_dtm <- DocumentTermMatrix(trainIMDBDataCorpus)
  train_tdm <- TermDocumentMatrix(trainIMDBDataCorpus)
  
  # Requires 7.7 Gb
  train_tdm_m <- as.matrix(train_tdm)
  
  term_tdm_freq <- rowSums(train_tdm_m)
  
  term_tdm_freq <- sort(term_tdm_freq, decreasing = TRUE)
  
  # View the top 10 most common words
  term_tdm_freq[1:10]
  
  # Plot a barchart of the 10 most common words
  barplot(term_tdm_freq[1:10], col = "tan", las = 2)
  
  trainData2Corpus <- convertTextToCorpus(combinedTrainData2$review)
  
  train_dtm2 <- DocumentTermMatrix(trainData2Corpus)
  train_tdm2 <- TermDocumentMatrix(trainData2Corpus)
  
  # Requires 22.0 Gb
  train2_tdm_m <- as.matrix(train_tdm2)
  
  term2_tdm_freq <- rowSums(train2_tdm_m)
  
  term2_tdm_freq <- sort(term2_tdm_freq, decreasing = TRUE)
  
  term2_tdm_freq[1:10]
  
  barplot(term2_tdm_freq[1:10], col = "tan", las = 2)
}

# RTextTools --------------------------------------------------------------
set.seed(123)

attempt1 <- function() {
  trainIMDBData <- readFirstDataset()
  trainIMDBData <- shuffleDataframe(trainIMDBData)
  trainIMDBData$sentiment <- as.factor(trainIMDBData$sentiment)
  
  corpus <- Corpus(VectorSource(trainIMDBData$review))
  
  corpus <- corpus %>%
    tm_map(content_transformer(tolower)) %>%
    tm_map(removePunctuation) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords(kind = "en")) %>%
    tm_map(stripWhitespace) %>%
    tm_map(stemDocument)
  
  dtm <- DocumentTermMatrix(corpus)
  dtm
  
  threeForth <- paste0(1:6084)
  oneForth <- paste0(6085:8693)
  
  # ?removeSparseTerms
  # removeSparseTerms(dtm, 0.95)
  
  # 70% of the sample size
  # sample_size_dtm <- floor(0.70 * nrow(dtm))
  
  # floor(0.5 * nrow(dtm))
  # train_ind <- sample((nrow(dtm)), size = sample_size_dtm)
  
  df_train <- trainIMDBData[oneForth,]
  df_test <- trainIMDBData[threeForth,]
  
  # plyr::count(df_train$sentiment)
  # plyr::count(df_test$sentiment)
  
  dtm_train <- dtm[oneForth,]
  dtm_test <- dtm[threeForth,]
  
  corpus_train_set <- corpus[oneForth]
  corpus_test_set <- corpus[threeForth]
  
  fivefreq <- findFreqTerms(dtm_train, 5)
  
  dtm_train <-
    DocumentTermMatrix(corpus_train_set, control = list(dictionary = fivefreq))
  
  dtm_test <-
    DocumentTermMatrix(corpus_test_set, control = list(dictionary = fivefreq))
  
  system.time(classifier <-
                naiveBayes(as.matrix(dtm_train),
                           (df_train$sentiment),
                           laplace = 0))
  
  system.time(pred <-
                predict(classifier, newdata = as.matrix(dtm_test)))
  
  table("Predictions" = pred,  "Actual" = df_test$sentiment)
  
  # 1st run 0.506687, laplace = 1
  # 2nd run 0.5051586, laplace = 0, without commoncleaning
  # 3rd run 0.5179158, laplace = 0, findfreq = 20, with cleaning, 30 train 70 test split
  recall_accuracy(pred, df_test$sentiment)
}

attempt2 <- function() {
  trainIMDBData <- readFirstDataset()
  trainIMDBData <- shuffleDataframe(trainIMDBData)
  trainIMDBData$sentiment <- as.factor(trainIMDBData$sentiment)
  trainIMDBData$review <-
    removeCommonStopWords(trainIMDBData$review)
  trainIMDBData$review <- commonCleaning(trainIMDBData$review)

  matrix = create_matrix(
    trainIMDBData[, 3],
    language = "english",
    removeNumbers = TRUE,
    stemWords = FALSE,
    removePunctuation = TRUE,
    stripWhitespace = TRUE
  )
  matrix <- removeSparseTerms(matrix, 0.995)
  
  # data split ratio of 70 /30
  threeForth <- paste0(1:6084)
  oneForth <- paste0(6085:8693)
  
  mat = as.matrix(matrix)
  system.time(classifier <-
                naiveBayes(mat[oneForth,], trainIMDBData[oneForth, 2]))
  system.time(predicted <- predict(classifier, mat[threeForth,]))
  
  # attempt 1: 0.5289383, with create_matrix cleaning and removeSparseTerms = 0.995
  # attempt 2: 0.563051, with precleaning and create_matrix cleaning, removeSparseTerms = 0.995
  # attempt 3: 0.5120736, with precleaning and removeSparseTerms = 0.995
  recall_accuracy(trainIMDBData[threeForth, 2], predicted)
}

dtm = create_matrix(
  trainIMDBData[1:1000, 3],
  language = "english",
  removeNumbers = TRUE  ,
  removePunctuation = TRUE,
  removeStopwords = TRUE,
  stripWhitespace = TRUE
)

matrix = as.matrix(dtm)
# matrix[1:10,]
classifier = naiveBayes(matrix[1:800, ], as.factor(trainIMDBData[1:800, 2]))
predicted = predict(classifier, matrix[801:1000, ])

table(trainIMDBData[801:1000, 2], predicted)
recall_accuracy(trainIMDBData[801:1000, 2], predicted)

# matrixData= create_matrix(trainIMDBData[, 3], language="english")
# trainDataMatrix = as.matrix(matrixData[1:1000, ])
#
# classifier = naiveBayes(trainDataMatrix[1:500, ], as.factor(trainIMDBData[1:500, 2]))
#
# predicted = predict(classifier, trainDataMatrix[501:600, ]);
# # trainIMDBData[11:15, 3]
# # predicted
#
# table(trainIMDBData[501:600, 2], predicted)
# recall_accuracy(trainIMDBData[501:600, 2], predicted)

testCaseNaiveBase <- function() {
  pos_tweets = rbind(
    c("I love this car", "positive"),
    c("This view is amazing",
      "positive"),
    c("I feel great this morning", "positive"),
    c("I am so excited about the concert",
      "positive"),
    c("He is my best friend", "positive")
  )
  
  
  neg_tweets = rbind(
    c("I do not like this car", "negative"),
    c("This view is horrible",
      "negative"),
    c("I feel tired this morning", "negative"),
    c("I am not looking forward to the concert",
      "negative"),
    c("He is my enemy", "negative")
  )
  
  
  test_tweets = rbind(
    c("feel happy this morning", "positive"),
    c("larry friend",
      "positive"),
    c("not like that man", "negative"),
    c("house not great", "negative"),
    c("your song annoying", "negative")
  )
  
  tweets = rbind(pos_tweets, neg_tweets, test_tweets)
  
  # native bayes
  matrix = create_matrix(
    tweets[, 1],
    language = "english",
    removeStopwords = FALSE,
    removeNumbers = TRUE,
    stemWords = FALSE,
    tm::weightTfIdf
  )
  mat = as.matrix(matrix)
  classifier = naiveBayes(mat[1:10,], as.factor(tweets[1:10, 2]))
  predicted = predict(classifier, mat[11:15,])
  recall_accuracy(tweets[11:15, 2], predicted)
}
