Sys.setenv(lang = "en")

library(config)
library(qdap)
library(tm)
library(readtext)
library(RTextTools)
library(e1071)
config <- config::get(file = "config.yml")

# General functions -------------------------------------------------------

# Convert review rating to binary
convertToBinary <- function(number){
  if(number >= 7){
    return(1)
  } else {
    return(0)
  }
}

removeCommonStopWords <- function(textToClean, customWords = NULL){
  if(!is.null(customWords)){
    total_stops <- c(customWords, stopwords("en"))
    return(removeWords(textToClean, total_stops))
  } else {
    return(removeWords(textToClean, stopwords("en")))
  }
}

# Convert the review data using the utf-8 encoding
convertToUtf8Enc <- function(text){
  return(iconv(enc2utf8(as.character(text)),sub = "byte"))
}

# Shuffle dataframe rowwise
shuffleDataframe <- function(dataframe){
  dataframe <- dataframe[sample(nrow(dataframe)), ]
  return(dataframe)
}

convertTextToCorpus <- function(text){
  vectorSource <- VectorSource(text)
  corpus <- VCorpus(vectorSource)
  
  return(corpus)
}

# Reading data ------------------------------------------------------------

# Data set 1
trainIMDBData <- read.table(config$firstDataFile, sep = '\t', header = TRUE)
trainIMDBData$id <- sub("^", "1_", trainIMDBData$id)

# Data set 2
readSecondDataset <- function(){
  trainData2positive <- readtext(paste0(config$trainDataDirectoryPositive,"*"))
  trainData2positive$V3 <- 0
  trainData2positive <- trainData2positive[c("doc_id", "V3", "text")]
  colnames(trainData2positive) <- colnames(trainIMDBData)
  
  trainData2negative <- readtext(paste0(config$trainDataDirectoryNegative,"*"))
  trainData2negative$V3 <- 0
  trainData2negative <- trainData2negative[c("doc_id", "V3", "text")]
  colnames(trainData2negative) <- colnames(trainIMDBData)
  
  combinedTrainData2 <- rbind(trainData2positive, trainData2negative)
  
  # Formatting the second data set
  combinedTrainData2$id <- gsub(combinedTrainData2$id, pattern = ".txt", replacement = "")
  combinedTrainData2$id <- sub("^", "2_", combinedTrainData2$id)
  combinedTrainData2$sentiment <- sub(".*_","",combinedTrainData2$id)
  combinedTrainData2$sentiment <- as.numeric(combinedTrainData2$sentiment)
  combinedTrainData2$sentiment <- lapply(combinedTrainData2$sentiment, convertToBinary)
  
  return(combinedTrainData2)
}

combinedTrainData2 <- readSecondDataset()

# Basic cleaning function
commonCleaning <- function(textToClean){
  # All lowercase
  textToClean <- tolower(textToClean)
  # Remove punctuation
  textToClean <- removePunctuation(textToClean)
  # Remove numbers
  textToClean <- removeNumbers(textToClean)
  # Remove whitespace
  textToClean <- stripWhitespace(textToClean)
  # Remove text within brackets
  textToClean <- bracketX(textToClean)
  # Replace numbers with words
  textToClean <- replace_number(textToClean)
  # Replace abbreviations
  textToClean <- replace_abbreviation(textToClean)
  # Replace contractions
  textToClean <- replace_contraction(textToClean)
  # Replace symbols with words
  textToClean <- replace_symbol(textToClean)
  
  return(textToClean)
}

trainIMDBData$review <- convertToUtf8Enc(trainIMDBData$review)
trainIMDBData$review <- commonCleaning(trainIMDBData$review)
trainIMDBData$review <- removeCommonStopWords(trainIMDBData$review)

combinedTrainData2$review <- convertToUtf8Enc(combinedTrainData2$review)
combinedTrainData2$review <- commonCleaning(combinedTrainData2$review)
combinedTrainData2$review <- removeCommonStopWords(combinedTrainData2$review)

# TDM & DTM creation ---------------------------------

trainIMDBDataCorpus <- convertTextToCorpus(shuffleDataframe(trainIMDBData)$review)

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


# RTextTools --------------------------------------------------------------

matrixData= create_matrix(trainIMDBData[, 3], language="english") 
trainDataMatrix = as.matrix(matrixData[1:1000, ])

classifier = naiveBayes(trainDataMatrix[1:500, ], as.factor(trainIMDBData[1:500, 2]))

predicted = predict(classifier, trainDataMatrix[501:600, ]); 
# trainIMDBData[11:15, 3]
# predicted

table(trainIMDBData[501:600, 2], predicted)
recall_accuracy(trainIMDBData[501:600, 2], predicted)
