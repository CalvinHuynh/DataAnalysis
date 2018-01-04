Sys.setenv(lang = "en")

library(config)
library(qdap)
library(tm)
library(readtext)
library(RTextTools)
library(e1071)
library(caret)
config <- config::get(file = "config.yml")

# General functions -------------------------------------------------------

# Convert review rating to binary
convertToBinary <- function(number) {
  if (number >= 7) {
    return(1)
  } else {
    return(0)
  }
}

removeCommonStopWords <- function(textToClean, customWords = NULL) {
  if (!is.null(customWords)) {
    total_stops <- c(customWords, stopwords("en"))
    return(removeWords(textToClean, total_stops))
  } else {
    return(removeWords(textToClean, stopwords("en")))
  }
}

# Convert the review data using the utf-8 encoding
convertToUtf8Enc <- function(text) {
  return(iconv(enc2utf8(as.character(text)), sub = "byte"))
}

# Shuffle dataframe rowwise
shuffleDataframe <- function(dataframe) {
  dataframe <- dataframe[sample(nrow(dataframe)),]
  return(dataframe)
}

convertTextToCorpus <- function(text) {
  vectorSource <- VectorSource(text)
  corpus <- VCorpus(vectorSource)
  
  return(corpus)
}

# Reading data ------------------------------------------------------------

# Data set 1
readFirstDataset <- function(){
  trainIMDBData <-
    read.table(config$firstDataFile, sep = '\t', header = TRUE)
  trainIMDBData$id <- sub("^", "1_", trainIMDBData$id)
  return(trainIMDBData)
}

# Data set 2
readSecondDataset <- function(columnNames) {
  trainData2positive <-
    readtext(paste0(config$trainDataDirectoryPositive, "*"))
  trainData2positive$V3 <- 0
  trainData2positive <-
    trainData2positive[c("doc_id", "V3", "text")]
  colnames(trainData2positive) <- colnames(columnNames)
  
  trainData2negative <-
    readtext(paste0(config$trainDataDirectoryNegative, "*"))
  trainData2negative$V3 <- 0
  trainData2negative <-
    trainData2negative[c("doc_id", "V3", "text")]
  colnames(trainData2negative) <- colnames(columnNames)
  
  combinedTrainData2 <-
    rbind(trainData2positive, trainData2negative)
  
  # Formatting the second data set
  combinedTrainData2$id <-
    gsub(combinedTrainData2$id,
         pattern = ".txt",
         replacement = "")
  combinedTrainData2$id <- sub("^", "2_", combinedTrainData2$id)
  combinedTrainData2$sentiment <-
    sub(".*_", "", combinedTrainData2$id)
  combinedTrainData2$sentiment <-
    as.numeric(combinedTrainData2$sentiment)
  combinedTrainData2$sentiment <-
    lapply(combinedTrainData2$sentiment, convertToBinary)
  
  return(combinedTrainData2)
}

# Basic cleaning function
commonCleaning <- function(textToClean) {
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
