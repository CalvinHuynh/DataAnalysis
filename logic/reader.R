Sys.setenv(lang = "en")
Sys.setlocale(category = "LC_ALL", locale = "English_United States.1252")

library(config)
library(readtext)
library(plyr)
library(dplyr)
library(stringi)
config <- config::get(file = "config.yml")

# General functions -------------------------------------------------------

# Convert review rating to binary
convertToBinaryScale1to10 <- function(number) {
  if (number >= 7) {
    return(1)
  } else {
    return(0)
  }
}

# Converts the Amazon review to binary
convertToBinaryScale1to5 <- function(number) {
  if (number >= 4) {
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
readFirstDataset <- function(stringAsFactor = FALSE){
  trainIMDBData <-
    read.table(config$firstDataFile, sep = '\t', header = TRUE, stringsAsFactors = stringAsFactor)
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
    lapply(combinedTrainData2$sentiment, convertToBinaryScale1to10)
  
  return(combinedTrainData2)
}

readLargeTextFile <- function(){
  largeDataFile <- reconstructColumnNames(read.horizontalTextfile(paste0(config$largeDataFile)))
  largeDataFile$sentiment <- as.numeric(sapply(largeDataFile$sentiment, convertToBinaryScale1to5))
  # write.csv(largeDataFile, file = "preparedAmazonReviews.csv")
  return(largeDataFile)
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
  
  textToClean <- stemDocument(textToClean)
  
  return(textToClean)
}

# function derived from https://stackoverflow.com/questions/17288197/reading-a-csv-file-organized-horizontally
read.tcsv = function(file, header=TRUE, sep="\n", nrow = 3000,...) {
  
  n = max(count.fields(file, sep=sep), na.rm=TRUE)
  n = nrow
  x = readLines(file)
  
  .splitvar = function(x, sep, n) {
    var = unlist(strsplit(x, split=sep))
    length(var) = n
    return(var)
  }
  
  x = do.call(cbind, lapply(x, .splitvar, sep=sep, n=n))
  x = apply(x, 1, paste, collapse=sep) 
  out = read.csv(text=x, sep=sep, header=header, ...)
  return(out)
}

# Data is from amazon movie reviews
# functon to create a dataframe from text with horizontal column names, written specifically for the following url:
# https://snap.stanford.edu/data/web-Movies.html 
read.horizontalTextfile <- function(inputFile, dataStartPosn = 12, nfields = 8, TXTmaxLen = 3e3, eachColnameLen = 11){
  dataStartPosn <- dataStartPosn
  nfields <- nfields
  TXTmaxLen <- TXTmaxLen
  eachColnameLen <- eachColnameLen
  
  #download and read lines
  dataLines <- readLines(file(inputFile, "r"))

  #extract data
  data <- stri_sub(dataLines, dataStartPosn, length=TXTmaxLen)
  
  #extract colnames
  colnames <- unname(sapply(dataLines[1:(nfields+1)], function(x) substring(x, 1, eachColnameLen)))
  
  #form table
  df <- data.frame(do.call(rbind, split(data, ceiling(seq_along(data)/(nfields+1)))))
  
  #formatting
  df <- setNames(df, colnames)
  df[-(nfields+1)]
}

reconstructColumnNames <- function(dataframe){
  dataframe <- dataframe %>%
    select("review/scor", "review/text")
  
  colnames(dataframe) <- c("sentiment","review")
  
  dataframe <- dataframe %>%
    mutate(sentiment = gsub('.*:',"", sentiment)) %>%
    mutate(review = gsub('.*:',"", review)) %>%
    mutate(sentiment = as.numeric(sentiment)) %>%
    filter(!is.na(sentiment))
  
  return(dataframe)
}

countScoreDistribution <- function(dataframe){
  scoreDistribution <- dataframe %>%
    group_by(sentiment) %>%
    summarise(number_of_people = n())
  return(scoreDistribution)
}
