Sys.setenv(lang = "en")

library(config)
library(qdap)
library(tm)
config <- config::get(file = "config.yml")

# Reading tab seperated file
trainData <- read.table(config$data, sep = '\t',header = TRUE)

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

removeCommonStopWords <- function(textToClean, customWords = NULL){
  if(!is.null(customWords)){
    total_stops <- c(customWords, stopwords("en"))
    return(removeWords(textToClean, total_stops))
  } else {
    return(removeWords(textToClean, stopwords("en")))
  }
}

# Convert the review data using the utf-8 encoding
trainData$review <- iconv(enc2utf8(as.character(trainData$review)),sub = "byte")
trainData$review <- commonCleaning(trainData$review)
trainData$review <- removeCommonStopWords(trainData$review)

term_count <- freq_terms(trainData$review, 10)
plot(term_count)

trainDataSource <- VectorSource(trainData$review)
trainDataCorpus <- VCorpus(trainDataSource)

train_dtm <- DocumentTermMatrix(trainDataCorpus)
train_tdm <- TermDocumentMatrix(trainDataCorpus)


### Not enough memory
train_tdm_m <- as.matrix(train_tdm)

term_tdm_freq <- rowSums(train_tdm_m)

term_tdm_freq <- sort(term_tdm_freq, decreasing = TRUE)

# View the top 10 most common words
term_tdm_freq[1:10]

# Plot a barchart of the 10 most common words
barplot(term_tdm_freq[1:10], col = "tan", las = 2)