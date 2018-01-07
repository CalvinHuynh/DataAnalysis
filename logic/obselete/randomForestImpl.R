Sys.setenv(lang = "en")

library(tm)
library(SnowballC)
library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)
source("logic/reader.R")

trainIMDBData <- readFirstDataset()
combinedTrainData2 <- readSecondDataset(trainIMDBData)

trainIMDBData$review <- convertToUtf8Enc(trainIMDBData$review)
# trainIMDBData$review <- commonCleaning(trainIMDBData$review)
# trainIMDBData$review <- removeCommonStopWords(trainIMDBData$review)

# combinedTrainData2$review <-
#   convertToUtf8Enc(combinedTrainData2$review)
# combinedTrainData2$review <-
#   commonCleaning(combinedTrainData2$review)
# combinedTrainData2$review <-
#   removeCommonStopWords(combinedTrainData2$review)

# Test with a small sample size
attempt1 <- function() {
  random2000rows <- shuffleDataframe(trainIMDBData)[1:2000,]
  table(random2000rows$sentiment)
  
  corpus <- Corpus(VectorSource(random2000rows$review))
  
  corpus <- corpus %>%
    tm_map(content_transformer(tolower)) %>%
    tm_map(removePunctuation) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords(kind = "en")) %>%
    tm_map(stripWhitespace)
  
  corpus[[1]]$content
  frequencies <- DocumentTermMatrix(corpus)
  inspect(frequencies[1995:2000, 505:515])
  
  str(findFreqTerms(frequencies, lowfreq = 20))
  sparse <- removeSparseTerms(frequencies, 0.995)
  
  IMDBtest <- as.data.frame(as.matrix(sparse))
  colnames(IMDBtest) <- make.names(colnames(IMDBtest))
  IMDBtest$sentiment <- random2000rows$sentiment
  
  set.seed(1)
  # split <- sample.split(IMDBtest$sentiment, SplitRatio= .70)
  # trainSparse <- subset(IMDBtest, split == TRUE)
  # testSparse <- subset(IMDBtest, split == FALSE)
  
  splitIndex = sample(1:nrow(IMDBtest),
                      size = round(0.7 * nrow(IMDBtest)),
                      replace = FALSE)
  trainSparse <- IMDBtest[splitIndex,]
  testSparse <- IMDBtest[-splitIndex,]
  
  IMDBCart <- rpart(sentiment ~ ., data = trainSparse, method = 'class')
  prp(IMDBCart)
  
  predictCART <- predict(IMDBCart, newdata = testSparse, type = 'class')
  table(testSparse$sentiment, predictCART)
  # Accuracy calculation
  # (203 + 208) / nrow(testSparse)
  
  table(testSparse$sentiment)
  # Accuracy calculation
  # 296 / nrow(testSparse)
  
  IMDBRF <- randomForest(sentiment ~ ., data = trainSparse)
  
  predictRF <- predict(IMDBRF, newdata = testSparse)
  table(testSparse$sentiment, predictRF)
  recall_accuracy(testSparse$sentiment, predictRF)
}
