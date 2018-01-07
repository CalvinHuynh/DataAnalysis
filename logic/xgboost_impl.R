Sys.setenv(lang = "en")
Sys.setlocale(category = "LC_ALL", locale = "English_United States.1252")

library(xgboost)
library(FeatureHashing)
library(config)
library(caret)
source("logic/reader.R")

imdbSet <- readFirstDataset()

# retrieves the amazon movie review data
# amazonSet <- shuffleDataframe(readAmazonReviews())
amazonSet <- shuffleDataframe(amazonReviewsDf)


# Written reviews (might have been insipred from other sources)
posWrittenReviews <- rbind(
  c(1, 'Surprisingly really not terrible'),  #This Charming Man (2006)
  c(1, 'Powerfull, stunning and exceptionally crafted'),  #Dunkirk (2017)
  c(1, 'ROFLED on the floor'),  #Deadpool (2016)
  c(1, 'Loved the atmosphere of the movie'),  #Ju-on: The Grudge (2002)
  c(1,'Space Jam has the potential to be a 5 star movie yet loses a star because Daffy Duck is underutilized') #Space Jam (1996)
)

negWrittenReviews <- rbind(
  c(0, 'Theres nothing good about the fellas in this movie'), #Goodfellas (1990)
  c(0,'I watched this to review for my GRANDAUGHTER age 8. It started out hopeful that it was going to be good. Then it kept getting darker and then "nude" animals entered into the movie'),  #Zootopia (2016)
  c(0,'The film is a bit unrealistic since the existence of dinosaurs has not been proven. There may be some bones but these are easy enough to fake'),  #Jurrasic World (2015)
  c(0, 'This is the worst movie ever made, it should have never been shown'), #Sausage Party (2016)
  c(0, 'oh yeah a boat this big could never sink') #The Titanic (1997)
)

writtenTestReviewsDf <-
  data.frame(rbind(posWrittenReviews, negWrittenReviews))
colnames(writtenTestReviewsDf) <- c("sentiment", "review")

# merging dataframes
# combinedDf <- rbind.fill(writtenTestReviewsDf, amazonSet, imdbSet)
combinedDf <- rbind.fill(imdbSet)

# remove unused column
combinedDf$id <- NULL

# system.time(combinedDf$review <-
#               removeCommonStopWords(commonCleaning(combinedDf$review, stemData = TRUE)))

# system.time(combinedDf$review <-
#               removeCommonStopWords(commonCleaning(combinedDf$review, stemData = TRUE)))

ownWrittenReviewDataFrame <- combinedDf[1:10, ]
otherReviewDf <- combinedDf[11:nrow(combinedDf), ]

set.seed(3)
# trainingData <- c(ownWrittenReviewDataFrame$review,
#                   otherReviewDf$review[sample(nrow(otherReviewDf), 8000)])
# 
# trainingLabel <- c(ownWrittenReviewDataFrame$sentiment,
#                    otherReviewDf$sentiment[sample(nrow(otherReviewDf), 8000)])

trainingData <- otherReviewDf$review[1:5000]

trainingLabel <- otherReviewDf$sentiment[1:5000]


# split data 25% test, 75% train
size = floor(0.25 * length(trainingData))
sizeNext = size + 1
total = length(trainingData)

train <- c(sizeNext:total) # 75%
test <- c(1:size) # 25%

corpus <- Corpus(VectorSource(trainingData))

# data cleaning using the tm package
corpus <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind = "en")) %>%
  tm_map(stripWhitespace)

# Inspecting the result of the cleaning
corpus[[1]]$content
frequencies <- DocumentTermMatrix(corpus)

# remove words that appear in less than 0,5% of the reviews
sparseDTM <- removeSparseTerms(frequencies, 0.995)
sparseDTM
matrixDTM <- as.matrix(sparseDTM)

dtrain <-
    xgb.DMatrix(matrixDTM[train, ], label = trainingLabel[train])
dtest <- xgb.DMatrix(matrixDTM[test, ], label = trainingLabel[test])

watch <- list(train = dtrain, valid = dtest)

m1 <- xgb.train(
  booster = "gblinear",
  nrounds = 200,
  nthread = 4,
  eta = 0.02,
  max.depth = 20,
  data = dtrain,
  objective = "binary:logistic",
  watchlist = watch,
  eval_metric = "error"
)


# m2 <- xgb.train(
#   data = dtrain,
#   nthread = 4,
#   nrounds = 100,
#   eta = 0.02,
#   max.depth = 45,
#   colsample_bytree = 0.1,
#   subsample = 0.95,
#   objective = "binary:logistic",
#   watchlist = watch,
#   eval_metric = "error"
# )
# 
# m3 <-
#   xgboost(
#     data = dtrain,
#     max.depth = 45,
#     eta = 1,
#     nthread = 4,
#     nround = 100,
#     objective = "binary:logistic"
#   )

# predict the sentiment using the created model and the test set
xgbpred <- predict(m1, dtest)

# Convert all the predicted values above 0.5 as positive
xgbpred <- ifelse(xgbpred > 0.5, 1, 0)

confusionMatrix(xgbpred, (as.numeric(trainingLabel[test])))

prepForXgbDMatrix <- function(dataframe){
  corpus <- Corpus(VectorSource(dataframe$review))
  
  # data cleaning using the tm package
  corpus <- corpus %>%
    tm_map(content_transformer(tolower)) %>%
    tm_map(removePunctuation) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords(kind = "en")) %>%
    tm_map(stripWhitespace)
  
  # Inspecting the result of the cleaning
  corpus[[1]]$content
  frequencies <- DocumentTermMatrix(corpus)
  
  # remove words that appear in less than 0,5% of the reviews
  # sparseDTM <- removeSparseTerms(frequencies, 0.995)
  # sparseDTM
  
  matrixDTM <- as.matrix(frequencies)
 
  dMatrix <- xgb.DMatrix(matrixDTM, label = dataframe$sentiment)
  return(dMatrix)
}

### Unfortunately the writtenTestReviewDf suddenly gained a new class, no idea why yet.
# xgbpredCustom <- predict(m1, prepForXgbDMatrix(writtenTestReviewsDf))
# 
# # Convert all the predicted values above 0.5 as positive
# xgbxgbpredCustompred <- ifelse(xgbpredCustom > 0.5, 1, 0)
# 
# confusionMatrix(xgbpredCustom, writtenTestReviewsDf$sentiment)
# 
# unique_results <- union(xgbpred, as.numeric(writtenTestReviewsDf$sentiment))
# 
# comparison_table <-
#   table(factor(xgbpred, unique_results),
#         factor(as.numeric(writtenTestReviewsDf$sentiment), unique_results))
# 
# confusionMatrix(comparison_table)

