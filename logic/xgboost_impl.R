Sys.setenv(lang = "en")
Sys.setlocale(category = "LC_ALL", locale = "English_United States.1252")

library(xgboost)
library(FeatureHashing)
library(config)
library(caret)
source("logic/reader.R")

imdbSet <- readFirstDataset()

# # retrieves the amazon movie review data
# amazonSet <- shuffleDataframe(readAmazonReviews())
# 
# # select the 10k rows from the amazon set
# amazonSet <- amazonSet[1:10000, ]

# Written reviews (might have been insipred from other sources)
posWrittenReviews <- rbind(
  c('1', 'Surprisingly really not terrible'),
  #This Charming Man (2006)
  c('1', 'Powerfull, stunning and exceptionally crafted'),
  #Dunkirk (2017)
  c('1', 'ROFLED on the floor'),
  #Deadpool (2016)
  c('1', 'Loved the atmosphere of the movie'),
  #Ju-on: The Grudge (2002)
  c(
    '1',
    'Space Jam has the potential to be a 5 star movie yet loses a star because Daffy Duck is underutilized'
  ) #Space Jam (1996)
)

negWrittenReviews <- rbind(
  c('0', 'Theres nothing good about the fellas in this movie'),
  #Goodfellas (1990)
  c(
    '0',
    'I watched this to review for my GRANDAUGHTER age 8. It started out hopeful that it was going to be good. Then it kept getting darker and then "nude" animals entered into the movie'
  ),
  #Zootopia (2016)
  c(
    '0',
    'The film is a bit unrealistic since the existence of dinosaurs has not been proven. There may be some bones but these are easy enough to fake'
  ),
  #Jurrasic World (2015)
  c('0', 'NOT FOR KIDS!!!! PORNO MOVIE using grocery items. OMG'),
  #Sausage Party (2016)
  c('0', 'oh yeah a boat this big could never sink') #The Titanic (1997)
)

writtenTestReviewsDf <-
  data.frame(rbind(posWrittenReviews, negWrittenReviews))
colnames(writtenTestReviewsDf) <- c("sentiment", "review")

# merging dataframes
combinedDf <- rbind.fill(writtenTestReviewsDf, amazonSet, imdbSet)
# combinedDf <- imdbSet

# remove unused column
combinedDf$id <- NULL

system.time(combinedDf$review <-
              removeCommonStopWords(combinedDf$review))

ownWrittenReviewDataFrame <- combinedDf[1:10, ]
otherReviewDf <- combinedDf[11:nrow(combinedDf), ]

set.seed(3)
trainingData <- c(ownWrittenReviewDataFrame$review,
                  otherReviewDf$review[sample(nrow(otherReviewDf), 1000)])

trainingLabel <- c(ownWrittenReviewDataFrame$sentiment,
                   otherReviewDf$sentiment[sample(nrow(otherReviewDf), 1000)])


# split data 25% test, 75% train
size = floor(0.25 * length(trainingData))
sizeNext = size + 1
total = length(trainingData)

train <- c(sizeNext:total) # 75%
test <- c(1:size) # 25%

hashedDTM <-
  hashed.model.matrix(
    ~ split(review, delim = " ", type = "tf-idf"),
    data = otherReviewDf,
    hash.size = 2 ^ 16,
    signed.hash = FALSE
  )

dtrain <-
  xgb.DMatrix(hashedDTM[train, ], label = trainingLabel[train])
dvalid <- xgb.DMatrix(hashedDTM[test, ], label = trainingLabel[test])

watch <- list(train = dtrain, valid = dvalid)

# cv <- xgb.cv(data = hashedDTM[train, ],
#              label = trainingLabel[train],
#              objective = "reg:linear",
#              nrounds = 200,
#              nfold = 5,
#              eta = 0.3,
#              depth = 10)
# 
# elog <- as.data.frame(cv$evaluation_log)
# nrounds <- which.min(elog$test_rmse_mean)

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

m2 <- xgb.train(
  data = dtrain,
  nthread = 4,
  nrounds = 300,
  eta = 0.02,
  max.depth = 20,
  colsample_bytree = 0.1,
  subsample = 0.95,
  objective = "binary:logistic",
  watchlist = watch,
  eval_metric = "error"
)

m3 <-
  xgboost(
    data = dtrain,
    max.depth = 20,
    eta = 1,
    nthread = 4,
    nround = 45,
    objective = "binary:logistic"
  )

xgbpred <- predict(m1, dvalid)
xgbpred <- ifelse(xgbpred > 0.5, 1, 0)

# length(xgbpred)
# table(xgbpred)
# str(xgbpred)
#
# length(as.numeric(trainingLabel[test]))
# table(trainingLabel[test])
# str(as.numeric(trainingLabel[test]))

unique_results <- union(xgbpred, as.numeric(trainingLabel[test]))

comparison_table <-
  table(factor(xgbpred, unique_results),
        factor(as.numeric(trainingLabel[test]), unique_results))

confusionMatrix(comparison_table)

# # Errors: "the data cannot have more levels than the reference"
# confusionMatrix(xgbpred, length(as.numeric(trainingLabel[test])))
