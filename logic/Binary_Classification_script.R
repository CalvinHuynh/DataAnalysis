Sys.setenv(lang = "en")
Sys.setlocale(category = "LC_ALL", locale = "English_United States.1252")

library(RTextTools)
library(plyr)
library(dplyr)
library(tm)
source("logic/reader.R")

imdbSet <- readFirstDataset()

# retrieves the amazon movie review data
# amazonSet <- shuffleDataframe(readAmazonReviews())

# select the 10k rows from the amazon set
# amazonSet <- amazonSet[1:10000,]

# Written reviews (might have been insipred from other sources)
posWrittenReviews <- rbind(
  c('1', 'Surprisingly really not terrible'), #This Charming Man (2006)
  c('1', 'Powerfull, stunning and exceptionally crafted'), #Dunkirk (2017)
  c('1', 'ROFLED on the floor'), #Deadpool (2016)
  c('1', 'Loved the atmosphere of the movie'), #Ju-on: The Grudge (2002)
  c('1', 'Space Jam has the potential to be a 5 star movie yet loses a star because Daffy Duck is underutilized') #Space Jam (1996)
)

negWrittenReviews <- rbind(
  c('0', 'Theres nothing good about the fellas in this movie'), #Goodfellas (1990)
  c('0', 'I watched this to review for my GRANDAUGHTER age 8. It started out hopeful that it was going to be good. Then it kept getting darker and then "nude" animals entered into the movie'),#Zootopia (2016)
  c('0', 'The film is a bit unrealistic since the existence of dinosaurs has not been proven. There may be some bones but these are easy enough to fake'), #Jurrasic World (2015)
  c('0', 'NOT FOR KIDS!!!! PORNO MOVIE using grocery items. OMG'), #Sausage Party (2016)
  c('0', 'oh yeah a boat this big could never sink') #The Titanic (1997)
)

writtenTestReviewsDf <-
  data.frame(rbind(posWrittenReviews, negWrittenReviews))
colnames(writtenTestReviewsDf) <- c("sentiment", "review")

# merging dataframes
# combinedDf <- rbind.fill(writtenTestReviewsDf, amazonSet, imdbSet)
combinedDf <- imdbSet

# remove unused column
combinedDf$id <- NULL

####splitting data and checking distibution
ownWrittenReviewDataFrame <- combinedDf[1:10,]
otherReviewDf <- combinedDf[11:nrow(combinedDf),]

set.seed(4242)
trainingData <- c(ownWrittenReviewDataFrame$review,
                  otherReviewDf$review[sample(nrow(otherReviewDf), 500)])

trainingLabel <- c(ownWrittenReviewDataFrame$sentiment,
                   otherReviewDf$sentiment[sample(nrow(otherReviewDf), 500)])

# trainingData <- otherReviewDf$review[sample(nrow(otherReviewDf), 500)]
# 
# trainingLabel <- otherReviewDf$sentiment[sample(nrow(otherReviewDf), 500)]

# split data 25% test, 75% train
size = floor(0.25 * length(trainingData)) 
sizeNext = size + 1
total = length(trainingData)
length(trainingData)
count(trainingLabel)
count(trainingLabel[1:size])
count(trainingLabel[sizeNext:total])

# Creating matrix with cleaning
matrix <-
  create_matrix(
    trainingData,
    language = "english",
    removeNumbers = TRUE,
    removePunctuation = TRUE,
    removeStopwords = TRUE,
    stemWords = TRUE,
    stripWhitespace = TRUE,
    toLower = TRUE,
    weighting=tm::weightTfIdf
  )

# Creating model
container <-
  create_container(
    matrix,
    as.factor(t(trainingLabel)),
    trainSize = sizeNext:total,
    testSize = 1:size,
    virgin = FALSE
  )

# Train model using random forest
model <- train_models(container, "RF") 

# Predict sentiment using random forest
results <- classify_models(container, model)
head(results, 5)

analytics <- create_analytics(container, results)
precisionSummary <-
  create_precisionRecallSummary(container, results, b_value = 1)
precisionSummary

# ####Cross validation for testing overfitting and validation
# ####Data too big if nrow >= 650
# options(expressions = 5e5)
# crossValidation <-
#   cross_validate(container, 3, algorithm = "RF", seed = 123)
# ? cross_validate
# crossValidation$meanAccuracy
