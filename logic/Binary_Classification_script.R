
library(RTextTools)
library(plyr)
source("logic/reader.R")

imdbSet <- readFirstDataset()
head(imdbSet)

# machiganSet <- read.csv(file = "machiganTraining2.csv", sep = ';', header = TRUE)
# 
# #### 10 own written review
# ownWrittenReview =  rbind(
#   c('1', 'Surprisingly really not terrible'), #This Charming Man (2006)
#   c('1', 'Powerfull, stunning and exceptionally crafted'), #Dunkirk (2017)
#   c('1', 'ROFL'ED on the floor), #Deadpool (2016)
#   c('1', 'Loved the atmosphere of the movie'), #Ju-on: The Grudge (2002)
#   c('1', 'Space Jam has the potential to be a 5 star movie yet loses a star because Daffy Duck is underutilized'), #Space Jam (1996)
#   
#   c('0', 'There's nothing 'good' about the 'fellas' in this movie), #Goodfellas (1990)
#   c('0', 'I watched this to review for my GRANDAUGHTER age 8. It started out hopeful that it was going to be good. Then it kept getting darker and then "nude" animals entered into the movie'), #Zootopia (2016)
#   c('0', 'The film is a bit unrealistic since the existence of dinosaurs has not been proven. There may be some bones but these are easy enough to fake'), #Jurrasic World (2015)
#   c('0', 'NOT FOR KIDS!!!! PORNO MOVIE using grocery items. OMG'), #Sausage Party (2016)
#   c('0', 'oh yeah a boat this big could never sink') #The Titanic (1997)
# )
# 
# ownWrittenReviewDataFrame <- data.frame(ownWrittenReview)
# colnames(ownWrittenReviewDataFrame) <- c("sentiment", "review")

#### merging dataframes
# allData.DataFrame <- rbind.fill(ownWrittenReviewDataFrame, machiganSet, imdbSet)
allData.DataFrame <- imdbSet

allData.DataFrame$id <- NULL

####cleaning
url_pattern <- "...http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

allData.DataFrame$review <- gsub(url_pattern, "", allData.DataFrame$review)
allData.DataFrame$review <- tolower(allData.DataFrame$review)
allData.DataFrame$review <- gsub("[[:punct:][:blank:]]+", " ", allData.DataFrame$review) 
allData.DataFrame$review <- gsub("[[:digit:]]", "", allData.DataFrame$review)
allData.DataFrame <- na.omit(allData.DataFrame)

####splitting data and checking distibution
ownWrittenReviewDataFrame <- allData.DataFrame[1:10,] #10 reviews that was binded into all data
internetReviewDataFrame <- allData.DataFrame[11:nrow(allData.DataFrame),] #rows that were from the internet

set.seed(4242)
trainingData <- c(ownWrittenReviewDataFrame$review, 
                   internetReviewDataFrame$review[sample(nrow(internetReviewDataFrame), 500)])
set.seed(4242)
trainingLabel <- c(ownWrittenReviewDataFrame$sentiment,
                   internetReviewDataFrame$sentiment[sample(nrow(internetReviewDataFrame), 500)])

size = floor(0.25 * length(trainingData)) ## first 25% for test, rest 75% for train
sizeNext = size + 1
total = length(trainingData)

length(trainingData)
count(trainingLabel)
count(trainingLabel[1:size])
count(trainingLabel[sizeNext:total])

####Creating matrix
matrix <- create_matrix(trainingData, language="english", removeNumbers=TRUE, 
                        removePunctuation=TRUE, removeStopwords=TRUE, stemWords = TRUE,
                        stripWhitespace=TRUE, toLower=TRUE)

####Creating model
container <- create_container(matrix,as.factor(t(trainingLabel)),trainSize = sizeNext:total, 
                              testSize = 1:size,virgin=FALSE)

model <- train_models(container, "RF") #for random forest
results <- classify_models(container, model)
head(results, 5)

analytics <- create_analytics(container, results)
precisionSummary <- create_precisionRecallSummary(container, results, b_value = 1)
precisionSummary

####Cross validation for testing overfitting and validation
####Data too big if nrow >= 650
options(expressions = 5e5)
crossValidation <- cross_validate(container, 3, algorithm = "RF", seed = 123)
?cross_validate
crossValidation$meanAccuracy
