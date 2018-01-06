library(tm)
library(RTextTools)
library(e1071)
library(dplyr)
library(caret)
source("logic/reader.R")

# section 1 ---------------------------------------------------------------
df <- readFirstDataset()

glimpse(df)
str(df)

df$review <- as.character(df$review)
df$id <- NULL

set.seed(2)
df <- df[sample(nrow(df)), ]
df <- df[sample(nrow(df)), ]
#glimpse(df)

# Convert the 'class' variable from character to factor.
#df$class <- as.factor(df$class)
df$sentiment <- as.factor(df$sentiment)

#corpus <- Corpus(VectorSource(df$text))
corpus <- Corpus(VectorSource(df$review))
# Inspect the corpus
corpus


inspect(corpus[1:3])


# clean data --------------------------------------------------------------

# Use dplyr's  %>% (pipe) utility to do this neatly.
corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corpus.clean)
# Inspect the dtm
inspect(dtm)


# partition data ----------------------------------------------------------
floor(0.7 * nrow(df))

split30 <- paste0(1:1000)
split70 <- paste0(1001:8693)

df.train <- df[split30,]
df.test <- df[split70,]

dtm.train <- dtm[split30,]
dtm.test <- dtm[split70,]

corpus.clean.train <- corpus.clean[split30]
corpus.clean.test <- corpus.clean[split70]

# Feature selection
dim(dtm.train)

fivefreq <- findFreqTerms(dtm.train, 9)
length((fivefreq))
## [1] 12144

# Use only 5 most frequent words (fivefreq) to build the DTM

dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))

dim(dtm.train.nb)

## [1]  1500 12144

dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))

dim(dtm.train.nb)
## [1]  1500 12144


# Naive Bayes -------------------------------------------------------------

# Function to convert the word frequencies to yes (presence) and no (absence) labels
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}
dim(dtm.test.nb)
str(dtm.test.nb)
colnames(dtm.test.nb)
# Apply the convert_count function to get final training and testing DTMs
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)

# Train the model
# Train the classifier
system.time( classifier <- naiveBayes(trainNB, df.train$sentiment, laplace = 1) )

# Use the NB classifier we built to make predictions on the test set.
system.time( pred <- predict(classifier, newdata=testNB) )

# Create a truth table by tabulating the predicted class labels with the actual class labels 
table("Predictions"= pred,  "Actual" = df.test$sentiment )

# Prepare the confusion matrix
conf.mat <- confusionMatrix(pred, df.test$sentiment)

conf.mat

conf.mat$byClass

conf.mat$overall

conf.mat$overall['Accuracy']
