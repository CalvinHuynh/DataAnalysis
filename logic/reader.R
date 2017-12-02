Sys.setenv(lang = "en")

library(config)
library(qdap)
library(tm)
config <- config::get(file = "config.yml")

trainData <- read.table(config$data, sep = '\t',header = TRUE)

term_count <- freq_terms(trainData$review, 10)
plot(term_count)

trainDataSource <- VectorSource(trainData$review)