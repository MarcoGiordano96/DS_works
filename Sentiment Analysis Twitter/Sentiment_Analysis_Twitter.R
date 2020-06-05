#----------------------------IMPORT LIBRARIES-----------------------------------------------------------------
 
library(rtweet)
library(tidyverse)
library(ggplot2)
library(wordcloud)
library(RColorBrewer)
library(quanteda)
library(stringr)
library(readtext)
library(tidytext)
library(syuzhet)
library(quanteda.textmodels)
library(stm)
library(SentimentAnalysis)
library(textclean)
library(tm)
library(caret)

#-----------------------------AUTHENTICATE YOURSELF TO TWITTER---------------------------------

#authentication
api_key <- "value"
api_secret_key <- "value"

token <- create_token(
  app = "app_name",
  consumer_key = api_key,
  consumer_secret = api_secret_key)

token

#--------------------------TWITTER API CALLS---------------------------------------------------

#retrieve last 3200 tweets from UK politicians
bj <- get_timeline("BorisJohnson", n = 3200, includeRts = FALSE)
# 
#Theresa May
m <- get_timeline("theresa_may", n = 3200, includeRts = FALSE)
# 
#David Cameron
dc <- get_timeline("David_Cameron", n = 3200, includeRts = FALSE)
# 
#Jeremy Corbin
jc <- get_timeline("jeremycorbyn", n = 3200, includeRts = FALSE)
# 
#Nigel Farage
nf <- get_timeline("Nigel_Farage", n = 3200, includeRts = FALSE)
# 
#Rishi Sunak
rs <- get_timeline("RishiSunak", n = 3200, includeRts = FALSE)
# 
#Sadiq Khan
sk <- get_timeline("SadiqKhan",n = 3200, includeRts = FALSE)
# 
# 
#Dominic Raab
dr <- get_timeline("DominicRaab",n = 3200, includeRts = FALSE)
# 
#Nicola Sturgeon
ns <- get_timeline("NicolaSturgeon",n = 3200, includeRts = FALSE)
# 
#Matt Hancock
mh <- get_timeline("MattHancock",n = 3200, includeRts = FALSE)
# 
#Ed Miliband
em <- get_timeline("Ed_Miliband",n = 3200, includeRts = FALSE)

#lab_party <- get_timeline("UKLabour", n = 3200, includeRts = FALSE)

cons <- get_timeline("Conservatives", n = 3200, includeRts = FALSE)



#------------------------DATASET CREATION------------------------------------------------------
#merge all tweet datasets into one
pol <- bind_rows(bj,tm, dc, jc, nf, rs, sk, dr, ns, mh, em, lab_party, cons)

#select meaningful variables
pol <- pol %>% select(screen_name, created_at, text)

#build 2 different datasets (before and after covid outbreak)

#one for covid
pol_covid <- pol %>%
  filter(pol$created_at >= as.Date("2020-01-28"))


#plot by date frequency
pol_covid %>% group_by(created_at) %>% summarise(n = length(created_at)) %>%
  ggplot(aes(x = created_at, y = n)) + geom_point() +
  xlab("Month") 
   


#the other for before covid
pol_bef_covid <- pol %>%
  filter(pol$created_at >= as.Date("2019-01-01") & pol$created_at < as.Date("2020-01-28"))



#----------------------------------DATA CLEANING-----------------------------------------------

#data cleaning with RegEx (for covid)
pol_covid$text <- gsub("t\\.co", " ", pol_covid$text)
pol_covid$text <- gsub("\\&amp\\;", " ", pol_covid$text)
pol_covid$text <- gsub('[[:punct:] ]+',' ', pol_covid$text)
pol_covid$text <- gsub("http\\S+\\s*", "", pol_covid$text)
pol_covid$text <- tolower(pol_covid$text)
pol_covid$text <- gsub("[[:digit:]]+","", pol_covid$text)
pol_covid$text <- gsub("([一-龯]))","", pol_covid$text)
pol_covid$text <- replace_emoji(pol_covid$text, emoticon_dt = lexicon::emojis_sentiment)
pol_covid$text <- gsub("\bt\b", "", pol_covid$text)
pol_covid$text <- gsub("<.*?>", " ", pol_covid$text)
pol_covid$text <- removeWords(pol_covid$text, stopwords("english"))


#data cleaning with RegEx (for before covid)
pol_bef_covid$text <- gsub("t\\.co", " ", pol_bef_covid$text)
pol_bef_covid$text <- gsub("\\&amp\\;", " ", pol_bef_covid$text)
pol_bef_covid$text <- gsub("[[:punct:]]+"," ", pol_bef_covid$text) 
pol_bef_covid$text <- gsub("http\\S+\\s*", "", pol_bef_covid$text) 
pol_bef_covid$text <- tolower(pol_bef_covid$text)
pol_bef_covid$text <- gsub("[[:digit:]]+","", pol_bef_covid$text) 
pol_bef_covid$text <- gsub("([一-龯]))","", pol_bef_covid$text) 
pol_bef_covid$text <- replace_emoji(pol_bef_covid$text, emoticon_dt = lexicon::emojis_sentiment) 
pol_bef_covid$text <- gsub("\bt\b", "", pol_bef_covid$text)
pol_bef_covid$text <- gsub("<.*?>", " ", pol_bef_covid$text)
pol_bef_covid$text <- removeWords(pol_bef_covid$text, stopwords("english"))
   


#----------------------SENTIMENT ANALYSIS-------------------------------------------------------

#-----------AFTER COVID------------  
  
#perform a sentiment analysis on text to get mean sentiment scores (covid)
sentiment_pol <- analyzeSentiment(pol_covid$text)


#select columns with computed scores
sentiment_pol <- select(sentiment_pol, SentimentGI, SentimentHE, 
                        SentimentLM, SentimentQDAP)


#compute average of every row to get overall score
sentiment_pol <- mutate(sentiment_pol, mean_sentiment = rowMeans(sentiment_pol))

#select the column you need (mean_sentiment)
sentiment_pol <- sentiment_pol %>% select(mean_sentiment)

ggplot(data = sentiment_pol, aes(mean_sentiment)) +
  geom_histogram() +
  labs(title = "Mean sentiment score after COVID-19 outbreak")

#add sentiment score to original datasets
pol_covid <- cbind(pol_covid, sentiment_pol)

ggplot(data = pol_covid, aes(x = mean_sentiment)) +
  geom_histogram() +
  labs(title = "Mean sentiment score after COVID-19 outbreak", y = "frequency")


#break sentiment score into classes
pol_covid <-pol_covid %>%
  mutate(sentiment = case_when(
    mean_sentiment < 0 ~ "Negative",
    mean_sentiment >= 0 ~ "Positive"))


#-----------------BEFORE COVID---------


#perform a sentiment analysis on text to get mean sentiment scores (before covid)
sentiment_pol_bef <- analyzeSentiment(pol_bef_covid$text)

#select columns with computed scores
sentiment_pol_bef <- select(sentiment_pol_bef, SentimentGI, SentimentHE, 
                        SentimentLM, SentimentQDAP)


#compute average of every row to get overall score
sentiment_pol_bef <- mutate(sentiment_pol_bef, mean_sentiment = rowMeans(sentiment_pol_bef))


#select the column you need (mean_sentiment)
sentiment_pol_bef <- sentiment_pol_bef %>% select(mean_sentiment)


ggplot(data = sentiment_pol_bef, aes(x = mean_sentiment)) +
  geom_histogram() +
  labs(title = "Mean sentiment score before COVID-19 outbreak", y = "frequency")


#--------------------TEXT ANALYSIS---------------------------------------------------

#create corpus
pol_covid_corpus <- corpus(pol_covid, text_field = "text")
kwic(pol_covid_corpus, phrase("coronavirus")) %>% head(50)
kwic(pol_covid_corpus, phrase("health")) %>% head()
kwic(pol_covid_corpus, phrase("coronavirus")) %>% head(10) %>% textplot_xray()



#create tokens
pol_covid_tokens <- tokens(pol_covid_corpus, "fasterword")


#data cleaning/preprocessing at token level
pol_covid_tokens <- tokens_wordstem(pol_covid_tokens, language = ("en"))

#create dfm
pol_covid_dfm <- dfm(pol_covid_tokens)
topfeatures(pol_covid_dfm)


#wordcloud visualization
set.seed(132)
textplot_wordcloud(pol_covid_dfm, max_words = 100)


#Naives Bayes (NB)

#build train dataset (~80% of total)
pol_covid_dfm_train <- dfm_sample(pol_covid_dfm, 6182)

#build test dataset (other ~20% remaining, no overlap!)
pol_covid_dfm_test <- dfm_subset(pol_covid_dfm, !(docnames(pol_covid_dfm) %in% docnames(pol_covid_dfm_train)))

#adjusting samples
pol_covid_dfm_train <- pol_covid_dfm_train %>% dfm_trim(1)
pol_covid_dfm_test <- dfm_match(pol_covid_dfm_test, featnames(pol_covid_dfm_train))

#training the model
nb_model_covid <- textmodel_nb(pol_covid_dfm_train, y = docvars(pol_covid_dfm_train, "sentiment"))

#predicting test data
test_predictions_covid <- predict(nb_model_covid, newdata = pol_covid_dfm_test)
head(test_predictions_covid)

#confusion matrix
confmat_covid <- confusionMatrix(as.factor(docvars(pol_covid_dfm_test, "sentiment")), test_predictions_covid)

actual_class_c <- docvars(pol_covid_dfm_test, "sentiment")
table_class_c <- table(actual_class_c, test_predictions_covid)
table_class_c


#plot to visualize data
table_class_c <- as.data.frame(table_class_c)

ggplot(data= table_class_c, aes(x = test_predictions_covid, y = actual_class_c)) +
  geom_tile(aes(fill = Freq)) + scale_fill_gradient(high = "black",
                                                    low = "white", name = "Value") + xlab("Predicted Class") +
  ylab("Actual Class") + scale_y_discrete(expand = c(0,0)) +
  theme_classic() + theme(axis.text.x = element_text(angle = 90)) +
  theme(axis.text.x = element_text(vjust = 0.4))

# LDA method

#prune dataset
pruned_pol_dfm <- pol_covid_dfm %>%
  dfm_trim(min_termfreq = 20,
           min_docfreq = 5,
           verbose = T)

#simplest form of execution

pol_stm_1 <- stm(pruned_pol_dfm, K = 20)

#list of topics
labelTopics(pol_stm_1)

#interpreting results
par(bty="n", col="grey40",lwd=5)
plot(pol_stm_1, topics = 1:5, type = "summary", xlim = c(0,0.2))

pol_stm_2 <- stm(pruned_pol_dfm, K = 10, content = ~text, data = docvars(pruned_pol_dfm))



#----------------FINAL SENTIMENT ANALYSIS-------------------------
#we will be using this dictionary to perform sentiment analysis
dictionary(data_dictionary_LSD2015)
pol_covid_dfm_dic <- dfm_lookup(pol_covid_dfm, data_dictionary_LSD2015)

#check sentiment scores for accounts
pol_covid_dfm_dic %>%
  dfm_group("screen_name")


#sentiment analysis with syuzhet for after covid outbreak
pol_covid_text <- pol_covid$text
mysentiment_pol <- get_nrc_sentiment((pol_covid_text))
sentscores_pol <- data.frame(colSums(mysentiment_pol[,]))

names(sentscores_pol) <- "Score"
sentscores_pol <- cbind("sentiment"=rownames(sentscores_pol), sentscores_pol)
rownames(sentscores_pol) <- NULL

#plot it!
ggplot(data = sentscores_pol, aes(x=sentiment, y = Score)) +
  geom_bar(aes(fill= sentiment), stat = "identity") +
  xlab("Sentiments") + ylab("scores")+ggtitle("Sentiments of politicians")



