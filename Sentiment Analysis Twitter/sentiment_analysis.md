Sentiment Analysis
================
Marco Giordano
1/5/2021

## Introduction

My final project for Text Analysis and Spatial Data last year involved a
sentiment analysis of UK politicians’ tweets to uncover the main
sentiments behind the nation. Actually, this is a fixed version that
includes tidytext support due to its awesome functions and integration
within tidyverse. Moreover, conversions to document-term matrix (tm
package) is also allowed and this is great for our analysis!

I admit that I am not an expert in UK politics, but this was a good
chance to train myself on something that was a bit unusual and to start
practicing text mining and sentiment analysis. Data were obtained from
the Twitter API by querying it in R with the aid of the “rtweet”
package, which I suggest you to use. I skipped this part for the moment,
but I’ll eventually cover it although it is very very simple and can be
gone through with a simple tutorial on Google.

We will start with basic data cleaning for text data and then show
visually some concepts to make some conclusions about what we found. For
instance, we may like to understand what are the main sentiments across
the period to understand how politicians communicated during this very
challenging period. Furthermore, we want an high-level overview of the
main topics discussed by those politicians on Twitter, since I don’t
know that much about politics!

``` r
library(rtweet)
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.0 ──

    ## ✓ ggplot2 3.3.2     ✓ purrr   0.3.4
    ## ✓ tibble  3.0.4     ✓ dplyr   1.0.2
    ## ✓ tidyr   1.1.2     ✓ stringr 1.4.0
    ## ✓ readr   1.4.0     ✓ forcats 0.5.0

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## x dplyr::filter()  masks stats::filter()
    ## x purrr::flatten() masks rtweet::flatten()
    ## x dplyr::lag()     masks stats::lag()

``` r
library(RColorBrewer)
library(stringr)
library(tidytext)
library(broom)
library(scales)
```

    ## 
    ## Attaching package: 'scales'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     discard

    ## The following object is masked from 'package:readr':
    ## 
    ##     col_factor

``` r
library(tm)
```

    ## Loading required package: NLP

    ## 
    ## Attaching package: 'NLP'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     annotate

``` r
library(topicmodels)
library(stm)
```

    ## stm v1.3.6 successfully loaded. See ?stm for help. 
    ##  Papers, resources, and other materials at structuraltopicmodel.com

``` r
library(caret)
```

    ## Loading required package: lattice

    ## 
    ## Attaching package: 'lattice'

    ## The following object is masked from 'package:stm':
    ## 
    ##     cloud

    ## 
    ## Attaching package: 'caret'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
library(mlbench)
```

``` r
pol <- readRDS('sentiment_pol_uk')

pol <- pol %>% select(screen_name, created_at, text) %>% filter(created_at >= as.Date("2020-01-28"))
```

Our cleaned dataset is made up by 15260 rows and 3 features, which
should be fine enough for what we have to do. A lot of features won’t be
necessary at all, and the same goes for old tweets before the beginning
of 2020, because they are out of scope. I decided to collect data from 8
Twitter accounts for the Right (7 politicians and 1 party) and 5 from
Left, wherein one is the Labour Party account. I think this number is
good enough for our kind of analysis and this should not pose a class
imbalance problem later. Actually, Left posted more than Right but the
difference is not that large. Consider that I just picked up the most
popular accounts in UK for politics, so this is not a thorough analysis
of all the political scene.

Also, remember that what said on Twitter may not reflect what was said
on other social media or channels and I think that this is the most
important point to interpret the results. It is also a possibility that
some of the accounts used are managed by ghostwriters.

After importing the necessary libraries we can load our data and select
only the meaningful variables. We just need the screen name of the
politician, the data of creation and obviously, the actual text of each
tweet. Since we are interested in getting information about COVID-19, we
may filter starting from 28 January 2020, that is the first day since
when airports in UK closed. This is just a quick way to filter out
non-relevant results, while keeping our data set relatively manageable
on a local machine. The last date recorded is 17 November 2020.

You can also scrape yourself the tweets, as explained below:

``` r
pol <- get_timeline(c("BorisJohnson", "theresa_may", "David_Cameron", "jeremycorbyn", "Nigel_Farage", "RishiSunak", "SadiqKhan", "DominicRaab", "NicolaSturgeon", "MattHancock", "Ed_Miliband", "UKLabour", "Conservatives"), n =3200, includeRts = F)

saveRDS(pol, file = 'sentiment_pol_uk')
```

## Data cleaning and preprocessing

The next step is to clean our data to prepare them for the algorithms we
will use. Although the data extracted from the Twitter API may seem
already fine and cleaned, it is not the case since we should first
adjust rows with empty text and remove stopwords/punctuation and Twitter
parameters like &amp. Recall that when dealing with textual data you
must always be sure to check for this kind of factors, otherwise you
risk inflating your models with a bunch of useless and inexpressive
words.

``` r
reg <- "([^A-Za-z\\d#@']|'(?![A-Za-z\\d#@]))"
clean_text <- function(df) {
  df <- df %>%
      filter(!str_detect(text, '^"')) %>%
      mutate(text  = str_replace_all(text, "https://t.co/[A-Za-z\\d]+|&amp;", "")) %>%
      unnest_tokens(word, text, token = "regex", pattern = reg, collapse = FALSE) %>%
      filter(!word %in% stop_words$word, str_detect(word, "[a-z]"))
}

clean_pol <- clean_text(pol)
```

A basic function that cleans your data and then applies tokenization is
perfect for what we have to do. We are now interested in seeking
information about single words, so unnesting tokens by word is the
correct way to proceed.

As seen below, the words that occured the most during the period mainly
involve public healthcare and ideas related to protecting and supporting
people. Please, consider that the word coronavirus isn’t actually
duplicated but it is reported twice since one is an hashtag. That means
that it was the most used hastag to communicate within tweets, as it is
the only one in the top 10. There is nothing particularly strange or
surprising at the moment, we can just notice that there are words
related to healthcare and more generic one like UK and London.

``` r
clean_pol %>%
  count(word, sort = T) %>%
  head(10) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_bar(stat = 'identity', fill = 'blue') +
  coord_flip() +
  theme_minimal()
```

![](sentiment_analysis_files/figure-gfm/Top%2010%20used%20words-1.png)<!-- -->

A more interesting overview may be given by splitting the
politicians/parties into two groups, Right and Left. Our hypothesis is
that their communication has to be somehow different due to their
different ideologies. We can use the log odds ratio to show which words
are more likely to be tweeted by one of the two political parties.

``` r
clean_pol <- clean_pol %>%
  mutate(group = case_when(
    screen_name %in% c('BorisJohnson', 'Nigel_Farage', 'David_Cameron', 'RishiSunak', 'DominicRaab', 'MattHancock', 'theresa_may', 'Conservatives') ~ 'Right',
    screen_name %in% c('UKLabour', 'NicolaSturgeon', 'jeremycorbyn', 'SadiqKhan', 'Ed_Miliband') ~ 'Left'
  ))



right_left_ratio <- clean_pol %>%
  count(word, group) %>%
  filter(sum(n) >= 5) %>%
  spread(group, n, fill = 0) %>%
  ungroup() %>%
  mutate_each(funs((. +1) / sum(. +1)), -word) %>%
  mutate(logratio = log2(Right / Left)) %>%
  arrange(desc(logratio))

right_left_ratio %>%
  group_by(logratio > 0) %>%
  top_n(15, abs(logratio)) %>%
  ungroup() %>%
  mutate(word = reorder(word, logratio)) %>%
  ggplot(aes(word, logratio, fill = logratio < 0)) +
  geom_bar(stat = 'identity') +
  coord_flip() +
  ylab('Right/Left log ratio') +
  scale_fill_manual(name='', labels = c('Right', 'Left'), values = c('lightblue', 'red'))
```

![](sentiment_analysis_files/figure-gfm/Log%20ratio%20of%20Right%20and%20Left-1.png)<!-- -->

It is pretty interesting to notice that both parties tend to mention
candidates within their political spectrum instead of opponents, which
reminds us of the “echo chamber” phenomenon. We can also notice how the
right is more likely to tweet about East and labour. On contrary, the
Left is more likely to tweet about London and related entities, such as
his Mayor.

Now that we have a clear overview of the single words used and by which
party, we can move on to something more complex. We can employ a very
simple statistical model to compare sentiments across the two parties is
a Poisson Regression, since we are dealing with count data. In this
case, the dependent variable Y is the count of words within a group
(sentiment) for each of the two parties. The covariate is represented by
the political party, which may be expressed as an indicator variable.
So, to measure the sentiment of both the parties, we can just count the
number of words found in each category:

``` r
nrc <- get_sentiments('nrc') 

groups <- clean_pol %>%
  group_by(group) %>%
  mutate(total_words = n()) %>%
  ungroup() %>%
  distinct(screen_name, group, total_words)

by_group_sentiment <- clean_pol %>%
  inner_join(nrc, by = 'word') %>%
  count(sentiment, screen_name) %>%
  ungroup() %>%
  complete(sentiment, screen_name, fill = list(n = 0)) %>%
  inner_join(groups) %>%
  group_by(group, sentiment, total_words) %>%
  summarise(words = sum(n)) %>%
  ungroup()
```

    ## Joining, by = "screen_name"

    ## `summarise()` regrouping output by 'group', 'sentiment' (override with `.groups` argument)

``` r
head(by_group_sentiment)
```

    ## # A tibble: 6 x 4
    ##   group sentiment    total_words words
    ##   <chr> <chr>              <int> <dbl>
    ## 1 Left  anger             113221  3380
    ## 2 Left  anticipation      113221  6961
    ## 3 Left  disgust           113221  1755
    ## 4 Left  fear              113221  6950
    ## 5 Left  joy               113221  4597
    ## 6 Left  negative          113221  9998

We may want to measure, for instance, how much more likely the Right is
to use some emotional words relative to the Left. A Poisson test may be
a useful tool to assess the aforementioned difference:

``` r
sentiment_differences <- by_group_sentiment %>%
  arrange(desc(group)) %>%
  group_by(sentiment) %>%
  do(tidy(poisson.test(.$words, .$total_words)))

sentiment_differences
```

    ## # A tibble: 10 x 9
    ## # Groups:   sentiment [10]
    ##    sentiment estimate statistic   p.value parameter conf.low conf.high method
    ##    <chr>        <dbl>     <dbl>     <dbl>     <dbl>    <dbl>     <dbl> <chr> 
    ##  1 anger        0.877      2234 1.48e-  6     2412.    0.831     0.926 Compa…
    ##  2 anticipa…    0.922      4834 1.23e-  5     5069.    0.888     0.956 Compa…
    ##  3 disgust      0.789      1043 9.56e- 10     1202.    0.730     0.852 Compa…
    ##  4 fear         0.743      3891 5.97e- 51     4659.    0.714     0.773 Compa…
    ##  5 joy          1.04       3599 8.78e-  2     3522.    0.994     1.09  Compa…
    ##  6 negative     0.696      5247 3.49e-103     6551.    0.673     0.720 Compa…
    ##  7 positive     1.10      12115 9.78e- 14    11511.    1.07      1.12  Compa…
    ##  8 sadness      0.716      2184 2.07e- 37     2679.    0.679     0.754 Compa…
    ##  9 surprise     1.01       1613 8.69e-  1     1608.    0.942     1.07  Compa…
    ## 10 trust        1.05       7323 1.89e-  3     7124.    1.02      1.08  Compa…
    ## # … with 1 more variable: alternative <chr>

Now we are ready to plot what we got before with a 95% confidence
interval:

``` r
sentiment_differences %>%
  ungroup() %>%
  mutate(sentiment = reorder(sentiment, estimate)) %>%
  mutate_each(funs(. -1), estimate, conf.low, conf.high) %>%
  ggplot(aes(estimate, sentiment)) +
  geom_point() +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high)) +
  scale_x_continuous(labels = percent_format()) +
  labs(x = "% increase in Right relative to Left", y = "Sentiment" )
```

![](sentiment_analysis_files/figure-gfm/Confidence%20Interval%20of%20Poisson%20Test-1.png)<!-- -->
We can notice that there for positive sentiments there is almost none
significant statistical difference from Right to Left, but the same
doesn’t hold true for the most negative sentiments, like disgust, fear
and sadness!

But what caused this difference in sentiment? We can visualize the words
with the largest changes within each category:

``` r
right_left_ratio %>%
  inner_join(nrc, by = 'word') %>%
  filter(!sentiment %in% c("positive", "negative")) %>%
  mutate(sentiment = reorder(sentiment, -logratio),
         word = reorder(word, -logratio)) %>%
  group_by(sentiment) %>%
  top_n(10, abs(logratio)) %>%
  ungroup() %>%
  ggplot(aes(word, logratio, fill = logratio < 0)) +
  facet_wrap(~ sentiment, scales = 'free', nrow = 2) +
  geom_bar(stat = 'identity') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "", y = "Right / Left log ratio") +
  scale_fill_manual(name='', labels = c('Right', 'Left'), values = c('lightblue', 'red'))
```

![](sentiment_analysis_files/figure-gfm/Largest%20changes%20for%20sentiment%20according%20to%20the%202%20parties-1.png)<!-- -->

We can notice that lot of words annotated with positive sentiments are
common in the Right.

The final step is to find a list of topics that can describe what we
told. We believe that 5 is a good enough number given the size of our
dataset.

``` r
clean_pol_2 <- clean_pol %>%
  filter(created_at > '2020-01-01') %>%
  count(screen_name, word, sort = T) %>%
  ungroup() 

pol_dtm <- clean_pol_2 %>% cast_dtm(screen_name, word, n)


ap_lda <- LDA(pol_dtm, k = 5, control = list(seed = 1234))
ap_topics <- tidy(ap_lda, matrix = "beta")

ap_top_terms <- ap_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

ap_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()
```

![](sentiment_analysis_files/figure-gfm/The%205%20main%20topics%20found-1.png)<!-- -->

The first topic is all about support from the government for the
COVID-19 crisis and for the labour market, The second one is mainly
about the security of the country due to COVID-19. The third one takes a
totally different flavor and expresses that there were some interest in
London and Sadiq Khan, the mayor of the city. Number 4 is related to
Scotland and health, as we can see from the words, mentioning Nicola
Sturgeon and the account of the Scotland Government. The last topic is
about work/business, it is quite similar to the first one but more
focused on economics than all the others.

After having checked the topics, we can say that 5 was indeed a good
enough number, that allowed us to uncover the main topics discussed
since the COVID-19 outbreak. What we can infer is that the labour market
was clealry discussed by UK politicians, as well as related
restrictions. Strangely, no word like lockdown was found with an high
frequency, barring home which probably refers to the concept of
literally staying at home to prevent contagion.

## Conclusions and main findings

This analysis highlighted how the most common words throughout the
period involved the labour market and public healthcare, nothing
surprising we could say. What was more interesting was that I actually
had to search for some info on this domain to understand some
information, such as the two different stances of the parties. This
analysis actually confirmes the presence of an echo chamber in both
parties, sinde the most likely words for both include some of their
politicians. I expected a more negative result in terms of sentiment but
it was actually the other way around!

This report will be continued and expanded later on, but I am pretty
satisfied by the outcome of this analysis.
