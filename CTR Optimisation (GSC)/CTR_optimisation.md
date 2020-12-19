CTR\_optimization
================
Marco Giordano
12/18/2020

## Introduction \[Goals and Why\]

This project is just an example of how it is possible to predict the CTR
of a website by using Google Search Console data, extracted via Google
API. The presented example can be considered as a “toy model”, in the
sense that it is just a demonstration to show the method on a relatively
small dataset. The main reason to predict CTR via Machine Learning
techniques is because SEO Specialists usually decide what to optimize by
looking at pages with high Impressions and low CTR. However, this method
is obviously biased, since CTR varies with position\!

I try to test some popular methods to predict the CTR, given a small
sample of 25K rows from 3 months from Google Search Console, in order to
create a better way to make decisions on what could be optimized.

## Descriptive Statistics

As we can quickly notice, clicks and impressions have a high range, due
to the effect of most popular articles, featuring very right-skewed
distributions too. Data from CTR have the correct values, because the
maximum value is equal to 1 and the minimum is not below 0. Another
large gap may be found by taking a look at the position, which is
normal, since we expect that there are some outliers that rank pretty
bad for some short-head query, probably.

``` r
summary(gsc_queries_all[,-c(1:3)])
```

    ##      clicks         impressions          ctr               position      
    ##  Min.   :  1.000   Min.   :   1.0   Min.   :0.0001864   Min.   :  1.000  
    ##  1st Qu.:  1.000   1st Qu.:   2.0   1st Qu.:0.2000000   1st Qu.:  1.111  
    ##  Median :  2.000   Median :   5.0   Median :0.4166667   Median :  2.286  
    ##  Mean   :  3.327   Mean   :  16.3   Mean   :0.4794322   Mean   :  3.815  
    ##  3rd Qu.:  3.000   3rd Qu.:  13.0   3rd Qu.:0.7142857   3rd Qu.:  4.875  
    ##  Max.   :453.000   Max.   :7528.0   Max.   :1.0000000   Max.   :180.000

# Data Cleaning

Even though data extracted from GSC API are in a tidy format, there are
some fixes to do before we start with our analysis. For instance,
details about the date, the query and the page are not requested to
predict the CTR. Hence, we remove them and create a new data set without
them. Moreover, we perform scaling on all the variables that will be
used as predictors to ensure they are in the same scale.

## Plotting

Although the data set is not particularly rich in terms of covariates,
we can still visualize something to understand its composition.

``` r
ggplot(gsc_queries_all, aes(clicks, ctr)) +
  geom_bin2d(bins = 20, color = "white") +
  scale_fill_gradient(low =  "#00AFBB", high = "#FC4E07")+
  theme_minimal()
```

![](CTR_optimisation_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

We can use a 2D bin plot to check the relationship between clicks and
CTR. It seems that a lot of articles with high CTR tend to have low
clicks too. A possible explanation is that pages with higher CTR may
rank for long-tail queries that have a low volume of impressions.

To investigate this phenomenon, we can visualize the click distribution,
we can use an histogram. We decided to use a vertical line to show the
median value, which was 2.00 (check summary).

``` r
ggplot(gsc_queries_all, aes(clicks, fill = "red", color = "black")) +
  geom_histogram() +
  geom_vline(xintercept = median(gsc_queries_all$clicks), color = "black") +
  xlim(0, 100) +
  theme_minimal() +
  theme(legend.position = "none")
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

    ## Warning: Removed 26 rows containing non-finite values (stat_bin).

    ## Warning: Removed 2 rows containing missing values (geom_bar).

![](CTR_optimisation_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

It is pretty clear that the distribution is very skewed to the right and
50% of our pages for some queries receive more than 2 clicks. This also
translate to 50% of the pages receiving less than 2 clicks, which is
coherent with what we found before.

We are interested in getting position as a categorical variable just to
plot it. Recall that the position on SERP cannot be continuous, but GSC
gives you the average position\! We used the ceiling function to round
values to the next integer. We didn’t use floor to be more conservative
and avoiding overestimating the position.

As said in the introduction, CTR varies with position. In our 3-month
sample, we can notice that getting the 1st position was crucial in terms
of mean CTR and we can easily state that this is similar to what is
reported by SEO studies about the relationship between position and CTR.
Please, note that pages at position 8 got even more than those at 6/7
and as much as 5 in our example. This also tells that aiming for the top
spots is truly beneficial, since there is a clear gap in the first 3
positions and even more among position 1 and the rest.

``` r
colors = c("red", "blue", "orange", "yellow", "purple", "lightblue", "pink", "green")
  
gsc_queries_all %>% 
  filter(position <= 8) %>%
  mutate(pos_cat = as.factor(ceiling(position))) %>%
  group_by(pos_cat) %>%
  summarize(mean_ctr = mean(ctr)) %>%
  ungroup() %>%
  ggplot(aes(reorder(pos_cat,desc(pos_cat)), mean_ctr, fill = colors)) +
  geom_col() +
  theme_minimal() +
  coord_flip() +
  theme(legend.position = "none") +
  ggtitle("Mean CTR by Average Position") +
  ylab("Mean CTR") +
  xlab("Average Position")
```

    ## `summarise()` ungrouping output (override with `.groups` argument)

![](CTR_optimisation_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

## Measuring correlation among variables

We check the correlation matrix to see if there are meaningful
associations between variables. We expected a strong correlation between
clicks or impressions and CTR, but the results tell us the opposite.

``` r
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))

p.mat <- cor.mtest(gsc)$p

corrplot(cor(gsc), , method = "color", col = col(200),
         type = "upper", order = "hclust", number.cex = .7,
         addCoef.col = "black", 
         tl.col = "black", tl.srt = 90,
         p.mat = p.mat, sig.level = 0.01, insig = "blank",
         diag = FALSE)
```

![](CTR_optimisation_files/figure-gfm/unnamed-chunk-5-1.png)<!-- --> CTR
is negatively correlated to clicks but the value is close to 0, so it’s
not that relevant. On contrary, there is a weak negative correlation
between impressions and CTR, which makes sense, because CTR is inversely
proportional to impressions. The negative correlation between CTR and
position is also sound because a lower position is usually associated
with a higher CTR. Be careful however, correlation doesn’t imply
causation, we are just talking about linear association.

## Train/Test split

Before we start training any model, we need to create a train/test split
to avoid overfitting. We will use a 75/25 ratio, chosen arbitrarily and
we should have enough data to train a model. To do this, we simply
create a Uniform distribution which stores

``` r
set.seed(55744747)
gp <- runif(nrow(gsc))

train_gsc <- gsc[gp < 0.75,]
test_gsc <- gsc[gp >= 0.75,]
```

## Linear Regression

We start with the most basic model to assess its accuracy. I suppose
that this model won’t lead to good results because I suspect that the
relationship is not linear at all\!

``` r
lm_mod <- lm(ctr ~ ., data = train_gsc)
train_gsc$pred <- predict(lm_mod, train_gsc)
test_gsc$pred <- predict(lm_mod, test_gsc)
```

``` r
summary(lm_mod)
```

    ## 
    ## Call:
    ## lm(formula = ctr ~ ., data = train_gsc)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -0.4639 -0.2716 -0.0622  0.2247  3.3755 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  0.478690   0.002317 206.612   <2e-16 ***
    ## clicks       0.004413   0.002366   1.865   0.0622 .  
    ## impressions -0.042336   0.002113 -20.035   <2e-16 ***
    ## position    -0.069991   0.002410 -29.040   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.3167 on 18680 degrees of freedom
    ## Multiple R-squared:  0.06584,    Adjusted R-squared:  0.06569 
    ## F-statistic: 438.9 on 3 and 18680 DF,  p-value: < 2.2e-16

We can visualize our results to assess how well the regression line fits
the data points.

``` r
ggplot(test_gsc, aes(pred, ctr)) +
  geom_point(alpha = 0.2, color = "darkgray") +
  geom_smooth(color = "darkblue") +
  geom_line(aes(ctr, ctr), color = "blue", linetype = 2) 
```

    ## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'

![](CTR_optimisation_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

The result is awful but that’s perfectly fine, our hypothesis that
linear models are not suitable is confirmed. We need something that can
capture non-linear relationships and that is more complex.

We can measure the results of the linear regression with some useful
metrics.

R squared tells us the amount of variance explained by the model and
ranges between 0 and 1, where 1 is the best theoretical value. We expect
to see R squared values that are close between train and test and that
the test one is at least larger than 0.70.

On contrary, RMSE has to be as low as possible since we are talking
about errors. Basically, it’s the square root of the variance of
residuals and refers to the absolute fit of the model to the data. It
tells us the accuracy with which our model predicts the response and it
is very important for our goal, namely prediction.

``` r
R2_train <- 1 - (sum((train_gsc$ctr-train_gsc$pred)^2)/sum((train_gsc$ctr-mean(train_gsc$ctr))^2))
R2_test <- 1 - (sum((test_gsc$ctr-test_gsc$pred)^2)/sum((test_gsc$ctr-mean(test_gsc$ctr))^2))

RMSE_train <- sqrt(mean( (train_gsc$ctr - train_gsc$pred)^2 ))
RMSE_test<- sqrt(mean( (test_gsc$ctr - test_gsc$pred)^2 ))
```

Both the two metrics for each of the 2 sets display totally unacceptable
values, in line with what we saw before.

## Random Forest

The previous result is very disappointing, so we go for an ensemble
method that can easily model non-linear relationships and is way more
powerful than standard decision trees. I ensure that the experiment is
repeatable by setting a seed value.

``` r
set.seed(55744747)
rand_gsc <- randomForest(ctr ~., data = train_gsc, importance = TRUE)

train_gsc$pred <- predict(rand_gsc, train_gsc)
test_gsc$pred <- predict(rand_gsc, test_gsc)
```

``` r
summary(rand_gsc)
```

    ##                 Length Class  Mode     
    ## call                4  -none- call     
    ## type                1  -none- character
    ## predicted       18684  -none- numeric  
    ## mse               500  -none- numeric  
    ## rsq               500  -none- numeric  
    ## oob.times       18684  -none- numeric  
    ## importance          8  -none- numeric  
    ## importanceSD        4  -none- numeric  
    ## localImportance     0  -none- NULL     
    ## proximity           0  -none- NULL     
    ## ntree               1  -none- numeric  
    ## mtry                1  -none- numeric  
    ## forest             11  -none- list     
    ## coefs               0  -none- NULL     
    ## y               18684  -none- numeric  
    ## test                0  -none- NULL     
    ## inbag               0  -none- NULL     
    ## terms               3  terms  call

After we fit the model, it is important to create new columns for the
predicted values in both the two sets. However, remind that we will only
use values from the test data set later on\!

``` r
varImp <- importance(rand_gsc)
varImpPlot(rand_gsc, type = 1)
```

![](CTR_optimisation_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

You may want to take a look at the importance of the variables according
to Random Forest. We used the percentage of increase in MSE when a given
variable variable is randomly permuted as a measure. We can quickly
notice that impressions and clicks are essential for our model, although
either didn’t show any strong correlation with CTR at the beginning of
our analysis. In cases where you have a lot of variables, you can
consider using this screening procedure to pick the most important
variables and work with a smaller feature set.

``` r
R2_train <- 1 - (sum((train_gsc$ctr-train_gsc$pred)^2)/sum((train_gsc$ctr-mean(train_gsc$ctr))^2))
R2_test <- 1 - (sum((test_gsc$ctr-test_gsc$pred)^2)/sum((test_gsc$ctr-mean(test_gsc$ctr))^2))

RMSE_train <- sqrt(mean( (train_gsc$ctr - train_gsc$pred)^2 ))
RMSE_test<- sqrt(mean( (test_gsc$ctr - test_gsc$pred)^2 ))
```

## Hyperparameter tuning for Random Forest

We cold try to improve the performance of our RM model by tuning the
mtry (i.e. number of variables randomly sampled as candidate at each
split) hyperparameter, if we had more features to work with. We should
not see that much difference in this example, since we just have 3
covariates and the default value for regression is set to p/3, where p
is the number of features.

``` r
mtry <- tuneRF(train_gsc[, -4],train_gsc$ctr, ntreeTry=500,
               stepFactor=1.5,improve=0.05, trace=TRUE, plot=TRUE)
```

    ## mtry = 1  OOB error = 0.001084383 
    ## Searching left ...
    ## Searching right ...

![](CTR_optimisation_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

As expected, the best value for mtry (based on OoB error) is 1, which is
already in use when building a model. Consider that this step is
essential for bigger models with a larger dimension space.

## Merge train and test

Now that we have assessed that RM is a valid model, we can merge back
train and test data sets, featuring the new columns with predicted
values. Then, we can reorder the indexes of the new merged dataframe and
create a new column that is given by the difference between the
prediction of our model and the actual CTR times 100. Finally, we are
ready to combine this dataframe with the one we started with, to recover
pages and queries.

``` r
df_gsc <- rbind(train_gsc, test_gsc)


df_gsc <- df_gsc[ order(as.numeric(row.names(df_gsc))), ]
df_gsc$diff_perc <- (df_gsc$pred - df_gsc$ctr)*100


final_df <- merge(gsc_queries_all, df_gsc, by = 0, all = T)
```

## Back to the original dataset

We are now ready for the final step. We filter for the rows with values
higher than 0 in the percentage difference and then apply some basic
transformations. Recall that since we are working with the very first
dataset, we don’t have scaled values.

``` r
final_df <- final_df[, -c(1:2, 9:12)]
final_df <- final_df[final_df$diff_perc > 0.00 ,]
final_df <- final_df %>%
  arrange(desc(diff_perc))

final_df$ctr <- final_df$ctr.x * 100
final_df$pred <- final_df$pred * 100

final_df <- final_df %>%
  group_by(page) %>%
  arrange(desc(diff_perc)) %>%
  ungroup() %>%
  select(-c(clicks.x, impressions.x, ctr.x, position.x))
```

Now, we are ready for some interpretation of the results. Our interest
is the column of the percentage difference, necessary to understand what
has to be optimized. Clearly, we are not interested in the value per se,
but in the sign and the relative value compared to others. For this
reason, arranging rows in descending order by that column is a good to
immediately see the pages with bigger differences, meaning that they
have a predicted CTR higher than the actual and you should definitely
work on them. On contrary, negative values express that those pages are
overperforming, so you can leave them be (for the moment).

As mentioned by the original author of this approach, there are some
limitations such as the fact that every query is treated as if it was
the same. Nonetheless, it is still a better and less biased solution
than traditional ones\! My personal advice is to use the fitted model
just for the website you tested for and not others, since you may want
to avoid comparing different data and most likely ranges.

Future improvements may include expanding the original data set with
other sources, such as Screaming Frog, Google Analytics or other SEO
tools (ie. SEMRush or Hrefs).
