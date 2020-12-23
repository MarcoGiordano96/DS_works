CTR Optimization with the aid of Machine Learning
================
Marco Giordano
12/18/2020

## Introduction \[Goals and Why\]

This project is just an example of how it is possible to predict the CTR
of a website by using Google Search Console data, extracted via Google
API. What we present can be considered as a “toy example”, in the sense
that it is just a demonstration to show two methods on a relatively
small dataset. The main reason to predict CTR via Machine Learning
techniques is because SEO Specialists usually decide what to optimize by
looking at pages with high Impressions and low CTR. However, this method
is obviously biased, since CTR varies with position\!

I try to test some popular methods to predict the CTR, given a small
sample of 25K rows from 3 months from Google Search Console, in order to
create a better way to make decisions on what could be optimized. The
inspiration came from an article you can find below.

Original Source:
[here](https://understandingdata.com/ctr-optimisation-with-machine-learning/)

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.0 ──

    ## ✓ ggplot2 3.3.2     ✓ purrr   0.3.4
    ## ✓ tibble  3.0.4     ✓ dplyr   1.0.2
    ## ✓ tidyr   1.1.2     ✓ stringr 1.4.0
    ## ✓ readr   1.4.0     ✓ forcats 0.5.0

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## x dplyr::combine()  masks randomForest::combine()
    ## x dplyr::filter()   masks stats::filter()
    ## x dplyr::lag()      masks stats::lag()
    ## x ggplot2::margin() masks randomForest::margin()

``` r
library(ggplot2)
library(corrplot)
```

    ## corrplot 0.84 loaded

``` r
library(gt)
library(stargazer)
```

    ## 
    ## Please cite as:

    ##  Hlavac, Marek (2018). stargazer: Well-Formatted Regression and Summary Statistics Tables.

    ##  R package version 5.2.2. https://CRAN.R-project.org/package=stargazer

``` r
library(caret)
```

    ## Loading required package: lattice

    ## 
    ## Attaching package: 'caret'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
library(ModelMetrics)
```

    ## 
    ## Attaching package: 'ModelMetrics'

    ## The following objects are masked from 'package:caret':
    ## 
    ##     confusionMatrix, precision, recall, sensitivity, specificity

    ## The following object is masked from 'package:base':
    ## 
    ##     kappa

After loading the necessary libraries, we are ready to start\! We have
already loaded our file and we won’t show this step.

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
some small fixes to do before we start with our analysis. For instance,
details about the date, the query and the page are not requested to
predict the CTR. Hence, we remove them and create a new data set without
them.

``` r
gsc <- gsc_queries_all[,-c(1:3)]
gsc <- gsc %>%
   select(-ctr) %>%
   cbind(gsc_queries_all$ctr) 
colnames(gsc)[4] <- "ctr"
```

## (Short) Exploratory Data Analysis

Although the data set is not particularly rich in terms of covariates,
we can still visualize something to understand its composition.

``` r
ggplot(gsc_queries_all, aes(clicks, ctr)) +
  geom_bin2d(bins = 20, color = "white") +
  scale_fill_gradient(low =  "#00AFBB", high = "#FC4E07")+
  theme_minimal()
```

![](CTR_optimisation_files/figure-gfm/2D%20Bin-1.png)<!-- -->

We can use a 2D bin plot to check the relationship between clicks and
CTR. It seems that a lot of articles with high CTR tend to have low
clicks too. A possible explanation is that pages with higher CTR may
rank for long-tail queries that have a low volume of impressions.

To investigate this phenomenon, we can visualize the click distribution,
we can use an histogram. We decided to use a vertical line to show the
median value, which was 2.00 (check summary). We zoomed on the range
0-100 to have a more clear representation.

``` r
ggplot(gsc_queries_all, aes(clicks, fill = "red", color = "black")) +
  geom_histogram() +
  geom_vline(xintercept = median(gsc_queries_all$clicks), color = "black") +
  xlim(0, 100) +
  theme_minimal() +
  theme(legend.position = "none")
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](CTR_optimisation_files/figure-gfm/Clicks%20distribution-1.png)<!-- -->

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

![](CTR_optimisation_files/figure-gfm/Mean%20CTR%20by%20Average%20Position-1.png)<!-- -->

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

![](CTR_optimisation_files/figure-gfm/Correlation%20Plot-1.png)<!-- -->

CTR is negatively correlated to clicks but the value is close to 0, so
it’s not that relevant. On contrary, there is a weak negative
correlation between impressions and CTR, which makes sense, because CTR
is inversely proportional to impressions. The negative correlation
between CTR and position is also sound because a lower position is
usually associated with a higher CTR. Be careful however, correlation
doesn’t imply causation, we are just talking about linear association.

## Train/Test split

Before we start training any model, we need to create a train/test split
to avoid overfitting. We will use a 75/25 ratio, chosen arbitrarily and
we should have enough data to train a model. To do this, we simply
create a random variable to group and partition data.

``` r
set.seed(55744747)
gp <- runif(nrow(gsc))
train_gsc <- gsc[gp < 0.75,]
test_gsc <- gsc[gp >= 0.75,]
```

After the train/test split, we can scale our two sets to adjust their
scale and make algorithms life easier. We use a function from the caret
package to process data from the training data set and then apply the
same preprocessing to the test data set. Be careful, as you need to
scale test data by what seen in train data\!

``` r
preProcValues <- preProcess(train_gsc[-4], method = c("center", "scale"))
train_gsc[-4] <- predict(preProcValues, train_gsc[-4])
test_gsc[-4] <- predict(preProcValues, test_gsc[-4])
```

## Linear Regression

We start with the most basic model to assess its accuracy. I suppose
that this model won’t lead to good results because I suspect that the
relationship is not linear at all, although linear regression allows for
interpolation. We can directly fit a model with all the covariates.

``` r
lm_mod <- lm(ctr ~ ., data = train_gsc)
train_Preds <- predict(lm_mod, train_gsc)
Preds <- predict(lm_mod, test_gsc)
```

Here follows a quick overview of the Linear Regression model.

``` r
stargazer(lm_mod, type = "text")
```

    ## 
    ## ===============================================
    ##                         Dependent variable:    
    ##                     ---------------------------
    ##                                 ctr            
    ## -----------------------------------------------
    ## clicks                        0.004*           
    ##                               (0.002)          
    ##                                                
    ## impressions                  -0.048***         
    ##                               (0.002)          
    ##                                                
    ## position                     -0.068***         
    ##                               (0.002)          
    ##                                                
    ## Constant                     0.478***          
    ##                               (0.002)          
    ##                                                
    ## -----------------------------------------------
    ## Observations                  18,684           
    ## R2                             0.066           
    ## Adjusted R2                    0.066           
    ## Residual Std. Error     0.317 (df = 18680)     
    ## F Statistic         438.866*** (df = 3; 18680) 
    ## ===============================================
    ## Note:               *p<0.1; **p<0.05; ***p<0.01

By taking a quick look at the summary, we can clearly notice that
converting variables to log or doing feature engineering would be a
waste of time.

We can visualize our results to assess how well the regression line fits
the data points.

``` r
ggplot(test_gsc, aes(Preds, ctr)) +
  geom_point(alpha = 0.2, color = "darkgray") +
  geom_smooth(color = "darkblue") +
  geom_line(aes(ctr, ctr), color = "blue", linetype = 2) 
```

    ## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'

![](CTR_optimisation_files/figure-gfm/Linear%20Regression-1.png)<!-- -->

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

Both the two metrics for each of the 2 sets display totally unacceptable
values, in line with what we saw before.

``` r
reg_rmse_train <- rmse(train_gsc$ctr, train_Preds)
reg_rmse_test <- rmse(test_gsc$ctr, Preds)
reg_R2_train <- 1 - (sum((train_gsc$ctr-train_Preds)^2)/sum((train_gsc$ctr-mean(train_gsc$ctr))^2))
reg_R2_test <- 1 - (sum((test_gsc$ctr-Preds)^2)/sum((test_gsc$ctr-mean(test_gsc$ctr))^2))


Reg_visual <- tibble(
  Dataset = c("train", "test"),
  RMSE = c(reg_rmse_train, reg_rmse_test),
  R_squared = c(reg_R2_train, reg_R2_test)
)

Reg_visual %>%
  gt() %>%
  tab_header(
    title = md("Summary table for Linear Regression metrics")
  )
```

<!--html_preserve-->

<style>html {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
}

#ysfdolteun .gt_table {
  display: table;
  border-collapse: collapse;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#ysfdolteun .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#ysfdolteun .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#ysfdolteun .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 0;
  padding-bottom: 4px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#ysfdolteun .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#ysfdolteun .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#ysfdolteun .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#ysfdolteun .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#ysfdolteun .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#ysfdolteun .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#ysfdolteun .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#ysfdolteun .gt_group_heading {
  padding: 8px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
}

#ysfdolteun .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#ysfdolteun .gt_from_md > :first-child {
  margin-top: 0;
}

#ysfdolteun .gt_from_md > :last-child {
  margin-bottom: 0;
}

#ysfdolteun .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#ysfdolteun .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 12px;
}

#ysfdolteun .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#ysfdolteun .gt_first_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
}

#ysfdolteun .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#ysfdolteun .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#ysfdolteun .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#ysfdolteun .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#ysfdolteun .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#ysfdolteun .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding: 4px;
}

#ysfdolteun .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#ysfdolteun .gt_sourcenote {
  font-size: 90%;
  padding: 4px;
}

#ysfdolteun .gt_left {
  text-align: left;
}

#ysfdolteun .gt_center {
  text-align: center;
}

#ysfdolteun .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#ysfdolteun .gt_font_normal {
  font-weight: normal;
}

#ysfdolteun .gt_font_bold {
  font-weight: bold;
}

#ysfdolteun .gt_font_italic {
  font-style: italic;
}

#ysfdolteun .gt_super {
  font-size: 65%;
}

#ysfdolteun .gt_footnote_marks {
  font-style: italic;
  font-size: 65%;
}
</style>

<div id="ysfdolteun" style="overflow-x:auto;overflow-y:auto;width:auto;height:auto;">

<table class="gt_table">

<thead class="gt_header">

<tr>

<th colspan="3" class="gt_heading gt_title gt_font_normal" style>

Summary table for Linear Regression metrics

</th>

</tr>

<tr>

<th colspan="3" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style>

</th>

</tr>

</thead>

<thead class="gt_col_headings">

<tr>

<th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1">

Dataset

</th>

<th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1">

RMSE

</th>

<th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1">

R\_squared

</th>

</tr>

</thead>

<tbody class="gt_table_body">

<tr>

<td class="gt_row gt_left">

train

</td>

<td class="gt_row gt_right">

0.3166501

</td>

<td class="gt_row gt_right">

0.06584103

</td>

</tr>

<tr>

<td class="gt_row gt_left">

test

</td>

<td class="gt_row gt_right">

0.3201127

</td>

<td class="gt_row gt_right">

0.06242624

</td>

</tr>

</tbody>

</table>

</div>

<!--/html_preserve-->

## Random Forest

The previous result is very disappointing, so we go for an ensemble
method that can easily model non-linear relationships and is way more
powerful than standard decision trees. I ensure that the experiment is
repeatable by setting a seed value.

``` r
set.seed(55744747)
rand_gsc <- randomForest(ctr ~., data = train_gsc, importance = TRUE, ntree = 500)
train_gsc$pred <- predict(rand_gsc, train_gsc)
test_gsc$pred <- predict(rand_gsc, test_gsc)
```

After we fit the model, it is important to create new columns for the
predicted values in both the two sets. However, remind that we will only
use values from the test data set later on\!

``` r
summary(rand_gsc)
```

    ##                 Length Class  Mode     
    ## call                5  -none- call     
    ## type                1  -none- character
    ## predicted       18684  -none- numeric  
    ## mse               500  -none- numeric  
    ## rsq               500  -none- numeric  
    ## oob.times       18684  -none- numeric  
    ## importance          6  -none- numeric  
    ## importanceSD        3  -none- numeric  
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

``` r
plot(rand_gsc, main = "Random Forest")
```

![](CTR_optimisation_files/figure-gfm/Error%20vs%20Number%20of%20Trees%20-%20RF-1.png)<!-- -->

As you can see above, after \~10 trees, error starts to increase and
then flatten as we move towards 500. We could also optimize the number
of trees used by setting a lower number but the model is already fine
this way. Please consider to test different combinations if you need to
get better results. In our case, we could just reduce the number of
trees to increase the speed of our algorithm. This is also part of
hyperparameter tuning if you need to improve your model predictions.

You may want to take a look at the importance of the variables according
to Random Forest. We used the percentage of increase in MSE when a given
variable variable is randomly permuted as a measure. We can quickly
notice that impressions and clicks are essential for our model, although
either didn’t show any strong correlation with CTR at the beginning of
our analysis. In cases where you have a lot of variables, you can
consider using this screening procedure to pick the most important
variables and work with a smaller feature set.

``` r
varImp <- importance(rand_gsc)
varImpPlot(rand_gsc, type = 1, main = "Variable Importance")
```

![](CTR_optimisation_files/figure-gfm/Variable%20importance%20-%20RF-1.png)<!-- -->

We can visualize how Random Forest performed by taking a look at the
same metrics we used for Linear Regression.

``` r
R2_train <- 1 - (sum((train_gsc$ctr-train_gsc$pred)^2)/sum((train_gsc$ctr-mean(train_gsc$ctr))^2))
R2_test <- 1 - (sum((test_gsc$ctr-test_gsc$pred)^2)/sum((test_gsc$ctr-mean(test_gsc$ctr))^2))
RMSE_train <- sqrt(mean( (train_gsc$ctr - train_gsc$pred)^2 ))
RMSE_test<- sqrt(mean( (test_gsc$ctr - test_gsc$pred)^2 ))

RF_visual <- tibble(
  Dataset = c("train", "test"),
  RMSE = c(RMSE_train, RMSE_test),
  R_squared = c(R2_train, R2_test)
)

RF_visual %>%
  gt() %>%
  tab_header(
    title = md("Summary table for Random Forest metrics")
  )
```

<!--html_preserve-->

<style>html {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
}

#nbfunssdtb .gt_table {
  display: table;
  border-collapse: collapse;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#nbfunssdtb .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#nbfunssdtb .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#nbfunssdtb .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 0;
  padding-bottom: 4px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#nbfunssdtb .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#nbfunssdtb .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#nbfunssdtb .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#nbfunssdtb .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#nbfunssdtb .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#nbfunssdtb .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#nbfunssdtb .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#nbfunssdtb .gt_group_heading {
  padding: 8px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
}

#nbfunssdtb .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#nbfunssdtb .gt_from_md > :first-child {
  margin-top: 0;
}

#nbfunssdtb .gt_from_md > :last-child {
  margin-bottom: 0;
}

#nbfunssdtb .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#nbfunssdtb .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 12px;
}

#nbfunssdtb .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#nbfunssdtb .gt_first_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
}

#nbfunssdtb .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#nbfunssdtb .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#nbfunssdtb .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#nbfunssdtb .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#nbfunssdtb .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#nbfunssdtb .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding: 4px;
}

#nbfunssdtb .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#nbfunssdtb .gt_sourcenote {
  font-size: 90%;
  padding: 4px;
}

#nbfunssdtb .gt_left {
  text-align: left;
}

#nbfunssdtb .gt_center {
  text-align: center;
}

#nbfunssdtb .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#nbfunssdtb .gt_font_normal {
  font-weight: normal;
}

#nbfunssdtb .gt_font_bold {
  font-weight: bold;
}

#nbfunssdtb .gt_font_italic {
  font-style: italic;
}

#nbfunssdtb .gt_super {
  font-size: 65%;
}

#nbfunssdtb .gt_footnote_marks {
  font-style: italic;
  font-size: 65%;
}
</style>

<div id="nbfunssdtb" style="overflow-x:auto;overflow-y:auto;width:auto;height:auto;">

<table class="gt_table">

<thead class="gt_header">

<tr>

<th colspan="3" class="gt_heading gt_title gt_font_normal" style>

Summary table for Random Forest metrics

</th>

</tr>

<tr>

<th colspan="3" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style>

</th>

</tr>

</thead>

<thead class="gt_col_headings">

<tr>

<th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1">

Dataset

</th>

<th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1">

RMSE

</th>

<th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1">

R\_squared

</th>

</tr>

</thead>

<tbody class="gt_table_body">

<tr>

<td class="gt_row gt_left">

train

</td>

<td class="gt_row gt_right">

0.08961063

</td>

<td class="gt_row gt_right">

0.9251864

</td>

</tr>

<tr>

<td class="gt_row gt_left">

test

</td>

<td class="gt_row gt_right">

0.09212720

</td>

<td class="gt_row gt_right">

0.9223439

</td>

</tr>

</tbody>

</table>

</div>

<!--/html_preserve-->

We cold try to improve the performance of our RM model by tuning the
mtry (i.e. number of variables randomly sampled as candidate at each
split) hyperparameter, if we had more features to work with. We should
not see that much difference in this example, since we just have 3
covariates and the default value for regression is set to p/3, where p
is the number of features.

``` r
mtry <- tuneRF(train_gsc[, -4],train_gsc$ctr, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
```

    ## mtry = 1  OOB error = 0.001267998 
    ## Searching left ...
    ## Searching right ...

![](CTR_optimisation_files/figure-gfm/Hyperparameter%20tuning%20-%20RF-1.png)<!-- -->

As expected, the best value for mtry (based on OoB error) is 1, which is
already in use when building a model. Consider that this step is
essential for bigger models with a larger dimension space.

``` r
visual <- tibble(
  Model = c("regression train", "regression test", "RF train", "Rf test"),
  RMSE = c(reg_rmse_train, reg_rmse_test, RMSE_train, RMSE_test),
  R_squared = c(reg_R2_train, reg_R2_test, R2_train, R2_test)
)
visual %>%
  gt() %>%
  tab_header(
    title = md("Summary table with model metrics")
  )
```

<!--html_preserve-->

<style>html {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
}

#fovpsykgwf .gt_table {
  display: table;
  border-collapse: collapse;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#fovpsykgwf .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#fovpsykgwf .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#fovpsykgwf .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 0;
  padding-bottom: 4px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#fovpsykgwf .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#fovpsykgwf .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#fovpsykgwf .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#fovpsykgwf .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#fovpsykgwf .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#fovpsykgwf .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#fovpsykgwf .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#fovpsykgwf .gt_group_heading {
  padding: 8px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
}

#fovpsykgwf .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#fovpsykgwf .gt_from_md > :first-child {
  margin-top: 0;
}

#fovpsykgwf .gt_from_md > :last-child {
  margin-bottom: 0;
}

#fovpsykgwf .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#fovpsykgwf .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 12px;
}

#fovpsykgwf .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#fovpsykgwf .gt_first_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
}

#fovpsykgwf .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#fovpsykgwf .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#fovpsykgwf .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#fovpsykgwf .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#fovpsykgwf .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#fovpsykgwf .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding: 4px;
}

#fovpsykgwf .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#fovpsykgwf .gt_sourcenote {
  font-size: 90%;
  padding: 4px;
}

#fovpsykgwf .gt_left {
  text-align: left;
}

#fovpsykgwf .gt_center {
  text-align: center;
}

#fovpsykgwf .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#fovpsykgwf .gt_font_normal {
  font-weight: normal;
}

#fovpsykgwf .gt_font_bold {
  font-weight: bold;
}

#fovpsykgwf .gt_font_italic {
  font-style: italic;
}

#fovpsykgwf .gt_super {
  font-size: 65%;
}

#fovpsykgwf .gt_footnote_marks {
  font-style: italic;
  font-size: 65%;
}
</style>

<div id="fovpsykgwf" style="overflow-x:auto;overflow-y:auto;width:auto;height:auto;">

<table class="gt_table">

<thead class="gt_header">

<tr>

<th colspan="3" class="gt_heading gt_title gt_font_normal" style>

Summary table with model metrics

</th>

</tr>

<tr>

<th colspan="3" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style>

</th>

</tr>

</thead>

<thead class="gt_col_headings">

<tr>

<th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1">

Model

</th>

<th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1">

RMSE

</th>

<th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1">

R\_squared

</th>

</tr>

</thead>

<tbody class="gt_table_body">

<tr>

<td class="gt_row gt_left">

regression train

</td>

<td class="gt_row gt_right">

0.31665010

</td>

<td class="gt_row gt_right">

0.06584103

</td>

</tr>

<tr>

<td class="gt_row gt_left">

regression test

</td>

<td class="gt_row gt_right">

0.32011271

</td>

<td class="gt_row gt_right">

0.06242624

</td>

</tr>

<tr>

<td class="gt_row gt_left">

RF train

</td>

<td class="gt_row gt_right">

0.08961063

</td>

<td class="gt_row gt_right">

0.92518640

</td>

</tr>

<tr>

<td class="gt_row gt_left">

Rf test

</td>

<td class="gt_row gt_right">

0.09212720

</td>

<td class="gt_row gt_right">

0.92234393

</td>

</tr>

</tbody>

</table>

</div>

<!--/html_preserve-->

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
  select(-c(clicks.x, impressions.x, ctr.x, position.x)) %>%
  relocate(diff_perc, .after = last_col())
```

We will show how our final data set looks like by taking a peak at only
the first row since these data are private.

``` r
head(final_df, 1)
```

    ## # A tibble: 1 x 5
    ##   query                page                                 pred   ctr diff_perc
    ##   <chr>                <chr>                               <dbl> <dbl>     <dbl>
    ## 1 schemi halloween an… https://ilovevg.it/2020/09/animal-…  39.5  5.88      33.6

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

## What’s next and what to consider

Random Forest cannot do extrapolation, since the average of samples
cannot fall outside their ranges. For this reason, you should totally
avoid applying this kind of model to new data that fall outside the
range on which you trained. Linear Regression can extrapolate data but
it performed really poorly and it is not a viable option.

Nonetheless, there are some solutions to try next, for instance neural
nets, SVM Regression or Regression-Enhanced Random Forests. The
possibility to extrapolate data is something that may improve your CTR
optimization process.
