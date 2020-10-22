    library(ggplot2)
    library(caTools)
    library(plotROC)
    library(ROCR)

    ## Warning: package 'ROCR' was built under R version 3.6.2

    library(rpart) # CART
    library(rpart.plot) # CART plotting
    library(caret) # c
    library(randomForest)
    library(gbm)

    ## Warning: package 'gbm' was built under R version 3.6.2

1.1 Simple Intro on is “B”
--------------------------

To make the problem easier, as well as getting to know the big picture
of the data, we’ll start by predicting whether or Not the letter is “B”

    ## load the data
    Letters = read.csv("Letters.csv",sep = ",")

#### add an “isB” column on the dataset

    Letters$isB = as.factor((Letters$letter == "B"))

#### Train Test Split

Put 65% of the data in the training set and 35% in the test set

    train.ids = sample(nrow(Letters), 0.65* nrow(Letters))
    Letters.train = Letters[train.ids, ]
    Letters.test = Letters[-train.ids, ]

#### 1 i) Baseline Model

This model always predicts “not B”, we’ll derive its performance in test
set

    table(Letters.train$isB)

    ## 
    ## FALSE  TRUE 
    ##  1515   510

    Accuracy = sum(Letters.train$isB == FALSE) / nrow(Letters.train)
    table(Letters.test$isB)

    ## 
    ## FALSE  TRUE 
    ##   835   256

    Base_Accuracy = sum(Letters.test$isB == FALSE) / nrow(Letters.test)

    paste("The Accuracy of the baseline model:", Base_Accuracy) 

    ## [1] "The Accuracy of the baseline model: 0.765352887259395"

#### 1 ii) Logistic Regression

A typical logistic regression that include all the variables, threshold
is set at $ p = 0.5$

    ## remove the letter for training
    log_model = glm(data = subset(Letters.train, select=c(-letter) ), family = "binomial", isB ~.)
    summary(log_model)

    ## 
    ## Call:
    ## glm(formula = isB ~ ., family = "binomial", data = subset(Letters.train, 
    ##     select = c(-letter)))
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -3.2383  -0.1493  -0.0165   0.0217   3.6970  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -16.25619    2.60528  -6.240 4.38e-10 ***
    ## xbox          0.07562    0.12502   0.605  0.54526    
    ## ybox         -0.02671    0.08666  -0.308  0.75794    
    ## width        -1.18786    0.15354  -7.737 1.02e-14 ***
    ## height       -0.77635    0.14207  -5.465 4.64e-08 ***
    ## onpix         0.99704    0.13728   7.263 3.79e-13 ***
    ## xbar          0.60824    0.13196   4.609 4.04e-06 ***
    ## ybar         -0.62184    0.12437  -5.000 5.74e-07 ***
    ## x2bar        -0.40789    0.09513  -4.287 1.81e-05 ***
    ## y2bar         1.43810    0.13275  10.833  < 2e-16 ***
    ## xybar         0.29993    0.09429   3.181  0.00147 ** 
    ## x2ybar        0.64532    0.12952   4.982 6.29e-07 ***
    ## xy2bar       -0.35337    0.11121  -3.178  0.00148 ** 
    ## xedge        -0.25838    0.09329  -2.770  0.00561 ** 
    ## xedgeycor     0.09464    0.10054   0.941  0.34656    
    ## yedge         1.70070    0.12569  13.531  < 2e-16 ***
    ## yedgexcor     0.35546    0.07378   4.818 1.45e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2285.66  on 2024  degrees of freedom
    ## Residual deviance:  633.34  on 2008  degrees of freedom
    ## AIC: 667.34
    ## 
    ## Number of Fisher Scoring iterations: 8

    ## predict on the test set
    pisB <- predict(log_model, newdata = subset(Letters.test, select=c(-letter) ), type = "response")
    pisB = pisB > 0.5

    ## construct the confusion matrix
    confusion_m <- table(Letters.test$isB, pisB)
    confusion_m

    ##        pisB
    ##         FALSE TRUE
    ##   FALSE   809   26
    ##   TRUE     38  218

    log_accuracy = (confusion_m[1] + confusion_m[4])/sum(confusion_m)
    paste("The Accuracy of the logistic regression model:", log_accuracy)

    ## [1] "The Accuracy of the logistic regression model: 0.941338221814849"

We also calculate the AUC of the model

    predTestLog <- predict(log_model, newdata = subset(Letters.test, select=c(-letter) ), type = "response")
    rocr.log.pred <- prediction(predTestLog, Letters.test$isB)
    logPerformance <- performance(rocr.log.pred, "tpr", "fpr")
    plot(logPerformance, colorize = TRUE)
    abline(0, 1)

![](letter_files/figure-markdown_strict/unnamed-chunk-7-1.png)

    # AUC
    paste("As calulated, AUC = ",as.numeric(performance(rocr.log.pred, "auc")@y.values))

    ## [1] "As calulated, AUC =  0.976305202095809"

#### 1 iii) CART Model to predict “isB”

Cross Validation is applied to select the cp(complexity parameter) of
the model.

    ## A list of cp value will be test 
    cpVals = data.frame(cp = seq(0, .1, by=.001)) 

    train.cart <- train(isB ~ .,
                        data = subset(Letters.train, select=c(-letter) ),
                        method = "rpart",
                        tuneGrid = cpVals,
                        trControl = trainControl(method = "cv", number=10),
                        metric = "Accuracy")

    Letters.test.mm = as.data.frame(model.matrix(isB~.+0, data=Letters.test))

    ## a threshold of 0.5 is set for decision
    pred.cart <- predict(train.cart,newdata = Letters.test.mm, type="prob" )
    pred_cart = pred.cart[,2] > 0.5

    confusion_m <- table(Letters.test$isB, pred_cart)
    confusion_m

    ##        pred_cart
    ##         FALSE TRUE
    ##   FALSE   817   18
    ##   TRUE     51  205

    cart_accuracy = (confusion_m[1] + confusion_m[4])/sum(confusion_m)
    paste("The Accuracy of the CART model:", cart_accuracy)

    ## [1] "The Accuracy of the CART model: 0.936755270394134"

    # A cp vs Accuracy to help us on deciding cp
    ggplot(train.cart$results, aes(x=cp, y=Accuracy)) + geom_point(size=3) +
      xlab("Complexity Parameter (cp)") + geom_line()

![](letter_files/figure-markdown_strict/unnamed-chunk-9-1.png)

#### 1 iv) Random Forest to predict “isB”

    ## train the RF model
    mod.rf <- randomForest(isB ~ .,
                        data = subset(Letters.train, select=c(-letter)))

    pred.rf <- predict(mod.rf, newdata = Letters.test, type = "response") 

    confusion_m <- table(Letters.test$isB, pred.rf)
    confusion_m

    ##        pred.rf
    ##         FALSE TRUE
    ##   FALSE   828    7
    ##   TRUE     18  238

    rf_accuracy = (confusion_m[1] + confusion_m[4])/sum(confusion_m)
    paste("The Accuracy of the Random Forest :", rf_accuracy)

    ## [1] "The Accuracy of the Random Forest : 0.977085242896425"

#### 1 v) Summary

    data.frame(Model = c("Baseline","Logistic Regression","CART","Random Forest"), Accuracy = c(Base_Accuracy, log_accuracy, cart_accuracy, rf_accuracy))

    ##                 Model  Accuracy
    ## 1            Baseline 0.7653529
    ## 2 Logistic Regression 0.9413382
    ## 3                CART 0.9367553
    ## 4       Random Forest 0.9770852
