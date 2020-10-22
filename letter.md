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

    library(GGally)
    library(MASS)

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
    ##  1526   499

    Accuracy = sum(Letters.train$isB == FALSE) / nrow(Letters.train)
    table(Letters.test$isB)

    ## 
    ## FALSE  TRUE 
    ##   824   267

    Base_Accuracy = sum(Letters.test$isB == FALSE) / nrow(Letters.test)

    paste("The Accuracy of the baseline model:", Base_Accuracy) 

    ## [1] "The Accuracy of the baseline model: 0.755270394133822"

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
    ## -3.4033  -0.1515  -0.0182   0.0000   3.5348  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -11.87452    2.45584  -4.835 1.33e-06 ***
    ## xbox          0.05691    0.12539   0.454 0.649951    
    ## ybox          0.13242    0.08658   1.530 0.126140    
    ## width        -1.15004    0.15626  -7.360 1.84e-13 ***
    ## height       -0.83729    0.14550  -5.754 8.69e-09 ***
    ## onpix         0.89868    0.13650   6.584 4.58e-11 ***
    ## xbar          0.63528    0.13056   4.866 1.14e-06 ***
    ## ybar         -0.71092    0.12266  -5.796 6.79e-09 ***
    ## x2bar        -0.37972    0.09455  -4.016 5.92e-05 ***
    ## y2bar         1.35729    0.12619  10.756  < 2e-16 ***
    ## xybar         0.13621    0.09180   1.484 0.137856    
    ## x2ybar        0.52199    0.12519   4.170 3.05e-05 ***
    ## xy2bar       -0.60928    0.10986  -5.546 2.92e-08 ***
    ## xedge        -0.32815    0.09556  -3.434 0.000594 ***
    ## xedgeycor     0.08840    0.10334   0.855 0.392295    
    ## yedge         1.76508    0.13098  13.476  < 2e-16 ***
    ## yedgexcor     0.30845    0.07269   4.243 2.20e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2261.39  on 2024  degrees of freedom
    ## Residual deviance:  627.72  on 2008  degrees of freedom
    ## AIC: 661.72
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
    ##   FALSE   788   36
    ##   TRUE     39  228

    log_accuracy = (confusion_m[1] + confusion_m[4])/sum(confusion_m)
    paste("The Accuracy of the logistic regression model:", log_accuracy)

    ## [1] "The Accuracy of the logistic regression model: 0.931255728689276"

We also calculate the AUC of the model

    predTestLog <- predict(log_model, newdata = subset(Letters.test, select=c(-letter) ), type = "response")
    rocr.log.pred <- prediction(predTestLog, Letters.test$isB)
    logPerformance <- performance(rocr.log.pred, "tpr", "fpr")
    plot(logPerformance, colorize = TRUE)
    abline(0, 1)

![](letter_files/figure-markdown_strict/unnamed-chunk-7-1.png)

    # AUC
    paste("As calulated, AUC = ",as.numeric(performance(rocr.log.pred, "auc")@y.values))

    ## [1] "As calulated, AUC =  0.976769026580852"

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
    ##   FALSE   788   36
    ##   TRUE     48  219

    cart_accuracy = (confusion_m[1] + confusion_m[4])/sum(confusion_m)
    paste("The Accuracy of the CART model:", cart_accuracy)

    ## [1] "The Accuracy of the CART model: 0.923006416131989"

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
    ##   FALSE   813   11
    ##   TRUE     15  252

    rf_accuracy = (confusion_m[1] + confusion_m[4])/sum(confusion_m)
    paste("The Accuracy of the Random Forest :", rf_accuracy)

    ## [1] "The Accuracy of the Random Forest : 0.976168652612282"

#### 1 v) Summary

    data.frame(Model = c("Baseline","Logistic Regression","CART","Random Forest"), Accuracy = c(Base_Accuracy, log_accuracy, cart_accuracy, rf_accuracy))

    ##                 Model  Accuracy
    ## 1            Baseline 0.7552704
    ## 2 Logistic Regression 0.9312557
    ## 3                CART 0.9230064
    ## 4       Random Forest 0.9761687

2. Prediction on “A”, “B”, “P”, “R”
-----------------------------------

Now, let’s move on to a more general case: predicting a letter is one of
“A”, “B”, “P”, “R”. It can be extended to the full 26 letters case, but
this will inflate the model complexity and run time. For the sake of
simplicity, we’ll run the demo of four letter case here.

#### 2 i) Baseline Model

    table(Letters.train$letter)

    ## 
    ##   A   B   P   R 
    ## 494 499 541 491

    table(Letters.test$letter)

    ## 
    ##   A   B   P   R 
    ## 295 267 262 267

    ## Since "A" appears most frequently in the training set, we'll always predict the "A" in the test set
    Base_Accuracy = sum(Letters.test$letter == "A") / nrow(Letters.test)
    paste("Baseline Model Accuracy = " ,Base_Accuracy) 

    ## [1] "Baseline Model Accuracy =  0.270394133822181"

#### 2 ii) LDA (Latent Dirichlet allocation)

    LdaModel <- lda(letter ~ ., data=subset(Letters.train, select = c(-isB)))

    predTestLDA <- predict(LdaModel, newdata = Letters.test)

    confusion_m <- table(predTestLDA$class, Letters.test$letter)
    confusion_m

    ##    
    ##       A   B   P   R
    ##   A 269   0   0   0
    ##   B   5 236   9  29
    ##   P   5   0 248   0
    ##   R  16  31   5 238

    lda_accuracy = sum(predTestLDA$class == Letters.test$letter)/nrow(Letters.test)
    paste("LDA Accuracy:" ,lda_accuracy)

    ## [1] "LDA Accuracy: 0.908340971585701"

#### 2 iii) CART

    cpVals = data.frame(cp = seq(0, .001, by=.1))

    train.cart_2 <- train(letter ~ .,
                        data = subset(Letters.train, select=c(-isB) ),
                        method = "rpart",
                        tuneGrid = cpVals,
                        trControl = trainControl(method = "cv", number=5),
                        metric = "Accuracy")

    train.cart_2

    ## CART 
    ## 
    ## 2025 samples
    ##   16 predictor
    ##    4 classes: 'A', 'B', 'P', 'R' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1619, 1621, 1620, 1620, 1620 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.8938379  0.8583647
    ## 
    ## Tuning parameter 'cp' was held constant at a value of 0

    Letters.test.mm = as.data.frame(model.matrix(letter~.+0, data=Letters.test))

    pred.cart <- predict(train.cart_2, newdata = Letters.test.mm, type="prob" )
    pred.cart$results = colnames(pred.cart)[apply(pred.cart,1,which.max)]

    confusion_m <- table(Letters.test$letter, pred.cart$results)
    confusion_m

    ##    
    ##       A   B   P   R
    ##   A 282   5   1   7
    ##   B   2 226  10  29
    ##   P   3   9 248   2
    ##   R   4  22   6 235

    cart_accuracy_2 = sum(Letters.test$letter == pred.cart$results)/(nrow(Letters.test))
    paste("CART Accuracy" ,cart_accuracy_2)

    ## [1] "CART Accuracy 0.908340971585701"

    tree <- rpart(letter ~., data = Letters.train, method = "class")
    rpart.plot(tree)

![](letter_files/figure-markdown_strict/unnamed-chunk-15-1.png)

#### iv Bagging of CART models
