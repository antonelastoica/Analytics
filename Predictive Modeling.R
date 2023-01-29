#Set working directory, clear workspace
rm(list = ls())
library(MASS)
library(pROC)
library(rpart)
library(randomForest)
library(xgboost)
library(nnet)
library(corrplot)

#Load data
dat <- read.table("winequality-red.csv", sep=";", header=TRUE)
dat$quality <- I(dat$quality > 6) * 1
set.seed(652)
trn <- runif(nrow(dat)) < .7
train <- dat[trn==TRUE,]
test <- dat[trn==FALSE,]
test <- readRDS("wine_test.rds")

#Logistic regression
glm <- glm(quality ~ ., family="binomial", data=train)
summary(glm)
yhat_glm <- predict(glm, type="response")

test$yhat.glm <- predict(glm, test, type="response")
TPR <- function(y,yhat)  { sum(y==1 & yhat==1) / sum(y==1) }
TNR <- function(y,yhat)  { sum(y==0 & yhat==0) / sum(y==0) }

table(test$quality, (test$yhat.glm > 0.5))
TPR(test$quality, (test$yhat.glm > 0.5))
TNR(test$quality, (test$yhat.glm > 0.5))

#LDA/QDA
lda <- lda(quality ~ ., data=train)
yhat_lda <- predict(lda)$posterior[,2]

test$yhat.lda <- predict(lda, test)$posterior[,2]
table(test$quality, (test$yhat.lda > 0.5))
TPR(test$quality, (test$yhat.lda >0.5))
TNR(test$quality, (test$yhat.lda >0.5))

qda <- qda(quality ~ ., data=train)
yhat_qda <- predict(qda)$posterior[,2]

test$yhat.qda <- predict(qda, test)$posterior[,2]
table(test$quality, (test$yhat.qda > 0.5))
TPR(test$quality, (test$yhat.qda >0.5))
TNR(test$quality, (test$yhat.qda >0.5))

#Classification Trees
form1 <- formula(quality ~ fixed.acidity + volatile.acidity + citric.acid
                 + residual.sugar + chlorides + free.sulfur.dioxide 
                 + total.sulfur.dioxide + density + pH + sulphates + alcohol)

t1 <- rpart(form1, data=train, cp=.001, method="class")
plot(t1,uniform=T,compress=T,margin=.05,branch=0.3)
text(t1, cex=.7, col="navy",use.n=TRUE)
plotcp(t1)
CP <- printcp(t1)
cp <- CP[,1][CP[,4] == min(CP[,4])]
cp

t2 <- prune(t1,cp=cp[1])
plot(t2,uniform=T,compress=T,margin=.05,branch=0.3)
text(t2, cex=.7, col="navy",use.n=TRUE)
test$yhat.t2 <- predict(t2, test, type="prob")[,2]

table(test$yhat.t2>0.5,test$quality)
TPR(test$yhat.t2>0.5,test$quality)
TNR(test$yhat.t2>0.5,test$quality)

#Random Forest
xvars <- names(train)[1:11]
X <- as.matrix(train[,xvars])
X.tst <- as.matrix(test[,xvars])
Y <- factor(train$quality)
set.seed(652)

mtry <- round(ncol(X)^.5); mtry

rf1 <- randomForest(x=X, y=Y, data=train,
                    ntree=500, mtry=3, importance=T, na.action=na.omit)
rf1
summary(rf1)
names(rf1)
head(rf1$importance)

varImpPlot(rf1)

yhat.rf <- predict(rf1, test, type="prob")[,2]
test$yhat.rf <- yhat.rf
par(mfrow=c(1,1))

table(test$quality, (test$yhat.rf > 0.5))
TPR(test$quality, (test$yhat.rf >0.5))
TNR(test$quality, (test$yhat.rf >0.5))

#Gradient Boost
Y <- train$quality
parm <- list(nthread=2, max_depth=2, eta=0.10)
bt <- xgboost(parm, data=X, label=Y, verbose=2, objective='binary:logistic', nrounds=20)

imp <- xgb.importance(feature_names=colnames(X), model=bt)
imp

xgb.plot.importance(imp, rel_to_first = TRUE, xlab = "Relative importance")

test$yhat.bt <- predict(bt, X.tst) 
par(mfrow=c(1,1))

table(test$quality, (test$yhat.bt > 0.5))
TPR(test$quality, (test$yhat.bt >0.5))
TNR(test$quality, (test$yhat.bt >0.5))

#Neural Nets
dat[,1:11]  <- scale(dat[,1:11])
dat$quality <- factor(dat$quality)
train2 <- dat[trn==TRUE,]
test2 <- scale(test[,1:11])

form1 <- formula(quality ~ fixed.acidity + volatile.acidity + citric.acid   + residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol)
nn1 <- nnet(form1, data = train2, size = 7, maxit = 500, decay=0.002)
nn1

yhat.nn1 <- predict(nn1, test)
test$yhat.nn1 <- yhat.nn1 
table(test$quality, (test$yhat.nn1 > 0.5))
TPR(test$quality, (test$yhat.nn1 >0.5))
TNR(test$quality, (test$yhat.nn1 >0.5))

form1 <- formula(quality ~ fixed.acidity + volatile.acidity + citric.acid   + residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol)
nn2 <- nnet(form1, data = train2, size = 9, maxit = 500, decay=0.002)
nn2

yhat.nn2 <- predict(nn2, test)
test$yhat.nn2 <- yhat.nn2 
table(test$quality, (test$yhat.nn2 > 0.5))
TPR(test$quality, (test$yhat.nn2 >0.5))
TNR(test$quality, (test$yhat.nn2 >0.5))

#ROC Curves
par(mfrow=c(1,1))
glm.roc <- roc(test$quality, test$yhat.glm, direction="<")
glm.roc
lda.roc <- roc(test$quality, test$yhat.lda, direction="<")
lda.roc
qda.roc <- roc(test$quality, test$yhat.qda, direction="<")
qda.roc
tree.roc <- roc(test$quality, test$yhat.t2, direction="<")
tree.roc
rf.roc <- roc(test$quality, test$yhat.rf, direction="<")
rf.roc
bt.roc <- roc(test$quality, test$yhat.bt, direction="<")
bt.roc
nn.roc <- roc(test$quality, test$yhat.nn1, direction="<")
nn.roc
nn2.roc <- roc(test$quality, test$yhat.nn2, direction="<")
nn2.roc

plot(nn.roc, lwd=3)
lines(nn2.roc, lwd=3, col = "pink")
lines(bt.roc, lwd=3, col = "orange")
lines(rf.roc, lwd=3, col = "green")
lines(tree.roc, lwd=3, col = "purple")
lines(lda.roc, lwd=3, col = "yellow")
lines(qda.roc, lwd=3, col = "red")
lines(glm.roc, lwd=3, col = "blue")
legend("bottomright",title="ROC Curves",c("neural_net", "neural_net2", "boost", "forest", "tree", "lda", "qda", "glm"), fill=c("black", "pink", "orange", "green", "purple", "yellow","red","blue"))

temp <- data.frame(test$yhat.nn1[,1],test$yhat.nn2[,1],test$quality)
rho <- cor(temp)
corrplot(rho, method="number")
