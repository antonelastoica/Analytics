#Set working directory, clear workspace
rm(list = ls())
library(MASS)
library(pROC)
library(rpart)
library(randomForest)
library(xgboost)
library(nnet)
library(corrplot)
library(maps)
library(mapproj)
library(NbClust)
library(cluster)
library(psych)

#Read in data
dat <- read.table("winequality-red.csv", sep=";", header=TRUE)
nm <- names(dat)
dat <- scale(dat)

#Clustering
ss <- function(x)  sum( ( x-mean(x) )^2 )
wss <- NULL
wss[1] <- sum( apply(dat,2,ss) )
for (k in 2:10) {
  temp <- kmeans(dat, k)
  wss[k] <- sum(temp$withinss)
}

barplot(wss, col="dodgerblue", names.arg=1:length(wss)
        , xlab="Number of Clusters (k)"
        , ylab="Total Within Sum of Squares")
abline(h=0)
title("Within Sum-of-Squares Analysis", col.main="navy")

k <- 8
set.seed(652)
km <- kmeans(dat, k)
clust.km <- km$cluster


#PCA
pc1 <- prcomp(dat)
pc1 <- prcomp(scale(dat))
round(pc1$rotation[,1:2], 3)

pcs <- predict(pc1) 
describe(pcs)[,1:5]
corrplot(cor(pcs)) 

vars <- apply(pcs, 2, var)
sum(vars)

barplot(vars[1:10], col="lightblue", ylab="variance", las=1)
title("Principal Components Analysis Scree Plot", col.main="navy")
abline(h=1:7, col="darkcyan")
abline(h=0)

summary(pc1)

biplot(pc1, col=c("slategrey", "navy"), cex=c(.2, .8))

#Visualizing the clusters
col <- c("blue","dodgerblue","lightgreen","purple","pink","red","brown","orange")

clust <- clust.km
plot(pcs, type="n", main="k-means")
text(pcs, labels=clust, col=col[clust])
abline(h=0); abline(v=0)

for(i in 1:12){
  boxplot( dat[,i] ~ clust, col=col, varwidth=TRUE)
  abline(h=0, col="navy")
  title(nm[i])
}

