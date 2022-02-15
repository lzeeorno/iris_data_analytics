
library(plotrix)
# Download the Iris data set,Since it is buildin in R
## we can just import data iris
data(iris)
## we check the data,The data should be 3 classes of 50 instance each.
## total 150 instance,correct. first four rows are correct.
data1 <- iris
head(data1, 4)
attributes(data1)
data1 <- na.omit(data1)  #Removing the missing values
pie3D(table(data1$Species),labels=rownames(table(data1$Species)),explode=0.1)

# Preprocess the data by subtracting the mean and dividing by the
# standard deviation of each attribute value. The resulting data should
# be zero-mean with variance 1.

# get mean and standard deviation first
sepalL_mean = mean(data1$Sepal.Length) #5.843
sepalW_mean = mean(data1$Sepal.Width)  #3.057
PetalL_mean = mean(data1$Petal.Length) #3.758
PetalW_mean = mean(data1$Petal.Width)  #1.199

## check answer with summary which is correct
summary(iris)

sepalL_sd = sd(data1$Sepal.Length)
sepalW_sd = sd(data1$Sepal.Width)
PetalL_sd = sd(data1$Petal.Length)
PetalW_sd = sd(data1$Petal.Width)

data1$Sepal.Length <- (data1$Sepal.Length - sepalL_mean)  / sepalL_sd
data1$Sepal.Width <- (data1$Sepal.Width - sepalW_mean)  / sepalW_sd
data1$Petal.Length <- (data1$Petal.Length - PetalL_mean)  / PetalL_sd
data1$Petal.Width <- (data1$Petal.Width - PetalW_mean)  /PetalW_sd

data1
#check weather variance == 1
sd(data1$Sepal.Length)
sd(data1$Sepal.Width)
sd(data1$Petal.Length)
sd(data1$Petal.Width)

#Compute the covariance matrix.  
data_cov <- data.frame(Sepal.Length = data1$Sepal.Length,
                       Sepal.Width = data1$Sepal.Width,
                       Petal.Length = data1$Petal.Length,
                       Petal.Width = data1$Petal.Width)

cov_data <- cov(data_cov)
cov_data
#result:
#Sepal.Length Sepal.Width Petal.Length Petal.Width
#Sepal.Length    1.0000000  -0.1175698    0.8717538   0.8179411
#Sepal.Width    -0.1175698   1.0000000   -0.4284401  -0.3661259
#Petal.Length    0.8717538  -0.4284401    1.0000000   0.9628654
#Petal.Width     0.8179411  -0.3661259    0.9628654   1.0000000

#(d) Factorize the covariance matrix using singular value decomposition
#and obtain the eigenvalues and eigenvector.

#data_SVD
SVD <- svd(cov_data)
SVD
# eigenvalues
SVD_diag <- diag(SVD$d)
SVD_diag

#the The singular values are 25, 6.013, 3.41, 1.88.
#check variance:
vardata1 = sum(SVD_diag**2)
vardata1 #
t_var = sum(SVD_diag**2)
t_var
#total_var = SVD singular variance = 9.37, correct

#get eigenvalues and eigenvector:
eigenvalues <- c(SVD_diag[1,1],SVD_diag[2,2],SVD_diag[3,3],SVD_diag[4,4])
eigenvalues
eigenvector <- SVD$u
eigenvector





#Project the data onto its first two principal components and plot
#the results.


#get PCA from 3-32 b/c not included id and diagnosis
iris.pr <- prcomp(data_cov, center = TRUE, scale = TRUE)
#plot only pc1 and pc2
plot(iris.pr$x[,1],iris.pr$x[,2], xlab="PC1", ylab = "PC2", main = "PC1 / PC2 - plot")

library("factoextra") 
fviz_pca_ind(iris.pr, geom.ind = "point", pointshape = 21, 
             pointsize = 2, 
             fill.ind = iris$Species, 
             col.ind = "black", 
             palette = "jco", 
             addEllipses = TRUE, 
             label = "var", 
             col.var = "black",
             repel = TRUE, 
             legend.title = "Species") + 
  ggtitle("2D PCA-plot") + 
  theme(plot.title = element_text(hjust = 0.5))


#Classification using the Iris data: trained a classifier for the Iris data
#using (a) Bayes’ LDA and (b) any other method 
library(MASS)
library(ggplot2)
#(a): 
iris_data <- data.frame(Sepal.Length = iris$Sepal.Length,
                       Sepal.Width = iris$Sepal.Width,
                       Petal.Length = iris$Petal.Length,
                       Petal.Width = iris$Petal.Width)
#scale each predictor variable (i.e. first 4 columns)
#iris[1:4] <- scale(iris[1:4])

#define a train- / test-split, splite data
#Use 70% of dataset as training set and remaining 30% as testing set
sample <- sample(c(TRUE, FALSE), nrow(iris_data), replace=TRUE, prob=c(0.5,0.5))
train <- iris[sample, ]
test <- iris[!sample, ] 

#fit LDA model
model <- lda(Species ~ Sepal.Width + Sepal.Length + Petal.Width + Petal.Length , data=train)
#view model output
model
#Therefore, model is the linear combination of predictor variables 
#that are used to form the decision rule of the LDA model.
#LD1 = 0.34*Sepal.Length + 0.7*Sepal.Width - 3.44*Petal.Length - 2.11*Petal.Width
#LD2 = 0.17*Sepal.Length + 0.77*Sepal.Width - 2.15*Petal.Length + 2.43*Petal.Width

#use LDA model to make predictions on test data
predicted <- predict(model, test)
predicted

#find accuracy of model, since the sample size is small we use mean accuracy to avoid bias
acc = 0
for(i in 1:20){
  c = mean(predicted$class==test$Species)
  acc = acc + c
}

accuracy = acc/20
accuracy
#70%training-30%testing give 0.981 accuracy
#50%-50% give 0.945 acc
#30%training-70%testing give 0.93

#visualize result
#define data to plot
lda_plot <- cbind(train, predict(model)$x)

#create plot
ggplot(lda_plot, aes(LD1, LD2)) +
  geom_point(aes(color = Species))


#(b): we also have linear regression, logistic regression, poission regression,
#decision tree, support vector machines(SVM), knn, k-means

#here we try 
#support vector machines
# Fitting SVM to the Training set
#install.packages('e1071')
library(e1071)
model.svm <- svm(Species~., data=train,kernel = "linear", cost = 10, scale = FALSE)
print(model.svm)


#find accuracy of model, since the sample size is small we use mean accuracy to avoid bias
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) }

SVM_tab <- table(predict(model.svm,train),train[,5])
SVM_tab
accuracy(SVM_tab)
#accuracy = 0.963


#LDA assume data is normally distributed, but SVM makes no assumptions about data at all.
#Therefore, LDA is the best discriminator when all assumptions are actually met,
#SVM is a very flexible method so that it is hard to interpret the results from a SVM classifier, compared to LDA.
#In conclusion, SVM focuses on points that are difficult to classify, and LDA focuses on all data points. 
