train=read.csv("E:\\DA Projects(1)\\Loancrossselling problem\\train" ,na.strings = c(""," ","NA") )
str(train)
summary(train)
library(DataExplorer)
plot_missing(train)
create_report(train)
library(esquisse)
esquisser(train)
View(train)
sum(is.na(train$Loan_ID))
sum(is.na(train$Gender))
train$Gender[which(is.na(train$Gender))]<-'Male'
sum(is.na(train$Married))
train$Married[which(is.na(train$Married))]<-'Yes'
sum(is.na(train$Dependents))
train$Dependents[which(is.na(train$Dependents))]<-'0'
sum(is.na(train$Education))
sum(is.na(train$Self_Employed))
train$Self_Employed[which(is.na(train$Self_Employed))]<-'No'
sum(is.na(train$ApplicantIncome))
sum(is.na(train$CoapplicantIncome))
sum(is.na(train$LoanAmount))
train$LoanAmount[which(is.na(train$LoanAmount))]<-146.4
sum(is.na(train$Loan_Amount_Term))
train$Loan_Amount_Term[which(is.na(train$Loan_Amount_Term))]<-342
sum(is.na(train$Credit_History))
train$Credit_History=as.factor(train$Credit_History)
train$Credit_History[which(is.na(train$Credit_History))]<-'1'
sum(is.na(train$Property_Area))
sum(is.na(train$Loan_Status))
sum(is.na(train))
library(mosaic)
library(manipulate)
mplot(train)
1
mplot(train)
2
mplot(train)
3

train=train[c(2:13)]

library(GGally)
ggcorr(train,label = TRUE)

summary(train)
train[6:9]=scale(train[6:9])

library(dummies)
str(train)
train=cbind(train,dummy(train$Gender,sep='_'))
train=cbind(train,dummy(train$Married,sep='_'))
train=cbind(train,dummy(train$Dependents,sep='_'))
train=cbind(train,dummy(train$Education,sep='_'))
train=cbind(train,dummy(train$Self_Employed,sep='_'))
train=cbind(train,dummy(train$Credit_History,sep='_'))
train=cbind(train,dummy(train$Property_Area,sep='_'))
str(train)

train=train[c(6:9,12,13,15,17,18,19,21,23,25,27,28)]
str(train)
library(DataExplorer)
plot_missing(train)
create_report(train)
library(esquisse)
esquisser(train)

library(caTools)
set.seed(123)
split <- sample.split(train$Loan_Status, SplitRatio = 0.70)
training_set <- subset(train, split == TRUE)
test_set <- subset (train, split == FALSE)
View(training_set)
View(test_set)
#Various Model Fitting over Survived
#1. Logistic Regression Model
logisticmodel <- glm(Loan_Status ~ ., family = 'binomial', data = training_set)

summary(logisticmodel)
#Predict the model
prediction <- predict (logisticmodel, type = 'response', newdata = test_set[-5]) 
prediction

#Make Binary Predictions
y_pred <- ifelse(prediction > 0.5, 'Y', 'N')
y_pred
y_pred=as.factor(y_pred)
summary(y_pred)

#Make the Confusion Matrix
library(caret)
confusionMatrix(data = y_pred,reference = test_set$Loan_Status)

#Accuracy : 84.86%          
library(pROC)
roc_obj <- roc(as.numeric(test_set[, 5]),as.numeric (y_pred))
auc(roc_obj) 

#AUC = 0.7633

#2. k-NN model
library(class)
y_pred <- knn(train = training_set [, -5], 
              test = test_set[, -5],
              cl = training_set[, 5],
              k = 5)
y_pred
#Confusion Matrix
confusionMatrix(data = y_pred,reference = test_set$Loan_Status)
#Accuraccy=59.46%
#ROC Curve
#Plotting the ROC CUrve and AUC
library(pROC)
roc_obj <- roc(as.numeric(test_set[, 5]), as.numeric(y_pred))
auc(roc_obj)
#AUC = 0.4752

#3. SVM Model kernel=linear
library(e1071)
classifier <- svm(formula = Loan_Status ~ .,
                  data = training_set,
                  type = 'C-classification',
                  kernel = 'linear')
classifier
#Predicting the Test Set Result
y_pred <- predict (classifier, newdata = test_set[-5]) 
y_pred

#Confusion Matrix 
confusionMatrix(data = y_pred,reference = test_set$Loan_Status)
#Accuracy=84.86
#Find the ROC and AUC
library(pROC)
roc_obj <- roc(as.numeric(test_set[, 5]), as.numeric(y_pred))
auc(roc_obj)
#AUC = 0.7633

#4. SVM kernel=radial
library(e1071)
classifier <- svm(formula = Loan_Status ~ .,
                  data = training_set,
                  type = 'C-classification',
                  kernel = 'radial')
classifier

#Predicting the Test Set Result
y_pred <- predict (classifier, newdata = test_set[-5]) 
y_pred

#Confusion Matrix
confusionMatrix(data = y_pred,reference = test_set$Loan_Status)
#Accuracy=83.78
#Fitting the ROC and AUC
library(pROC)
roc_obj <- roc(as.numeric(test_set[, 5]), as.numeric(y_pred))
auc(roc_obj)
#AUC = 0.7461

#5. CART Model
library(rpart)
library(rpart.plot)
classifier <- rpart(formula = Loan_Status ~.,
                    data = training_set)

classifier
#Plotting the Decision Tree
plot(classifier)
text(classifier)
print(classifier)
summary(classifier)
rpart.plot(x =classifier, yesno = 2, type = 0, extra = 0)
#Predicting the Test Set Result
y_pred <- predict (classifier, newdata = test_set[-5], type = 'class') 
y_pred 

#Confusion Matrix 
confusionMatrix(data = y_pred,reference = test_set$Loan_Status)

#Accuracy = 84.86%
#Fitting the ROC and AUC
library(pROC)
roc_obj <- roc(as.numeric(test_set[, 5]), as.numeric(y_pred))
auc(roc_obj)
#AUC = 0.7633

#6. Random Forest Model
library(randomForest)
classifier <- randomForest(x = training_set[-5],
                           y = training_set$Loan_Status,
                           ntree = 1000)
classifier

#Predicting the Test Set Result
y_pred <- predict (classifier, newdata = test_set[-5]) 
y_pred

#Confusion Matrix
confusionMatrix(data = y_pred,reference = test_set$Loan_Status)

#Accuracy = 84.32%
#Fitting the ROC and AUC
library(pROC)
roc_obj <- roc(as.numeric(test_set[, 5]), as.numeric(y_pred))
auc(roc_obj)
#AUC = 0.7594

#7. Naive Bayes 
#Fitting the Naive Bayes Model
library(e1071)
classifier <- naiveBayes(x = training_set[-5],
                         y = training_set$Loan_Status)
classifier

#Predicting the Test Set Result
y_pred <- predict (classifier, newdata = test_set[-5]) 
y_pred

#Making the Confusion Matrix. 
confusionMatrix(data = y_pred,reference = test_set$Loan_Status)

#Accuracy = 85.41%
#Fitting the ROC and AUC
library(pROC)
roc_obj <- roc(as.numeric(test_set[,5]),as.numeric(y_pred))
auc(roc_obj)
#AUC = 0.7719

#8. C5.0 
#Fitting the C5.0 Model
library(C50)
C50_mod<- C5.0(training_set[,-5],training_set[,5])
print(C50_mod)
summary(C50_mod)
plot(C50_mod)
pred=predict(C50_mod,test_set,type="class")
table(test_set[,5],pred)
confusionMatrix(data = pred,
                reference = test_set$Loan_Status) 
#Accuracy : 0.8486
#Fitting the ROC and AUC
library(pROC)
roc_obj <- roc(as.numeric(test_set[, 5]), as.numeric(y_pred))
auc(roc_obj)
#AUC = 0.7719
#Conclusion:We came to a conclusion that naive bayes is the most suitable model having Accuracy = 85.41% and 0.7719 area under ROC curve= 0.7557  
