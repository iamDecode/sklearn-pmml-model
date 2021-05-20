library(MASS)
library(pmml)
#library(glmnet)
library(randomForest)
library(e1071)

data(Pima.tr2)
#impute
Pima.tr2 <- rfImpute(type ~ ., Pima.tr2)
# undersample
Pima.tr2 = Pima.tr2[order(Pima.tr2$type!="Yes"),]
Pima.tr2 = Pima.tr2[1:212,]
target = 'Yes'
# make attribute categorical
Pima.tr2$age = as.factor(cut(x=Pima.tr2$age, breaks=c(20,30,40,50,60,70)))
Pima.tr2

# test train
#library(caret)
#set.seed(123)
#indices = createDataPartition(Pima.tr2$type, times=1, p=0.6)$Resample1
#train = Pima.tr2[indices,]
#test = Pima.tr2[-indices,]
#indices = createDataPartition(test$type, times=1, p=0.6)$Resample1
#dashboardRecords = test[-indices,]
#test = test[indices,]
#write.csv(train, '/Users/dennis/Downloads/linear-model-train.csv', row.names=FALSE)
##write.csv(test, '/Users/dennis/Downloads/linear-model-test.csv', row.names=FALSE)
#test = read.csv('/Users/dennis/git/sklearn-pmml-model/models/categorical-test.csv', header=TRUE, sep=",")
#x = model.matrix(type ~ ., train)[,-1]
#y = ifelse(train$type == "Yes", 1, 0)
#train$type = y
#test$type = ifelse(test$type == "Yes", 1, 0)


#   cv.clf2 <- cv.glmnet(x, y, alpha = 0)
#   clf2 = glmnet(x, y, alpha = 0, lambda = cv.clf2$lambda.min)
#   # alpha 0 ridge, 1 lasso, in between: elasticnet.
#   # , family = "binomial" for binary outcome
#   pmml_clf2 = pmml(cv.clf2)
#   saveXML(pmml_clf2, "/Users/dennis/Downloads/linear-model-ridge.pmml")
#   predict(cv.clf2, model.matrix(type ~ ., test)[,-1])
#   cv.clf3 <- cv.glmnet(x, y, alpha = 1)
#   clf3 = glmnet(x, y, alpha = 1, lambda = cv.clf3$lambda.min)
#   # alpha 0 ridge, 1 lasso, in between: elasticnet.
#   # , family = "binomial" for binary outcome




#test = read.csv('/Users/decode/Developer/sklearn-pmml-model/models/categorical-test.csv', header=TRUE, sep=",")

#clf = naiveBayes(type ~., data=Pima.tr2)
#pmml_clf = pmml(clf, predicted_field = "type")
#saveXML(pmml_clf, "/Users/dennis/Downloads/naive_bayes.pmml")


#predict(clf, model.matrix(type ~ ., test)[,-1], type="raw")

#preds <- predict(clf, newdata = test)
#conf_matrix <- table(preds, test$type)
#conf_matrix





library(gbm)
library(pmml)
test = read.csv('/Users/decode/Developer/sklearn-pmml-model/models/categorical-test.csv', header=TRUE, sep=",")
test$age = as.factor(test$age)
test$type = as.factor(test$type)
#test$type = ifelse(test$type == "Yes", 1, 0)

clf = gbm::gbm(type ~., data=test, distribution="multinomial")
#clf = xgboost(data=x, label=y, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
#pmml_clf = pmml(clf, predicted_field = "type")
#saveXML(pmml_clf, "/Users/decode/Downloads/gbm.pmml")

options(digits=16)
predict(clf, test)

r2pmml(clf, "pima_gbm.pmml", fmap = as.fmap(as.matrix(test)), response_name = "type", response_levels = c("Yes", "No"), missing = NULL, compact = TRUE)





# LightGBM, unfortunately no (intuitive) PMML export supported at the moment.

#library(lightgbm)
#
#test = read.csv('/Users/decode/Developer/sklearn-pmml-model/models/categorical-test.csv', header=TRUE, sep=",")
#
#x = test[,-1]
#y = ifelse(test$type == "Yes", 1, 0)
#
#clf = lgb.train(data=lgb.Dataset(as.matrix(x), label=y), obj="binary")
##clf = xgboost(data=x, label=y, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
#pmml_clf = pmml(clf, predicted_field = "type")
#saveXML(pmml_clf, "/Users/decode/Downloads/xgboost.pmml")
#
#
#preds = ifelse(predict(clf, as.matrix(x))<0.5, 0, 1)
#
##preds <- predict(clf, newdata = test)
#conf_matrix <- table(preds, test$type)
#conf_matrix




library(xgboost)
test = read.csv('/Users/decode/Developer/sklearn-pmml-model/models/categorical-test.csv', header=TRUE, sep=",")
test$age = as.factor(test$age)
#test$age = as.numeric(as.factor(test$age))
#test$type = ifelse(test$type == "Yes", 1, 0)
#x = as.matrix(test[,-1])
#y = as.matrix(test$type)
x = model.matrix(type ~ ., test)[,-1]
y = ifelse(test$type == "Yes", 1, 0)

#clf = xgboost(data=x, label=y, missing = NULL, max.depth = 100, eta = 1, nthread = 2, nrounds = 100, objective = "binary:logistic")
#clf = xgboost(data=x, label=y, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")

clf = xgboost(data=x, label=y, missing = NULL, nrounds = 5, num_class = 2, objective = "multi:softmax")

# PMML WITH XGBOOST IS ACTUALLY BROKEN!
#pmml_clf = pmml(
#  clf, 
#  input_feature_names=c("npreg","glu","bp","skin","bmi","ped","age"),
#  output_categories=c("Yes", "No"),
#  output_label_name="target",
#  xgb_dump_file = "/Users/decode/Downloads/xgboost.pmml"
#)
#saveXML(pmml_clf, "/Users/decode/Downloads/xgboost.pmml")

library(r2pmml)

r2pmml(clf, 
       "xgboost2.pmml", 
       fmap = as.fmap(as.matrix(x)), 
       response_name = "target", 
       response_levels = c("Yes", "No"), 
       missing = NULL, 
       compact = TRUE
)

preds = as.factor(predict(clf, as.matrix(x)))
levels(preds) <- c('No', 'Yes')
library(caret)
confusionMatrix(preds, as.factor(test$type))






library("xgboost")
library("r2pmml")

data(iris)

iris_X = iris[, 1:4]
iris_y = as.integer(iris[, 5]) - 1

# Generate R model matrix
iris.matrix = model.matrix(~ . - 1, data = iris_X)

# Generate XGBoost DMatrix and feature map based on R model matrix
iris.DMatrix = xgb.DMatrix(iris.matrix, label = iris_y)
iris.fmap = as.fmap(iris.matrix)

# Train a model
iris.xgb = xgboost(data = iris.DMatrix, missing = NULL, objective = "multi:softprob", num_class = 3, nrounds = 13)

predict(iris.xgb, iris.matrix)
library(caret)
confusionMatrix(as.factor(predict(iris.xgb, iris.matrix)), as.factor(iris_y))

# Export the model to PMML.
# Pass the feature map as the `fmap` argument.
# Pass the name and category levels of the target field as `response_name` and `response_levels` arguments, respectively.
# Pass the value of missing value as the `missing` argument
# Pass the optimal number of trees as the `ntreelimit` argument (analogous to the `ntreelimit` argument of the `xgb::predict.xgb.Booster` function)
r2pmml(iris.xgb, "iris_xgb.pmml", fmap = iris.fmap, response_name = "Species", response_levels = c("setosa", "versicolor", "virginica"), missing = NULL, compact = TRUE)

#xgb.plot.tree(model=iris.xgb, trees=0, show_node_id=TRUE)


#library(pmml)
#pmml_iris = pmml(
#  iris.xgb, 
#  input_feature_names=c('Sepal.Length', 'Sepal.Width', 'Petal.Length','Petal.Width'),
#  output_categories=c('setosa', 'versicolor', 'virginica'),
#  output_label_name="Species",
#  xgb_dump_file = "/Users/decode/xgboost.pmml"
#)
#saveXML(pmml_iris, "/Users/decode/xgboost.pmml")



iris.gbm = gbm::gbm(Species ~., data=iris)
predict(iris.gbm, iris_X)


r2pmml(iris.gbm, "iris_gbm.pmml", fmap = iris.fmap, response_name = "Species", response_levels = c("setosa", "versicolor", "virginica"), missing = NULL, compact = TRUE)

#clf = xgboost(data=x, label=y, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
pmml_clf = pmml(iris.gbm, predicted_field = "Species")
saveXML(pmml_clf, "/Users/decode/Downloads/gbm.pmml")
