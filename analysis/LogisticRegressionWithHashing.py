import math
import datetime

from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil

class Weights:
  def __init__(self, featuredim):
    self.featuredim = featuredim
    self.w0 = self.w_age = self.w_gender = self.w_depth = self.w_position = 0
    # hashed feature weights
    self.w_hashed_features = [0.0 for _ in range(featuredim)]
    # to keep track of the access timestamp of feature weights.
    #   use this to do delayed regularization.
    self.access_time = {}

  def __str__(self):
    formatter = "{0:.2f}"
    string = ""
    string += "Intercept: " + formatter.format(self.w0) + "\n"
    string += "Depth: " + formatter.format(self.w_depth) + "\n"
    string += "Position: " + formatter.format(self.w_position) + "\n"
    string += "Gender: " + formatter.format(self.w_gender) + "\n"
    string += "Age: " + formatter.format(self.w_age) + "\n"
    string += "Hashed Feature: "
    string += " ".join([str(val) for val in self.w_hashed_features])
    string += "\n"
    return string

  # @return {Double} the l2 norm of this weight vector
  def l2_norm(self):
    l2 = self.w0 * self.w0 +\
          self.w_age * self.w_age +\
          self.w_gender * self.w_gender +\
          self.w_depth * self.w_depth +\
          self.w_position * self.w_position
    for w in self.w_hashed_features:
      l2 += w * w
    return math.sqrt(l2)


class LogisticRegressionWithHashing:
  # ==========================
  # Helper function to compute inner product w^Tx.
  # @param weights {Weights}
  # @param instance {DataInstance}
  # @return {Double}
  # ==========================
  def compute_weight_feature_product(self, weights, instance):
    result = weights.w0 * 1 + \
             weights.w_age * instance.age + \
             weights.w_gender * instance.gender + \
             weights.w_depth * instance.depth + \
             weights.w_position * instance.position
    for x in instance.hashed_text_feature:   # x should be int
      result += weights.w_hashed_features[x] * instance.hashed_text_feature[x]
    return result
  
  # ==========================
  # Apply delayed regularization to the weights corresponding to the given
  # tokens.
  # @param featureids {[Int]} list of feature ids
  # @param weights {Weights}
  # @param now {Int} current iteration
  # @param step {Double} step size
  # @param lambduh {Double} lambda
  # ==========================
  def perform_delayed_regularization(self, featureids, weights, now, step, lambduh):
    for x in featureids:   # x is int
      if x in weights.access_time:
        weights.w_hashed_features[x] *= pow(1-step*lambduh, now-weights.access_time[x]-1)
      else:
        weights.w_hashed_features[x] = 0.0
      weights.access_time[x] = now
    return
  
  # ==========================
  # Train the logistic regression model using the training data and the
  # hyperparameters. Return the weights, and record the cumulative loss.
  # @return {Weights} the final trained weights.
  # ==========================
  def train(self, dataset, dim, lambduh, step, avg_loss, personalized):
    weights = Weights(dim)
    loss = 0.0
    count = 0
    while dataset.hasNext():
      count += 1
      instance = dataset.nextHashedInstance(dim, personalized)

      # perform delayed regularizaton
      if lambduh:
        self.perform_delayed_regularization(instance.hashed_text_feature,weights,count,step,lambduh)

      inner_prod = -1*self.compute_weight_feature_product(weights, instance)
      sigmoid = 1.0/(1+math.exp(inner_prod))
      #yhat = 1 if sigmoid > 0.5 else 0

      update1 = step*(instance.clicked-sigmoid)
      update2 = lambduh * step

      weights.w0 += update1
      weights.w_age += update1 * instance.age - update2 * weights.w_age
      weights.w_gender += update1 * instance.gender - update2 * weights.w_gender
      weights.w_depth += update1 * instance.depth - update2 * weights.w_depth
      weights.w_position += update1 * instance.position - update2 * weights.w_position
      for x in instance.hashed_text_feature:   # x is int
        weights.w_hashed_features[x] += instance.hashed_text_feature[x] * update1 - update2 * weights.w_hashed_features[x]

      if not count % 100000:
        print(str(count), "data points trained...")
      if count == dataset.size:
        print(str(count), "total data points trained !")

      # perform lazy regularization
      if lambduh:
        for i in range(dim):
          weights.w_hashed_features[i] *= pow(1 - update2, count - weights.access_time.get(i,count))

      #loss += pow(instance.clicked-yhat,2)

    dataset.reset()
    return weights

  # ==========================
  # Use the weights to predict the CTR for a dataset.
  # @param weights {Weights}
  # @param dataset {DataSet}
  # @param personalized {Boolean}
  # ==========================
  def predict(self, weights, dataset, personalized):
    y_predicted = []
    count = 0
    while dataset.hasNext():
      count += 1
      instance = dataset.nextHashedInstance(weights.featuredim, personalized)
      inner_prod = -1*self.compute_weight_feature_product(weights,instance)
      prob_1 = 1.0/(1+math.exp(inner_prod))
      y_predicted.append(prob_1)
      #y_predicted.append(1) if prob_1 > 0.5 else y_predicted.append(0)

      if not count % 100000:
        print(str(count), "data points predicted")
      if count == dataset.size:
        print(str(count), "total data points predicted !")

    dataset.reset()
    return y_predicted
  
  
if __name__ == '__main__':
  print ("Training Logistic Regression with Hashed Features...\n")
  start = datetime.datetime.now()

  training = DataSet("../data/train.txt", True, 2335859)
  test = DataSet("../data/test.txt", False, 1016552)
  step = 0.01
  lambduh = 0.001
  #dims = [101,12277,1573549]
  #personalized = 0

  training_LR = LogisticRegressionWithHashing()

  weights_train = training_LR.train(dataset=training,dim=12277,lambduh=lambduh,step=step,avg_loss=0,personalized=False)
  y_predicted = training_LR.predict(weights=weights_train,dataset=test,personalized=False)
  RMSE = EvalUtil.eval("../data/test_label.txt", y_predicted)
  print("Rooted mean square error (RMSE) between the prediction and the true labels is %.8f \n" % RMSE)
  F = EvalUtil.error_analysis("../data/test_label.txt", y_predicted)
  print("F-score with hash kernels is %.5f " % F,'\n')

  file = open("rmse1.txt","w")
  file.write("RMSE is: %f." % RMSE)
  file.close()

  print("Train Logistic Regression with Personalization...\n")
  #dim = 12277
  #personalized = 1
  LR_Per = LogisticRegressionWithHashing()
  weights_Per = LR_Per.train(dataset=training,dim=12277,lambduh=lambduh,step=step,avg_loss=0,personalized=True)
  y_predicted_Per = LR_Per.predict(weights=weights_Per,dataset=test,personalized=True)
  RMSE_Per = EvalUtil.eval("../data/test_label.txt", y_predicted_Per)
  print("Rooted mean square error (RMSE) between the prediction under personalization and the true labels is %.8f \n" % RMSE_Per)
  F_Per = EvalUtil.error_analysis("../data/test_label.txt", y_predicted_Per)
  print("F-score under personalization is %.5f" % F_Per,'\n')

  # for subset of the data sharing the same users
  common_list = []
  training_users = set()
  while training.hasNext():
    training_users.add(training.nextInstance().userid)
  training.reset()
  while test.hasNext():
    uid = test.nextInstance().userid
    if uid in training_users:
      common_list.append(1)
    else:
      common_list.append(0)
  test.reset()

  print("Predict for common users...\n")
  RMSE_Comm = EvalUtil.eval_with_including_list("../data/test_label.txt", y_predicted_Per, common_list)
  print ('RMSE on subset: %.8f ' % RMSE_Comm,'\n')

  end = datetime.datetime.now()
  elapsed = end - start
  print("Elapsed time is :", elapsed)