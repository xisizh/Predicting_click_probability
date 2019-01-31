import math
import matplotlib.pyplot as plt
import datetime

from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil
from analysis.BasicAnalysis import BasicAnalysis



# This class represents the we ights in the logistic regression model.
class Weights:
  def __init__(self):
    self.w0 = self.w_age = self.w_gender = self.w_depth = self.w_position = 0
    # token feature weights
    self.w_tokens = {}
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
    string += "Tokens: " + str(self.w_tokens) + "\n"
    return string
  
  # @return {Double} the l2 norm of this weight vector
  def l2_norm(self):
    l2 = self.w0 * self.w0 +\
          self.w_age * self.w_age +\
          self.w_gender * self.w_gender +\
          self.w_depth * self.w_depth +\
          self.w_position * self.w_position
    for w in self.w_tokens.values():
      l2 += w * w
    return math.sqrt(l2)
  
  # @return {Int} the l2 norm of this weight vector
  def l0_norm(self):
    return 4 + len(self.w_tokens)


class LogisticRegression:

  def __init__(self):
    self.average_loss = []

  # ==========================
  # Helper function to compute inner product w^Tx.
  # @param weights {Weights}
  # @param instance {DataInstance}
  # @return {Double}
  # ==========================
  def compute_weight_feature_product(self, weights, instance):  # assume weights have same len with unique_tokens and instance.tokens are binary list
    result = weights.w0 * 1 + \
             weights.w_age * instance.age + \
             weights.w_gender * instance.gender + \
             weights.w_depth * instance.depth + \
             weights.w_position * instance.position
    for x in instance.tokens:   # should loop in instance.tokens, because we only care non-zero value features
      result += weights.w_tokens.get(x,0)
    return result

  # ==========================
  # Apply delayed regularization to the weights corresponding to the given
  # tokens.
  # @param tokens {[Int]} list of tokens
  # @param weights {Weights}
  # @param now {Int} current iteration
  # @param step {Double} step size
  # @param lambduh {Double} lambda
  # ==========================
  def perform_delayed_regularization(self, tokens, weights, now, step, lambduh):
    for x in tokens:
      if x in weights.access_time:
        weights.w_tokens[x] *= pow(1-step*lambduh, now-weights.access_time[x]-1)
      else:
        weights.w_tokens[x] = 0.0
      weights.access_time[x] = now
    return

  # ==========================
  # Train the logistic regression model using the training data and the
  # hyperparameters. Return the weights, and record the cumulative loss.
  # @return {Weights} the final trained weights.
  # ==========================
  def train(self, dataset, lambduh, step, avg_loss):
    weights = Weights()

    loss = 0.0
    count = 0
    while dataset.hasNext():
      count += 1
      instance = dataset.nextInstance()
      y = instance.clicked

      # perform delayed regularization if lambduh is not 0
      if lambduh:
        self.perform_delayed_regularization(instance.tokens,weights,count,step,lambduh)

      inner_prod = -1.0*self.compute_weight_feature_product(weights, instance)
      sigmoid = 1.0/(1+math.exp(inner_prod))  # prob of y = 1
      yhat = 1 if sigmoid > 0.5 else 0

      # update weights (gradient ascent not gradient descent)
      # avoid over compute step * (y - sigmoid) and lambduh * step
      update1 = step * (y - sigmoid)
      update2 = lambduh * step

      weights.w0 += update1
      weights.w_age += update1 * instance.age - update2 * weights.w_age
      weights.w_gender += update1 * instance.gender - update2 * weights.w_gender
      weights.w_depth += update1 * instance.depth - update2 * weights.w_depth
      weights.w_position += update1 * instance.position - update2 * weights.w_position
      for x in instance.tokens:
        #weights.w_tokens[x] += update1 - update2 * weights.w_tokens[x]
        weights.w_tokens[x] = weights.w_tokens.get(x, 0) + update1 - update2 * weights.w_tokens.get(x, 0)

      if not count % 100000:
        print(str(count), "data points trained...")
      if count == dataset.size:
        print(str(count), "total data points trained !")

      # computer average loss
      loss += pow(y - yhat,2)
      if not count % 100:
        self.average_loss.append(loss/count)

    # perform lazy regularization
    if lambduh:
      for x in weights.w_tokens:
        weights.w_tokens[x] *= pow(1-update2, count-weights.access_time[x])

    dataset.reset()
    return weights

  # ==========================
  # Helper function to plot the average losses
  # ==========================
  def plot_ave_losses(self, eta):
    x = range(100, (len(self.average_loss)+1) * 100, 100)
    plt.plot(x, self.average_loss, 'ob-')
    plt.xlabel('Steps of iteration')
    plt.ylabel('Average Loss')
    plt.title('Average Losses $\overline{L}$ vs. Steps $T$ , $\lambda = 0$, $\eta = $' + str(eta))
    plt.show()

  # ==========================
  # Use the weights to predict the CTR for a dataset.
  # @param weights {Weights}
  # @param dataset {DataSet}
  # ==========================
  def predict(self, weights, dataset):
    y_predict = []
    count = 0
    while dataset.hasNext():
      count += 1
      instance = dataset.nextInstance()
      inner_prod = -1*self.compute_weight_feature_product(weights, instance)
      prob_1 = 1/(1 + math.exp(inner_prod))
      y_predict.append(1) if prob_1>0.5 else y_predict.append(0)

      if not count % 100000:
        print(str(count), "data points predicted")
      if count == dataset.size:
        print(str(count), "total data points predicted !")

    dataset.reset()
    return y_predict
  
  
if __name__ == '__main__':
  start = datetime.datetime.now()
  print ("Training Logistic Regression...")

  training = DataSet("../data/train.txt", True, 2335859)
  test = DataSet("../data/test.txt", False, 1016552)

  training_LR = LogisticRegression()

  #step = 0.001
  #weights_training = training_LR.train(training,0,0.001,0)
  #print("L2-norm when eta = 0.001: ", weights_training.l2_norm())
  #training_LR.plot_ave_losses(0.001)

  step = 0.05
  lambduhs = [i*0.002 for i in range(8)]
  l2_norms = []
  for lambduh in lambduhs:
    print ("Training Logistic Regression with lambda = %.3f..." % lambduh)
    weights_training = training_LR.train(training, lambduh, step, 0)
    l2_norms.append(weights_training.l2_norm())
  plt.plot(lambduhs, l2_norms, 'ob-')
  plt.xlabel('$\lambda$')
  plt.ylabel('$l_2$ Norm')
  plt.title('$l_2$ Norms vs. $\lambda$\'s, $\eta = 0.05$')
  plt.show()

  end = datetime.datetime.now()
  elapsed = end - start
  print("Elapsed time is :", elapsed)

  #y_predicted = training_LR.predict(weights_training, test)
  #print ("RMSE: %f " % EvalUtil.eval("/Users/xisizhang/Desktop/data/test_label.txt", y_predicted))