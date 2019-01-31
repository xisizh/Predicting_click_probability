from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil

class BasicAnalysis:
  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique tokens in the dataset
  # ==========================


  def __init__(self):
    self.ave_crt = 0.0
    self.unique_tokens = set()
    self.unique_users = set()
    self.user_by_age = dict([(i, 0) for i in range(7)])

  def analysis(self, dataset):
    count_click, count = 0.0, 0
    token_list = []
    user_list = [[], [], [], [], [], [], []]
    while dataset.hasNext():
      instance = dataset.nextInstance()
      count_click += instance.clicked
      count += 1
      if not count % 500000:
        print(str(count), "lines load...")
      if count == dataset.size:
        print(str(count), "total lines load!")

      #for x in instance.tokens:
          #self.unique_tokens.add(x)
      token_list += instance.tokens

      user_list[instance.age].append(instance.userid)

    self.ave_crt = count_click / dataset.size

    self.unique_tokens = set(token_list)

    self.unique_users = [set(x) for x in user_list]
    count_by_age = list([len(x) for x in self.unique_users])
    for x in self.user_by_age:
      self.user_by_age[x] = count_by_age[int(x)]

    dataset.reset()

  def uniq_tokens(self):
    return self.unique_tokens
  
  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique user ids in the dataset
  # ==========================
  def uniq_users(self):
    return self.unique_users

  # ==========================
  # @param dataset {DataSet}
  # @return {Int: [{Int}]} a mapping from age group to unique users ids
  #                        in the dataset
  # ==========================
  def uniq_users_per_age_group(self):
    return self.user_by_age

  # ==========================
  # @param dataset {DataSet}
  # @return {Double} the average CTR for a dataset
  # ==========================
  def average_ctr(self):
    return self.ave_crt

if __name__ == '__main__':
  print ("Basic Analysis...")
  train_analysis = BasicAnalysis()
  test_analysis = BasicAnalysis()

  size_train = 2335859
  size_test = 1016552
  
  training = DataSet("../data/train.txt", True, size_train)
  test = DataSet("../data/test.txt", False, size_test)

  print("Load training data")
  train_analysis.analysis(training)
  print("Load test data")
  test_analysis.analysis(test)

  ave_crt = train_analysis.average_ctr()
  print("Average CTR for training data is %.8f" % ave_crt)

  unique_tokens_train = train_analysis.uniq_tokens()
  unique_tokens_test = test_analysis.uniq_tokens()
  
  print("# of unique tokens for training data is %d" % len(unique_tokens_train))
  print("# of unique tokens for test data is %d" % len(unique_tokens_test))
  print("# of unique tokens in both set is %d" % len(unique_tokens_train & unique_tokens_test))

  user_train = train_analysis.uniq_users_per_age_group()
  user_test = test_analysis.uniq_users_per_age_group()
  print("# of unique users in age group 0 to 6 in training data : ", user_train)
  print("# of unique users in age group 0 to 6 in test data : ", user_test)


