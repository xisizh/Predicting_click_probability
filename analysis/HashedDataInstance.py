from util.HashUtil import HashUtil

class HashedDataInstance:
  
  # =====================
  # constructor
  # @param self
  # @param line {String}
  # @param has_label {Boolean}
  # @param dim {Int}
  # @param personal {Boolean}
  # =====================
  def __init__(self, line, has_label, dim, personal):
    fields = line.split("|")
    offset = 0
    
    if has_label:
      # whether clicked: 0 or 1
      self.clicked = int(fields[0])
      offset = 1
    else:
      self.clicked = -1

    # depth of the session
    self.depth = int(fields[offset])
    # position of the ad
    self.position = int(fields[offset + 1])
    self.userid = int(fields[offset + 2])
    # user gender indicator (-1 for male, 1 for female)
    self.gender = int(fields[offset+3])
    if self.gender != 0:
      self.gender = int((self.gender - 1.5) * 2)
    # user age indicator:
    #   '1' for (0, 12],
    #   '2' for (12, 18],
    #   '3' for (18, 24],
    #   '4' for (24, 30],
    #   '5' for (30, 40], and
    #   '6' for greater than 40.
    self.age = int(fields[offset + 4])
    # list of token ids
    tokens = [int(xx) for xx in fields[offset + 5].split(",")]
    
    self.featuredim = dim
    self.hashed_text_feature = {}

    for xx in tokens:
        self.update_feature(key=HashUtil.hash_to_range(xx,dim), val=HashUtil.hash_to_sign(xx))

    if personal:
      for xx in tokens:
        # in order to distinguish token and userid, we multiply token by a large num such as 10000000. Then convert to string
        self.update_feature(key=HashUtil.hash_to_range(xx * 10000000 + self.userid, dim),
                            val=HashUtil.hash_to_sign(xx * 10000000 + self.userid))
      pass

  # =====================
  # Helper function. Updates the feature hashmap with a given key and value.
  # You can use HashUtil.hash_to_range as h, and HashUtil.hash_to_sign as \xi. 
  # @param key {String}   But key should be {int}?
  # @param val {Int}
  # =====================
  def update_feature(self, key, val):
    self.hashed_text_feature[key] = self.hashed_text_feature.get(key,0) + val*1.0
    pass