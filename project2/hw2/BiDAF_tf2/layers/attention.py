import tensorflow as tf

class C2QAttention(tf.keras.layers.Layer):

    def call(self, similarity, qencode):
        # homework 
        # print(similarity.shape)
        # print(qencode.shape)
        
        # 根据论文对similarity matrix做row-wise softmax
        softmax_similarity_matrix = tf.keras.activations.softmax(similarity, axis=-1)

        # 给similarity matrix加一个column维度，给Query matrix加一个row维度，才能使两个matrix相乘
        softmax_similarity_matrix = tf.expand_dims(softmax_similarity_matrix, axis=-1)
        qencode = tf.expand_dims(qencode, axis=1)
        
        # 相乘后对row维度进行reduce sum变回原来的维度
        c2q_att = tf.math.reduce_sum(softmax_similarity_matrix * qencode, axis=-2)

        return c2q_att

class Q2CAttention(tf.keras.layers.Layer):

    def call(self, similarity, cencode):
        # homework 
        # 按照论文对每个row取最大值从而得到一个column vector
        max_similarity = tf.math.reduce_max(similarity, axis=-1)
        print('max_similarity:', max_similarity.shape)
        # 通过softmax来得到attention distribution
        att_distribution = tf.keras.activations.softmax(max_similarity)
        print('att_distribution:', att_distribution.shape)
        # 增加一个column维度
        att_distribution = tf.expand_dims(att_distribution, axis=-1)
        print('att_distribution:', att_distribution.shape)

        # attention distribution与context matrix相乘并reduce sum来求得在context里 每个词的weighted sum用以找到最重要的词
        weighted_sum = tf.math.reduce_sum(att_distribution * cencode, axis=-2)
        print('weighted_sum:', weighted_sum.shape)
        weighted_sum = tf.expand_dims(weighted_sum, 1)
        print('weighted_sum:', weighted_sum.shape)
        # 重复cencode.shape[1]次，因为cencode里每一个column代表context里的每一个词
        num_repeat = cencode.shape[1]
        print('cencode:', cencode.shape)
        # 通过tf.tile按照论文描述把weighted sum重复num_repeat次，生成q2c的attention matrix
        q2c_att = tf.tile(weighted_sum, [1, num_repeat, 1])
        print('q2c_att:', q2c_att.shape)
        
        return q2c_att
