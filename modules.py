import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense,LayerNormalization,Embedding,Dropout
from tokeniser import Subword_Tokeniser

#**************FOR TF 2.11***************

class Positional_Encode(tf.Module):
    def __init__(self):
        self.encoding = 0

    def build(self,input_shape):
        batch,window_size,emb_dim = input_shape
        if batch == None: batch = 1

        encoding = np.zeros((window_size,emb_dim))

        for i in range(window_size):
            for j in range(int(emb_dim / 2)):
                encoding[i][2 * j] = np.sin(i / np.power(10000,2 * j / emb_dim))
                encoding[i][2 * j + 1] = np.cos(i / np.power(10000,2 * j / emb_dim))
        print("hi")
        self.encoding = tf.convert_to_tensor(np.repeat(encoding.astype("float32"),batch,axis = 1))

    def __call__(self,inputs):
        return inputs + self.encoding

class Attention_Head(tf.Module):
    def __init__(self,emb_dim,head_size = 32,window_size = 64):
        self.emb_dim = emb_dim
        self.head_size = head_size

        self.query = Dense(self.head_size,use_bias = False)
        self.key = Dense(self.head_size,use_bias = False)
        self.value = tf.Variable(np.random.rand(emb_dim).astype("float32"),trainable = True)

        self.attention_mask = tf.experimental.numpy.triu(np.full((window_size,window_size),-np.inf).astype("float32"),1)

    def __call__(self,inputs):
        b,t,v = inputs.shape

        queries = self.query(inputs)
        keys = self.key(inputs)
        values = tf.einsum("btv,v->btv",inputs,self.value)

        queries /= (self.emb_dim ** (0.25))
        keys /= (self.emb_dim ** (0.25))

        weights = tf.matmul(queries,tf.transpose(keys,perm = [0,2,1])) + self.attention_mask
        softmax_weights = tf.nn.softmax(weights,axis = 2)

        output = tf.einsum("btt,btv->btv",softmax_weights,values)

        return output

class Multi_Head_Attention(tf.Module):
    def __init__(self,emb_dim,num_heads = 8,head_size = 32,window_size = 64):
        self.emb_dim = emb_dim
        self.heads = [Attention_Head(emb_dim,head_size = head_size,window_size = window_size) for _ in range(num_heads)] 
        self.concat = tf.Variable(np.random.rand(emb_dim,emb_dim).astype("float32"),trainable = True)
        self.dropout = Dropout(0.15)

    def __call__(self,inputs):
        head_outputs = tf.transpose(tf.convert_to_tensor([head(inputs) for head in self.heads]),perm = [1,2,3,0])
        concatenated_outputs = tf.einsum("btvh,vv->btv",head_outputs,self.concat)
        output = self.dropout(concatenated_outputs)

        return output

class Transformer_Block(tf.Module):
    def __init__(self,emb_dim,num_heads = 8,head_size = 32,window_size = 64):
        self.multi_head_attention = Multi_Head_Attention(emb_dim,num_heads = num_heads,head_size = head_size,window_size = window_size)
        self.norm_1 = LayerNormalization()
        self.norm_2 = LayerNormalization()
        self.feedforward = tf.keras.Sequential([Dense(4 * emb_dim,activation = "relu"),Dense(emb_dim),Dropout(0.15)])

    def __call__(self,inputs):
        attention = self.multi_head_attention(inputs)
        norm_1 = self.norm_1(attention + inputs)
        feedforward = self.feedforward(norm_1)
        norm_2 = self.norm_2(feedforward + norm_1)

        return norm_2

if __name__ == "__main__":
    window_size = 150
    vocab_size = 920
    embedding_height = 128

    model = tf.keras.Sequential()
    model.add(Embedding(vocab_size,embedding_height,mask_zero = True,input_length = window_size))
    model.add(Positional_Encode())
    model.add(Transformer_Block(embedding_height,window_size = window_size))
    model.add(Dense(vocab_size,activation = "softmax"))

    model.compile(loss = "sparse_categorical_crossentropy",optimizer = "adam",metrics = ["accuracy"])

    model.fit(np.random.rand(1000,150).astype("float32"),np.random.rand(1000,150).astype("int64"))