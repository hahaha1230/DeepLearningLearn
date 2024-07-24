import math

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy


# 构建embedding类来实现文本嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        # d_model词嵌入的维度
        # vocab 词表大小
        super(Embeddings, self).__init__()
        # 定义embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # 将参数传入类中
        self.d_model = d_model

    def forward(self, x):
        # x代表输入进模型的文本通过词汇映射后的数字张量
        return self.lut(x) * math.sqrt(self.d_model)


d_model = 512
vocab = 1000
x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
emb = Embeddings(d_model, vocab)
embr = emb(x)
print("embr:", embr)
print(embr.shape)


# 构建位置编码器类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        # d_model代表词嵌入的维度
        # dropout 代表置零的比例
        # max_len 代表每个句子的最大长度
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置编码矩阵，大小是max_len*d_model
        pe = torch.zeros(max_len, d_model)
        # 初始化一个绝对位置矩阵，max_len*1
        position = torch.arange(0, max_len).unsqueeze(1)
        # 定义哟个变化矩阵div_term，跳跃式的初始化
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # 将前面定义的变化矩阵进行奇数和偶数的分别赋值,pe[:, 1::2]表示从第2列开始,每隔一个元素进行选择。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将二维张量扩充成三维张量
        pe = pe.unsqueeze(0)

        # 将位置编码矩阵注册成模型的buffer，这个buffer不是模型中的参数，不和优化器同步更新
        # 注册成buffer后我们就可以在模型保存后重新加载的时候，将这个位置编码器和模型的参数一同加载进来
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x代表文本序列的词嵌入表示
        # 首先明确pe的编码太长了，将第二个维度，也就是max——len对应的那个维度缩小成x的句子长度同等的长度
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x


d_model = 512
dropout = 0.1
max_len = 60

x = embr
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
print((pe_result))
print(pe_result.shape)


# 注意力机制

def attention(query, key, value, mask=None, dropout=None):
    # query,key,value代表注意力的三个输入张量
    # 先把query的最后一个维度提取出来，代表词嵌入的维度
    d_k = query.size(-1)

    # 按照注意力的计算公式，将query和key的转置进行矩阵乘法，然后除以缩放系数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 判断是否使用掩码张量
    if mask is not None:
        # 利用masked_fill方法，将掩码张量和0进行位置的意义比较，如果等于0，替换成一个非常小的值
        scores = scores.masked_fill(mask == 0, -1e9)
    # 对scores的最后一个维度进行softmax操作
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后完成p_attn和value的乘法，并返回query注意力表示
    return torch.matmul(p_attn, value), p_attn


# query = key = value = pe_result
# mask = Variable(torch.zeros(2, 4, 4))
# attn, p_attn = attention(query, key, value, mask=mask)
# print('attn:', attn)
# print(attn.shape)
# print('p_attn:', p_attn)
# print(p_attn.shape)


# 定义克隆函数，因为在多头注意力机制的实现中，用到多个结构相同的线性层
# 我们将使用clone函数将他们一同初始化在一个网络层列表对象中，之后的结构中也会用到该函数
def clones(module, N):
    """用于生成相同网络层的克隆函数，他的参数module表示要clone的目标网络层，n代表clone的数量"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 实现多头注意力机制的类
class MultiHeaderAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        # head代表几个头
        # embedding_dim代表词嵌入的维度
        # dropout代表置零比例
        super(MultiHeaderAttention, self).__init__()

        # 要确认一个事实，多头的数量需要整除embedding_dim
        assert embedding_dim % head == 0
        # 计算每个头获取的词向量维度
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim
        # 获取线性层，要获取4个，分别是QKV以及最终的输出的线性层
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        # 初始化注意力张量
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query,key,value是注意力机制的三个输入张量，mask代表掩膜张量
        # 首先判断是否使用掩码张量
        if mask is not None:
            # 使用squeeze将掩码张量进行维度扩充，代表多头中的第n个头
            mask = mask.unsqueeze(1)
        # 计算batch_size
        batch_size = query.size(0)
        # 首先使用zip将网络层和输入数据连接在一起，模型的输出利用view和transpose进行维度和形状的变化
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value))]

        # 将每个头的输出传入到注意力层
        x, self_attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 得到每个头的计算结果是4维张量，需要进行形状的转换
        # 前面已经将1,2维度进行了转置，这里必须先转置回来
        # 注意，经过transpose方法后，必须要使用contiguous方法，不然无法使用view方法
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后将x输入到线性层列表中的最后一个线性层进行处理，得到最终的多头注意力结构的输出
        return self.linears[-1](x)


head = 8
embedding_dim = 512
dropout = 0.2

query = key = value = pe_result
# mask = Variable(torch.zeros(8, 4, 4))
mask = Variable(torch.zeros(2, 4, 4))
mha = MultiHeaderAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)
print((mha_result))
print(mha_result.shape)


# 构建前馈全连接网络类
class PositionwiseFeedForward(nn.Module):
    # n_module词嵌入的维度
    # d_ff代表第一个线性层输出维度，和第二个线性层的输入维度
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 先将x送入到第一个线性网络，然后经过relu激活函数，在经过dropout，最后送入第二个线性层
        return self.w2(self.dropout(F.relu(self.w1(x))))


d_model = 512
d_ff = 64
dropout = 0.2

x = mha_result
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
ff_result = ff(x)
print(ff_result)
print(ff_result.shape)


# 构建规范化层的类
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        # features 词嵌入的维度
        # eps 一个足够小的正数，用来在规范化公式的分母中，防止除0
        super(LayerNorm, self).__init__()
        # 初始化两个参数张量a2，b2，用于对结果进行规范化操作的计算
        # 将其用nn.parameter进行封装，代表他们是模型中的参数
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        # x代表上一层网络的输出
        # 首先对x进行最后一个维度上进行求均值操作，同时保证输出维度和输入维度相同
        mean = x.mean(-1, keepdim=True)
        # 接着对x进行最后一个维度上求标准差的操作，保持输出维度和输入维度相同
        std = x.std(-1, keepdim=True)
        # 按照规范化公式进行计算并返回结果
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


features = d_model = 512
eps = 1e-6
x = ff_result
ln = LayerNorm(features, eps)
ln_result = ln(x)
print(ln_result)
print(ln_result.shape)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        # 实例化规范化对象slef.norm
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)
        self.size = size

    def forward(self, x, sublayer):
        # 向前传播的逻辑，接收上一层或者子层的输入作为第一个参数，将该子层连接中的子层函数作为第二个函数
        # 首先对输入进行规范化，然后将结果传递给子层处理，之后再对子层进行dropout操作
        # 随机停止一些网络中神经元的作用，来防止过拟合，最后还有一个add操作
        # 因为存在跳跃连接，所以是将输入x和dropout后的子层输出结果相加作为最终的子层连接输出
        return x + self.dropout(sublayer(self.norm(x)))


size = d_model = 512
head = 8
dropout = 0.2
x = pe_result
mask = Variable(torch.zeros(2, 4, 4))
self_attn = MultiHeaderAttention(head, d_model)
sublayer = lambda x: self_attn(x, x, x, mask)
sc = SublayerConnection(size, dropout)
sc_result = sc(x, sublayer)
print(sc_result)
print(sc_result.shape)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        # size是词嵌入的维度
        # self_attn代表传入的多头自注意力子层的实例化对象
        # feed_forword代表前馈全连接层的实例化对象
        # dropout代表置零比例
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size

        self.subLayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        # x是上一层传入的张量
        # mask 代表掩码张量
        # 首先让x经过第一个子层的连接结构，内部包含多头自注意力机制子层
        # 再让张量经过第二个子层连接结构，其中包含前馈神经网络
        x = self.subLayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.subLayer[0](x, self.feed_forward)


# size = d_model = 512
# head = 8
# d_ff = 64
# x = pe_result
# dropout = 0.2
# self_attn = MultiHeaderAttention(head, d_model)
# ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# mask = Variable(torch.zeros(8, 4, 4))


# 使用encoder类来实现编码器
class Encoder(nn.Module):
    # 初始化函数的两个参数分别代表编码器层和编码器层的个数
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 首先使用clone函数克隆n个编码器层放在self.layers中
        self.layers = clones(layer, N)
        # 在初始化一个规范化层，他将用在编码器的后面
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # x代表上一层的输出，mask代表掩码张量
        # 首先就是对于我们clone的编码器层进行循环，每次都会得到一个新的x，
        # 这个循环的过程中，就相当于输出的x经过了n个编码器层的处理，
        # 最后在通过规范化层的对象self。norm进行处理，最后返回结果
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


size = d_model = 512
d_ff = 64
head = 8
c = copy.deepcopy
attn = MultiHeaderAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
dropout = 0.2
layer = EncoderLayer(size, c(attn), c(ff), dropout)
N = 8
mask = Variable(torch.zeros((2, 4, 4)))

en = Encoder(layer, N)
en_result = en(x, mask)
print(en_result)
print(en_result.shape)


# 使用decoderlayer的类实现解码器层
class DecoderLayer(nn.Module):
    # 初始化参数有5个，分别是size，代表词嵌入维度大小，同时代表解码器的尺寸
    # 第二个是self_attn，多头自注意对象，该注意力机制Q=K=V
    # 第三个是src_attn，多头注意力对象，这里Q!=K=V，
    # 第4个事前馈全连接层对象
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    # x来自上一层的输出，memory来自编码器的语义存储变量，源数据掩码张量和目标数据集掩码张量
    def forward(self, x, memory, source_mask, target_mask):
        # 将memory表示为m方便后续使用
        m = memory
        # 将x传入第一个子层结构，第一个子层结构的输入分别是x和self_attn函数，并且对目标数据进行遮掩
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # 接着进入第二个子层，这个子层中常规的注意力机制，q是输入的x，k和v是编码器层输出的memory
        # 同样也传入source_mask，但是进行源数据遮掩的原因并非是抑制信息泄露，而是遮蔽对结果用处不大的信息
        # 以此来提升模型的效果和训练速度，这样就完成了第二个子层的处理
        x = self.sublayer[1](x, lambda x: self.self_attn(x, m, m, source_mask))

        # 最后一个子层就是前馈全连接子层，经过处理后皆可以返回结果
        return self.sublayer[2](x, self.feed_forward)


size = d_model = 512
d_ff = 64
head = 8
dropout = 0.2

self_attn = src_attn = MultiHeaderAttention(head, d_model, dropout)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)

x = pe_result
memory = en_result

mask = Variable(torch.zeros((2, 4, 4)))
source_mask = target_mask = mask

dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
dl_result = dl(x, memory, source_mask, target_mask)
print(dl_result)
print(dl_result.shape)
