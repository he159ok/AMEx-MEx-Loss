import  torch
from    torch import nn
from    torch.nn import functional as F
#from    utils import sparse_dropout, dot


class GraphConvolution(nn.Module):


    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()


        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs

        # if self.training and self.is_sparse_inputs:
        #     x = sparse_dropout(x, self.dropout, self.num_features_nonzero)    #dropout rate = 0,所以其实是可以删掉这块的
        # elif self.training:
        #     x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless: # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight

        out = torch.sparse.mm(support, xw)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), support

class EdgeGraphConvolution(nn.Module):


    def __init__(self, input_dim, output_dim, edge_emb_dim, num_features_nonzero, FixDegree=3,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        super(EdgeGraphConvolution, self).__init__()


        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero
        self.edge_emb_dim = edge_emb_dim

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        self.FixDegree = FixDegree

        # if bias:
        #     self.bias = nn.Parameter(torch.zeros(output_dim))

    def GenEmbI(self, N, P):
#N: the number of nodes
#P: the Dimensions of Edge embedding
        mid = torch.zeros((N, P, N))
        for i in range(N):
            mid[i, :, i] = torch.ones(P, 1).squeeze()
        return mid.cuda()


    def forward(self, inputs):                  #假定的输入是 adjacent mat A 和 点的表达 H或者X 以及度值 N
        # print('inputs:', inputs)
        x, support = inputs           #support is A, adjecent mat
        A_shape = support.shape
        assert A_shape[0] == A_shape[2]
        NumNodes = A_shape[0]                      #N
        NumDimEdgeEmb = self.edge_emb_dim                 #P
        #FD = self.FixDegree                        #在我的设置中，FD=3
        FD = 3                      #这里实现的太生硬，需要考虑 最开始定义EGCN的地方的numOFnonZero这一参数
        I_embedding = self.GenEmbI(NumNodes, NumDimEdgeEmb)
        support = (1/FD)*(FD*I_embedding + support) # A,此时是support : npn I:nn I_embedding: npn

        # if self.training and self.is_sparse_inputs:
        #     x = sparse_dropout(x, self.dropout, self.num_features_nonzero)    #dropout rate = 0,所以其实是可以删掉这块的
        # elif self.training:
        #     x = F.dropout(x, self.dropout)

        # convolve


        # suitalbe for bmm
        if len(x.shape) != 2:
            _, _, NumDimOutput = x.shape
            PoolingMid = torch.nn.AdaptiveAvgPool2d([1, NumDimOutput])
            x = PoolingMid(x)
            x = x.squeeze(1)
        #
        # if len(x.shape) == 2:
        #     x = x.unsqueeze(0)
        #     x = x.repeat(NumNodes, 1, 1)
        #
        # weight = self.weight.unsqueeze(0)
        # weight = weight.repeat(NumNodes, 1, 1)

        # bias = self.bias.unsqueeze(0)
        # bias = bias.repeat(NumNodes, 1, 1)

        # if not self.is_sparse_inputs:
        #     support_x = torch.sparse.bmm(support, x)
        #     out = torch.sparse.bmm(support_x, weight)
        # else:
        #     support_x = torch.bmm(support, x)
        #     out = torch.bmm(support_x, weight)
        support_x = torch.matmul(support, x)
        out = torch.matmul(support_x, self.weight)






        # if self.bias is not None:
        #     out += bias

        return self.activation(out), support
