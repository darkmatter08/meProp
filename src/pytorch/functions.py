'''
Define new functional operations that using meProp
Both meProp and unified meProp are supported
'''
import torch
from torch.autograd import Function


class linearUnified(Function):
    '''
    linear function with meProp, unified top-k across minibatch
    y = f(x, w, b) = xw + b
    '''

    def __init__(self, k):
        '''
        k is top-k in the backprop of meprop
        '''
        super(linearUnified, self).__init__()
        self.k = k

    # @staticmethod
    def forward(self, x, w, b):
        '''
        forward propagation
        x should be of size [minibatch, input feature]
        w should be of size [input feature, output feature]
        b should be of size [output feature]

        This is slightly different from the default linear function in PyTorch.
        In that implementation, w is of size [output feature, input feature].
        We find that that implementation is slower in both forward and backward propagation on our devices.
        '''
        self.save_for_backward(x, w, b)
        y = x.new(x.size(0), w.size(1))
        y.addmm_(0, 1, x, w)
        self.add_buffer = x.new(x.size(0)).fill_(1)
        y.addr_(self.add_buffer, b)
        return y

    # @staticmethod
    def backward(self, dy):
        '''
        backprop with meprop
        if k is invalid (<=0 or > output feature), no top-k selection is applied
        '''
        x, w, b = self.saved_tensors
        dx = dw = db = None

        if self.k > 0 and self.k < w.size(1):  # backprop with top-k selection
            _, inds = dy.abs().sum(0).topk(self.k)  # get top-k across examples in magnitude
            inds = inds.view(-1)  # flat
            pdy = dy.index_select(
                -1, inds
            )  # get the top-k values (k column) from dy and form a smaller dy matrix

            # compute the gradients of x, w, and b, using the smaller dy matrix
            if self.needs_input_grad[0]:
                dx = torch.mm(pdy, w.index_select(-1, inds).t_())
            if self.needs_input_grad[1]:
                dw = w.new(w.size()).zero_().index_copy_(
                    -1, inds, torch.mm(x.t(), pdy))
            if self.needs_input_grad[2]:
                # torch.mv(): matrix-vector product
                db = torch.mv(dy.t(), self.add_buffer)
        else:  # backprop without top-k selection
            if self.needs_input_grad[0]:
                dx = torch.mm(dy, w.t())
            if self.needs_input_grad[1]:
                dw = torch.mm(x.t(), dy)
            if self.needs_input_grad[2]:
                db = torch.mv(dy.t(), self.add_buffer)

        return dx, dw, db


class linear(Function):
    '''
    linear function with meProp, top-k selection with respect to each example in minibatch
    y = f(x, w, b) = xw + b
    '''

    def __init__(self, k, sparse=True):
        '''
        k is top-k in the backprop of meprop
        '''
        super(linear, self).__init__()
        self.k = k
        self.sparse = sparse

    def forward(self, x, w, b):
        '''
        forward propagation
        x should be of size [minibatch, input feature]
        w should be of size [input feature, output feature]
        b should be of size [output feature]
        
        This is slightly different from the default linear function in PyTorch.
        In that implementation, w is of size [output feature, input feature].
        We find that that implementation is slower in both forward and backward propagation on our devices.
        '''
        self.save_for_backward(x, w, b)
        y = x.new(x.size(0), w.size(1))
        y.addmm_(0, 1, x, w)
        self.add_buffer = x.new(x.size(0)).fill_(1)
        y.addr_(self.add_buffer, b)
        return y

    def backward(self, dy):
        '''
        backprop with meprop
        if k is invalid (<=0 or > output feature), no top-k selection is applied
        '''
        x, w, b = self.saved_tensors
        dx = dw = db = None

        if self.k > 0 and self.k < w.size(1):  # backprop with top-k selection
            _, indices = dy.abs().topk(self.k)
            if self.sparse:  # using sparse matrix multiplication
                values = dy.gather(-1, indices).view(-1)
                row_indices = torch.arange(
                    0, dy.size()[0]).long().cuda().unsqueeze_(-1).repeat(
                        1, self.k)
                indices = torch.stack([row_indices.view(-1), indices.view(-1)])
                pdy = torch.cuda.sparse.FloatTensor(indices, values, dy.size())
                if self.needs_input_grad[0]:
                    dx = torch.dsmm(pdy, w.t())
                if self.needs_input_grad[1]:
                    dw = torch.dsmm(pdy.t(), x).t()
            else:
                pdy = torch.cuda.FloatTensor(dy.size()).zero_().scatter_(
                    -1, indices, dy.gather(-1, indices))
                if self.needs_input_grad[0]:
                    dx = torch.mm(pdy, w.t())
                if self.needs_input_grad[1]:
                    dw = torch.mm(x.t(), pdy)
        else:  # backprop without top-k selection
            if self.needs_input_grad[0]:
                dx = torch.mm(dy, w.t())
            if self.needs_input_grad[1]:
                dw = torch.mm(x.t(), dy)

        if self.needs_input_grad[2]:
            db = torch.mv(dy.t(), self.add_buffer)

        return dx, dw, db


class linear_crs(Function):
    '''
    linear function with meProp, top-k selection with respect to each example in minibatch
    y = f(x, w, b) = xw + b
    '''

    def __init__(self, k, strategy='random'):
        '''
        k is top-k in the backprop of meprop
        '''
        super(linear_crs, self).__init__()
        self.k = k
        self.strategy = strategy

    def forward(self, x, w, b):
        '''
        Forward propagation with Column Row Sampling
        x should be of size [minibatch, input feature] (e.g. input vector)
        w should be of size [input feature, output feature] (e.g. weight matrix)
        bias should be of size [output feature] (will be broadcasted across batch dimension).

        returns (w @ x) + b
        
        This is slightly different from the default linear function in PyTorch.
        In that implementation, w is of size [output feature, input feature].
        We find that that implementation is slower in both forward and backward propagation on our devices.
        '''
        self.save_for_backward(x, w, b)

        # batch_size=10
        # x.shape, w.shape, b.shape
        # (torch.Size([10, 784]), torch.Size([512, 784]), torch.Size([512]))

        batch_size = x.shape[0]

        if self.k <= 0:  # shortcut for baseline case
            result = x @ w.T + b
            return result

        B = x
        A = w
        k = self.k
        strategy = self.strategy
        # In the batched case, A.shape = (out_features, in_features), B.shape = (batch_size, in_features)
        # What matmul operation to support this case? How can I specify broadcasting over the (implict) batch dim?
        # Should I do A.unsqueeze(0) to get a leading 1, and do a matmul to broadcast over that?
        assert A.shape[1] == B.shape[0]
        common_dimension = A.shape[1]
        assert common_dimension >= k

        if strategy == 'random':
            # Random Sampling (w/o replacement)
            # indexes = np.random.choice(common_dimension, size=k, replace=False)
            indexes = torch.randperm(common_dimension)[:k]
            indexes, inds = torch.sort(indexes)
            # Scale by 1 / (k*p_i)  # Eq. 1 in [1]
            scaling = 1 / (k * 1/common_dimension)
        elif strategy == 'det_top_k':
            # Deterministic top-k
            # Compute norms of all cols of A
            col_norms_A = torch.norm(A, dim=0)  # TODO VERIFY: dim=0 for cols.
            # Compute norms of all rows of B
            row_norms_B = torch.norm(B, dim=1)  # TODO VERIFY: dim=1 for rows.
            assert col_norms_A.shape == row_norms_B.shape

            # Compute norm-products of all corresponding col-row pairs (not all pairs!)
            norm_products = col_norms_A * row_norms_B
            assert norm_products.shape == col_norms_A.shape

            # Pick the indexes of the largest norm-products
            # np.argpartition(arr, k) returns the indexes of the k smallest elements in arr followed by other unsorted indexes.
            # Take the last k of these values to get the indexes of the largest values.
            # Ref: https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
            # indexes = np.argpartition(norm_products, -k)[-k:]
            _, indexes = torch.topk(norm_products, k)
            indexes, inds = torch.sort(indexes)

            # "In addition, we introduce a deterministic top-k sampling, which chooses the k column-row pairs
            # with the largest product of their euclidean norms without scaling." [1]
            scaling = None
        elif strategy == 'nps':
            # Norm-Proportional Sampling w/o replacement
            # Compute a probability distribution over column-row pairs, according to Eq. 3 from [1].
            # p_i = |A[:, i]| * |B[i, :]| / summation_j [|A[:, j]| * |B[j, :]|]

            # Compute norms of all cols of A
            col_norms_A = torch.norm(A, dim=0)  # TODO VERIFY: dim=0 for cols.
            # Compute norms of all rows of B
            row_norms_B = torch.norm(B, dim=1)  # TODO VERIFY: dim=1 for rows.
            assert col_norms_A.shape == row_norms_B.shape

            # Compute norm-products of all corresponding col-row pairs (not all pairs!)
            norm_products = col_norms_A * row_norms_B
            assert norm_products.shape == col_norms_A.shape

            # Normalize by sum of norm products.
            p_i = norm_products / torch.sum(norm_products)
            assert p_i.shape == (common_dimension,)

            # select k random samples w/o replacement, from the distribution p_i, from the set of col-row pairs.
            # indexes = np.random.choice(common_dimension, size=k, replace=False, p=p_i)
            indexes = torch.multinomial(p_i, num_samples=k, replacement=False)
            indexes, inds = torch.sort(indexes)
            assert indexes.shape == (k,)

            # "when one matrix or both have i.i.d. entries with zero mean, random individual column-row
            # products are unbiased estimators of the result. In this case, multiplying by the scaling
            # factor 1 / (k*p_{i_t}) only increases the error" [1]
            # TL;DR -- No scaling is better.
            scaling = None
            # TODO implement scaling, just for comparison...
            if 0:  # scaling seems to improve error metric...
                scaling = 1 / (k * p_i)
                scaling = torch.diag(scaling[indexes])
        else:
            raise NotImplementedError

        # Select `indexes` cols from A and `indexes` rows from B
        assert indexes.shape == (k,)
        cols_A = A[:, indexes]
        rows_B = B[indexes, :]
        assert cols_A.shape[0] == A.shape[0]
        assert rows_B.shape[1] == B.shape[1]
        assert cols_A.shape[1] == rows_B.shape[0]

        if scaling is not None:
            if strategy == 'nps':
                D = cols_A @ scaling @ rows_B  # Eq. 3 from [1]
            else:
                D = cols_A @ rows_B * scaling
        else:
            # Simply take outer product of cols_A and rows_B
            D = cols_A @ rows_B

        # What exactly do we need to save for backward???
        # self.save_for_backward(x, w, b)  # moved to top and it started working...
        self.indexes_scaling = indexes, scaling

        # return D + b
        return D


    def backward(self, dy):
        '''
        backward() with CRS sampling
        The sampling "mask" is saved from forward().
        TODO: What about multiple calls to forward()? How does gradient accum work with forward?
        '''
        # PyTorch convention is dy is a col vector, i.e. dy = (dL/dy).T
        x, w, b = self.saved_tensors

        dx = dw = db = None
        if self.k <= 0:  # baseline, exact MatMul instead of CRS.
            if self.needs_input_grad[1]:  # w
                dw = dy.T @ x
                assert dw.shape == w.shape

            if self.needs_input_grad[0]:  # x
                dx = dy @ w
                assert dx.shape == x.shape

            if self.needs_input_grad[2]:  # b
                # TODO: work out the formula for this.
                db = torch.zeros_like(b)
                assert db.shape == b.shape

            return dx, dw, db

        # TODO: figure out where scaling fits in with
        # the gradients. For now assume no scaling, which is usually preferred.
        indexes, scaling = self.indexes_scaling

        if self.needs_input_grad[1]:
            partial_dw = x[indexes] @ dy.T
            full_dw = torch.zeros_like(w.T)
            # TODO: need to be very careful in copying... which axes, indexes are rows or cols
            #   given the shape of partial_dw...
            full_dw.index_copy_(0, indexes, partial_dw)

            dw = full_dw.T  # need to make dw in the same shape of w.
            assert dw.shape == w.shape

        if self.needs_input_grad[0]:
            # using denom. convention. partial_dx has only populated entries of dx. dx has shape x.shape
            partial_dx = w[:, indexes].T @ dy
            print('w[:, indexes].shape', w[:, indexes].shape, '\nw[:, indexes].T.shape', w[:, indexes].T.shape)
            full_dx = torch.zeros_like(x)
            full_dx.index_copy_(0, indexes, partial_dx)
            dx = full_dx
            # dx = w.T @ dy
            assert dx.shape == x.shape

        if self.needs_input_grad[2]:
            # TODO: what is the right formula for this?
            # db = dy.T @ torch.ones(dy.shape, b.shape[0])
            db = torch.zeros_like(b)
            assert db.shape == b.shape

        # Why is their formula still different/backwards than ours?
        # Reason: They actually compute x @ w or x @ w.T (depending on shape of w).
        # This gives a different order of partial derivaties.
        '''
        pdy = torch.cuda.FloatTensor(dy.size()).zero_().scatter_(
            -1, indices, dy.gather(-1, indices))
        if self.needs_input_grad[0]:
            dx = torch.mm(pdy, w.t())
        if self.needs_input_grad[1]:
            dw = torch.mm(x.t(), pdy)

        else:  # backprop without top-k selection
            if self.needs_input_grad[0]:
                dx = torch.mm(dy, w.t())
            if self.needs_input_grad[1]:
                dw = torch.mm(x.t(), dy)
        '''

        return dx, dw, db


def test_linear_crs_fw(k=50):
    A = torch.rand(1000, 1000)
    B = torch.rand(1000, 1000) + 1
    bias = torch.zeros(1000, 1)

    C = A @ B + bias
    exact_norm = torch.norm(C)

    for strategy in ('random', 'det_top_k', 'nps'):
        D = linear_crs(k=k, strategy=strategy)(B, A, bias)  # x=B, w=A, bias=bias

        norm = torch.norm(D)
        norm_diff = torch.norm(C - D)
        norm_diff_ratio = norm_diff / (torch.norm(A) * torch.norm(B))  # Error metric in [1]

        print('Approximate Result (k={}, strategy={}):'.format(k, strategy))
        print('D ~= A @ B =\n', D)
        print('|D| =', norm)
        print('|C - D| =', norm_diff)
        print('(|C - D|) / (|A| |B|) =', norm_diff_ratio)


def test_linear_crs_fw_bw(k=50):
    A = torch.rand(1000, 500, requires_grad=True)  # output_features, input_features
    B = torch.rand(500, 200, requires_grad=True) + 1  # input_features, batch_size
    bias = torch.zeros(1000, 1, requires_grad=True)  # output_features, broadcasted across batch_size

    C = A @ B + bias
    exact_norm = torch.norm(C)
    resultsum = torch.sum(C)
    resultsum.backward()

    for strategy in ('random', 'det_top_k', 'nps'):
        D = linear_crs(k=k, strategy=strategy)(B, A, bias)  # x=B, w=A, bias=bias
        dy = torch.ones_like(D)
        outputs = D.backward(dy)

        norm = torch.norm(D)
        norm_diff = torch.norm(C - D)
        norm_diff_ratio = norm_diff / (torch.norm(A) * torch.norm(B))  # Error metric in [1]

        print('Approximate Result (k={}, strategy={}):'.format(k, strategy))
        print('D ~= A @ B =\n', D)
        print('|D| =', norm)
        print('|C - D| =', norm_diff)
        print('(|C - D|) / (|A| |B|) =', norm_diff_ratio)


def main():
    test_linear_crs_fw()
    test_linear_crs_fw_bw()


if __name__ == '__main__':
    main()
