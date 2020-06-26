'''
Define new functional operations that using meProp
Both meProp and unified meProp are supported
'''
import torch
from torch.autograd import Function


FULL_DW_EXPERIMENT = False

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
        # This code was originally written with pyTorch=0.3.0 according to README.md
        # https://pytorch.org/docs/0.3.0/tensors.html#torch.Tensor.addmm_
        # addmm_(beta=0, mat=1, alpha=1, mat1=x, mat2=w) → Tensor
        # mat + (mat1 @ mat2)
        # Here, we run the code on pyTorch=1.5.0
        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.addmm_
        # y.addmm_(mat1, mat2, *, beta=1, alpha=1) → Tensor
        # y = beta * y + alpha * (mat1 @ mat2)
        # Use explict arguments to resolve ambiguity.
        y.addmm_(beta=0, alpha=1, mat1=x, mat2=w)
        self.add_buffer = x.new(x.size(0)).fill_(1)
        # Adding the bias, but in a way that multiplies the bias
        # based on the batch size.
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
                if FULL_DW_EXPERIMENT:
                    dw = torch.mm(x.t(), dy)
                else:
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


class linearUnified_shawn(Function):
    '''
    Shawn's implementation of
    linear function with meProp, unified top-k across minibatch
    y = f(x, w, b) = xw + b
    '''

    def __init__(self, k):
        '''
        k is top-k in the backprop of meprop
        '''
        super(linearUnified_shawn, self).__init__()
        self.k = k

    def forward(self, x, w, b):
        '''
        Forward propagation with Column Row Sampling
        x should be of size [minibatch, input feature] (e.g. input vector)
        w should be of size [input feature, output feature] (e.g. weight matrix)
        bias should be of size [output feature] (will be broadcasted across batch dimension).

        return x @ w.T + b

        This is slightly different from the default linear function in PyTorch.
        In that implementation, w is of size [output feature, input feature].
        We find that that implementation is slower in both forward and backward propagation on our devices.
        '''
        self.save_for_backward(x, w, b)
        result = torch.addmm(b, x, w.T, alpha=1, beta=1)
        return result

    def backward(self, dy):
        '''
        backward() with CRS sampling
        The sampling "mask" is saved from forward().
        TODO: What about multiple calls to forward()? How does gradient accum work with forward?
        '''
        # PyTorch convention is dy is a col vector, i.e. dy = (dL/dy).T
        x, w, b = self.saved_tensors

        dx = dw = db = None
        if self.k <= 0 or dy.shape[1] <= self.k:  # exact backprop, no top-k selection
            if self.needs_input_grad[1]:  # w
                # dw = dy.T @ x
                dw = torch.mm(dy.T, x)
                assert dw.shape == w.shape

            if self.needs_input_grad[0]:  # x
                # dx = dy @ w
                dx = torch.mm(dy, w)
                assert dx.shape == x.shape

            if self.needs_input_grad[2]:  # b
                # TODO: work out the formula for this.
                # db = dy.T @ torch.ones(dy.shape[0], device=dy.device)
                db = torch.mv(dy.T, torch.ones(dy.shape[0], device=dy.device))
                assert db.shape == b.shape

            return dx, dw, db

        else:  # Do the top-k selection
            # get top-k across examples in magnitude
            _, indexes = dy.abs().sum(0).topk(self.k)  # .topk applies to which dim? last dim only?

            if self.needs_input_grad[1]:  # w
                # partial_dw = dy.T[indexes, :] @ x
                partial_dw = torch.mm(dy.T[indexes, :], x)
                dw = torch.zeros_like(w)
                # alternative to scatter_ or index_copy_
                dw[indexes, :] = partial_dw
                assert dw.shape == w.shape

            if self.needs_input_grad[0]:  # x
                # dx = dy[:, indexes] @ w[indexes, :]
                dx = torch.mm(dy[:, indexes], w[indexes, :])
                assert dx.shape == x.shape

            if self.needs_input_grad[2]:  # b
                # TODO: confirm formula for bias with batches
                # db = dy.T @ torch.ones(dy.shape[0], device=dy.device)
                db = torch.mv(dy.T, torch.ones(dy.shape[0], device=dy.device))
                assert db.shape == b.shape

        return dx, dw, db


class linear_crs(Function):
    '''
    linear function CRS sampling in forward().
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

        return x @ w.T + b

        This is slightly different from the default linear function in PyTorch.
        In that implementation, w is of size [output feature, input feature].
        We find that that implementation is slower in both forward and backward propagation on our devices.
        '''
        self.save_for_backward(x, w, b)

        # batch_size=10
        # x.shape, w.shape, b.shape
        # (torch.Size([10, 784]), torch.Size([512, 784]), torch.Size([512]))

        if self.k <= 0:  # shortcut for baseline case
            # result = x @ w.T + b
            result = torch.addmm(b, x, w.T)
            return result

        D, indexes, scaling = crs_mm(x, w.T, self.k, strategy=self.strategy)
        self.indexes_scaling = indexes, scaling

        return D + b


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
                db = dy.T @ torch.ones(dy.shape[0], device=dy.device)
                assert db.shape == b.shape

            return dx, dw, db

        # TODO: figure out where scaling fits in with
        # the gradients. For now assume no scaling, which is usually preferred.
        indexes, scaling = self.indexes_scaling

        if self.needs_input_grad[1]:  # w
            partial_dw = dy.T @ x[:, indexes]
            dw = torch.zeros_like(w)
            dw[:, indexes] = partial_dw  # alternative to scatter_ or index_copy_
            assert dw.shape == w.shape

        if self.needs_input_grad[0]:  # x
            partial_dx = dy @ w[:, indexes]
            dx = torch.zeros_like(x)
            dx[:, indexes] = partial_dx
            assert dx.shape == x.shape

        if self.needs_input_grad[2]:  # b
            # TODO: what is the right formula for this?
            # bias with batches???
            db = dy.T @ torch.ones(dy.shape[0], device=dy.device)
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


def crs_mm(A, B, k, strategy='random'):
    """ Returns A @ B, computed using `k` outer products.
    `k` <= A.shape[1] == B.shape[0]
    `strategy` in ('random', 'det_top_k', 'nps')
    """

    assert A.shape[1] == B.shape[0]
    common_dimension = A.shape[1]
    assert common_dimension >= k

    if strategy == 'random':
        # Random Sampling (w/o replacement)
        # indexes = np.random.choice(common_dimension, size=k, replace=False)
        indexes = torch.randperm(common_dimension)[:k]
        # indexes, inds = torch.sort(indexes)  # only needed for the index_copy_ strategy.
        # Scale by 1 / (k*p_i)  # Eq. 1 in [1]
        scaling = 1 / (k * 1/common_dimension)
    elif strategy == 'det_top_k':
        # Deterministic top-k
        # Compute norms of all cols of A
        # A.shape = (m,n); torch.norm(A, dim=0).shape = (n)  # i.e. cols
        # torch.norm(A, dim=1).shape = (m)  # i.e rows
        col_norms_A = torch.norm(A, dim=0)
        # Compute norms of all rows of B
        row_norms_B = torch.norm(B, dim=1)
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
        # indexes, inds = torch.sort(indexes)  # only needed for the index_copy_ strategy.

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
        # indexes, inds = torch.sort(indexes)  # only needed for the index_copy_ strategy.
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

    return D, indexes, scaling


def test_linear_crs_fw(k=50):
    # TODO remove this test in favor of `test_linear_crs_fw_diffsize`.
    print('START test_linear_crs_fw')
    print('WARNING: DEPRECATED TEST')
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
    print('END test_linear_crs_fw')


def test_linear_crs_fw_diffsize(k=50):
    print('START test_linear_crs_fw_diffsize')
    A = torch.rand(1000, 500)  # output_features, input_features -- weight matrix
    B = torch.rand(200, 500) + 1  # batch_size, input_features -- input matrix
    # batch size as leading dim (pytorch convetion)
    bias = torch.zeros(1000)  # output_features, implicitly broadcasted across batch_size

    C = B @ A.T + bias
    exact_norm = torch.norm(C)

    for strategy in ('random', 'det_top_k', 'nps'):
        D = linear_crs(k=k, strategy=strategy)(B, A, bias)  # x=B, w=A, bias=bias
        # returns B @ A.T + bias

        norm = torch.norm(D)
        norm_diff = torch.norm(C - D)
        norm_diff_ratio = norm_diff / (torch.norm(A) * torch.norm(B))  # Error metric in [1]

        print('Approximate Result (k={}, strategy={}):'.format(k, strategy))
        print('D ~= A @ B =\n', D)
        print('|D| =', norm)
        print('|C - D| =', norm_diff)
        print('(|C - D|) / (|A| |B|) =', norm_diff_ratio)
    print('END test_linear_crs_fw_diffsize')


def test_linear_crs_fw_bw(k=50):
    print('START test_linear_crs_fw_bw')
    A = torch.rand(1000, 500, requires_grad=True)  # output_features, input_features
    B = torch.rand(200, 500, requires_grad=True)  # batch_size, input_features -- input matrix
    bias = torch.zeros(1000, requires_grad=True)  # output_features, broadcasted across batch_size

    C = B @ A.T + bias
    exact_norm = torch.norm(C)
    resultsum = torch.sum(C)
    resultsum.backward()
    print('A.grad.shape =', A.grad.shape, '\nB.grad.shape =', B.grad.shape, '\nbias.grad.shape =', bias.grad.shape)

    for strategy in ('random', 'det_top_k', 'nps'):
        D = linear_crs(k=k, strategy=strategy)(B, A, bias)  # x=B, w=A, bias=bias
        dy = torch.ones_like(D)
        outputs = D.backward(dy)
        print('A.grad.shape =', A.grad.shape, '\nB.grad.shape =',
            B.grad.shape, '\nbias.grad.shape =', bias.grad.shape)

        norm = torch.norm(D)
        norm_diff = torch.norm(C - D)
        norm_diff_ratio = norm_diff / (torch.norm(A) * torch.norm(B))  # Error metric in [1]

        print('Approximate Result (k={}, strategy={}):'.format(k, strategy))
        print('D ~= A @ B =\n', D)
        print('|D| =', norm)
        print('|C - D| =', norm_diff)
        print('(|C - D|) / (|A| |B|) =', norm_diff_ratio)
    print('END test_linear_crs_fw_bw')


def test_linearUnified_shawn(k=50):
    print('START test_linearUnified_shawn')
    # output_features, input_features
    A = torch.rand(1000, 500, requires_grad=True)
    # batch_size, input_features -- input matrix
    B = torch.rand(200, 500, requires_grad=True)
    # output_features, broadcasted across batch_size
    bias = torch.zeros(1000, requires_grad=True)

    C = B @ A.T + bias
    exact_norm = torch.norm(C)
    resultsum = torch.sum(C)
    resultsum.backward()
    print('A.grad.shape =', A.grad.shape, '\nB.grad.shape =',
          B.grad.shape, '\nbias.grad.shape =', bias.grad.shape)

    for strategy in ('random', 'det_top_k', 'nps'):
        D = linearUnified_shawn(k=k)(
            B, A, bias)  # x=B, w=A, bias=bias
        dy = torch.ones_like(D)
        outputs = D.backward(dy)
        print('A.grad.shape =', A.grad.shape, '\nB.grad.shape =',
              B.grad.shape, '\nbias.grad.shape =', bias.grad.shape)

        norm = torch.norm(D)
        norm_diff = torch.norm(C - D)
        norm_diff_ratio = norm_diff / \
            (torch.norm(A) * torch.norm(B))  # Error metric in [1]

        print('Approximate Result (k={}, strategy={}):'.format(k, strategy))
        print('D ~= A @ B =\n', D)
        print('|D| =', norm)
        print('|C - D| =', norm_diff)
        print('(|C - D|) / (|A| |B|) =', norm_diff_ratio)
    print('END test_linearUnified_shawn')



def main():
    if 0:
        test_linear_crs_fw()
        test_linear_crs_fw_diffsize()
        test_linear_crs_fw_bw()
    if 1:
        test_linearUnified_shawn()

if __name__ == '__main__':
    main()
