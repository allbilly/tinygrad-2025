from tinygrad import Tensor, dtypes
# rand = Tensor.rand(2, 3) # create a tensor of shape (2, 3) filled with random values from a uniform distribution
# Create two tensors

# t1 = Tensor([[0, 1, 2], [4, 5, 6]], dtype=dtypes.float)
# t2 = Tensor([[7, 8, 9], [10, 11, 12]], dtype=dtypes.float)

t1 = Tensor([[2.3, 2.1, 1.3]], dtype=dtypes.float)
t2 = Tensor([[2.1, 2.3, 3.1]], dtype=dtypes.float)
# t1 = Tensor([[2.3, 2.1]], dtype=dtypes.float)
# t2 = Tensor([[2.1, 2.3]], dtype=dtypes.float)
t6 = t1 + t2 

print(t6.numpy())

