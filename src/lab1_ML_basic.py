import tensorflow as tf

print(tf.__version__)


### --------------- Hello, TensorFlow! ---------------
# hello = tf.constant("Hello, TensorFlow!")

# # TF 1.x 방식 (TF 2.x에서는 사용 불가)
# # sess = tf.Session()

# # print(sess.run(hello))

# # TF 2.x에서는 세션 없이 바로 실행 가능
# print(hello.numpy())

### --------------- Computational Graph ---------------
# node1 = tf.constant(3.0, dtype=tf.float32)
# node2 = tf.constant(4.0)  # 디폴트 타입은 tf.float32 (implicitly)
# node3 = tf.add(node1, node2)

# print("node1:", node1, "node2:", node2)
# print("node3:", node3)

# print("node1, node2 (numpy):", node1.numpy(), node2.numpy())
# print("node3 (numpy):", node3.numpy())

### --------------- Placeholder --------------- (대체 사용법 포함)
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# adder_node = a + b # tf.add(a, b) 와 동일

# # placeholder 개념이 TF 2.x에서는 제거되었기 때문에, 아래 코드는 오류가 발생합니다.
# print(adder_node.numpy()) # 오류 발생

### placeholder 대신, 함수를 사용하여 동일한 효과를 얻을 수 있습니다.
# float32 타입의 입력만 받도록 제한 (placeholder의 선언부와 기능적으로 동일)
@tf.function(input_signature=[
    tf.TensorSpec(shape=None, dtype=tf.float32), 
    tf.TensorSpec(shape=None, dtype=tf.float32)
])
def add_nodes(a, b):    # 함수 정의 (이것이 하나의 거대한 Node 역할을 합니다)
    return a + b

# 실행 시점에 원하는 값을 바로 전달 (feed_dict와 동일한 효과)
result = add_nodes(3.0, 4.0)
print(result.numpy())  # 결과: 7.0

### --------------- Tensor Rank, Shape, Type ---------------
tensor_rank0 = tf.constant(3.0)
tensor_rank1 = tf.constant([1.0, 2.0, 3.0])
tensor_rank2 = tf.constant([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
tensor_rank3 = tf.constant([[[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0]],
                            [[7.0, 8.0, 9.0],
                             [10.0, 11.0, 12.0]]])

# 단순 출력
print("Tensor_rank0:", tensor_rank0.numpy())
print("Tensor_rank1:", tensor_rank1.numpy())
print("Tensor_rank2:", tensor_rank2.numpy())
print("Tensor_rank3:", tensor_rank3.numpy())

# 랭크(차원) 출력
print("Rank 0 rank:", tf.rank(tensor_rank0).numpy())
print("Rank 1 rank:", tf.rank(tensor_rank1).numpy())
print("Rank 2 rank:", tf.rank(tensor_rank2).numpy())
print("Rank 3 rank:", tf.rank(tensor_rank3).numpy())

# 형태(shape) 출력
print("Rank 0 shape:", tensor_rank0.shape)
print("Rank 1 shape:", tensor_rank1.shape)
print("Rank 2 shape:", tensor_rank2.shape)
print("Rank 3 shape:", tensor_rank3.shape)

# Size 출력 (전체 원소 개수)
print("Rank 0 size:", tf.size(tensor_rank0).numpy())
print("Rank 1 size:", tf.size(tensor_rank1).numpy())
print("Rank 2 size:", tf.size(tensor_rank2).numpy())
print("Rank 3 size:", tf.size(tensor_rank3).numpy())

# Type 출력
print("Rank 0 type:", tensor_rank0.dtype)
print("Rank 1 type:", tensor_rank1.dtype)
print("Rank 2 type:", tensor_rank2.dtype)
print("Rank 3 type:", tensor_rank3.dtype)

