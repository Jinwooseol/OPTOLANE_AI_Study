import tensorflow as tf

# TF 2.x 코드 (New Version)

# # 1. 가중치(W)와 편향(b) 변수 선언
# # tf.random_normal -> tf.random.normal로 변경 
# W = tf.Variable(tf.random.normal([1]), name='weight')  # 평균 0, 표준편차 1인 정규분포로부터 무작위 숫자 추출 [1]: 1개의 요소를 가진 텐서 생성
# b = tf.Variable(tf.random.normal([1]), name='bias')

# # 2. 데이터 정의 (Placeholder를 사용하지 않고 직접 정의)
# # x_train = [1, 2, 3]
# # y_train = [1, 2, 3]

# x_train = [1, 2, 3, 4, 5]
# y_train = [2.1, 3.1, 4.1, 5.1, 6.1]

# # 3. 최적화 도구 설정 (Keras Optimizer 사용)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# # 4. 학습 루프 (Session과 feed_dict 대신 GradientTape 사용)
# for step in range(2001):
#     # tf.GradientTape는 연산 과정을 기록하여 경사(Gradient)를 자동 계산함
#     with tf.GradientTape() as tape:
#         # 가설 H(x) = Wx + b
#         hypothesis = x_train * W + b
#         # cost/loss function(MSE: Mean Squared Error)
#         cost = tf.reduce_mean(tf.square(hypothesis - y_train))

#     # 기록된 연산을 바탕으로 W, b에 대한 경사값 계산
#     gradients = tape.gradient(cost, [W, b])
    
#     # 계산된 경사값을 변수에 적용하여 업데이트 (기존 optimizer.minimize 역할)
#     optimizer.apply_gradients(zip(gradients, [W, b]))

#     if step % 20 == 0:
#         # sess.run() 없이 .numpy()로 값을 바로 확인
#         print(f"Step: {step:4}, Cost: {cost.numpy():.6f}, W: {W.numpy()[0]:.4f}, b: {b.numpy()[0]:.4f}")

# # 5. 예측 테스트 (sess.run 대신 직접 연산)
# def predict(x):
#     return x * W + b

# print("\n[Testing model]")
# print(f"X가 5일 때 예측: {predict(5).numpy()}")
# print(f"X가 2.5일 때 예측: {predict(2.5).numpy()}")


# ========== 시각화를 위한 추가 코드 시작 ==========
import matplotlib.pyplot as plt
import numpy as np

# 1. 가중치(W)와 편향(b) 변수 선언
# tf.random_normal -> tf.random.normal로 변경 
W = tf.Variable(tf.random.normal([1]), name='weight')  # 평균 0, 표준편차 1인 정규분포로부터 무작위 숫자 추출 [1]: 1개의 요소를 가진 텐서 생성
b = tf.Variable(tf.random.normal([1]), name='bias')

# 2. 데이터 정의 (Placeholder를 사용하지 않고 직접 정의)
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

x_train = [1, 2, 3, 4, 5]
y_train = [2.1, 3.1, 4.1, 5.1, 6.1]

# 3. 최적화 도구 설정 (Keras Optimizer 사용)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 4. 그래프 시각화 설정
plt.ion()  # Interactive mode 활성화
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Linear Regression Training', fontsize=14)
ax.grid(True, alpha=0.3)

# 데이터 포인트 그리기 (한 번만)
ax.scatter(x_train, y_train, color='red', s=100, zorder=5, label='Training Data')
ax.legend()

# x 범위 설정 (직선을 그리기 위해)
x_min, x_max = min(x_train) - 1, max(x_train) + 1
x_line = np.linspace(x_min, x_max, 100)

# 초기 직선 그리기
line, = ax.plot(x_line, x_line * W.numpy()[0] + b.numpy()[0], 'b-', linewidth=2, label='Fitted Line')
ax.legend()

plt.tight_layout()

# 4. 학습 루프 (Session과 feed_dict 대신 GradientTape 사용)
for step in range(2001):
    # tf.GradientTape는 연산 과정을 기록하여 경사(Gradient)를 자동 계산함
    with tf.GradientTape() as tape:
        # 가설 H(x) = Wx + b
        hypothesis = x_train * W + b
        # cost/loss function(MSE: Mean Squared Error)
        cost = tf.reduce_mean(tf.square(hypothesis - y_train))

    # 기록된 연산을 바탕으로 W, b에 대한 경사값 계산
    gradients = tape.gradient(cost, [W, b])
    
    # 계산된 경사값을 변수에 적용하여 업데이트 (기존 optimizer.minimize 역할)
    optimizer.apply_gradients(zip(gradients, [W, b]))

    # 그래프 업데이트 (매 스텝마다 또는 특정 간격마다)
    if step % 20 == 0:  # 10 스텝마다 그래프 업데이트
        # 직선 업데이트
        y_line = x_line * W.numpy()[0] + b.numpy()[0]
        line.set_ydata(y_line)
        
        # 제목에 현재 정보 표시
        ax.set_title(f'Linear Regression Training (Step: {step}, Cost: {cost.numpy():.6f})', fontsize=14)
        
        # 그래프 새로고침
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)  # 짧은 딜레이로 애니메이션 효과

    if step % 20 == 0:
        # sess.run() 없이 .numpy()로 값을 바로 확인
        print(f"Step: {step:4}, Cost: {cost.numpy():.6f}, W: {W.numpy()[0]:.4f}, b: {b.numpy()[0]:.4f}")

plt.ioff()  # Interactive mode 비활성화
plt.show()  # 최종 그래프 유지

# ========== 시각화를 위한 추가 코드 끝 ==========