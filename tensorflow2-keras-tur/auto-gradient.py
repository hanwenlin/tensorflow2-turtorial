import tensorflow as tf
import  numpy as np


x = tf.ones((2,2))
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
dz_dx = t.gradient(z, x)
print(dz_dx)

# if there is not permeter 'persistent=True', tape.gradient can be used only once
dz_dy = t.gradient(z, y)
print(dz_dy)


# 二、记录控制流

# 因为tapes记录了整个操作，所以即使过程中存在python控制流（如if， while），梯度求导也能正常处理。
def f(x, y):
    output = 1.0
    # 根据y的循环
    for i in range(y):
        # 根据每一项进行判断
        if i> 1 and i<5:
            output = tf.multiply(output, x)
    return output

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
        # 返回梯度
        return t.gradient(out, x)
# x为固定值
x = tf.convert_to_tensor(2.0)

print(grad(x, 6))
print(grad(x, 5))
print(grad(x, 4))

# 三、高阶梯度

# GradientTape上下文管理器在计算梯度的同时也会保持梯度，
# 所以GradientTape也可以实现高阶梯度计算，

x = tf.Variable(1.0)
with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = x * x * x
    dy_dx = t2.gradient(y, x)
    print(dy_dx)
d2y_d2x = t1.gradient(dy_dx, x)
print(d2y_d2x)
