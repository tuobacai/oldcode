# coding:utf-8
'''
Variable/placeholder/feeds/fetches/Lazy loading
1、TensorFlow Variables 是内存中的容纳 tensor 的缓存。
尝试用它们在模型训练时(during training)创建、保存和更新模型参数(model parameters)
建模时它们需要被明确地初始化，模型训练后它们必须被存储到磁盘。这些变量的值可在之后模型训练和分析是被加载。
2-1、除了在计算图中引入了 tensor, 以常量或变量的形式存储.
 TensorFlow 还提供了 feed 机制, 该机制可以临时替代图中的任意操作中的 tensor 。
 可以对图中任何操作提交补丁, 直接插入一个 tensor。
2-2、feed 使用一个 tensor 值临时替换一个操作的输出结果.
 你可以提供 feed 数据作为 run() 调用的参数.
 feed 只在调用它的方法内有效, 方法结束, feed 就会消失.
 最常见的用例是将某些特殊的操作指定为 "feed" 操作, 标记的方法是使用 tf.placeholder() 为这些操作创建占位符.
3-1、fetch:为了取回操作的输出内容, 可以在使用 Session 对象的 run() 调用 执行图时,
传入一些 tensor, 这些 tensor 会帮助你取回结果.
3-2.eg.sess.run([e,d], feed_dict={a: [3, 3, 3]})中[e,d]就是fetch操作部分
4、 lazy loading是指你推迟变量的创建直到你必须要使用他的时候
'''
import tensorflow as tf
#1:variable

#step 1:创建变量,tf.Variable()
state=tf.Variable(0,name='counter')
one=tf.constant(1)

value=tf.add(state,one)
update=tf.assign(state,value)

#step 2:变量初始化
init=tf.initialize_all_variables()

#step 3:保存变量,tf.train.Saver()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print sess.run(state)
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print "Model saved in file: ", save_path

    #step 4:加载存储的变量值
    saver.restore(sess, "/tmp/model.ckpt")
    print "Model restored."


#2:placeholder/feeds/fetches
a = tf.placeholder (tf.float32 ,shape =[ 3 ])
b = tf.constant([5,5,5],tf.float32)
c=tf.constant([1,3,5],tf.float32)
# use the placeholder as you would a constant or a variable
d = a + b
e=c+d

with tf . Session () as sess:
    print (sess.run(d,feed_dict={a:[3,3,3]}))
    print (sess.run([e,d], feed_dict={a: [3, 3, 3]}))


#3:normal and lazing loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = x+y # you create the node for add node before executing the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('normal', sess.graph)
    sess.run(z)
    writer.close()

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('lazing', sess.graph)
    print sess.run(tf.add(x, y)) # someone decides to be clever to save one line of code
    writer.close()
'''
normal loading在session中不管做多少次x+y，只需要执行z定义的加法操作就可以了，
而lazy loading在session中每进行一次x+y，就会在图中创建一个加法操作，如果进行1000次x+y的运算，
normal loading的计算图没有任何变化，而lazy loading的计算图会多1000个节点，每个节点都表示x+y的操作。
这就是lazy loading造成的问题，这会严重影响图的读入速度。
'''
# normal loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(z)
# lazy loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(tf.add(x, y))

# W = tf.Variable(10)
# assign_op = W.assign(100)
# with tf.Session() as sess:
#     sess.run(W.initializer)
#     sess.run(assign_op)
#     print W.eval()


# W = tf.Variable(10)
# assign_op = W.assign_add(1)
# with tf.Session() as sess:
#     sess.run(W.initializer)
#     for _ in range(3):
#         sess.run(assign_op)
#         print W.eval()






