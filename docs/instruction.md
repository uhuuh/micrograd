重构反向传播流程
- tensor的执行分为两个层次，顶层是tensor之间调用算子，底层是算子之间的执行
- 在算子之间执行中（Function的forward方法）
-- 如果任意一个输入tensor的requred grad为True
--- 创建一个上下文，根据算子反向传播要求将必要输入或输出或其他内容保存到上下文中
--- 计算得到的输出tensor也为requred grad为True，grad fn指向该上下文
--- 过滤除所有requred grad为True的输入tensor，上下文的next属性执行这些输入tensor的grad fn，同时输入tensor的prev输入该上下文，如果输入tensor没有grad fn，则使用leaf属性指向所有的输入tensor
-- 否则只计算输出tensor

- 单独创建一个note对象，而不是像之前那样将所有信息保存到Function中

- 设计一个no grad，可以同时用于装饰器和上下文管理，在该过程中禁用计算图的构建。注意在Function的forward方法中，应该处理no grad环境

- 在tensor的backward执行过程中调用grad fn的backward方法(将tensor backward逻辑挪至note backward中)

- 在note backward执行过程中，进行反向传播
-- 现在只需要一次bfs就可以使用，因为prev已经保存了连接信息
-- 注意执行过程中销毁next和prev属性中多其他node的引用
-- 梯度不断反向传播，最终累计到leaf属性中的tensor的grad属性中


现在计算图由所有中间tensor构建，使得所有中间tensor都被相互引用而不能释放，一些backward过程不需要的中间tensor也不被是放。重构后，计算图有note节点构成，note节点只保存必要的tensor

使用superpower skill