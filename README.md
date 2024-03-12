Encoder模块

经典的Transformer架构中的Encoder模块包含6个Encoder Block.
每个Encoder Block包含两个子模块, 分别是多头自注意力层, 和前馈全连接层.
多头自注意力层采用的是一种Scaled Dot-Product Attention的计算方式, 实验结果表明, Mul ti-head可以在更细致的层面上提取不同head的特征, 比单一head提取特征的效果更佳.
前馈全连接层是由两个全连接层组成, 线性变换中间增添一个Relu激活函数, 具体的维度采用4倍关系, 即多头自注意力的d_model=512, 则层内的变换维度d_ff=2048.
Decoder模块

经典的Transformer架构中的Decoder模块包含6个Decoder Block.
每个Decoder Block包含3个子模块, 分别是多头自注意力层, Encoder-Decoder Attention层, 和前馈全连接层.
多头自注意力层采用和Encoder模块一样的Scaled Dot-Product Attention的计算方式, 最大的 区别在于需要添加look-ahead-mask, 即遮掩"未来的信息".
Encoder-Decoder Attention层和上一层多头自注意力层最主要的区别在于Q != K = V, 矩阵Q来源于上一层Decoder Block的输出, 同时K, V来源于Encoder端的输出.
前馈全连接层和Encoder中完全一样.
Add & Norm模块

Add & Norm模块接在每一个Encoder Block和Decoder Block中的每一个子层的后面.
对于每一个Encoder Block, 里面的两个子层后面都有Add & Norm.
对于每一个Decoder Block, 里面的三个子层后面都有Add & Norm.
Add表示残差连接, 作用是为了将信息无损耗的传递的更深, 来增强模型的拟合能力.
Norm表示LayerNorm, 层级别的数值标准化操作, 作用是防止参数过大过小导致的学习过程异常, 模型收敛特别慢的问题.
位置编码器Positional Encoding

Transformer中采用三角函数来计算位置编码.
因为三角函数是周期性函数, 不受序列长度的限制, 而且这种计算方式可以对序列中不同位置的编码的重要程度同等看待.
