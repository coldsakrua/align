# Tokenization

## BPE

### 完善tokenizer类（tokenizer.py）

- `get_stats`

	统计词汇表 `vocab` 中所有相邻词对（二元组）的出现频率。

- `merge_vocab`

	在给定的词汇表 `vocab` 里，将指定的二元组（`pair`）合并成一个新的标记（`token`），然后返回更新后的词汇表。

- `train`

	给每个字符之间加上空白格，并且初始化`vocab`，然后不断合并频率最高的相邻词对。同时给词汇表`vocab`里添加未知符，用于处理未知情况。

- `encode`

	对每个字符不断向后匹配，直到字符串不在符号表里。

- `decode`

	将得到的`ids`对应到符号表即可。

- `save`&&`load`

	把`vocab`存储为json格式，以及实现加载功能。

### 在文本上训练并测试（test_tokenizer.ipynb）

- 用`bpe/manual.txt`进行训练，encode和再decode得到和原文一致文本。

- 调用gpt2的tokenizer，在英文文本是encode得到的长度要远小于自己的tokenizer，在中文文本（疑似校规原文）得到的长度要大于自己的tokenizer。

	原因分析：

	- 英文上是因为校规以中文为主，并未有大规模的英文词汇和句子，所以自己的tokenizer得到的英文token以单个单词为主，同时长度要大于gpt得到的encode ids
	- 中文上由于测试样本和训练样本高度一致，所以自己的tokenizer得到的长度要小于gpt2。

### 问题回答

- Python中使用什么函数查看字符的Unicode，什么函数将Unicode转换成字符？并使用它们查看“北”“大”的Unicode，查看Unicode为22823、27169、22411对应的字符
	- `ord() `，`chr()`  21271  22823  大模型
- Tokenizer的`vocab size`大和小分别有什么好处和坏处
	- 大`vocab size`
		- 优点
			- 更多的词汇，即可以有更丰富的表达能力
			- 减少未识别词汇
		- 缺点
			- 计算资源大，需要更大的训练文本
			- 可能存在稀疏的文本，比如生僻字或词汇等，使得词汇之间关联性降低
			- 过拟合
	- 小`vocab size`
		- 优点
			- 对训练文本没有太多要求，小样本也可以。
			- 数据出现会比较密集
			- 计算资源喜爱
		- 缺点
			- 表达效果不足，类似词汇可能归为一类
			- 容易遇到未识别词汇
- 为什么 LLM 不能处理非常简单的字符串操作任务，比如反转字符串？
	- 因为在`vocab.json`里有可能不存在反转后的字符串。比如`LlaVa`和`aValL`

- 为什么 LLM 在非英语语言（例如日语）上表现较差？
	- 训练数据多以英文为主。
	- 许多LLM都以英文训练为主，在此基础上的其他语种fine tune可能和英语语法有所出入，会导致性能下降。
- 为什么 LLM 在简单算术问题上表现不好？
	- 训练数据以文本为主，而且生成是next token prediction，是基于概率模型的预测，而非逻辑上的运算，所以在简单的算术问题是会有较差的表现。
- 为什么 GPT-2 在编写 Python 代码时遇到比预期更多的困难？
	- 训练数据以文本为主，代码上的训练较少。
	- GPT-2的模型设计较为简单，缺乏深度思考或较强的逻辑性，对于代码不够擅长。
	- 编写代码需要较长的上下文联系（比如全局变量），GPT-2对于长文本生成可能存在遗忘。
	- GPT-2的任务主要是文本生成，而非代码生成。
- 为什么 LLM 遇到字符串 “<|endoftext|>” 时会突然中断？
	- 标记的一种规则，遇到该符号停止生成。
- 为什么当问 LLM 关于 “SolidGoldMagikarp” 的问题时 LLM 会崩溃？
	- 这种有可能是在`vocab`里出现但是没有参与或较少地参与到了训练过程，导致LLM遇到该词汇崩溃。
	- 也有可能是在前向传播时精度不够导致推理过程中崩溃。
- 为什么在使用 LLM 时应该更倾向于使用 YAML 而不是 JSON？
	- YAML可读性更强，而且可以添加注释，更利于开发人员进行阅读和修改。
	- YAML格式简单。
- 为什么 LLM 实际上不是端到端的语言建模？
	- 需要tokenizer进行预处理，而且可能需要外部数据库等工具的集成。

## LLM training

根据Andrej Karpathy的视频，复现了GPT2的模型框架，git commit 提交记录如下

![image-20250405003706219](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20250405003706219.png)![image-20250405003721941](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20250405003721941.png)

提交记录可以[点击此处](https://github.com/coldsakrua/align/commits/master/)基本就是按照youtube上章节划分，commit的message就是实现的功能。前半部分侧重于模型的实现，后半部分侧重于训练+推理的优化以及evaluation。比较有收获的点：`torch.compile`可以加速训练（不过并不常用）；`vocab_size`的修改可以加速训练；`kernel fusion`的使用和`ddp`的实现（之前都是直接套）。

以及GPT2相比与算法，更像是一种工程实现，在算法实现的基础上不断套优化（后半部分几乎都在优化）。

- Section1：模型的构造+初始化，包括load hf中要改一些字典键名（我记得megatron2pytorch也需要）
- Section2：一些加速trick，更倾向于底层算法（精度、conpile、flash attention等，包括vocab_size也是为了加速GPU内的运算）
- Section3：优化模型的效果，并且分布式训练。比如使用fusionAdamW，梯度累计，warmup等，分布式训练使用DDP，最后来评估模型效果。
- Section4：Summary，查看了nanogpt和124M的gpt2、gpt3的效果比较，并且证明了c比py快。一些小总结
- PS：Karpathy真有钱，随手8张a100。。。

## LoRA finetune

