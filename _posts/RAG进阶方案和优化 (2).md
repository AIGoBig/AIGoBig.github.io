# 目录
这次主要进行RAG相关的分享，注意分为三个部分

- RAG 应用优化思路：介绍调研和基于模块化、主动检索等思想进行的一些优化方法的总结
- RAG 应用典型流程：这个主要除了固定的pipeline，更加灵活的甚至主动的编排流程
- RAG 应用优化实验：主要进行了灵活的甚至主动的编排流程
# RAG应用优化思路
## 动机
**背景**
针对RAG应用：

1. **RAG应用探索及实验**：**由于之前做的问答应用固定流程的pipeline基础上，**目前**通过这种模块化及流程优化**可以让rag通过模型训练有更好的效果，**因此进行了一些调研和探索。**
2. **可以解决和缓解一些问题**（比如防止不应召回的进行召回影响输出，非单一搜索可以解决的问题，无法为段落提供引文的可解释性问题等）

**方案**
根据最新的文献和工作，主要调研如下优化方案：

- 模块化RAG：基于原始的固定流程pipeline，增加可扩展性。模块化，易于扩展、组合 ，针对不同业务场景可以轻松实现更优的方案。
- 主动检索：RAG系统可以主动确定检索的时机，并决定何时结束整个过程并生成最终结果。
> - 实现自主编排：结合agent思想，让模型作为决策者参与完整的RAG链路。
> - 组件的进一步扩充：联网搜索组件、query扩展等等组件。


## 从Naive到高级、模块的RAG-1
从这篇综述开始今年发这个综述中，之前一芒也在群里发过，**总结了整个RAG发展过程，以及进行了按时间先后的三种分类**
**朴素**RAG在三个关键领域面临**重大挑战:“检索”、“生成”和“增强”**。
**检索质量：**带来了各种各样的挑战，**包括：精度低，导致检索块错位，以及出现幻觉等潜在问题**。**低召回率**，导致无法检索所有相关块，从而阻碍llm构建全面响应的能力。**过时的信息**进一步加剧了问题，可能会产生不准确的检索结果。
**生成质量：**主要是幻觉问题，模型生成的答案在提供的**上下文中没有根据**，以及**不相关的上下文和模型输出中的潜在毒性或偏差问题**。
**增强过程：**

- **不连贯的输出：**在有效整合检索段落的上下文与当前生成任务方面，可能导致**脱节或不连贯的输出**。
- **冗余和重复**问题：特别是当**多个检索到的段落包含相似的信息时**，导致生成的响应中出现重复的内容。
- **任务相关性问题：识别多个检索段落对生成任务的重要性和相关性**是另一个挑战，需要对每个段落的价值进行适当的平衡。
- **输出的一致性：协调写作风格和语调的差异以确保输出的一致性**也是至关重要的。

**RAG研究范式**是不断发展的，本节主要描述了它的发展过程。本文将其分为三种类型:朴素RAG、高级RAG和模块化RAG。
**高级RAG和模块化RAG的开发是对朴素RAG中局限性和的处理。**
> 最后，生成模型**存在过度依赖增强信息的风险**，可能导致输出只是重申检索到的内容，而没有提供新的价值或合成的信息。



> RAG优势：针对幻觉、过时的知识和不透明、不可追溯的推理过程，增强了模型的准确性和可信度 贡献如下: 
> - 对最先进的RAG进行了全面和系统的回顾，通过包括幼稚RAG、高级RAG和模块化RAG在内的范式描述了它的演变。这篇综述的背景下，更广泛的范围内的法学硕士研究RAG的景观。
> - 我们确定并讨论了RAG过程中不可或缺的核心技术，特别关注“检索”，“生成器”和“增强”方面，并深入研究了它们的协同作用，阐明了这些组件如何复杂地协作以形成一个有凝聚力和有效的RAG框架。
> - 我们为RAG构建了一个全面的评估框架，概述了评估目标和指标。我们的对比分析从不同的角度阐明了RAG与微调相比的优缺点。
> 
此外，我们预测了RAG的未来方向，强调潜在的增强以应对当前的挑战，扩展到多模式设置，以及其生态系统的发展。
> 
> 第2节和第3节定义了RAG并详细说明了其发展过程。 第4节到第6节探讨了核心组件——检索、“生成”和“增强”——重点介绍了各种嵌入式技术。 第7节重点介绍RAG的评估体系。
> 第8节将RAG与其他LLM优化方法进行了比较，并提出了其可能的发展方向。 


## 从Naive到高级、模块的RAG-2
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1712044267827-071ba24c-5ad1-45ce-b75e-d12d0399c618.png#clientId=u0ce91fe0-0789-4&from=paste&height=908&id=uaefa1be5&originHeight=1635&originWidth=2576&originalType=binary&ratio=1.7999999523162842&rotation=0&showTitle=false&size=491014&status=done&style=none&taskId=u2b44a991-58d6-416e-ae2d-7903dc1db68&title=&width=1431.1111490226097)



**RAG的进化轨迹在四个不同的阶段展开**
初始阶段：在**2017年成立之初**，**与Transformer架构的出现相一致**，主要的推动力是通过**预训练模型(PTM)吸收额外的知识**来增强语言模型。**这主要是针对优化预训练方法。**
休眠阶段：在这个初始阶段之后，在chatGPT出现之前，有一段相对的**休眠期**，在此期间，对RAG的相关研究进展甚微。
关键时刻：随后，**chatGPT的出现标志着这一发展轨迹的关键时刻**，将LLMs推向了前沿。社区的焦点转向利用LLMs的能力，以达到更高的可控性和解决不断变化的需求。因此，**RAG的大部分努力都集中在推理上，少数致力于微调过程**。
**混合发展：**随着LLM技术的不断发展，特别是**GPT-4的引入**，RAG技术的前景发生了重大变化。**重点发展成为一种混合方法，结合RAG和微调的优势，以及专门的少数人继续专注于优化预训练方法**。
**以及出现了一些方法来研究关键问题，诸如“检索什么”、“何时检索”和“如何使用检索到的信息”。**

> 尽管RAG研究发展迅速，但该领域缺乏系统的整合和抽象，这对理解RAG进展的全面前景提出了挑战。本调查旨在概述整个RAG过程，并通过提供对法学硕士检索增强的彻底检查，涵盖RAG研究的当前和未来方向。

> 在技术上，RAG通过各种创新方法得到了丰富，**这些方法解决了诸如“检索什么”、“何时检索”和“如何使用检索到的信息”等关键问题。**
> 对于“检索什么”的研究已经从**简单的令牌**[Khandelwal等人，2019]和**实体检索**[Nishikawa等人，2022]发展到**更复杂的结构**，如块[Ram等人，2023]和知识图谱[Kang等人，2023]，研究重点是**检索的粒度和数据结构的水平**。粗粒度带来更多的信息，但精度较低。检索结构化文本可以在牺牲效率的同时提供更多信息。
“何时检索”的问题导致了从单一[Wang等人，2023e, Shi等人，2023]到**自适应**[Jiang等人，2023b, Huang等人，2023]和**多重检索**[Izacard等人，2022]方法的策略。**检索频率高，信息量大，效率低。**
> 至于“如何使用”检索到的数据，已经在模型架构的各个层次上开发了**集成技术**，包括输入层[Khattab等人，2022]、中间层[Borgeaud等人，2022]和输出层[Liang等人，2023]。虽然**“中间层”和“输出层”更有效，但存在需要训练和效率低的问题**。

## 从Naive到高级、模块的RAG-3
高级RAG的开发具有**针对性的增强**，以解决naive RAG的缺点。在**检索质量**方面，高级RAG实现了**预检索以及后检索策略**。为了解决幼稚RAG遇到的索引挑战，高级RAG使用**滑动窗口、细粒度分割和元数据**等技术重新定义了其**索引方法**。它还引入了各种方法来优化检索过程[ILIN, 2023]。 Pre-Retrieval过程和post-Retrieval过程在一会的模块化总结里具体分享。
**模块化的RAG：**模块化RAG呈现出一种**高度可扩展**的范式，将RAG系统划分为**模块类型、模块和运算符**三层结构。每种模块类型代表着RAG系统中的一个核心流程，包含多个功能模块。

![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147250324-bf6864ad-f154-4968-b812-8a6a808519a3.png#clientId=udc119121-7e43-4&from=paste&id=AQaP5&originHeight=711&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=udc477b3e-b33b-4e08-9574-10ae40f4fb4&title=)

**每个功能模块又包含多个具体的运算符**。**整个RAG系统变成了多个模块和相应运算符的排列组合**，形成我们所说的RAG流程。
**在流程中，可以在每个模块类型中选择不同的功能模块，在每个功能模块内部，又可以选择一个或多个运算符。**

> **Pre-Retrieval过程**
> 优化数据索引。优化数据索引的目标是**提高被索引内容的质量**。这涉及到五个主要策略: 增强数据粒度、优化索引结构、添加元数据、对齐优化和混合检索。
> 增强数据粒度旨在提高文本的标准化、一致性、事实准确性和丰富的上下文，从而提高RAG系统的性能。这包括去除无关信息，**消除实体和术语中的歧义**，确认事实准确性，维护上下文，以及更新过时的文档。
> 优化索引结构包括**调整分块的大小以捕获相关上下文**，跨多个索引路径查询，以及通过利用图数据索引中节点之间的关系从图结构中合并信息以捕获相关上下文。
> 添**加元数据信息**涉及将引用的元数据(如日期和目的)集成到块中以实现过滤目的，并将引用的章节和子节等元数据合并以提高检索效率。
> **对齐优化通过在文档中引入“假定性问题”**[Li etal.， 2023d]来纠正对齐问题和差异，从而解决文档之间的对齐问题和差异。
> **检索**
> 在检索阶段，主要关注的是通过**计算查询和块之间的相似度**来识别适当的上下文。嵌入模型是这个过程的核心。在高级RAG中，有可能对嵌入模型进行优化。
> **微调嵌入。**微调嵌入模型会显著影响RAG系统中检索内容的相关性。这个过程涉及到定制嵌入模型，以增强特定领域上下文的检索相关性，特别是对于处理演变或
> 稀有术语的专业领域。BGE嵌入模型[BAAI, 2023]，如BAAI开发的**BGE-large- en2**，就是一个可以微调以优化检索相关性的高性能嵌入模型的例子。可以使用GPT-3.5-
> turbo等语言模型生成用于微调的训练数据，以制定基于文档块的问题，然后将其用作微调对。
> **动态嵌入适应单词使用的上下文**，不像静态嵌入，它为每个单词使用单个向量[Karpukhin等人，2020]。例如，在BERT这样的transformer模型中，同一个单词可以根据周围的单词具有不同的嵌入。OpenAI的embedding -ada-02模型3，建立在原理之上它是一个复杂的动态嵌入模型，可以捕捉上下文理解。
> 然而，它可能不会像GPT-4这样最新的全尺寸语言模型那样对上下文表现出同样的敏感性。
> **Post-Retrieval过程**
> 在从数据库中检索有价值的上下文之后，有必要将其与查询合并，作为llm的输入，同时解决上下文窗口限制带来的挑战。简单地将所有相关文件一次性呈现给
> LLM可能会超出上下文窗口限制，引入噪音，并阻碍对关键信息的关注。为了解决这些问题，**对检索到的内容进行额外的处理是必要的**。
> 重新评估。对检索到的信息重新排序，将最相关的内容重新定位到提示符的边缘是一个关键策略。这一概念已经在LlamaIndex4、LangChain5 和HayStack [Blagojevi,2023]等框架中得到了实现。例如，Diversity Ranker6根据文档多样性进行优先级重排序，而LostInTheMiddleRanker替代将最佳文档放置在上下文窗口的开始和结束位置。另外， 像cohereAIrerank [Cohere, 2023]、bge-rerank7 和LongLLMLingua [Jiang等人，2023a]这样的方法重新计算
> 了相关文本和查询之间的语义相似度，解决了解释基于向量的模拟搜索以获得语义相似度的挑战。
> **提示压缩。**研究表明，检索文档中的噪声会对RAG性能产生不利影响。在后处理中，重点在于压缩无关的上下文，突出关键段落，减少整体的上下文长度。选择性上下文和LLMLingua等方法[Litman et al.， 2020,Anderson et al.， 2022]利用小型语言模型计算提示互信息或困惑度，估计元素重要性。Recomp [Xu等人，2023a]通
> 过在不同粒度上训练压缩器来解决这个问题，而LongContext [Xu等人，2023b]和“在记忆迷宫中行走”[Chen等人，2023a]设计总结技术来增强LLM的关键信息感知，
> 特别是在处理广泛的上下文时。


# 🌟RAG模块化方案
## **RAG模块化思路下的优化方法总结**
针对这个思想，进行了一些更多的调研，总结了目前的一些模块和优化思路的总结。
其中蓝色部分是目前已经加入问答应用或者进行了测试的功能，加粗的是我认为会比较有效后面想着重去看的功能。
优势：

1. **技术整合和扩展**：RAG正在整合其他技术，包括微调、适配器模块和强化学习，以增强检索能力。
2. **可适应的检索过程**：自主判断和LLM的使用增加了通过确定检索的需求来回答问题的效率。所以需要进行模块化的优化，例如多轮检索增强时以及自适应检索时，需要模型自适应的去判断使用哪个模块。

> ### **什么是模块化RAG？**
> 模块化RAG是指检索增强生成技术的一种演进形式，其进展带来了更加多样化和灵活的过程，具体体现在以下关键方面：
> 1. **增强数据获取**：RAG已经超越了传统的非结构化数据，现在包括半结构化和结构化数据，在预处理结构化数据方面更加关注，以改善检索并减少模型对外部知识来源的依赖。
> 2. **整合技术**：RAG正在整合其他技术，包括微调、适配器模块和强化学习，以增强检索能力。
> 3. **可适应的检索过程**：检索过程已经发展到支持多轮检索增强，利用检索内容来指导生成，反之亦然。此外，自主判断和LLM的使用增加了通过确定检索的需求来回答问题的效率。

## **索引模块**
索引是将文本分解为可管理的块的过程，在组织系统中是一个至关重要的步骤，面临着三个主要挑战：

- **不完整的内容表示**。块的语义信息受到分割方法的影响，在更长的**上下文中会导致重要信息的丢失**或被淹没。
- **不准确的块相似性搜索**。随着数据量的增加，检索中的**噪音增加**，导致频繁与错误数据匹配，使检索系统变得脆弱且不可靠。
- **引用轨迹不清晰**。检索到的块可能来自任何文档，**缺乏引用轨迹**，可能导致存在来自多个不同文档的块，尽管在语义上相似，但包含的内容却是完全不同的主题。
### **块优化**
较大的块可以**捕捉更多的上下文**，但也会产生更多的噪音，需要更长的处理时间和更高的成本。而较小的块可能无法完全传达所需的上下文，但它们的噪音较少。
#### 滑动窗口
平衡这些需求的一种简单方法是使用**重叠的块**。通过使用滑动窗口，语义转换得到增强。然而，存在一些限制，包括对上下文大小的控制不够精确，截断单词或句子的风险，以及缺乏语义考虑。
#### **从小到大**
关键思想是**将用于检索的块与用于合成的块分开**。使用较小的块可以提高检索的准确性，而较大的块可以提供更多的上下文信息。
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147250306-ffe8e21e-485e-4b17-b0c4-8bfedb95647a.png#clientId=udc119121-7e43-4&from=paste&id=u65393fa4&originHeight=279&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=uf89c125d-79d7-4032-a9a8-379f8bf7972&title=)





#### **元数据附加**
块可以使用元数据信息进行丰富，例如页码、文件名、作者、时间戳、摘要，或者块可以回答的问题。
### **结构化组织**
提高信息检索效率的一种有效方法是为文档**建立分层结构**。通过构建块结构，RAG系统可以加速检索和处理相关数据。
#### **层次化索引**
在文档的层次结构中，**节点按照父子关系排列**，与之关联的块链接到这些节点上。**数据摘要存储在每个节点上，有助于快速遍历数据**，并帮助RAG系统确定要提取的块。这种方法还可以**缓解由块提取问题引起的幻觉。**
构建结构化索引的方法主要包括：

- **结构意识。**对文档进行段落和句子分割。
- **内容意识。**利用PDF、HTML、Latex等文件的内在结构。
- **语义意识。**基于NLP技术，如利用NLTK，对文本进行语义识别和分割。
#### **知识图谱文档组织**
在构建文档的层次结构时，利用**知识图谱（KGs）有助于保持一致性**。它描述了不同概念和实体之间的关系，显著降低了幻觉的可能性。
## **⭐️预检索模块**
RAG的主要挑战之一是用户**提出精确清晰的问题是困难的**，不明智的查询会导致检索效果不佳。

- **问题措辞不当。**问题本身复杂，语言组织不佳。
- **语言复杂性和歧义性。**当处理专业词汇或具有多重含义的模糊缩写时，语言模型经常会遇到困难。
### **查询扩展**
将单个查询扩展为多个查询丰富了查询的内容，为**解决特定细微差别的缺乏**提供了进一步的上下文，从而确保生成的答案的最佳相关性。
#### **多查询**
通过**使用提示工程来通过LLM扩展查询**，这些查询可以并行执行。查询的扩展不是随意的，而是经过精心设计的。这种设计的两个关键标准是查询的多样性和覆盖范围。
#### **子查询**
子问题规划的过程代表了**生成必要的子问题**，以在组合时为**原始问题提供上下文并完全回答**。这个添加相关上下文的过程原则上类似于查询扩展。具体来说，可以使用从最少提示到最多提示的方法，**将复杂问题分解为一系列更简单的子问题。**
#### **CoVe**
**CoVe（Chain-of-Verification）**是由Meta AI提出的**另一种查询扩展方法**。扩展的查询**经过LLM的验证**，以达到减少幻觉的效果。经过验证的扩展查询通常具有更高的**可靠性。**
### **查询转换**
#### **重写**
原始查询在实际场景中并不总是最佳的LLM检索条件。因此，我们**可以提示LLM重写查询**。1）使用LLM进行查询重写外，2）还可以利用**专门的较小语言模型，例如RRR（重写-检索-阅读）**。
#### **HyDE**
当响应查询时，**LLM构建假设文档（假定答案）**，而不是直接在向量数据库中搜索查询及其计算的向量。它**专注于从答案到答案的嵌入相似性**，而不是寻求问题或查询的嵌入相似性。此外，它还包括**反向HyDE**，它专注于**从查询到查询的检索。**
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147251046-b3c8b845-e21b-4547-aa44-2199170b7ea3.png#clientId=udc119121-7e43-4&from=paste&id=u9dd53736&originHeight=450&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=u577dd96c-fca4-4c87-90b5-1e4fee16445&title=)









### **查询路由**
**根据不同的查询，将其路由到不同的RAG管道**，这适用于一个灵活的RAG系统，设计用于适应多样化的场景。
#### **元数据路由器/过滤器**
第一步涉及**从查询中提取关键字（实体），然后基于块内的关键字和元数据进行过滤，**缩小搜索范围。
#### **语义路由器**
另一种路由方法涉及利**用查询的语义信息**。具体方法请参见语义路由器。当然，也可以采用混合路由方法，结合基于语义和基于元数据的方法，以增强查询路由的效果。
### **查询构建**
将用户的查询转换为另一种查询语言，以访问替代数据源。常见的方法包括：

- 文本到Cypher
- 文本到**SQL**
## **检索模块**
检索过程在RAG中起着至关重要的作用。利用**强大的PLM可以有效地在潜在空间中表示查询和文本**，从而促进问题和文档之间的语义相似性的建立，以支持检索。
### **检索模型选择**
Hugging Face的**MTEB排行榜评估**了几乎所有可用的嵌入模型在8个任务上的性能 C-MTEB侧重于评估中文嵌入模型的能力，涵盖了6个任务和35个数据集。
#### **稀疏检索器**
尽管稀疏编码模型可能被认为是一种略显过时的技术，通常基于诸如**词频统计**之类的统计方法，但由于其**更高的编码效率和稳定性**，它们仍然具有一定的地位。常见的系数编码模型包括**BM25和TF-IDF。**
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147251021-a6d27d96-ffe0-4b7d-85a6-0613705ed9a2.png#clientId=udc119121-7e43-4&from=paste&id=u08fa1322&originHeight=447&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=u5b562df4-1738-4c60-918f-51f5002aefb&title=)







#### **密集检索器**

- 建立在BERT架构上的编码器-解码器语言模型，例如ColBERT。
- 像BGE和Baichuan-Text-Embedding这样的全面多任务微调模型。
- 基于云API的模型，例如OpenAI-Ada-002和Cohere Embedding。
- 面向大规模数据应用的下一代加速编码框架Dragon+。
- **混合/混合检索**
### **检索器微调**
在某些情况下，上下文可能与预训练模型在嵌入空间中认为相似的内容有所偏离，特别是在**高度专业化的领域**，如医疗保健、法律和其他丰富专有术语的领域中，调整嵌入模型可以解决这个问题。虽然这种调整需要额外的工作，但它可以**大大提高检索效率和领域对齐度**。
#### **SFT（自我训练）**
可以基于领域**特定数据**构建自己的微调数据集，使用**LlamaIndex**可以迅速完成此任务。
#### **🚩LSR（语言模型监督检索器）？？**
与直接从数据集构建微调数据集不同，**LSR利用语言模型生成的结果作为监督信号**，在RAG过程中微调嵌入模型。
#### **RL（强化学习）**
受RLHF（从人类反馈中进行强化学习）的启发，利用基于语言模型的反馈通过强化学习来强化检索器。
#### **Adapter**
有时，对整个检索器进行微调可能成本高昂，特别是在处理无法直接微调的基于API的检索器时。在这种情况下，我们可以通过**引入Adapter模块并进行微调**来缓解这一问题。
## **⭐后处理模块**
将整个文档块检索并直接输入LLM的上下文环境并不是一个最佳选择。**对文档进行后处理**可以帮助LLM更好地利用上下文信息。
主要挑战包括：

1. 中间丢失。与人类类似，LLM倾向于只记住长文本的**开头和结尾**，而忘记中间部分。
2. 噪声/反事实块。检索到的**嘈杂或事实相互矛盾的文档**可能会影响最终的检索生成。
3. 上下文窗口。尽管检索了大量相关内容，但大型模型对上下文信息长度的限制阻止了所有内容的包含。
### **重新排序**
重新排序已检索到的文档块，而不改变其内容或长度，以**增强LLM对更关键的文档块的可见性**。具体来说：
#### **基于规则的重新排序**
根据某些规则，计算指标以重新排序文档块。常见的指标包括：

- 多样性
- 相关性
- MRR

MMR的背后思想是**减少冗余并增加结果的多样性**，它常用于文本摘要。MMR根据查询相关性和信息新颖性的综合标准，在最终的关键短语列表中选择短语。
#### **模型基础的重新排序**
**利用语言模型对文档块进行重新排序**，可选的模型包括：

- 来自BERT系列的编码器-解码器模型，例如SpanBERT
- 专门的重新排序模型，例如Cohere rerank或bge-raranker-large
- 通用的大型语言模型，例如GPT-4
### **压缩和选择**
在RAG过程中的一个**常见误解**是认为尽可能检索更多相关文档并**将它们连接起来形成一个冗长的检索提示是有益的**。然而，过多的上下文可能会引入更多的噪音，降低LLM对关键信息的感知，并导致诸如“**中间丢失**”之类的问题。解决这个问题的常见方法是压缩和选择检索到的内容。
#### **LLMLingua**
通过**利用对齐和训练良好的小型语言模型**，例如GPT-2 Small或LLaMA-7B，可以实现**从提示中检测和删除不重要的标记，**将其转换为人类难以理解但LLM很好理解的形式。
#### **Recomp**
Recomp引入了两种类型的压缩器：**一种是抽取式压缩器**，从检索到的文档中选择相关的句子；**另一种是生成式压缩器**，通过将多个文档中的信息融合产生简洁的摘要。这两种压缩器都经过训练，以在生成的摘要被添加到语言模型的输入时提高语言模型在最终任务上的性能，同时确保摘要的简洁性
#### **选择性上下文**
**通过识别并删除输入上下文中的冗余内容，**可以简化输入，从而提高语言模型的推理效率。选择性上下文类似于“停用词移除”策略。
#### **LLM批评**
另一种直观且有效的方法是让LLM在生成最终答案之前评估已检索的内容。这使得LLM可以**通过LLM批评**过滤掉相关性较差的文档。
## **生成模块**
利用LLM根据用户的查询和检索到的上下文信息生成答案。
### **生成器选择**
根据场景的不同，LLM的选择可以分为以下两种类型：
#### **云API基础生成器**
基于云API的生成器利用第三方LLM的API，例如OpenAI的ChatGPT、GPT-4和Anthropic Claude等。优势包括：

- 无服务器压力
- 高并发性
- 能够使用更强大的模型

缺点包括：

- 数据通过网络传递，存在数据隐私问题
- 无法调整模型（在绝大多数情况下）
#### **本地部署**
本地部署的开源或自行开发的LLM，例如Llama系列、GLM等。其优势和劣势与基于云API的模型相反。本地部署的模型提供**更大的灵活性和更好的隐私保护，但需要更高的计算资源**。
### **生成器微调**
除了直接使用LLM外，根据场景和数据特征进行目标微调可以获得更好的结果。这也是使用本地部署设置的最大优势之一。常见的微调方法包括以下几种：
#### **SFT**
当LLM在特定领域缺乏数据时，可以通过微调向LLM提供额外的知识。Huggingface的微调数据也可以作为一个初始步骤。
微调的另一个好处是能够**调整模型的输入和输出**。例如，它可以使LLM**适应特定的数据格式**，并按照指示以**特定的风格生成响应**。
#### **RL**
通过**强化学习将LLM的输出与人类或检索器的偏好进行对齐是一个潜在的方法**。例如，手动注释最终生成的答案，然后通过强化学习提供反馈。**除了与人类偏好保持一致外，还可以与微调模型和检索器的偏好保持一致。**
#### **蒸馏**
当情况阻止访问强大的专有模型或更大参数的开源模型时**，一种简单有效的方法是将更强大的模型（例如GPT-4）蒸馏为更小的模型**。
## **⭐调度模块**
Orchestration指的是控制RAG过程的模块。与以前固定的过程不同，**RAG现在涉及在关键点做出决策，并根据结果动态选择下一步。**与Naive RAG相比，这也是**模块化RAG的主要特点之一。**
### **调度**
Judge模块评估RAG过程中的关键点，确定是否需要检索外部文档存储库，答案是否满意，以及是否需要进一步探索。**它通常用于递归、迭代和自适应检索。**具体来说，它主要包括以下两种操作符：
#### **基于规则**
下一步的行动基于**预定义的规则确定**。通常，生成的答案会得分，然后**根据得分是否达到预定义的阈值**来决定是否继续或停止。常见的阈值包括令牌的置信水平。
#### **基于提示**
**LLM自主确定下一步的行动**。主要有两种方法实现这一点。**第一种方法涉及提示LLM反思或根据对话历史进行判断，如ReACT框架所示。**这里的好处是消除了对模型进行微调的需要。然而，判断的输出格式取决于LLM是否遵循指令。基于提示的案例是FLARE。
#### **基于调整**
第二种方法**涉及LLM生成特定的令牌来触发特定的操作**，这种方法可以追溯到Toolformer，并应用于RAG，**例如Self-RAG。**
### **检索融合**
如前面关于查询扩展的部分所述，当前的RAG过程不再是一个单一的管道。它通常需要通过多个分支来扩展检索范围或多样性。因此，在扩展到多个分支之后，融合模块被依赖于来合并多个答案。
#### **概率集成**
融合方法**基于从多个分支生成的不同令牌的加权值**，从而全面选择最终的输出。加权平均是主要采用的方法。**参见REPLUG。**
#### **RRF（互惠排名融合）**
RRF是一种将**多个搜索结果列表的排名结合起来生成单一统一排名的技术**。与满足任何单一分支下的重新排序相比，与满足任何单一分支下的重新排序相比，**RRF产生的结果更为有效。**
# 案例 - OpenAI RAG优化路线
### 案例-1
[OpenAI的DevDay最新披露-”精进大型语言模型性能的各种技巧”专场ppt分享](https://zhuanlan.zhihu.com/p/666867807)
OpenAI Dev day中，在精进大型语言模型性能的各种技巧分享中，分享了他们的优化路线，其中就包括RAG。
首先，他们提出了**优化LLM的性能并不总是线性的**，关于RAG和微调之前的选择我认为可以参考：

- 很多时候，人们会先进行提示工程（prompt engineering），然后进行检索增强生成/RAG（retrieval-augmented generation），然后进行微调，这是优化LLM的常见方式。
- 这种方式有问题，因为**RAG和微调解决的是不同的问题**。有时你需要前者，有时你需要后者，有时你可能都需要，这取决于你要处理的问题的类别：
   - **优化有两个轴线方向可以考虑**：
      - 一个是上下文context优化，即模型需要了解什么信息才能解决你的问题。
      - 另一个是LLM优化，即模型需要以何种方式行动，才能真正解决你的问题；
   - 方案：
      - 你应该做的第一件事是从提示词优化开始进行评估，找出你如何持续评估输出的质量。
      - 直到你可以确定，**这到底是一个上下文问题还是一个我们需要模型如何行动的问题**？
      - 如果是前者，请转向RAG，如果是后者，请转向微调。
      - 当然有些时候你会同时需要两者，而且应该再次强调这不是一个线性的过程，而可能是类似下图这样的有反复横跳的过程；
- **RAG/检索增强生成 - 给模型提供特定领域全部内容**
   - 尝试了提示词优化后，你的下一步是要看到底先尝试RAG还是微调（**并不总是微调优于RAG！**）。**这里可以将其理解为短期记忆与长期记忆的选择问题** 
> 如果我们把这个问题想象成为准备考试，你的提示就是给他们需要完成考试的指导。微调就像你之前学习的所有方法和需要回答这些问题的实际框架。而RAG就像在他们实际参加考试时给他们一个开卷用的课本。如果他们知道方法和需要寻找的内容，那么RAG意味着他们可以打开书本，翻到正确的页面，轻松找到他们需要的内容。

![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1711993339320-6ef3966e-5b6c-4e2d-a31f-04ea016c2942.png#clientId=u3cc00b57-e268-4&from=paste&id=ev0YP&originHeight=472&originWidth=1080&originalType=url&ratio=2&rotation=0&showTitle=false&status=done&style=none&taskId=u18f64adf-09f9-478b-8fcf-d8e8770d8e1&title=)



微调的优点：

- 提高模型在**特定任务上的性能**
   - 经常是比提示工程化或FSL更有效的提高模型性能的方式；
- **提高模型效率**
   - 减少模型在您的任务上表现良好所需的token数量；
   - 将大型模型的专业知识提炼到较小的模型中；
> 微调的最佳实践：
> - 从提示工程化和 FSI 开始 - 从低投入的技术开始，快速迭代并验证您的用例；
> - 建立基线 - 确保你有一个性能基线来对比你的微调模型；
> - 从小处开始，注重质量 - 数据集可能难以构建，从小开始并有意识地投入。优化较少的高质量训练示例；

**微调+RAG**

- 微调模型以理解复杂的指令；
- 最小化提示工程token数，为检索的上下文提供更多空间；
- 使用 RAG 将相关知识注入上下文；
### 案例-2
**对RAG进行优化能大幅提升回答质量 - 需要专家来进行多轮尝试**

OpenAI团队提到，虽然随便通过某个LLMops你也能获得一个RAG能力的AI-bot，但是最初的回答质量其实是很容易出问题的。
参考下图，即使是OpenAI的顶级工程师，依然是在尝试了多次不同的优化后才为某个用户的RAG应用案例的回答质量从最开始的45%提升到最后的98%
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147251923-af3a287e-b237-4cda-81a0-6feb6919a91b.png#clientId=udc119121-7e43-4&from=paste&id=zFahQ&originHeight=214&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=u58e23547-64a7-4d41-a07c-aa24d307224&title=)







他们探索了**假设性文档嵌入（HyDE）**、**微调嵌入**和其他方法，但结果并不令人满意。（_顺道提一下他们提到了针对这个案例也试了做微调，效果不太好，_**_正好再次说明这类型特定领域的回答用RAG效果更好的大原则_**）
通过**尝试不同大小的信息块和嵌入不同的内容部分**，他们将准确率提高到了65%。
通过**重新排名**和**针对不同类型问题定制**的方法，他们进一步将准确率提高到了85%。
最终，通过**结合提示工程、查询扩展和其他方法**，他们实现了98%的准确率。
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/webp/65956778/1711993339296-e36e5618-5e9a-4855-b6ad-06e4d689e201.webp#clientId=u3cc00b57-e268-4&from=paste&id=PBMD9&originHeight=487&originWidth=1080&originalType=url&ratio=2&rotation=0&showTitle=false&status=done&style=none&taskId=ueafe7b04-8ffe-4ab1-9602-556b51f377d&title=)











> **Ragas - RAG评估框架**
> 测量四个指标。其中两个衡量 LLM 回答问题的效果，另外两个衡量内容和问题的相关性。通过观察分值会对如何优化RAG产生良好指导意义。
> ![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1711993339304-b25bde89-e75b-4665-8d88-4838057c823f.png#clientId=u3cc00b57-e268-4&from=paste&id=z0DLy&originHeight=427&originWidth=1080&originalType=url&ratio=2&rotation=0&showTitle=false&status=done&style=none&taskId=ub58db5f3-92e5-4f1d-b5b2-8effcf064a5&title=)
>
> 
>
> 
>
> 
>
> 
>
> **微调 - 在更小、特定领域的数据集上继续训练过程，以优化模型以完成特定任务。请注意 - 如果提示词工程优化的效果不好，很可能就不用尝试微调了...**


# 典型的RAG流程模式
### **微调阶段**
#### **🚩检索器微调**

- **直接微调检索器。** 构建用于检索的专门数据集，并对密集检索器进行微调。例如，使用开源检索数据集或基于特定领域数据构建数据集。
- **添加可训练的适配器模块。** 直接微调基于云API的嵌入模型。
- **LLM监督检索（LSR）。** **根据LLM生成的结果对检索器进行微调。**
- **LLM奖励RL：** 仍然使用LLM输出结果作为监督信号。利用强化学习来使检索器与生成器对齐。

![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147251066-5b100919-ca06-48d8-8652-7fd55140f872.png#clientId=udc119121-7e43-4&from=paste&id=u6c3a625e&originHeight=340&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=u67c45077-d066-4f14-8aaf-14d3b5b3a9d&title=)
#### **生成器微调**

- **直接微调**。通过外部数据集的微调可以**为生成器提供额外的知识**。另一个好处是能够**定制输入和输出格式。**通过设置问答格式，LLM可以理解特定的数据格式，并根据指令进行输出。
- **GPT-4蒸馏**。在使用开源模型的本地部署时，一个简单有效的方法是使用GPT-4**批量构建微调数据，以增强开源模型的能力**。
- **LLM/人类反馈的强化学习**。基于最终生成的答案的反馈进行强化学习。除了使用人类评估之外，**GPT-4还可以作为评估法官**。

![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147251099-96bcd312-d296-42a8-9be9-82c30bbd9f04.png#clientId=udc119121-7e43-4&from=paste&id=u7b0adb0c&originHeight=314&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=u6b7b6c6f-15c6-449c-8228-8d813ea31cd&title=)
### **推理阶段**
#### **顺序结构**
RAG流的顺序结构将RAG的模块和操作以线性管道的形式组织起来，如下图所示。如果包括了前检索和后检索模块类型，则代表了典型的高级RAG范式；否则，它体现了典型的简单RAG范式。
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147251315-15f9b510-d28d-438d-b1f9-41d455f3c7dc.png#clientId=udc119121-7e43-4&from=paste&id=ubb03f3ee&originHeight=206&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=uec644e8d-a747-408b-bbf0-7059b7b1df3&title=)
最广泛使用的RAG流水线目前是顺序结构，通常**在检索之前包括查询重写或HyDE**，并在**检索后包括重新排序操作**，例如QAnything案例。
在RRR中，Query Rewrite模块是一个较小的可训练语言模型，在强化学习的背景下，重写器的优化被形式化为马尔可夫决策过程，LLM的最终输出作为奖励。检索器利用稀疏编码模型BM25。
#### **条件结构**
具有条件结构的RAG流**涉及根据不同条件选择不同的RAG路径**。通常，这是**通过路由模块**实现的，该模块根据查询关键词或语义确定路径。
基于问题类型选择不同路径，针对特定情景导向不同的流。**例如**，当用户询问严肃问题、政治问题或娱乐话题时，来自大模型的答案容忍度不同。**不同的路由分支**通常在检索源、检索过程、配置、模型和提示方面有所不同。
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147251459-cf0a77f0-5909-42ad-b767-3bbbd8b91d77.png#clientId=udc119121-7e43-4&from=paste&id=u7536dd56&originHeight=320&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=u8f68dbef-aeaf-48b6-831e-3b5ab626725&title=)







#### **分支结构**
具有分支结构的RAG流与条件方法不同之处在于，它**涉及多个平行分支**，**而不是**在条件方法中从多个选项中**选择**一个分支。在结构上，它可以分为两种类型：
**前检索分支（多查询，并行检索）**。这涉及**扩展原始查询以获取多个子查询**，然后针对每个子查询进行单独的检索。在检索之后，该方法允许基于子问题和相应的检索内容立即生成答案。或者，它可能仅涉及使用扩展的检索内容，**并将其合并为生成的统一上下文。**
**后检索分支（单一查询，并行生成）**。**该方法保留原始查询并检索多个文档块。**随后，**它同时使用原始查询和每个文档块进行生成**，**最终将生成的结果合并在一起**。
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147251476-f6207c69-56cf-4423-8cad-39e527fdec42.png#clientId=udc119121-7e43-4&from=paste&id=u12046070&originHeight=431&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=u7f3ff4e1-7fb6-49e1-b52a-b2ebe1893f7&title=)









REPLUG体现了经典的后检索分支结构，其中**为每个分支预测了每个标记的概率**。通过加权可能性合集，将不同的分支聚合在一起，最终生成的结果用于通过反馈对检索器进行微调，称为Contriever。

#### **循环结构**
具有循环结构的RAG Flow是**模块化RAG**的一个重要特征，**涉及相互依赖的检索和推理步骤**。通常**包括一个用于流程控制的Judge模块**。这**可以进一步分为迭代、递归和自适应（主动）检索方法。**
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147251521-68bc751f-73f3-4ca5-9421-fbe58c5b7542.png#clientId=udc119121-7e43-4&from=paste&id=u7622ba04&originHeight=761&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=u93115156-b7a5-4118-b901-612d1e76822&title=)















#### **迭代结构**
有时，**单次检索和生成可能无法有效解决需要广泛知识的复杂问题**。因此，在RAG中可以使用迭代方法，通常**涉及固定数量的迭代进行检索。**
> 在prompt优化里也有应用

![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147251563-eded1f43-f09e-4ab1-be13-ee42ab4052af.png#clientId=udc119121-7e43-4&from=paste&id=u43b41a77&originHeight=281&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=uaa1fd784-0765-4bae-96f4-806c96c4388&title=)
迭代检索的一个典型案例是ITER-RETGEN，它迭代进行检索增强生成和生成增强检索。检索增强生成根据**所有检索到的知识输出任务**输入的响应。在每次迭代中，ITER-RETGEN**利用上一次迭代的模型输出作为特定上下文**，帮助检索更相关的知识。**循环的终止由预定义的迭代次数确定。**

#### **递归结构**
递归检索的**特征特点与迭代检索相反**，其清晰依赖于先前步骤并持续深化检索。通常，递归检索具有**终止机制作为递归检索的退出条件**。在RAG系统中，**递归检索通常涉及查询转换**，依赖于每次检索的新重写查询。
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147251783-44d322cc-192c-424d-ae29-7a1dc10b29e7.png#clientId=udc119121-7e43-4&from=paste&id=u41edc066&originHeight=170&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=u61739718-39b8-438d-8c36-9674ed5438d&title=)
递归检索的典型实现，如ToC，涉及递归执行RAC（递归增强澄清），逐渐将子节点插入澄清树中，从初始模糊问题（AQ）开始。在每个扩展步骤中，根据当前查询执行段落重新排名，以生成明确的问题（DQ)。在达到最大有效节点数或最大深度时，树的探索结束。一旦构建了澄清树，ToC收集所有有效节点并生成全面的长文本答案来解决AQ。

#### **🚩**🌟**自适应结构**
随着RAG的发展，逐渐**从被动检索转向了自适应检索的出现，也称为主动检索**，这在一定程度上归功于LLM的强大能力。**这与LLM Agent共享一个核心概念。**

RAG系统可以主动确定检索的时机，**并决定何时结束整个过程并生成最终结果。基于判断标准，这可以进一步分为基于提示和基于调整的方法。**
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147251885-f8066ae6-5481-4759-8f96-392e2f0acafe.png#clientId=udc119121-7e43-4&from=paste&id=u5a756523&originHeight=261&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=u297834e4-c021-496f-a628-53d413febe6&title=)
**基于调整的方法涉及对LLM进行微调以生成特殊标记，从而触发检索或生成。**这个概念可以追溯到**Toolformer**，其中生成特定内容有助于调用工具。在RAG系统中，这种方法用于**控制检索和生成步骤**。一个典型的案例是Self-RAG。具体来说：

1. 给定一个输入提示和前一个生成结果，首先预测特殊标记“Retrieve”是否有助于通过段落检索增强持续的生成。
2. 如果需要检索，**模型生成**：评价标记来评估检索段的相关性，下一个响应段，以及评价标记来评估响应段中的信息是否得到段的支持。
3. 最后评价标记评估响应的整体效用，并选择最佳结果作为最终输出。
- 实验结果显示，Self-RAG 在多种任务上，如开放领域的问答、推理和事实验证，均表现得比现有的LLMs（如 ChatGPT）和检索增强模型（如检索增强的 Llama2-chat）更好，特别是在事实性和引用准确性方面有显著提高。



**这一过程不同于传统的 RAG（图 1 左），后者无论检索的必要性如何（例如，下图示例不需要事实性知识），都会持续检索固定数量的文档进行生成，而且从不对生成质量进行二次检查。**
此外，SELF-RAG 还会**为每个段落提供引文，并对输出结果是否得到段落支持进行自我评估**，从而更容易进行事实验证。

# 🌟self-reflection类方法/自己的方法
**但RAG也有其局限性，例如不加选择地进行检索和只整合固定数量的段落，可能导致生成的回应不够准确或与问题不相关。**
为了进一步改进，作者提出了自反思检索增强生成（Self-RAG, Self-Reflective Retrieval-Augmented Generation）。

- 可以根据需要**自适应地检索段落**（即：模型可以判断是否有必要进行检索增强）
- 还引入了名为**反思令牌**（reflection tokens）的特殊令牌，使LM在推理阶段可控。



1. **检索**：首先，Self-RAG **解码检索令牌（retrieval token）以评估是否需要检索，并控制检索组件**。如果需要检索，LM 将调用外部检索模块查找相关文档。
2. **生成**：如果不需要检索，**模型会预测下一个输出段**。如果需要检索，模型首先**生成批评令牌（critique token）来评估检索到的文档是否相关，然后根据检索到的段落生成后续内容。**
3. **批评**：如果需要检索，模型进一步评估段落是否支持生成。最后，**一个新的批评令牌（critique token）评估响应的整体效用。**



具体来说，在给定输入提示和前几代的情况下，会判断用检索到的段落来增强继续生成是否有帮助。

1. 如果有帮助，**它就会输出一个检索标记，**按需调用检索模型（步骤 1）。
2. 随后，SELF-RAG 同时处理多个检索到的段落，**评估它们的相关性，然后生成相应的任务输出（步骤 2）。**
3. 然后，**它生成批判标记来批判自己的输出**，**并从事实性和整体质量方面选择最佳输出**（第 3 步）。

**这一过程不同于传统的 RAG（图 1 左），后者无论检索的必要性如何（例如，下图示例不需要事实性知识），都会持续检索固定数量的文档进行生成，而且从不对生成质量进行二次检查。**
此外，SELF-RAG 还会**为每个段落提供引文，并对输出结果是否得到段落支持进行自我评估**，从而更容易进行事实验证。


# 结束
> 



# 🌟self-reflection类方法
## 🚩可以解决的业务问题

1. 可以防止不应召回的进行召回影响输出。
2. 非通过单一搜索可以解决的。
3. 为每个段落提供引文

## **前言**
大型语言模型（LLMs）具有出色的能力，但由于完全依赖其内部的参数化知识，它们经常产生包含事实错误的回答，尤其在长尾知识中。为了解决这一问题，之前的研究人员提出了检索增强生成（RAG），它通过检索相关知识来增强LMs的效果，尤其在需要大量知识的任务，如问答中，表现出色。
**但RAG也有其局限性，例如不加选择地进行检索和只整合固定数量的段落，可能导致生成的回应不够准确或与问题不相关。**
为了进一步改进，作者提出了自反思检索增强生成（Self-RAG, Self-Reflective Retrieval-Augmented Generation）。这是一个新框架，**它不仅可以根据需要自适应地检索段落（即：模型可以判断是否有必要进行检索增强）**，还引**入了名为反思令牌（reflection tokens）的特殊令牌，使LM在推理阶段可控。**
实验结果显示，Self-RAG 在多种任务上，如开放领域的问答、推理和事实验证，均表现得比现有的LLMs（如 ChatGPT）和检索增强模型（如检索增强的 Llama2-chat）更好，特别是在事实性和引用准确性方面有显著提高。
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1711014453364-a36cc4f1-5cc4-46bd-8774-6a2391d62ac8.png#clientId=u5bc522dc-dd8a-4&from=paste&id=u76aae4f8&originHeight=383&originWidth=1440&originalType=url&ratio=1.7999999523162842&rotation=0&showTitle=false&status=done&style=none&taskId=u656d7b67-ebd8-4a1d-8ac4-280b2a65453&title=)

## **方法**
引入了自我反思检索增强生成（SELF-RAG），通过按需检索和自我反思来提高LLM的生成质量，包括其事实准确性，而不损害其通用性。以端到端方式训练任意LLM，**使其学会在任务输入时，通过生成任务输出和间歇性特殊标记（即反思标记）来反思自己的生成过程**。反思标记分为检索标记和批判标记，分别表示检索需求和生成质量。
Self-RAG 是一个新的框架，通过**自我反思令牌**（Self-reflection tokens）来训练和控制任意 LM。它主要分为三个步骤：检索、生成和批评。

1. **检索**：首先，Self-RAG **解码检索令牌（retrieval token）以评估是否需要检索，并控制检索组件**。如果需要检索，LM 将调用外部检索模块查找相关文档。
2. **生成**：如果不需要检索，**模型会预测下一个输出段**。如果需要检索，模型首先**生成批评令牌（critique token）来评估检索到的文档是否相关，然后根据检索到的段落生成后续内容。**
3. **批评**：如果需要检索，模型进一步评估段落是否支持生成。最后，**一个新的批评令牌（critique token）评估响应的整体效用。**

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1710249554111-f9cab000-d917-49e1-93ce-a2fa7410cd2c.png#clientId=ubbf2d517-b558-4&from=paste&height=531&id=u71cl&originHeight=956&originWidth=1800&originalType=binary&ratio=1.7999999523162842&rotation=0&showTitle=false&size=437463&status=done&style=none&taskId=uec37a7fc-6f02-4f4d-a704-a00f406bbfe&title=&width=1000.0000264909539)





具体来说，在给定输入提示和前几代的情况下，SELF-RAG 首先**会判断用检索到的段落来增强继续生成是否有帮助。**

1. 如果有帮助，**它就会输出一个检索标记，**按需调用检索模型（步骤 1）。
2. 随后，SELF-RAG 同时处理多个检索到的段落，**评估它们的相关性，然后生成相应的任务输出（步骤 2）。**
3. 然后，**它生成批判标记来批判自己的输出**，**并从事实性和整体质量方面选择最佳输出**（第 3 步）。

**这一过程不同于传统的 RAG（图 1 左），后者无论检索的必要性如何（例如，下图示例不需要事实性知识），都会持续检索固定数量的文档进行生成，而且从不对生成质量进行二次检查。**
此外，SELF-RAG 还会**为每个段落提供引文，并对输出结果是否得到段落支持进行自我评估**，从而更容易进行事实验证。
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1710247038820-946f97dd-6fbf-439a-ac55-1bdcd1fe3cd6.png#clientId=ubbf2d517-b558-4&from=paste&id=pOrdc&originHeight=1015&originWidth=1080&originalType=url&ratio=1.7999999523162842&rotation=0&showTitle=false&status=done&style=none&taskId=uae733014-4f7a-443f-9911-98c0618adf4&title=)
SELF-RAG 通过**将任意 LM 统一为扩展模型词汇表中的下一个标记预测，训练其生成带有反射标记的文本。**

### **反思令牌（reflection tokens）**
作者使用的反思令牌（reflection tokens）如下：
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/webp/65956778/1711014453400-81f637f3-fabe-4fa9-ba26-851b0f9d9a36.webp#clientId=u5bc522dc-dd8a-4&from=paste&id=ub30ea9f9&originHeight=402&originWidth=1440&originalType=url&ratio=1.7999999523162842&rotation=0&showTitle=false&status=done&style=none&taskId=u76bd0b41-54a2-42f5-ade1-ac78b41269a&title=)

### **训练**
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/webp/65956778/1711014453411-9589e84e-6d2a-4e8d-9d22-d9598387169f.webp#clientId=u5bc522dc-dd8a-4&from=paste&id=uf89b7e4f&originHeight=329&originWidth=1440&originalType=url&ratio=1.7999999523162842&rotation=0&showTitle=false&status=done&style=none&taskId=uc938c8a9-9d39-433a-9435-2d17cfce9d2&title=)
Self-RAG 的训练包括三个模型：**检索器（Retriever）、评论家（Critic）和生成器（Generator）。**
首先，训练评论家，使用检索器检索到的段落以及反思令牌增强指令-输出数据。
然后，使用标准的下一个 token 预测目标来训练生成器 LM，以学习生成 自然延续(continuations)以及特殊 tokens (用来检索或批评其自己的生成内容).
### **推理**
Self-RAG 通过学习生成反思令牌，使得在不需要训练LMs的情况下为各种下游任务或偏好量身定制模型行为。特别是：

1. 它可以适应性地使用检索令牌进行检索，因此模型可以自发判断是不是有必要进行检索。
2. 它引入了多种细粒度的批评令牌，这些令牌用于评估生成内容的各个方面的质量。在生成过程中，作者使用期望的批评令牌概率的线性插值进行segment级的beam search，以在每一个时间步骤中确定最佳的K个续写方案。
## **实验结果**
Self-RAG在六项任务中均超越了原始的 ChatGPT 或 LLama2-chat，并且在大多数任务中，其表现远超那些广泛应用的检索增强方法。
![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1711014453377-53ca0854-ba48-4611-a047-76df231b21b4.png#clientId=u5bc522dc-dd8a-4&from=paste&id=u310905ca&originHeight=465&originWidth=1440&originalType=url&ratio=1.7999999523162842&rotation=0&showTitle=false&status=done&style=none&taskId=u5b1d514c-cc33-40b0-a1e3-e24513de2e7&title=)
以上是一些消融实验，可以看到：每一个组件和技术在Self-RAG中都起到了至关重要的作用。调整这些组件可以显著影响模型的输出性质和质量，这证明了它们在模型中的重要性。

## **总结**
综上所述，Self-RAG 作为一种新型的检索增强生成框架，**通过自适应检索和引入反思令牌**，不仅增强了模型的生成效果，还提供了对模型行为的更高程度的控制。这项技术为提高开放领域问答和事实验证的准确性开辟了新的可能性，展示了模型自我评估和调整的潜力。
> ### **企业案例**
> #### **Baichuan**
> Baichuan是受Meta的**CoVe**启发而设计的，提出了一种将复杂提示拆解为多个独立并行可检索的查询的方法。Baichuan利用了自己的**专有TSF（Think-Step Further）**来推断和挖掘用户输入背后更深层次的问题，从而更精确、全面地理解用户意图。
> ![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1707147251961-fb3234f7-de56-4ac8-aa40-d72a544937f8.png#clientId=udc119121-7e43-4&from=paste&id=x9CSh&originHeight=459&originWidth=1080&originalType=url&ratio=1.100000023841858&rotation=0&showTitle=false&status=done&style=none&taskId=ua75dfd66-9f94-4aee-bd6c-bafb56e4255&title=)
> 在检索步骤中，百川智能开发了百川文本嵌入向量模型，该模型在高质量的中文数据上进行了预训练，包括超过1.5万亿个标记。他们通过专有的损失函数解决了对比学习中的批大小依赖性问题。这个向量模型已经超越了C-MTEB。
> 此外，他们引入了**稀疏检索和重新排名模型**（未透露），形成了一种混合检索方法，将向量检索与稀疏检索并行使用，显著提高了召回率达到95%。
> 此外，他们**引入了自我批评**，使大型模型**能够根据提示、相关性和效用**对检索到的内容进行内省，并进行二次审查以选择最匹配和高质量的候选内容。
> #### **Databricks**
> https://docs.databricks.com/en/generative-ai/retrieval-augmented-generation.html
> Databricks作为大数据领域的领先服务提供商，在RAG设计中保持了其独特的特点和优势。
> 当用户输入问题时，系统从预处理的文本向量索引中检索相关信息，并结合提示工程生成响应。上半部分，即非结构化数据管道，遵循主流的RAG方法，并没有显示出任何特殊性。


# 亮点构建
## 多模态编码
### 环境配置
visual bge：[https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/visual](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/visual)
可视化的bge提供了多种格式的多模态数据编码功能，无论是纯文本、纯图像还是两者的组合。
#### 需要进行源码安装
#### 报错NotImplementedError: No operator found for `memory_efficient_attention_forward` with inputs:
方案：卸载xformers
#### 包导入错误
代码要放置在子目录下，否则会报导入错误 FlagEmbedding/FlagEmbedding/visual/my_test_visual_bge.py
### 源码学习
```python
def encode(self, image=None, text=None):
        # used for simple inference
        if image is not None:
            image = self.preprocess_val(Image.open(image)).unsqueeze(0)
            if text is not None:
                text = self.tokenizer(text, return_tensors="pt", padding=True)
                return self.encode_mm(image, text) # 图文编码
            else:
                return self.encode_image(image)
        else:
            if text is not None:
                text = self.tokenizer(text, return_tensors="pt", padding=True)
                return self.encode_text(text) # 只进行文本编码
            else:
                return None
```
### 测试效果
embedding距离计算：
```python
sim_1 = query_emb @ candi_emb_1.T
sim_2 = query_emb @ candi_emb_2.T
```
| **query（img_query+text_query）** | **距离sim_1** | **距离sim_2** |
| --- | --- | --- |
| **img_query =**![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1711046846609-9df9d331-bba0-4223-bceb-d9c46394e61d.png#clientId=u5bc522dc-dd8a-4&from=paste&height=169&id=Zs8gP&originHeight=304&originWidth=498&originalType=binary&ratio=1.7999999523162842&rotation=0&showTitle=false&size=424676&status=done&style=none&taskId=u6f7c06b6-44ac-4472-9e3b-7c866fa55ae&title=&width=276.6666739958306) | **img1  =**![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1711046900649-ce20fdf1-6d0d-4d34-8a9a-4336ea707d45.png#clientId=u5bc522dc-dd8a-4&from=paste&height=152&id=uba591a48&originHeight=480&originWidth=640&originalType=binary&ratio=1.7999999523162842&rotation=0&showTitle=false&size=481282&status=done&style=none&taskId=ubf6c7aa6-54ea-4238-b31c-27dfebea455&title=&width=202.48959350585938) | **img2=**![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1711046918209-3aaa0a98-d0ca-4037-9150-a529259349ab.png#clientId=u5bc522dc-dd8a-4&from=paste&height=1298&id=u8462301a&originHeight=2336&originWidth=3504&originalType=binary&ratio=1.7999999523162842&rotation=0&showTitle=false&size=9219656&status=done&style=none&taskId=ub525375d-ede0-4607-a48b-d0f81891f71&title=&width=1946.6667182357237) |
| **text_query = **一匹马牵着这辆车 | 0.7026 | **0.8075** |
| **text_query = **这辆车上坐着人 | 0.6877 | **0.7368** |
| **text_query = **这辆车在晚上的时候 | **0.7501** | 0.6717 |

## 检索评判阶段
### 针对无需检索问题-直接进行回答
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1711017350117-0e055a0a-a927-4a84-8ff6-ebe759710308.png#clientId=u5bc522dc-dd8a-4&from=paste&height=200&id=u37fc4297&originHeight=360&originWidth=1358&originalType=binary&ratio=1.7999999523162842&rotation=0&showTitle=false&size=106348&status=done&style=none&taskId=u104a500a-c5f8-4f58-8c45-25703c342d5&title=&width=754.4444644303975)
### 针对需要检索问题-调用检索插件
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/65956778/1711018966402-f064cc97-ec69-45b1-b9d4-7d5e0f71c07c.png#clientId=u5bc522dc-dd8a-4&from=paste&height=233&id=ua969bdab&originHeight=420&originWidth=1922&originalType=binary&ratio=1.7999999523162842&rotation=0&showTitle=false&size=162352&status=done&style=none&taskId=udbb37298-2184-4ae0-8554-90fd9ed9983&title=&width=1067.7778060642297)

# 参考
Self-RAG论文链接：[https://arxiv.org/abs/2310.11511](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2310.11511)
Self-RAG项目主页：[https://selfrag.github.io/](https://link.zhihu.com/?target=https%3A//selfrag.github.io/)
推特爆火！超越ChatGPT和Llama2，新一代检索增强方法Self-RAG来了：[https://zhuanlan.zhihu.com/p/662654185](https://zhuanlan.zhihu.com/p/662654185)



1. 对齐优化沉淀和存放的位置
2. 搜索插件调用



中移动搜索大模型（GZW）项目，进行了知识库问答服务应用基础功能开发以及持续优化，目前将稳定算法镜像及接口文档进行输出：进行方案对齐并确定后续优化功能；**完成知识库问答服务应用基础功能开发，针对镜像服务进行了测试和并发调用出现问题的解决，目前完成算法镜像容器启动和镜像导出**，并完成接口文档的输出；持续对业务数据以及整体pipeline流程进行优化。

1. 项目方案对齐：对齐目前RAG应用的基础流程和亮点内容，针对rag应用的功能进行继续构建和测试，后续持续对如下功能进行优化：拒绝回答、自我认知、（主动检索）自反思批判、query改写（加入纠错）。
2. 知识库问答服务应用基础功能开发：处理答案&引用输出部分，加入构建的文档拼接召回部分代码及最新query设计加入现有系统，考虑到项目交付，进行基于plus接口的prompt优化测试确认效果等。
3. 接口文档对齐和输出：完成中移动GZW接口文档，并调整代码文档输出接口并优化输入输出接口参数。
4. 完成镜像的接口测试以及容器启动和镜像导出：目前发现容器启动后，并发频繁调用会存在问题，进行调试和修改后满足调用需求。

1. 1. 在服务器以容器形式进行启动，进行接口提供
   2. 调试接口启动后频繁调用存在问题，针对docker启动进行参数调整

1. 针对业务数据进行优化：进行知识库问答整体pipeline优化处理，并针对性进行组装组件的优化等。



中移动专利POC测试优化：利用客户提供的文档进行问答构建和测试标注，**并从RAG系统参数、更优的pdf解析、document组装链路及指令调试等方面，着重对pdf文档效果进行优化，在存在跨页总结的hardcase情况下实现pdf问答完全正确率提升30%。**

1. 移动专利问答构建和测试标注：构建测试数据并对进行问题打标：针对不同类型的问题进行准备和批量测试，包括pdf的问答、表格的问答等。针对效果进行初步问题处理，整理和分析测试结果ppt
2. 中移动pdf转docx效果测试：考虑到便于检索定位和阅读效果，进行pdf转docx处理和效果测试，直接转换有一些排版变化，经批量调整后转换出的word中ppt效果和原始pdf显示已经基本一致，且文本块和内容的结构也比较完整。但目前系统查看页面目前对转换出来的docx支持不是很友好，讨论后续进行对比验证和优化。
3. 移动专利pdf问答效果优化：

1. 1. 进行效果的初步优化方式：

1. 1. 1. 水印去除：pdf中存在的水印，进行预处理和水印批量去除
      2. 尝试更优的pdf解析：得到更好的结构化解析，减少标题和内容丢失。
      3. 进行图片文本及特殊符号的定制化处理等。

1. 1. 对模型参数等调整优化：

1. 1. 1. 调整参数：无召回情况较多，进行参数修改，调整chunck、相似度、topk等相关超参数
      2. 目前看document组装方式以及指令可以进一步优化。

1. 1. 修改文档组装链路及进行指令调试，以尝试进一步优化模型回复。
   2. 进行测试数据集评测，并反馈至应用调整。



知识库问答应用优化：将近期优化后的document组装链路及指令加入目前pipelines，进行大模型问答调用接口的拆分和增加使其更加模块化和可扩展；  //  在项目中完成容器的并发调用调试，能够启动稳定的算法服务；完成联调和兼容最新的检索接口，并开始与前端应用联调和开发迭代。

1. 知识库问答服务容器代码调试：完成联调和兼容最新的检索接口，进行大模型问答调用接口的拆分和增加。

1. 1. 联调和兼容最新的检索接口：

1. 1. 1. 联调最新的检索接口：与最新的检索接口联调和兼容
      2. 加入中间结果输出：检索插件的errorlog response输出，利于联调并增强鲁棒性。

1. 1. 进行大模型问答调用接口的拆分和增加：增加query改写、自适应检索、通用问答和检索测试等接口的预留
   2. 基础功能测试完成：进行处理答案&引用输出部分，调整代码文档输出接口并优化输入输出接口参数。

1. 与前端应用联调开发迭代：提供schema文件等开始与前端应用联调开发，进行应用侧知识问答的迭代



税务算力平台性能测试方面，支持内网完成训练启动，针对性能测试脚本完成初步准备，待后续进行现场测试：针对税务内网大模型训练，进行税务内网大模型训练启动支持，在内网中完成训练启动；针对DCU环境的性能指标批量评测方案，完成性能指标的调研，以及计算公式和计算代码的构建和整理，完成初版的评估模版文档的构建，并设计构建初步自动评测方案；针对性能指标批量评测代码，完成批量性能评测的批量脚本的构建。
