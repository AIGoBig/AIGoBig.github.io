

**GPT Researcher架构：**

- **Task**：整个研究任务是由一个特定的研究**查询**或任务驱动的。这一步骤确定了需要解决的问题，并为后续的任务制定了明确的目标。
- **Planner**：“计划者”代理的主要任务是**生成研究问题**。根据研究查询，计划者会制定一系列具体的研究问题，这些问题共同构成对任务的全面理解。计划者确保研究问题覆盖了任务的各个方面，从而为后续的信息搜集和分析打下基础。
- **Researcher**：“执行者”代理负责根据计划者生成的每个研究问题**寻找最相关的信息**。这个步骤中，执行者代理会触发爬虫代理，在网络上抓取与每个研究问题相关的资源。执行者代理利用注入gpt3.5-turbo和gpt-4-turbo的大模型来处理和分析这些信息
- **Query**：在执行代理搜集信息的过程中，系统会不断发出查询请求。这些查询是基于研究问题设计的，目的是在网络上找到**最相关和最新的信息资源**。每个查询都是一个精确的问题，确保获取的信息是高质量且相关的。
- **Publisher**：此时，“计划者”会过滤并**汇总**所有相关信息，创建最终的研究报告。这个步骤包括对所有抓取到的资源进行总结，并跟踪其来源，确保信息的**可靠性和可追溯性**。最终的研究报告由”发布者“进行发布，它整合了所有的总结信息，提供一个全面、详尽且公正的研究结果。

![GPT Researcher：破解复杂研究的AI利器-AI.x社区](https://cdn.jsdelivr.net/gh/AIGoBig/PicRepo@master/2024/08/b567c1b87d882b108514324260aee6cb7f302f_20240816194806J6Vh7D.png)

**GPT Researcher 的工作流程：**

- 生成研究问题**提纲**：形成对任何给定任务的客观意见。
- 触发**爬虫**代理：对于每个研究问题，从网上资源中抓取相关信息。
- **总结**和过滤：对抓取的资源进行总结和过滤，仅保留相关信息。
- 生成研究**报告**：聚合所有总结的资源，使用GPT-4生成最终的研究报告。

**GPT Researcher 的内部结构**：

- **Backend** 文件夹中的代码文件用来处理GPT Researcher的**后台网络请求**，这里利用FastAPI 框架创建应用程序，它具有处理静态文件、模板渲染和 WebSocket 连接的功能。
- Docs文件夹用来存放使用手册以及GPT Researcher最新的Blog 文章。
- Examples文件夹提供了简单的GPT Researcher例子。
- Frontend 文件夹包含 GPT Researcher前端界面的信息， 包括html、css、js 等文件。
- **Gpt_researcher** 文件夹存放**核心代码**，包括参数配置、上下文管理、大模型管理、代理、函数、提示词管理、记忆管理、网络爬虫工具以及搜索引擎工具等。
- **Mulit_agents** 用来支持**多代理模式**。
- Outputs 用来**保存**输出的研究文档，目前GPT Researcher支持研究结果以word、MD等多种文档的方式下载。
- Main.py 是 GPT Researcher Web 应用的**入口文件**，我们通过它**启动**整个应用。
- Requirements.txt 用来存放依赖的组件库，然后需要结合PIP 命令对其中包含的组件进行安装，从而保证GPT Researcher的运行。



由于GPT Researcher 需要借助大模型以及网络搜索技术完成研究工作，所以需要获取两者的访问权限。因此我们需要对大模型和搜索引擎的API密钥进行设置，可以使用两种方法设置API密钥：直接导出或将其存储在 .env 文件中。这里的API 密钥是用来访问大模型(OpenAI)和搜索引擎(Tavily Search API)的，需要通过API密钥的方式获取二者的访问权限。

可以通过如下命令完成：

```plain
export OPENAI_API_KEY={Your OpenAI API Key here}
export TAVILY_API_KEY={Your Tavily API Key here}
```

同时，还可以通过配置文件完成密钥的配置，在gpt-researcher目录中找到.env 文件，打开该文件进行API 密钥的设置。

并输入以下密钥：

```plain
OPENAI_API_KEY={Your OpenAI API Key here}
TAVILY_API_KEY={Your Tavily API Key here}
```

### 执行 GPT Researcher：研究报告与成果解析

GPT Researcher 的Web 应用启动，可以通过http://127.0.0.1:8000的地址访问GPT Researcher。

启动之后会看到如下图的界面，即GPT Researcher 的宣传语。

![GPT Researcher：破解复杂研究的AI利器-AI.x社区](https://cdn.jsdelivr.net/gh/AIGoBig/PicRepo@master/2024/08/36ad1f9287dc1680385873d52bcb44912f1b82_20240816195100PXq9o9.png)

接着会看到GPT Researcher 的主界面，如下图所示：

在 “What would you like me to research next? ”下方的文本框，可以输入你**要研究的内容**，也就是提示词信息。

在”What type of report would you like me to generate?”下方会提供一个下拉框，可以选择**研究报告的类型**，目前提供三类：

- Summary Short and fast (~2 min)：摘要类型，特点是内容较少但是生成速度较快。
- Detailed In depth and longer (~5 min)：详细类型，更加有深度，内容比较多，生成速度会慢一点。
- Resource Report：原始报告，将提供所有网络搜索的内容以供参考。

在”Agent Output”的部分会看到研究的计划和任务的执行情况。“Research Report” 中会生成最终的生成报告。

![GPT Researcher：破解复杂研究的AI利器-AI.x社区](https://cdn.jsdelivr.net/gh/AIGoBig/PicRepo@master/2024/08/f6d8ad9202e31e428de86003e53f8f54a06cf9_20240816195102dDKvce.png)

实测一下它的研究能力，如下图所示，输入“ai agent 在企业中的应用”作为我们的研究主题，并且选择Summary类型作为报告生成方式。

![GPT Researcher：破解复杂研究的AI利器-AI.x社区](https://cdn.jsdelivr.net/gh/AIGoBig/PicRepo@master/2024/08/1753cd671766246173958044105b8e88de6fce_20240816195156CQbtyP.png)

点击“Research”按钮之后开始执行研究任务，同时Agent Output输出如下图所示内容。

Thinking about research questions for the task...

GPT Researcher在生成研究问题，以便为特定任务形成客观和全面的研究框架。

Starting the research task for 'ai agent 在企业中的应用'...

GPT Researcher开始执行具体的研究任务，基于用户提供的主题 "ai agent 在企业中的应用"。

Business Analyst Agent

GPT Researcher采用了一个特定的“商业分析代理”来处理研究任务。

I will conduct my research based on the following queries: ['ai agent 在企业中的应用 2024', 'ai agent 在企业中的应用 中国 2024', 'ai agent 在企业中的应用 案例分析 2024', 'ai agent 在企业中的应用']...

GPT Researcher列出了将要使用的具体查询。这些查询反映了研究任务的多维度，包括时间、地点和具体应用案例。

Running research for 'ai agent 在企业中的应用 2024'...

显示GPT Researcher实际开始对指定查询执行研究任务。

![GPT Researcher：破解复杂研究的AI利器-AI.x社区](https://cdn.jsdelivr.net/gh/AIGoBig/PicRepo@master/2024/08/e4535aa710b7988bdf6039c895ba0af0b1dc6e_20240816195311oFbs8i.png)

执行过程中如果切换到控制台，从命令行的输出来看，发现如下内容。由于内容较多，我们截取其中一部分给大家解释。

> ��Starting the research task for 'ai agent 在企业中的应用'...
>
> [https://api.chatanywhere.tech/v1](https://api.chatanywhere.tech/v1)
>
> �� Business Analyst Agent
>
> [https://api.chatanywhere.tech/v1](https://api.chatanywhere.tech/v1)
>
> ��I will conduct my research based on the following queries: ['ai agent 在企业中的应用 2024', 'ai agent 在企业中的应用 中国 2024', 'ai agent 在企业中的应用 案例分析 2024', 'ai agent 在企业中的应用']...
>
> ��Running research for 'ai agent 在企业中的应用 2024'...
>
> ✅ Added source url to research: https://zhuanlan.zhihu.com/p/675595267
>
> ✅ Added source url to research: https://www.technologyreview.com/2024/05/14/1092407/googles-astra-is-its-first-ai-for-everything-agent/
>
> ✅ Added source url to research: https://zhuanlan.zhihu.com/p/676245844
>
> ✅ Added source url to research: https://www.infoq.cn/article/bqmoGzkvE4GwWsvruqHp
>
> ✅ Added source url to research: https://www.thepaper.cn/newsDetail_forward_27225624
>
> �� Researching for relevant information...
>
> ��Getting relevant content based on query: ai agent 在企业中的应用 2024...
>
> �� Source: https://www.thepaper.cn/newsDetail_forward_27225624
>
> Title:
>
> Content: 登录
>
> 《2024年AI Agent行业报告》——大模型时代的“APP”，探索新一代人机交互及协作范式
>
> 原创 刘瑶 甲子光年
>
> 60页报告，和100+场景梳理，可能依然赶不上飞速发展的AI Agent！
>
> 随着大型模型在各行各业的广泛应用，基于大型模型的人工智能体（AI Agent）迎来了快速发展的阶段。研究AI Agent是人类不断接近人工通用智能（AGI）的探索之一。知名AI Agent项目AutoGPT已经在GitHub的星星数已经达到 140,000 颗，进一步反映了用户对于AI Agents 项目的广泛兴趣和支持。
>
> 随着AI Agent变得越来越易用和高效，"Agent+"的产品越来越多，未来AI Agent有望成为AI应用层的基本架构，涵盖toC和toB产品等不同领域。
>
> 因此甲子光年推出《2024年AI Agent行业报告》，探讨AI Agent在概念变化，学术及商业界的尝试与探索，对各行业、各场景对于AIGC技术的需求进行调研及梳理，展示AI Agent领域近期的突破及商业实践范式，对未来行业的趋势进行研判。
>
> 原标题：《《2024年AI Agent行业报告》——大模型时代的“APP”，探索新一代人机交互及协作范式｜甲子光年智库》
>
> <省略部分内容……>

这里的输出结果更加详细，方便我们了解GPT Researcher的工作流程，大致包括如下5个步骤：

启动研究任务：

开始研究主题为“ai agent 在企业中的应用”的任务。

选择和执行商业分析代理：

选择了“Business Analyst Agent”作为执行任务的代理。

生成查询：

生成了一系列研究查询，此时会调用搜索引擎，查询内容包括：

- 'ai agent 在企业中的应用 2024'
- 'ai agent 在企业中的应用 中国 2024'
- 'ai agent 在企业中的应用 案例分析 2024'
- 'ai agent 在企业中的应用'

执行具体查询：

对每个查询分别进行研究，并添加相关的资源URL，例如：

https://zhuanlan.zhihu.com/p/675595267 等。

获取相关内容：

从添加的资源URL中获取相关内容，并展示部分内容摘要。例如：

《2024年AI Agent行业报告》：探讨AI Agent在各行各业的应用、概念变化、学术及商业界的尝试与探索。

上面这些研究、搜索、汇总工作都是由GPT Researcher 自动完成不需要人工干预，大约2分钟之后整个过程执行完毕。此时，我们可以通过**选择下载报告文件类型**的方式选择要下载的文件格式。这里我选择了**Word 和MD**的文件格式进行下载，如下图所示，与此同时在源代码目录下outputs 目录中也会生成对应的文件。

![GPT Researcher：破解复杂研究的AI利器-AI.x社区](https://cdn.jsdelivr.net/gh/AIGoBig/PicRepo@master/2024/08/b60d03142f50b5078642962fb9d9ce093d001d_20240816195417sGx2st.png)

我们打开Word 文件，查看其内容如下图所示。左边显示了研究报告的目录结构，包括：基本概念，应用场景，技术架构，优势和挑战等，看上去还比较全面。

![GPT Researcher：破解复杂研究的AI利器-AI.x社区](https://cdn.jsdelivr.net/gh/AIGoBig/PicRepo@master/2024/08/19d5e8776119adc6b048412c0ee512e70d3748_20240816195418gE2Mjc.png)

在研究报告的最后，还提供了参考文献的链接，如下图所示，这些文献都是来自于互联网搜索的结果。

![GPT Researcher：破解复杂研究的AI利器-AI.x社区](https://cdn.jsdelivr.net/gh/AIGoBig/PicRepo@master/2024/08/d9b79ea02cf641391631858a467f930337ae4a_20240816195417kc7s0X.png)

### 集成GPT Researcher：从应用到扩展

通过执行GPT Researcher让其为我们研究“ai agent 在企业中的应用”，这个操作过程是通过GPT Researcher提供的Web 应用界面完成的。如果我们需要将GPT Researcher的能力集成到自己的项目中，我们就需要将 GPTResearcher 引入到现有的 Python 项目中。

大致步骤如下，首先需要**导入相关模块并定义全局常量**，如研究查询和报告类型。然后，通过**定义异步函数**来初始化GPT Researcher 实例并执行研究任务。最后，运行主函数以生成和打印研究报告。

我们将上述扩展过程整理成如下代码：

```plain
from gpt_researcher import GPTResearcher
import asyncio
QUERY = "<填写需要研究的内容>"
REPORT_TYPE = "research_report"
async def fetch_report(query, report_type):
    researcher = GPTResearcher(query=query, report_type=report_type, config_path=None)
    await researcher.conduct_research()
    report = await researcher.write_report()
    return report
async def generate_research_report():
    report = await fetch_report(QUERY, REPORT_TYPE)
    print(report)
if __name__ == "__main__":
    asyncio.run(generate_research_report())
```

代码内容主要使用GPT Researcher调用研究任务的计划、执行、搜索、总结的能力，详细代码解释如下：

(1) 导入模块

```plain
from gpt_researcher import GPTResearcher
import asyncio
```

导入了自定义模块 gpt_researcher 中的 GPTResearcher 类，用于后续研究报告的生成。同时导入了 asyncio 模块，用于支持异步编程。

(2) 定义全局常量

```plain
QUERY = "<填写需要研究的内容>"
REPORT_TYPE = "research_report"
```

定义了两个全局常量：QUERY 保存查询问题，REPORT_TYPE 保存报告类型。

(3) 异步函数 fetch_report

```plain
async def fetch_report(query, report_type):
    researcher = GPTResearcher(query=query, report_type=report_type, config_path=None)
    await researcher.conduct_research()
    report = await researcher.write_report()
    return report
```

定义了一个名为 fetch_report 的异步函数，用于根据给定的查询和报告类型获取研究报告。

- 创建 GPTResearcher 实例，传入查询和报告类型。
- 调用 conduct_research 方法执行研究。
- 调用 write_report 方法生成报告并返回。

(4) 异步函数 generate_research_report

```plain
async def generate_research_report():
    report = await fetch_report(QUERY, REPORT_TYPE)
    print(report)1.2.3.
```

定义了一个名为 generate_research_report 的异步函数，用于执行生成研究报告的主要逻辑。调用 fetch_report 函数获取研究报告。打印生成的报告。

(5) 主程序入口

```plain
if __name__ == "__main__":
    asyncio.run(generate_research_report())
```

在脚本作为主程序运行时，调用 asyncio.run 方法执行 generate_research_report 函数，启动异步任务并生成报告。

整体而言这段代码，使用异步编程技术，通过 GPTResearcher 类生成关于指定查询的研究报告，并在主程序中执行生成并打印结果。

### 定制GPT Researcher：从配置到应用

我们不仅可以将GPT Researcher集成应用中，还可以对大模型，搜索引擎等信息进行定制。在源代码目录下面的gpt_researcher/config目录下面，存在配置文件 config.py，文件中描述了所有可以配置的环境变量。基于这些环境变量，你可以根据具体需求灵活定制 GPT Researcher。包括**选择不同的搜索引擎、嵌入提供商和大语言模型提供商**，以确保获得最佳的研究结果。而这些配置参数设置在.env 文件中保存。说白了，就是config.py 文件中定义配置的信息，具体的配置在.env 中描述。

config.py 文件中的配置类定义如下：

```plain
def __init__(self, config_file: str = None):
        """Initialize the config class."""
        self.config_file = os.path.expanduser(config_file) if config_file else os.getenv('CONFIG_FILE')
        self.retriever = os.getenv('RETRIEVER', "tavily")
        self.embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'openai')
        self.llm_provider = os.getenv('LLM_PROVIDER', "openai")
        self.fast_llm_model = os.getenv('', "gpt-3.5-turbo-16k")
        self.smart_llm_model = os.getenv('SMART_LLM_MODEL', "gpt-4o")
        self.fast_token_limit = int(os.getenv('FAST_TOKEN_LIMIT', 2000))
        self.smart_token_limit = int(os.getenv('SMART_TOKEN_LIMIT', 4000))
        self.browse_chunk_max_length = int(os.getenv('BROWSE_CHUNK_MAX_LENGTH', 8192))
        self.summary_token_limit = int(os.getenv('SUMMARY_TOKEN_LIMIT', 700))
        self.temperature = float(os.getenv('TEMPERATURE', 0.55))
        self.user_agent = os.getenv('USER_AGENT', "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                                   "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0")
        self.max_search_results_per_query = int(os.getenv('MAX_SEARCH_RESULTS_PER_QUERY', 5))
        self.memory_backend = os.getenv('MEMORY_BACKEND', "local")
        self.total_words = int(os.getenv('TOTAL_WORDS', 800))
        self.report_format = os.getenv('REPORT_FORMAT', "APA")
        self.max_iterations = int(os.getenv('MAX_ITERATIONS', 3))
        self.agent_role = os.getenv('AGENT_ROLE', None)
        self.scraper = os.getenv("SCRAPER", "bs")
        self.max_subtopics = os.getenv("MAX_SUBTOPICS", 3)

        self.load_config_file()

    def load_config_file(self) -> None:
        """Load the config file."""
        if self.config_file is None:
            return None
        with open(self.config_file, "r") as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(self, key.lower(), value)1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20.21.22.23.24.25.26.27.28.29.30.31.32.33.34.
```

由于配置参数内容比较多，这里我们挑选几个经常用的配置给大家介绍。这些配置参数可以在 .env 文件中设置，并在 Config 类初始化时加载。以下是主要配置选项的介绍：

**RETRIEVER：**用于检索资源的**网络搜索引擎**。默认值为 tavily，可选项包括 duckduckgo、bing、google、serper、searx。通过设置 RETRIEVER 环境变量，可以选择不同的搜索引擎。例如，在.env 中使用 Bing 搜索引擎：

```plain
RETRIEVER=bing
```

**EMBEDDING_PROVIDER：**嵌入模型的**提供商**。默认值为 openai，可选项包括 ollama、huggingface、azureopenai、custom。可以通过设置 EMBEDDING_PROVIDER 环境变量来选择不同的提供商：

```plain
EMBEDDING_PROVIDER=huggingface
```

**LLM_PROVIDER：**大语言模型(LLM)的提供商。默认值为 openai，可选项包括 google、ollama、groq 等。可以通过设置 LLM_PROVIDER 环境变量选择不同的 LLM 提供商：

```plain
LLM_PROVIDER= openai
```

**FAST_LLM_MODEL：**用于快速 LLM 操作(如摘要)的模型名称。默认值为 gpt-3.5-turbo-16k。可以通过设置 FAST_LLM_MODEL 环境变量调整此模型：

```plain
FAST_LLM_MODEL=gpt-3.5-turbo-16k
```

**SMART_LLM_MODEL：**用于复杂操作(**如生成研究报告和推理**)的模型名称。默认值为 gpt-4o。可以通过设置 SMART_LLM_MODEL 环境变量来选择合适的模型：

```plain
SMART_LLM_MODEL=gpt-4o
```

### 总结

GPT Researcher 在自动化研究领域中展示了显著的进步，解决了传统大型语言模型的局限性。通过采用 **Plan-and-Solve** 方法，GPT Researcher 能够**高效处理以前需要大量人力的复杂任务**。它能够**拆解任务、进行并行处理**，并从多个来源生成详细的报告，确保高准确性和客观性。此外，其灵活的配置使用户可以根据具体的研究需求对其进行定制，增强了其在各种领域的实用性。随着AI技术的不断发展，像GPT Researcher这样的工具将在简化研究过程方面发挥关键作用，使高质量的信息变得更加易于获取，并减少完成综合研究所需的时间和精力。

### 参考和来源

[https://papers.nips.cc/paper_files/paper/2022/file/8bb0d291acd4acf06ef112099c16f326-Paper-Conference.pdf](https://papers.nips.cc/paper_files/paper/2022/file/8bb0d291acd4acf06ef112099c16f326-Paper-Conference.pdf)

[https://ar5iv.labs.arxiv.org/html/2305.04091](https://ar5iv.labs.arxiv.org/html/2305.04091)

https://www.51cto.com/aigc/956.html

