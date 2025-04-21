





``` mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#2c3e50', 'edgeLabelBackground':'#fff' }} }%%

graph LR
    A[企业] -->|拥有/包含| B((技术部))
    A -->|拥有/包含| C((市场部))
    A -->|拥有/包含| D((人力资源部))

    subgraph 员工能力层
        E[张三 大五人格: 外向性70%<br>宜人性65%<br>尽责性80%] -->|擅长| F[Java开发]
        E -->|掌握| G[Spring框架]
        E -->|熟悉| H[MySQL]

        I[李四 大五人格: 开放性85%<br>尽责性75%<br>神经质50%] -->|精通| J[Python]
        I -->|擅长| K[数据分析]
        I -->|了解| L[机器学习]

        M[王五 大五人格: 宜人性90%<br>尽责性85%<br>外向性60%] -->|精通| N[前端开发]
        M -->|擅长| O[UI设计]
        M -->|熟悉| P[Vue.js]
    end

    subgraph 岗位技能层
        B -->|要求| Q[Java高级开发]
        B -->|要求| R[分布式系统]
        B -->|要求| S[微服务架构]

        C -->|要求| T[数字营销]
        C -->|要求| U[SEO优化]
        C -->|要求| V[社交媒体运营]

        D -->|要求| W[HR三支柱模型]
        D -->|要求| X[劳动法实务]
        D -->|要求| Y[员工测评]
    end

    subgraph 课程推荐层
        F -->|学习路径| Z[Java核心技术]
        F -->|进阶| AA[高并发编程实战]

        J -->|学习路径| AB[Python数据分析]
        J -->|拓展| AC[TensorFlow实战]

        N -->|学习路径| AD[全栈开发入门]
        N -->|提升| AE[响应式设计进阶]

        Q -->|补强| AF[设计模式精讲]
        S -->|补强| AG[Spring Cloud实战]
    end

    classDef employee fill:#4a90e2,stroke:#2c3e50,stroke-width:2px;
    classDef skill fill:#2ecc71,stroke:#27ae60,stroke-width:2px;
    classDef position fill:#f39c12,stroke:#d35400,stroke-width:2px;
    classDef department fill:#7f8c8d,stroke:#bdc3c7,stroke-width:2px;
    classDef course fill:#fff3cd,stroke:#f1c40f,stroke-width:2px,dashed;
    classDef assessment fill:#ba4a9a,stroke:#c0392b,stroke-width:2px,shape:hexagon;

    class A department;
    class B,C,D department;
    class E,I,M employee;
    class F,G,H,J,K,L,N,O,P skill;
    class Q,R,S,T,U,V,W,X,Y position;
    class Z,AA,AB,AC,AD,AE,AF,AG course;
```

``` mermaid
flowchart TD
    A[当前岗位: 高级开发工程师] -->|需要提升| B{技术经理}
    A --> C[优势项]
    A --> D[待提升项]
    
    C --> C1["技术能力 95/100"]
    C --> C2["系统设计 88/100"]
    
    D --> D1["沟通能力 60/100"]
    D --> D2["项目管理 72/10"]
    
    B --> B1["目标能力要求"]
    B1 --> B1a["技术深度 90+"]
    B1 --> B1b["团队管理 85+"]
    B1 --> B1c["跨部门协作 80+"]
    
    style C fill:#E8F5E9,stroke:#4CAF50
    style D fill:#FFF3E0,stroke:#FF9800
    style B fill:#FFF8E1,stroke:#FFC107
```









```mermaid
heatMap
    title 企业人力资源能力热力图 - 能力短板识别
    row ["部门/职能", "战略规划", "技术研发", "运营执行", "市场营销", "人力资源", "财务管理"]
    col ["能力维度", "专业深度", "团队协作", "创新力", "决策效率", "数字化水平", "人才储备"]
    
    
cell [
    [null, "一般", "优秀", "一般", "薄弱", "一般", "薄弱"],
    ["一般", "优秀", "良好", "薄弱", "一般", "薄弱", "优秀"],
    ["薄弱", "良好", "优秀", "一般", "薄弱", "优秀", "一般"],
    ["优秀", "薄弱", "一般", "优秀", "良好", "薄弱", "良好"],
    ["良好", "优秀", "薄弱", "优秀", "一般", "一般", "优秀"],
    ["薄弱", "一般", "优秀", "薄弱", "优秀", "良好", "一般"]
]

color
    薄弱: #FF6B6B
    一般: #FFD93D
    良好: #B6D7A8
    优秀: #4ECDC4
```