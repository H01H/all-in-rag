# app.py
# -*- coding: utf-8 -*-

"""
Streamlit 可视化问答页面

运行方式：
    cd /workspaces/all-in-rag/code/C8
    streamlit run app.py
"""

import streamlit as st
from main import RecipeRAGSystem


# ===============================
# 1. 页面基础配置
# ===============================

st.set_page_config(
    page_title="中文 Markdown 垂域 RAG 智能问答系统",
    page_icon="📚",
    layout="wide"
)


# ===============================
# 2. 初始化 RAG 系统
# ===============================

@st.cache_resource
def load_rag_system():
    """
    初始化并缓存 RAG 系统。

    使用 st.cache_resource 的好处：
    1. 第一次打开页面时加载模型和索引
    2. 后续提问不需要重复初始化
    3. 页面刷新时也尽量复用资源
    """

    system = RecipeRAGSystem()
    system.initialize_system()
    system.build_knowledge_base()

    return system


def convert_docs_to_display_items(docs):
    """
    把 LangChain Document 转成方便页面展示的字典。
    """

    display_items = []

    for i, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        page_content = getattr(doc, "page_content", "")

        item = {
            "rank": i,
            "dish_name": metadata.get("dish_name", "未知菜品"),
            "category": metadata.get("category", ""),
            "difficulty": metadata.get("difficulty", ""),
            "source": metadata.get("source", ""),
            "doc_type": metadata.get("doc_type", ""),
            "parent_id": metadata.get("parent_id", ""),
            "rrf_score": metadata.get("rrf_score", ""),
            "content": page_content
        }

        display_items.append(item)

    return display_items


def retrieve_for_display(system, question, top_k, enable_filter=True):
    """
    用于页面展示的检索函数。

    它会返回：
    1. 查询类型 route_type
    2. 查询重写 rewritten_query
    3. 过滤条件 filters
    4. 检索到的子块 relevant_chunks
    5. 回传的父文档 parent_docs
    """

    # 1. 查询路由
    route_type = system.generation_module.query_router(question)

    # 2. 查询重写
    if route_type == "list":
        rewritten_query = question
    else:
        rewritten_query = system.generation_module.query_rewrite(question)

    # 3. 提取过滤条件
    filters = system._extract_filters_from_query(question)

    # 4. 检索
    if enable_filter and filters:
        relevant_chunks = system.retrieval_module.metadata_filtered_search(
            query=rewritten_query,
            filters=filters,
            top_k=top_k
        )
    else:
        relevant_chunks = system.retrieval_module.hybrid_search(
            query=rewritten_query,
            top_k=top_k
        )

    # 5. 根据子块找回父文档
    parent_docs = system.data_module.get_parent_documents(relevant_chunks)

    return route_type, rewritten_query, filters, relevant_chunks, parent_docs


def generate_answer(system, question, route_type, parent_docs, stream=False):
    """
    根据查询类型生成回答。
    """

    if not parent_docs:
        return "抱歉，没有找到相关的知识库内容。"

    if route_type == "list":
        return system.generation_module.generate_list_answer(question, parent_docs)

    if route_type == "detail":
        if stream:
            return system.generation_module.generate_step_by_step_answer_stream(question, parent_docs)
        return system.generation_module.generate_step_by_step_answer(question, parent_docs)

    if stream:
        return system.generation_module.generate_basic_answer_stream(question, parent_docs)

    return system.generation_module.generate_basic_answer(question, parent_docs)


# ===============================
# 3. 页面标题
# ===============================

st.title("📚 中文 Markdown 垂域 RAG 智能问答系统")

st.markdown(
    """
    本系统基于中文 Markdown 文档构建垂域知识库，支持父子文本块切分、BGE Embedding、FAISS 向量检索、BM25 关键词检索、RRF 融合重排、元数据过滤和 LLM 答案生成。
    """
)


# ===============================
# 4. 侧边栏配置
# ===============================

with st.sidebar:
    st.header("⚙️ 检索配置")

    top_k = st.slider(
        "Top K 检索数量",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    enable_filter = st.checkbox(
        "启用元数据过滤",
        value=True
    )

    show_retrieval_process = st.checkbox(
        "显示检索过程",
        value=True
    )

    use_stream = st.checkbox(
        "启用流式输出",
        value=False
    )

    st.markdown("---")
    st.caption("建议演示时打开“显示检索过程”，这样更能体现项目的可解释性。")


# ===============================
# 5. 加载系统
# ===============================

with st.spinner("正在加载 RAG 系统，请稍等..."):
    rag_system = load_rag_system()

st.success("RAG 系统加载完成！")


# ===============================
# 6. 用户输入
# ===============================

question = st.text_input(
    "请输入你的问题：",
    placeholder="例如：有什么简单的甜品？宫保鸡丁怎么做？西红柿炒鸡蛋需要什么材料？"
)

ask_button = st.button("开始问答", type="primary")


# ===============================
# 7. 问答主流程
# ===============================

if ask_button:
    if not question.strip():
        st.warning("请先输入一个问题。")
    else:
        with st.spinner("正在检索相关文档..."):
            route_type, rewritten_query, filters, relevant_chunks, parent_docs = retrieve_for_display(
                system=rag_system,
                question=question,
                top_k=top_k,
                enable_filter=enable_filter
            )

        # 展示检索过程
        if show_retrieval_process:
            st.subheader("🔍 检索过程")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("查询类型", route_type)

            with col2:
                st.metric("检索子块数", len(relevant_chunks))

            with col3:
                st.metric("回传父文档数", len(parent_docs))

            st.markdown("**原始问题：**")
            st.code(question, language="text")

            st.markdown("**查询重写结果：**")
            st.code(rewritten_query, language="text")

            st.markdown("**元数据过滤条件：**")
            if filters:
                st.json(filters)
            else:
                st.info("未提取到过滤条件")

            st.markdown("**检索到的子文本块：**")

            display_items = convert_docs_to_display_items(relevant_chunks)

            if display_items:
                for item in display_items:
                    with st.expander(
                        f"Top {item['rank']} | {item['dish_name']} | {item['category']} | {item['difficulty']}"
                    ):
                        st.write(f"**菜品名称：** {item['dish_name']}")
                        st.write(f"**分类：** {item['category']}")
                        st.write(f"**难度：** {item['difficulty']}")
                        st.write(f"**来源：** {item['source']}")
                        st.write(f"**父文档 ID：** {item['parent_id']}")
                        st.write(f"**RRF 分数：** {item['rrf_score']}")
                        st.markdown("**文本片段：**")
                        st.write(item["content"][:800])
            else:
                st.warning("没有检索到相关子文本块。")

            st.markdown("**回传的父文档：**")
            parent_display_items = convert_docs_to_display_items(parent_docs)

            if parent_display_items:
                for item in parent_display_items:
                    st.write(
                        f"- {item['dish_name']} | {item['category']} | {item['difficulty']} | {item['source']}"
                    )
            else:
                st.warning("没有找到对应父文档。")

        # 生成回答
        st.subheader("💬 最终回答")

        if not relevant_chunks:
            st.warning("没有找到相关文档，无法生成可靠回答。")
        else:
            if use_stream:
                answer_area = st.empty()

                answer_text = ""
                with st.spinner("正在生成回答..."):
                    stream_result = generate_answer(
                        system=rag_system,
                        question=question,
                        route_type=route_type,
                        parent_docs=parent_docs,
                        stream=True
                    )

                    for chunk in stream_result:
                        answer_text += chunk
                        answer_area.markdown(answer_text)
            else:
                with st.spinner("正在生成回答..."):
                    answer = generate_answer(
                        system=rag_system,
                        question=question,
                        route_type=route_type,
                        parent_docs=parent_docs,
                        stream=False
                    )

                st.markdown(answer)