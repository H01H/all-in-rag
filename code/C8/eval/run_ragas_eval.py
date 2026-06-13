# eval/run_ragas_eval.py
# -*- coding: utf-8 -*-

"""
真正的 RAGAS 全量评测脚本

功能：
1. 读取 data/eval_dataset_full.csv
2. 对全部样本运行 RAG：检索 + 生成
3. 构造 RAGAS 数据集
4. 使用 DeepSeek 作为评测 LLM
5. 使用 BGE 中文向量模型作为评测 embedding
6. 输出：
   - ragas_input_dataset.csv
   - ragas_eval_result.csv
   - ragas_summary.csv

运行方式：
    cd /workspaces/all-in-rag/code/C8
    python eval/run_ragas_eval.py
"""

import os
import sys
import csv
import time
import traceback
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from main import RecipeRAGSystem


# ===============================
# 1. RAGAS 相关导入
# ===============================

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        answer_correctness,
        context_precision,
        context_recall,
    )
except Exception as e:
    raise ImportError(
        "RAGAS 导入失败，请先执行：pip install \"ragas==0.2.8\" datasets pandas langchain-openai\n"
        f"原始错误：{e}"
    )


# ===============================
# 2. LangChain 模型导入
# ===============================

try:
    from langchain_openai import ChatOpenAI
except Exception as e:
    raise ImportError(
        "langchain_openai 导入失败，请执行：pip install langchain-openai\n"
        f"原始错误：{e}"
    )

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    # 兼容旧版本
    from langchain_community.embeddings import HuggingFaceEmbeddings


# ===============================
# 3. 路径与参数配置
# ===============================

EVAL_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "eval_dataset_full.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "eval_reports")

RAGAS_INPUT_PATH = os.path.join(OUTPUT_DIR, "ragas_input_dataset.csv")
RAGAS_RESULT_PATH = os.path.join(OUTPUT_DIR, "ragas_eval_result.csv")
RAGAS_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "ragas_summary.csv")

TOP_K = 1

# 全量评测：None 表示不截断，直接跑全部 1023 条
MAX_SAMPLES = None

# RAGAS 一次 evaluate 太多样本容易超时，所以分批跑
BATCH_SIZE = 10

# 每批之间暂停，避免 API 请求过密
SLEEP_SECONDS = 1

rag_system = None


# ===============================
# 4. 初始化 RAG 系统
# ===============================

def init_rag_system():
    """
    初始化项目原有的 RAG 系统。
    """

    global rag_system

    if rag_system is not None:
        return rag_system

    print("正在初始化 RAG 系统，请稍等...")

    rag_system = RecipeRAGSystem()
    rag_system.initialize_system()
    rag_system.build_knowledge_base()

    print("RAG 系统初始化完成。")
    print()

    return rag_system


# ===============================
# 5. 初始化 RAGAS 使用的 LLM 和 Embedding
# ===============================

def init_ragas_llm():
    """
    使用 DeepSeek 作为 RAGAS 的评测 LLM。

    DeepSeek API 通常兼容 OpenAI 风格调用。
    """

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("未检测到 DEEPSEEK_API_KEY，请在 .env 或终端环境变量中设置。")

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0,
        max_tokens=1024,
        timeout=120,
        max_retries=3,
    )

    return llm


def init_ragas_embeddings():
    """
    使用 BGE 中文 embedding 作为 RAGAS 的 embedding。

    这样不用 OpenAI Embedding，成本更低。
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings


# ===============================
# 6. 读取评测集
# ===============================

def load_eval_dataset(path, max_samples=None):
    """
    读取 eval_dataset_full.csv。
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到评测集：{path}")

    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if max_samples is not None:
        rows = rows[:max_samples]

    return rows


# ===============================
# 7. 对单条样本运行 RAG
# ===============================

def run_rag_once(system, question, top_k=3):
    """
    对单个问题执行完整 RAG 流程：
    1. query_router
    2. query_rewrite
    3. metadata filter
    4. hybrid retrieval
    5. parent document recall
    6. answer generation

    返回：
        route_type
        rewritten_query
        filters
        contexts
        answer
    """

    route_type = system.generation_module.query_router(question)

    if route_type == "list":
        rewritten_query = question
    else:
        rewritten_query = system.generation_module.query_rewrite(question)

    filters = system._extract_filters_from_query(question)

    if filters:
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

    parent_docs = system.data_module.get_parent_documents(relevant_chunks)

    contexts = []

    for doc in relevant_chunks:
        content = getattr(doc, "page_content", "")
        metadata = getattr(doc, "metadata", {}) or {}

        dish_name = metadata.get("dish_name", "未知菜品")
        category = metadata.get("category", "")
        difficulty = metadata.get("difficulty", "")

        context_text = (
            f"菜品名称：{dish_name}\n"
            f"分类：{category}\n"
            f"难度：{difficulty}\n"
            f"正文：{content}"
        )

        if context_text.strip():
            contexts.append(context_text)

    

    if not parent_docs:
        answer = "抱歉，没有找到相关的知识库内容。"
        return route_type, rewritten_query, filters, contexts, answer

    if route_type == "list":
        answer = system.generation_module.generate_list_answer(question, parent_docs)
    elif route_type == "detail":
        answer = system.generation_module.generate_step_by_step_answer(question, parent_docs)
    else:
        answer = system.generation_module.generate_basic_answer(question, parent_docs)

    return route_type, rewritten_query, filters, contexts, answer


# ===============================
# 8. 构造 RAGAS 输入数据
# ===============================

def build_ragas_input_dataset(samples):
    """
    运行系统，构造 RAGAS 需要的数据格式。

    RAGAS 常用字段：
    - question
    - answer
    - contexts
    - ground_truth
    - reference
    """

    system = init_rag_system()

    ragas_rows = []
    debug_rows = []

    total = len(samples)

    for idx, row in enumerate(samples, start=1):
        question = row.get("question", "")
        reference_answer = row.get("reference_answer", "")
        ground_truth_docs = row.get("ground_truth_docs", "")
        query_type = row.get("query_type", "")

        print(f"[{idx}/{total}] 正在运行 RAG：{question}")

        try:
            route_type, rewritten_query, filters, contexts, answer = run_rag_once(
                system=system,
                question=question,
                top_k=TOP_K
            )

            if not contexts:
                contexts = [""]

            ragas_rows.append({
                # RAGAS 0.2.x / 0.3.x 推荐字段
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": reference_answer,

                # 兼容旧版本字段，保留也没事
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": reference_answer,
            })

            debug_rows.append({
                "question": question,
                "query_type": query_type,
                "route_type": route_type,
                "ground_truth_docs": ground_truth_docs,
                "reference_answer": reference_answer,
                "answer": answer,
                "contexts": "\n\n---\n\n".join(contexts),
                "rewritten_query": rewritten_query,
                "filters": str(filters),
                "status": "success",
                "error": "",
            })

        except Exception as e:
            print(f"    当前样本处理失败：{e}")

            ragas_rows.append({
                "user_input": question,
                "response": "",
                "retrieved_contexts": [""],
                "reference": reference_answer,

                "question": question,
                "answer": "",
                "contexts": [""],
                "ground_truth": reference_answer,
            })

            debug_rows.append({
                "question": question,
                "query_type": query_type,
                "route_type": "",
                "ground_truth_docs": ground_truth_docs,
                "reference_answer": reference_answer,
                "answer": "",
                "contexts": "",
                "rewritten_query": "",
                "filters": "",
                "status": "failed",
                "error": str(e),
            })

    return ragas_rows, debug_rows


# ===============================
# 9. 保存 RAGAS 输入数据
# ===============================

def save_ragas_input(debug_rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.DataFrame(debug_rows)
    df.to_csv(RAGAS_INPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"RAGAS 输入数据已保存：{RAGAS_INPUT_PATH}")


# ===============================
# 10. 分批运行 RAGAS
# ===============================

def chunk_list(data, batch_size):
    """
    将列表切成多个 batch。
    """

    for i in range(0, len(data), batch_size):
        yield i, data[i:i + batch_size]


def evaluate_one_batch(batch_rows, llm, embeddings):
    """
    对一个 batch 运行 RAGAS。
    """

    dataset = Dataset.from_list(batch_rows)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            #answer_correctness,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    result_df = result.to_pandas()

    return result_df


def run_ragas_in_batches(ragas_rows):
    """
    分批运行 RAGAS，并将结果合并。
    """

    llm = init_ragas_llm()
    embeddings = init_ragas_embeddings()

    all_result_dfs = []

    total = len(ragas_rows)
    batch_id = 0

    for start_idx, batch_rows in chunk_list(ragas_rows, BATCH_SIZE):
        batch_id += 1
        end_idx = min(start_idx + BATCH_SIZE, total)

        print("=" * 60)
        print(f"正在运行 RAGAS Batch {batch_id}: 样本 {start_idx + 1} - {end_idx} / {total}")
        print("=" * 60)

        try:
            batch_result_df = evaluate_one_batch(
                batch_rows=batch_rows,
                llm=llm,
                embeddings=embeddings
            )

            all_result_dfs.append(batch_result_df)

            # 每跑完一批，就保存临时结果，防止中途断掉全丢
            temp_df = pd.concat(all_result_dfs, ignore_index=True)
            temp_df.to_csv(RAGAS_RESULT_PATH, index=False, encoding="utf-8-sig")

            print(f"Batch {batch_id} 完成，临时结果已保存：{RAGAS_RESULT_PATH}")

        except Exception as e:
            print(f"Batch {batch_id} 失败：{e}")
            traceback.print_exc()

            # 当前 batch 失败时，写入空指标，保证流程继续
            failed_df = pd.DataFrame(batch_rows)
            failed_df["faithfulness"] = None
            failed_df["answer_relevancy"] = None
            failed_df["answer_correctness"] = None
            failed_df["context_precision"] = None
            failed_df["context_recall"] = None
            failed_df["ragas_error"] = str(e)

            all_result_dfs.append(failed_df)

        time.sleep(SLEEP_SECONDS)

    final_df = pd.concat(all_result_dfs, ignore_index=True)

    return final_df






# ===============================
# 11. 生成 summary
# ===============================

def save_summary(result_df):
    """
    保存整体平均指标。
    """

    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        #"answer_correctness",
        "context_precision",
        "context_recall",
    ]

    summary = {}

    for col in metric_cols:
        if col in result_df.columns:
            summary[col] = float(round(pd.to_numeric(result_df[col], errors="coerce").mean(), 4))
        else:
            summary[col] = None

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(RAGAS_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print()
    print("RAGAS 平均指标：")
    print(summary)
    print()
    print(f"RAGAS 汇总结果已保存：{RAGAS_SUMMARY_PATH}")


# ===============================
# 12. 主流程
# ===============================

def main():
    print("=" * 60)
    print("开始运行真正的 RAGAS 全量评测")
    print("=" * 60)
    print(f"评测集：{EVAL_DATASET_PATH}")
    print(f"Top K：{TOP_K}")
    print(f"MAX_SAMPLES：{MAX_SAMPLES}")
    print(f"BATCH_SIZE：{BATCH_SIZE}")
    print()

    samples = load_eval_dataset(
        path=EVAL_DATASET_PATH,
        max_samples=MAX_SAMPLES
    )
    # RAGAS 只评测 detail / general，先排除 list 推荐类问题
    samples = [
        item for item in samples
        if item.get("query_type", "") in ["detail", "general"]
    ]

    # 先取前 50 条做抽样评测
    samples = samples[:50]

    print(f"成功读取评测样本：{len(samples)} 条")
    print()

    ragas_rows, debug_rows = build_ragas_input_dataset(samples)

    save_ragas_input(debug_rows)

    print()
    print("开始调用 RAGAS 进行评测...")
    print()

    result_df = run_ragas_in_batches(ragas_rows)

    result_df.to_csv(RAGAS_RESULT_PATH, index=False, encoding="utf-8-sig")

    print()
    print("=" * 60)
    print("RAGAS 全量评测完成")
    print("=" * 60)
    print(f"详细结果已保存：{RAGAS_RESULT_PATH}")

    save_summary(result_df)


if __name__ == "__main__":
    main()