# eval/run_eval.py
# -*- coding: utf-8 -*-

"""
自动化检索评测脚本

作用：
1. 读取 data/eval_dataset.csv
2. 对每个问题调用 RAG 检索系统
3. 计算 Recall@K、Precision@K、HitRate@K、MRR
4. 保存每条问题的评测结果
5. 输出整体平均指标

运行方式：
    python eval/run_eval.py
"""

import os
import sys
import csv
from datetime import datetime



# ===============================
# 1. 处理项目路径
# ===============================

# 当前文件路径：C8/eval/run_eval.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录：C8/
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 把 C8/ 加入 Python 搜索路径
# 这样才能导入 rag_modules/evaluation.py
sys.path.append(PROJECT_ROOT)


from rag_modules.evaluation import evaluate_single_query, evaluate_all_queries
from main import RecipeRAGSystem

# ===============================
# 2. 配置路径
# ===============================

EVAL_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "eval_dataset_full.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "eval_reports")

EVAL_RESULT_PATH = os.path.join(OUTPUT_DIR, "eval_result.csv")
GROUP_RESULT_PATH = os.path.join(OUTPUT_DIR, "eval_group_result.csv")

TOP_K = 3

rag_system = None


# ===============================
# 3. 读取评测集
# ===============================

def load_eval_dataset(csv_path):
    """
    读取评测集 CSV 文件。

    参数：
        csv_path: str
            eval_dataset.csv 的路径

    返回：
        list[dict]
            每一行是一个评测样本
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到评测集文件：{csv_path}")

    samples = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            samples.append(row)

    return samples


def init_rag_system():
    """
    初始化原项目中的 RAG 系统。

    注意：
    initialize_system() 只初始化模块；
    build_knowledge_base() 才会真正加载索引、切分文档、创建 retrieval_module。
    """

    global rag_system

    if rag_system is not None:
        return rag_system

    print("正在初始化 RAG 系统，请稍等...")

    rag_system = RecipeRAGSystem()

    # 1. 初始化基础模块
    rag_system.initialize_system()

    # 2. 构建 / 加载知识库
    # 这一步非常关键，会创建 rag_system.retrieval_module
    rag_system.build_knowledge_base()

    print("RAG 系统初始化完成。")
    print()

    return rag_system

# ===============================
# 4. 调用检索系统
# ===============================

def retrieve_docs_by_system(question, top_k=3):
    """
    调用真实 RAG 检索系统，返回 top-k 检索结果。
    """

    system = init_rag_system()

    if system.retrieval_module is None:
        raise ValueError("retrieval_module 仍然是 None，请检查 build_knowledge_base() 是否成功执行。")

    # 1. 模仿 main.py 里的逻辑：先提取过滤条件
    filters = system._extract_filters_from_query(question)

    # 2. 如果有分类/难度过滤，就走 metadata_filtered_search
    if filters:
        docs = system.retrieval_module.metadata_filtered_search(
            query=question,
            filters=filters,
            top_k=top_k
        )
    else:
        docs = system.retrieval_module.hybrid_search(
            query=question,
            top_k=top_k
        )

    # 3. 把 LangChain Document 转成普通 dict
    retrieved_docs = []

    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        page_content = getattr(doc, "page_content", "")

        item = {
            "dish_name": metadata.get("dish_name", ""),
            "category": metadata.get("category", ""),
            "difficulty": metadata.get("difficulty", ""),
            "source": metadata.get("source", ""),
            "doc_type": metadata.get("doc_type", ""),
            "parent_id": metadata.get("parent_id", ""),
            "rrf_score": metadata.get("rrf_score", ""),
            "content": page_content
        }

        retrieved_docs.append(item)

    return retrieved_docs


# ===============================
# 5. 保存单条评测结果
# ===============================

def save_eval_results(results, output_path):
    """
    保存每条问题的评测结果到 CSV。
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "question",
        "query_type",
        "ground_truth_docs",
        "retrieved_docs",
        f"recall@{TOP_K}",
        f"precision@{TOP_K}",
        f"hit_rate@{TOP_K}",
        "mrr",
        "top1_accuracy",
        "reference_answer",
        "category_filter",
        "difficulty_filter",
        "notes"
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for item in results:
            writer.writerow(item)


def save_group_results(group_results, output_path, k=3):
    """
    保存分组评测结果到 CSV。
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "query_type",
        "count",
        f"avg_recall@{k}",
        f"avg_precision@{k}",
        f"avg_hit_rate@{k}",
        "avg_mrr",
        "avg_top1_accuracy"
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for query_type, result in group_results.items():
            row = {
                "query_type": query_type,
                **result
            }
            writer.writerow(row)


def evaluate_by_query_type(save_rows, k=3):
    """
    按 query_type 分组统计指标。

    会分别统计：
    1. detail
    2. list
    3. general
    """

    groups = {}

    for row in save_rows:
        query_type = row.get("query_type", "unknown")

        if not query_type:
            query_type = "unknown"

        if query_type not in groups:
            groups[query_type] = []

        groups[query_type].append(row)

    group_results = {}

    for query_type, rows in groups.items():
        recall_key = f"recall@{k}"
        precision_key = f"precision@{k}"
        hit_rate_key = f"hit_rate@{k}"

        recall_scores = [float(row.get(recall_key, 0)) for row in rows]
        precision_scores = [float(row.get(precision_key, 0)) for row in rows]
        hit_rate_scores = [float(row.get(hit_rate_key, 0)) for row in rows]
        mrr_scores = [float(row.get("mrr", 0)) for row in rows]
        top1_scores = [float(row.get("top1_accuracy", 0)) for row in rows]

        group_results[query_type] = {
            "count": len(rows),
            f"avg_recall@{k}": round(sum(recall_scores) / len(recall_scores), 4),
            f"avg_precision@{k}": round(sum(precision_scores) / len(precision_scores), 4),
            f"avg_hit_rate@{k}": round(sum(hit_rate_scores) / len(hit_rate_scores), 4),
            "avg_mrr": round(sum(mrr_scores) / len(mrr_scores), 4),
            "avg_top1_accuracy": round(sum(top1_scores) / len(top1_scores), 4)
        }

    return group_results
# ===============================
# 6. 主评测流程
# ===============================

def run_evaluation():
    """
    运行完整评测流程。
    """

    print("=" * 60)
    print("开始运行 RAG 检索评测")
    print("=" * 60)

    print(f"评测集路径：{EVAL_DATASET_PATH}")
    print(f"评测结果保存路径：{EVAL_RESULT_PATH}")
    print(f"Top K = {TOP_K}")
    print()

    # 1. 读取评测集
    samples = load_eval_dataset(EVAL_DATASET_PATH)

    print(f"成功读取评测集，共 {len(samples)} 条问题")
    print()

    all_query_results = []
    save_rows = []

    # 2. 逐条评测
    for idx, sample in enumerate(samples, start=1):
        question = sample.get("question", "")
        query_type = sample.get("query_type", "")
        ground_truth_docs = sample.get("ground_truth_docs", "")
        reference_answer = sample.get("reference_answer", "")
        category_filter = sample.get("category_filter", "")
        difficulty_filter = sample.get("difficulty_filter", "")
        notes = sample.get("notes", "")

        print(f"[{idx}/{len(samples)}] 正在评测问题：{question}")

        # 3. 调用检索系统
        retrieved_docs = retrieve_docs_by_system(
            question=question,
            top_k=TOP_K
        )

        # 4. 计算单条问题指标
        metric_result = evaluate_single_query(
            ground_truth_docs=ground_truth_docs,
            retrieved_docs=retrieved_docs,
            k=TOP_K
        )

        # 5. 为整体平均指标准备数据
        all_query_results.append({
            "question": question,
            "ground_truth_docs": ground_truth_docs,
            "retrieved_docs": retrieved_docs
        })

        # 6. 整理检索到的文档名称，方便保存到 CSV
        retrieved_doc_names = []

        for doc in retrieved_docs:
            if isinstance(doc, dict):
                dish_name = doc.get("dish_name", "")

                if not dish_name and "metadata" in doc:
                    metadata = doc.get("metadata", {})
                    dish_name = metadata.get("dish_name", "")

                retrieved_doc_names.append(dish_name)
            else:
                retrieved_doc_names.append(str(doc))

        retrieved_docs_str = ";".join(retrieved_doc_names)

        # 7. 保存单条结果
        save_rows.append({
            "question": question,
            "query_type": query_type,
            "ground_truth_docs": ground_truth_docs,
            "retrieved_docs": retrieved_docs_str,
            f"recall@{TOP_K}": metric_result[f"recall@{TOP_K}"],
            f"precision@{TOP_K}": metric_result[f"precision@{TOP_K}"],
            f"hit_rate@{TOP_K}": metric_result[f"hit_rate@{TOP_K}"],
            "mrr": metric_result["mrr"],
            "top1_accuracy": metric_result["top1_accuracy"],
            "reference_answer": reference_answer,
            "category_filter": category_filter,
            "difficulty_filter": difficulty_filter,
            "notes": notes
        })

        print(f"    正确文档：{ground_truth_docs}")
        print(f"    检索结果：{retrieved_docs_str}")
        print(f"    指标：{metric_result}")
        print()

    # 8. 计算整体平均指标
    avg_result = evaluate_all_queries(
        query_results=all_query_results,
        k=TOP_K
    )

    # 9. 保存结果
    save_eval_results(
        results=save_rows,
        output_path=EVAL_RESULT_PATH
    )

    group_results = evaluate_by_query_type(
        save_rows=save_rows,
        k=TOP_K
    )
    save_group_results(
        group_results=group_results,
        output_path=GROUP_RESULT_PATH,
        k=TOP_K
    )

    print("=" * 60)
    print("评测完成")
    print("=" * 60)

    print("整体平均指标：")
    print(avg_result)
    print()

    print("按 query_type 分组指标：")
    for query_type, result in group_results.items():
        print(f"[{query_type}]")
        print(result)
        print()

    print(f"详细结果已保存到：{EVAL_RESULT_PATH}")

    


if __name__ == "__main__":
    run_evaluation()