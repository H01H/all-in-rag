# rag_modules/evaluation.py
# -*- coding: utf-8 -*-

"""
检索评测模块

指标：
1. Recall@K
2. Precision@K
3. HitRate@K
4. MRR

注意：
这里按照“父文档 / 菜品名称”进行评测，而不是按照 chunk 评测。
因为同一个菜品可能被切成多个子块，如果不去重，Recall 会被错误放大。
"""


def parse_ground_truth_docs(ground_truth_docs):
    """
    将 ground_truth_docs 解析成列表。

    示例：
        "宫保鸡丁" -> ["宫保鸡丁"]
        "双皮奶;红豆沙;蛋挞" -> ["双皮奶", "红豆沙", "蛋挞"]
    """

    if ground_truth_docs is None:
        return []

    ground_truth_docs = str(ground_truth_docs).strip()

    if ground_truth_docs == "":
        return []

    docs = ground_truth_docs.split(";")
    docs = [doc.strip() for doc in docs if doc.strip()]

    return docs


def normalize_text(text):
    """
    简单标准化文本。
    """

    if text is None:
        return ""

    return str(text).strip().lower()


def extract_doc_name(result):
    """
    从检索结果中提取父文档名称 / 菜名。
    """

    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        if "dish_name" in result:
            return result.get("dish_name", "")

        if "title" in result:
            return result.get("title", "")

        if "doc_name" in result:
            return result.get("doc_name", "")

        if "name" in result:
            return result.get("name", "")

        metadata = result.get("metadata", {})
        if isinstance(metadata, dict):
            if "dish_name" in metadata:
                return metadata.get("dish_name", "")

            if "title" in metadata:
                return metadata.get("title", "")

            if "doc_name" in metadata:
                return metadata.get("doc_name", "")

            if "name" in metadata:
                return metadata.get("name", "")

    return ""


def get_unique_retrieved_doc_names(retrieved_docs, k=3):
    """
    获取 top-k 检索结果中的唯一父文档名称。

    关键点：
    同一道菜可能有多个 chunk 被检索出来。
    例如：
        宫保鸡丁-食材
        宫保鸡丁-做法
        宫保鸡丁-技巧

    它们都属于同一个父文档“宫保鸡丁”。

    所以这里必须去重，否则 Recall@K 会大于 1。
    """

    unique_names = []
    seen = set()

    for doc in retrieved_docs[:k]:
        doc_name = normalize_text(extract_doc_name(doc))

        if not doc_name:
            continue

        if doc_name not in seen:
            seen.add(doc_name)
            unique_names.append(doc_name)

    return unique_names


def calculate_recall_at_k(ground_truth_docs, retrieved_docs, k=3):
    """
    Recall@K = top-k 中命中的唯一正确文档数 / 正确文档总数
    """

    truth_docs = parse_ground_truth_docs(ground_truth_docs)
    truth_docs = set(normalize_text(doc) for doc in truth_docs)

    if len(truth_docs) == 0:
        return 0.0

    retrieved_names = get_unique_retrieved_doc_names(retrieved_docs, k=k)

    hit_docs = set()

    for doc_name in retrieved_names:
        if doc_name in truth_docs:
            hit_docs.add(doc_name)

    recall = len(hit_docs) / len(truth_docs)

    return min(recall, 1.0)


def calculate_precision_at_k(ground_truth_docs, retrieved_docs, k=3):
    """
    Precision@K = top-k 中命中的唯一正确文档数 / top-k 唯一文档数

    注意：
    这里不用固定除以 k，而是除以去重后的检索文档数。
    因为如果 top-3 都来自同一道菜，实际上只检索到了 1 个父文档。
    """

    truth_docs = parse_ground_truth_docs(ground_truth_docs)
    truth_docs = set(normalize_text(doc) for doc in truth_docs)

    retrieved_names = get_unique_retrieved_doc_names(retrieved_docs, k=k)

    if len(retrieved_names) == 0:
        return 0.0

    hit_docs = set()

    for doc_name in retrieved_names:
        if doc_name in truth_docs:
            hit_docs.add(doc_name)

    precision = len(hit_docs) / len(retrieved_names)

    return min(precision, 1.0)


def calculate_hit_rate_at_k(ground_truth_docs, retrieved_docs, k=3):
    """
    HitRate@K：
    top-k 中只要至少命中一个正确父文档，就记为 1。
    """

    truth_docs = parse_ground_truth_docs(ground_truth_docs)
    truth_docs = set(normalize_text(doc) for doc in truth_docs)

    if len(truth_docs) == 0:
        return 0.0

    retrieved_names = get_unique_retrieved_doc_names(retrieved_docs, k=k)

    for doc_name in retrieved_names:
        if doc_name in truth_docs:
            return 1.0

    return 0.0


def calculate_mrr(ground_truth_docs, retrieved_docs):
    """
    MRR = 1 / 第一个正确父文档的排名

    注意：
    排名也按照去重后的父文档序列计算。
    """

    truth_docs = parse_ground_truth_docs(ground_truth_docs)
    truth_docs = set(normalize_text(doc) for doc in truth_docs)

    if len(truth_docs) == 0:
        return 0.0

    # 这里不限制 k，按完整检索列表去重后计算第一个命中位置
    retrieved_names = get_unique_retrieved_doc_names(
        retrieved_docs,
        k=len(retrieved_docs)
    )

    for index, doc_name in enumerate(retrieved_names):
        if doc_name in truth_docs:
            rank = index + 1
            return 1.0 / rank

    return 0.0


def calculate_top1_accuracy(ground_truth_docs, retrieved_docs):
    """
    计算 Top1 Accuracy。

    含义：
    第一个检索结果是不是正确父文档。

    如果 Top1 命中正确文档，返回 1；
    否则返回 0。
    """

    truth_docs = parse_ground_truth_docs(ground_truth_docs)
    truth_docs = set(normalize_text(doc) for doc in truth_docs)

    if len(truth_docs) == 0:
        return 0.0

    retrieved_names = get_unique_retrieved_doc_names(
        retrieved_docs,
        k=len(retrieved_docs)
    )

    if not retrieved_names:
        return 0.0

    top1_name = retrieved_names[0]

    if top1_name in truth_docs:
        return 1.0

    return 0.0


def evaluate_single_query(ground_truth_docs, retrieved_docs, k=3):
    """
    对单个问题计算检索指标。
    """

    recall = calculate_recall_at_k(
        ground_truth_docs=ground_truth_docs,
        retrieved_docs=retrieved_docs,
        k=k
    )

    precision = calculate_precision_at_k(
        ground_truth_docs=ground_truth_docs,
        retrieved_docs=retrieved_docs,
        k=k
    )

    hit_rate = calculate_hit_rate_at_k(
        ground_truth_docs=ground_truth_docs,
        retrieved_docs=retrieved_docs,
        k=k
    )

    mrr = calculate_mrr(
        ground_truth_docs=ground_truth_docs,
        retrieved_docs=retrieved_docs
    )
    top1_accuracy = calculate_top1_accuracy(
    ground_truth_docs=ground_truth_docs,
    retrieved_docs=retrieved_docs
    )

    result = {
        f"recall@{k}": round(recall, 4),
        f"precision@{k}": round(precision, 4),
        f"hit_rate@{k}": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "top1_accuracy": round(top1_accuracy, 4)
    }

    return result


def evaluate_all_queries(query_results, k=3):
    """
    对整个评测集计算平均指标。
    """

    if not query_results:
        return {
            f"avg_recall@{k}": 0.0,
            f"avg_precision@{k}": 0.0,
            f"avg_hit_rate@{k}": 0.0,
            "avg_mrr": 0.0
        }

    recall_scores = []
    precision_scores = []
    hit_rate_scores = []
    mrr_scores = []
    top1_scores = []

    for item in query_results:
        ground_truth_docs = item.get("ground_truth_docs", "")
        retrieved_docs = item.get("retrieved_docs", [])

        single_result = evaluate_single_query(
            ground_truth_docs=ground_truth_docs,
            retrieved_docs=retrieved_docs,
            k=k
        )

        recall_scores.append(single_result[f"recall@{k}"])
        precision_scores.append(single_result[f"precision@{k}"])
        hit_rate_scores.append(single_result[f"hit_rate@{k}"])
        mrr_scores.append(single_result["mrr"])
        top1_scores.append(single_result["top1_accuracy"])

    avg_result = {
        f"avg_recall@{k}": round(sum(recall_scores) / len(recall_scores), 4),
        f"avg_precision@{k}": round(sum(precision_scores) / len(precision_scores), 4),
        f"avg_hit_rate@{k}": round(sum(hit_rate_scores) / len(hit_rate_scores), 4),
        "avg_mrr": round(sum(mrr_scores) / len(mrr_scores), 4),
        "avg_top1_accuracy": round(sum(top1_scores) / len(top1_scores), 4)
    }

    return avg_result


if __name__ == "__main__":
    test_ground_truth = "宫保鸡丁"

    test_retrieved_docs = [
        {"dish_name": "宫保鸡丁"},
        {"dish_name": "宫保鸡丁"},
        {"dish_name": "宫保鸡丁"}
    ]

    result = evaluate_single_query(
        ground_truth_docs=test_ground_truth,
        retrieved_docs=test_retrieved_docs,
        k=3
    )

    print("测试：top-3 都是同一个父文档时，Recall 不应该大于 1")
    print(result)

    test_ground_truth_2 = "双皮奶;红豆沙;蛋挞"

    test_retrieved_docs_2 = [
        {"dish_name": "双皮奶"},
        {"dish_name": "红豆沙"},
        {"dish_name": "宫保鸡丁"}
    ]

    result_2 = evaluate_single_query(
        ground_truth_docs=test_ground_truth_2,
        retrieved_docs=test_retrieved_docs_2,
        k=3
    )

    print("测试：多个正确文档时")
    print(result_2)