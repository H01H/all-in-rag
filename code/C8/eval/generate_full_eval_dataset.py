# eval/generate_full_eval_dataset.py
# -*- coding: utf-8 -*-

"""
自动生成全量评测集

作用：
1. 读取当前知识库中的全部父文档
2. 为每个菜品自动生成做法类、食材类问题
3. 根据 category / difficulty 自动生成推荐类问题
4. 输出 data/eval_dataset_full.csv

运行方式：
    cd /workspaces/all-in-rag/code/C8
    python eval/generate_full_eval_dataset.py
"""

import os
import sys
import csv
import re
from collections import defaultdict


# ===============================
# 1. 设置项目路径
# ===============================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from main import RecipeRAGSystem


# ===============================
# 2. 路径配置
# ===============================

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "eval_dataset_full.csv")


# ===============================
# 3. 文本处理函数
# ===============================

def clean_text(text):
    """
    清理文本，避免 CSV 中出现过多换行。
    """

    if not text:
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def truncate_text(text, max_len=500):
    """
    截断 reference_answer，避免一行太长。
    """

    text = clean_text(text)

    if len(text) > max_len:
        return text[:max_len] + "..."

    return text


def extract_section_by_keywords(content, keywords):
    """
    从 Markdown 文档中，根据标题关键词提取相关章节。

    比如：
    keywords = ["原料", "食材", "材料"]
    就尽量提取食材部分。

    keywords = ["步骤", "做法", "操作"]
    就尽量提取做法部分。
    """

    if not content:
        return ""

    lines = content.splitlines()

    sections = []
    current_title = ""
    current_content = []

    for line in lines:
        stripped = line.strip()

        # Markdown 标题，例如 #、##、###
        if stripped.startswith("#"):
            # 保存上一节
            if current_title or current_content:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_content)
                })

            current_title = stripped.replace("#", "").strip()
            current_content = []
        else:
            current_content.append(line)

    # 保存最后一节
    if current_title or current_content:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_content)
        })

    # 按标题关键词匹配
    for section in sections:
        title = section["title"]
        for kw in keywords:
            if kw in title:
                return section["content"]

    # 如果标题没匹配到，再在全文里简单找
    for kw in keywords:
        if kw in content:
            return content

    return ""


def build_reference_answer(doc, answer_type):
    """
    根据问题类型，从父文档中生成标准答案。

    注意：
    这里不是让大模型编答案，而是尽量从原始 Markdown 文档里抽取。
    这样更适合后面做 RAGAS 的 reference / ground_truth。
    """

    metadata = doc.metadata or {}
    dish_name = metadata.get("dish_name", "未知菜品")
    content = doc.page_content or ""

    if answer_type == "ingredients":
        section = extract_section_by_keywords(
            content,
            ["原料", "食材", "材料", "配料", "准备"]
        )

        if section:
            return truncate_text(f"{dish_name}的食材信息如下：{section}")

        return truncate_text(f"{dish_name}的食材信息可参考原文：{content}")

    if answer_type == "steps":
        section = extract_section_by_keywords(
            content,
            ["步骤", "做法", "操作", "制作", "流程"]
        )

        if section:
            return truncate_text(f"{dish_name}的做法步骤如下：{section}")

        return truncate_text(f"{dish_name}的做法可参考原文：{content}")

    if answer_type == "tips":
        section = extract_section_by_keywords(
            content,
            ["技巧", "小贴士", "注意", "提示", "关键"]
        )

        if section:
            return truncate_text(f"{dish_name}的制作技巧如下：{section}")

        return truncate_text(f"{dish_name}制作时可参考原文中的注意事项：{content}")

    return truncate_text(content)


# ===============================
# 4. 初始化系统并读取全部父文档
# ===============================

def load_all_parent_documents():
    """
    初始化 RAG 系统，并读取全部父文档。
    """

    print("正在初始化 RAG 系统并读取全部父文档...")

    system = RecipeRAGSystem()
    system.initialize_system()
    system.build_knowledge_base()

    docs = system.data_module.documents

    print(f"共读取到 {len(docs)} 个父文档。")

    return docs


# ===============================
# 5. 生成全量评测集
# ===============================

def generate_eval_rows(docs):
    """
    根据全部父文档生成 eval rows。
    """

    rows = []

    category_to_docs = defaultdict(list)
    difficulty_to_docs = defaultdict(list)
    category_difficulty_to_docs = defaultdict(list)

    seen_dish_names = set()

    for doc in docs:
        metadata = doc.metadata or {}

        dish_name = metadata.get("dish_name", "").strip()
        category = metadata.get("category", "").strip()
        difficulty = metadata.get("difficulty", "").strip()

        if not dish_name:
            continue

        # 避免重复父文档
        if dish_name in seen_dish_names:
            continue

        seen_dish_names.add(dish_name)

        if category:
            category_to_docs[category].append(dish_name)

        if difficulty:
            difficulty_to_docs[difficulty].append(dish_name)

        if category and difficulty:
            category_difficulty_to_docs[(category, difficulty)].append(dish_name)

        # 1. 每道菜：做法类问题
        rows.append({
            "question": f"{dish_name}怎么做？",
            "query_type": "detail",
            "ground_truth_docs": dish_name,
            "reference_answer": build_reference_answer(doc, "steps"),
            "category_filter": "",
            "difficulty_filter": "",
            "expected_answer_type": "做法步骤",
            "notes": "全量自动生成：具体菜品做法问题"
        })

        # 2. 每道菜：食材类问题
        rows.append({
            "question": f"{dish_name}需要哪些食材？",
            "query_type": "detail",
            "ground_truth_docs": dish_name,
            "reference_answer": build_reference_answer(doc, "ingredients"),
            "category_filter": "",
            "difficulty_filter": "",
            "expected_answer_type": "食材说明",
            "notes": "全量自动生成：具体菜品食材问题"
        })

        # 3. 每道菜：技巧类问题
        rows.append({
            "question": f"{dish_name}有什么制作技巧？",
            "query_type": "general",
            "ground_truth_docs": dish_name,
            "reference_answer": build_reference_answer(doc, "tips"),
            "category_filter": "",
            "difficulty_filter": "",
            "expected_answer_type": "技巧解释",
            "notes": "全量自动生成：具体菜品技巧问题"
        })

    # 4. 每个分类：推荐类问题
    for category, dish_names in category_to_docs.items():
        unique_names = list(dict.fromkeys(dish_names))

        if not unique_names:
            continue

        rows.append({
            "question": f"有哪些{category}可以推荐？",
            "query_type": "list",
            "ground_truth_docs": ";".join(unique_names),
            "reference_answer": f"可以推荐以下{category}：" + "、".join(unique_names) + "。",
            "category_filter": category,
            "difficulty_filter": "",
            "expected_answer_type": "菜品推荐",
            "notes": "全量自动生成：分类推荐问题"
        })

    # 5. 每个难度：推荐类问题
    for difficulty, dish_names in difficulty_to_docs.items():
        unique_names = list(dict.fromkeys(dish_names))

        if not unique_names:
            continue

        rows.append({
            "question": f"推荐几个{difficulty}的菜。",
            "query_type": "list",
            "ground_truth_docs": ";".join(unique_names),
            "reference_answer": f"可以推荐以下{difficulty}难度的菜品：" + "、".join(unique_names) + "。",
            "category_filter": "",
            "difficulty_filter": difficulty,
            "expected_answer_type": "菜品推荐",
            "notes": "全量自动生成：难度推荐问题"
        })

    # 6. 分类 + 难度组合推荐
    for (category, difficulty), dish_names in category_difficulty_to_docs.items():
        unique_names = list(dict.fromkeys(dish_names))

        if not unique_names:
            continue

        rows.append({
            "question": f"推荐几个{difficulty}的{category}。",
            "query_type": "list",
            "ground_truth_docs": ";".join(unique_names),
            "reference_answer": f"可以推荐以下{difficulty}的{category}：" + "、".join(unique_names) + "。",
            "category_filter": category,
            "difficulty_filter": difficulty,
            "expected_answer_type": "菜品推荐",
            "notes": "全量自动生成：分类加难度推荐问题"
        })

    return rows


# ===============================
# 6. 保存 CSV
# ===============================

def save_eval_dataset(rows, output_path):
    """
    保存全量评测集。
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "question",
        "query_type",
        "ground_truth_docs",
        "reference_answer",
        "category_filter",
        "difficulty_filter",
        "expected_answer_type",
        "notes"
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ===============================
# 7. 主函数
# ===============================

def main():
    docs = load_all_parent_documents()

    rows = generate_eval_rows(docs)

    save_eval_dataset(rows, OUTPUT_PATH)

    print("=" * 60)
    print("全量评测集生成完成")
    print("=" * 60)
    print(f"父文档数量：{len(docs)}")
    print(f"评测样本数量：{len(rows)}")
    print(f"保存路径：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()