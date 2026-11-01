import json
from functools import lru_cache
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi


CHUNKS_PATH = Path(__file__).parent / "data" / "all_chunks.jsonl"


def tokenize(text: str) -> list[str]:
    """统一处理资料和问题：英文转小写，然后分词。"""
    words = jieba.lcut_for_search(text.lower())

    tokens = []

    for word in words:
        word = word.strip()

        # 保留含文字或数字的词，去掉空白和纯标点
        if word and any(char.isalnum() for char in word):
            tokens.append(word)

    return tokens


@lru_cache(maxsize=1)
def load_bm25_index() -> tuple:
    """读取已有片段，构建可重复使用的 BM25 检索对象。"""
    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError("切片文件为空")

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    if not any(tokenized_chunks):
        raise ValueError("切片中没有可用于检索的词语")

    bm25 = BM25Okapi(tokenized_chunks)

    return chunks, tokenized_chunks, bm25


def search_bm25(question: str, top_k: int = 5) -> list[dict]:
    """返回按 BM25 分数排序的候选片段。"""
    question = question.strip()

    if not question:
        raise ValueError("检索问题不能为空")

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    chunks, tokenized_chunks, bm25 = load_bm25_index()

    # 得到每个片段的分数，顺序与 chunks 一致
    scores = bm25.get_scores(query_tokens)

    # 只保留至少匹配了一个查询词的片段
    query_terms = set(query_tokens)
    candidate_indices = []

    for index, tokens in enumerate(tokenized_chunks):
        if query_terms.intersection(tokens):
            candidate_indices.append(index)

    # 按分数从高到低排序，取前 top_k 个
    ranked_indices = sorted(
        candidate_indices,
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    matches = []

    for rank, index in enumerate(ranked_indices, start=1):
        chunk = chunks[index]

        matches.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source": chunk.get("source"),
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "source_url": chunk.get("source_url"),
            "bm25_score": float(scores[index]),
        })

    return matches


def main() -> None:
    question = input("请输入检索问题：").strip()

    if not question:
        print("问题不能为空。")
        return

    print("问题分词：", tokenize(question))

    matches = search_bm25(question, top_k=5)

    print(json.dumps(
        {
            "question": question,
            "matches": matches,
            "count": len(matches),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()