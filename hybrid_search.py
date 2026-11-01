import json
from pathlib import Path

from dotenv import load_dotenv

# 保持配置先加载
load_dotenv(Path(__file__).parent / ".env", override=False)

from vector_search import search_index
from document_tools import load_retriever
from bm25_search import search_bm25


def rrf_fuse(
    vector_matches: list[dict],
    bm25_matches: list[dict],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """按片段编号合并两路结果，再按 RRF 分数排序。"""
    if top_k <= 0 or rrf_k <= 0:
        raise ValueError("top_k 和 rrf_k 必须大于 0")

    # key 是片段编号，value 是该片段的融合信息
    merged = {}

    result_lists = [
        ("vector", vector_matches),
        ("bm25", bm25_matches),
    ]

    for method, matches in result_lists:
        seen_ids = set()

        for position, match in enumerate(matches, start=1):
            chunk_id = match["chunk_id"]

            # 同一路中，一个片段只计分一次
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            if chunk_id not in merged:
                merged[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": match["text"],
                    "source": match.get("source"),
                    "page": match.get("page"),
                    "section": match.get("section"),
                    "source_url": match.get("source_url"),
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "bm25_rank": None,
                }

            item = merged[chunk_id] 

            # 同一个编号应对应同一份原文
            if item["text"] != match["text"]:
                raise ValueError(
                    f"片段 {chunk_id} 的两路原文不一致，"
                    "请检查 JSONL 与 Chroma 是否使用同一版资料"
                )

            # 保存该片段在当前检索方式中的排名
            item[f"{method}_rank"] = position

            # 累加当前检索方式贡献的分数
            item["rrf_score"] += 1.0 / (rrf_k + position)

    # 分数降序；同分时按片段编号排序，使结果顺序稳定
    ranked = sorted(
        merged.values(),
        key=lambda item: (-item["rrf_score"], item["chunk_id"]),
    )

    final_matches = ranked[:top_k]

    for rank, item in enumerate(final_matches, start=1):
        item["rank"] = rank

    return final_matches


def search_hybrid(
    question: str,
    top_k: int = 5,
    candidate_k: int = 10,
) -> list[dict]:
    """分别检索，再融合为最终的候选片段列表。"""
    question = question.strip()

    if not question:
        raise ValueError("检索问题不能为空")

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    if candidate_k < top_k:
        raise ValueError("candidate_k 不能小于 top_k")

    model, collection = load_retriever()

    vector_matches = search_index(
        model=model,
        collection=collection,
        question=question,
        top_k=candidate_k,
    )

    bm25_matches = search_bm25(
        question=question,
        top_k=candidate_k,
    )

    return rrf_fuse(
        vector_matches=vector_matches,
        bm25_matches=bm25_matches,
        top_k=top_k,
        rrf_k=60,
    )


def main() -> None:
    question = input("请输入混合检索问题：").strip()

    if not question:
        print("问题不能为空。")
        return

    matches = search_hybrid(
        question=question,
        top_k=5,
        candidate_k=10,
    )

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