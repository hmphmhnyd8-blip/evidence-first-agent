from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv

# 先加载环境变量，再导入相关库
load_dotenv(Path(__file__).parent / ".env", override=False)


def search_index(
    model: SentenceTransformer,
    collection,
    question: str,
    top_k: int = 5,
) -> list[dict]:
    """
    使用已有模型和集合，检索与问题最相关的片段。

    返回类型：list[dict]。
    返回内容：按相关程度排序的片段列表，每项包含排名、
    片段编号、原文、来源信息和相似度。空集合返回空列表。
    """
    # 检查输入，避免对空问题进行检索
    question = question.strip()
    if not question:
        raise ValueError("检索问题不能为空")

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    count = collection.count()
    if count == 0:
        return []

    # 使用调用方传入的模型，不在函数中重复加载模型
    question_vectors = model.encode(
        ["query: " + question],
        normalize_embeddings=True,
    )

    # 查询已有片段向量，最多返回 top_k 条
    result = collection.query(
        query_embeddings=question_vectors.tolist(),
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    # 取出本次唯一一个问题对应的结果
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    matches = []

    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] or {}

        # 将分散在不同列表里的信息，整理成一个片段字典
        match = {
            "rank": index + 1,
            "chunk_id": chunk_id,
            "text": documents[index],
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "section": metadata.get("section"),
            "source_url": metadata.get("source_url"),
            "similarity": float(1 - distances[index]),
        }
        matches.append(match)

    return matches

def main() -> None:
    """
    打开已有的本地向量数据库，检索与问题最相关的五个片段。

    返回类型：None。
    返回内容：无；在终端打印集合数量、问题向量形状和检索结果。
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    # 读取数据库和模型配置
    persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME")
    model_name = os.getenv("EMBEDDING_MODEL_NAME")
    cache_dir = os.getenv("HF_HUB_CACHE")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    offline_mode = os.getenv("HF_HUB_OFFLINE", "0") == "1"

    if not all([persist_dir, collection_name, model_name, cache_dir]):
        raise ValueError("请检查 .env 中的数据库和模型配置")

    # 打开之前保存的数据库
    client = chromadb.PersistentClient(path=persist_dir)

    # 只获取已有集合；名称或目录错误时应报错，避免创建空集合
    collection = client.get_collection(
        name=collection_name,
        embedding_function=None,
    )

    # 查询问题使用的模型必须与建库时一致
    saved_model = (collection.metadata or {}).get("embedding_model")
    if saved_model != model_name:
        raise ValueError("查询模型与集合记录的模型不一致")

    count = collection.count()
    print("数据库中的片段数：", count)

    if count == 0:
        raise ValueError("集合为空，请先运行建库脚本")

    question = input("\n请输入检索问题：").strip()
    if not question:
        print("没有输入问题，程序结束。")
        return

    # 加载同一个 Embedding 模型，只计算本次问题的向量
    print("正在加载模型……")
    model = SentenceTransformer(
        model_name,
        cache_folder=cache_dir,
        device=device,
        local_files_only=offline_mode,
    )

    # 调用检索函数，接收它返回的列表
    matches = search_index(
        model=model,
        collection=collection,
        question=question,
        top_k=5,
    )

    # 展示结果由 main 负责
    for match in matches:
        print(f"\n--- 第 {match['rank']} 名 ---")
        print("片段编号：", match["chunk_id"])
        print("来源文件：", match["source"])
        print("来源页码：", match["page"])
        print("来源段落：", match["section"])
        print(f"相似度：{match['similarity']:.4f}")
        print("正文：", match["text"])

if __name__ == "__main__":
    main()