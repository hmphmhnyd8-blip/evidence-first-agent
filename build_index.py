import os
from pathlib import Path

from dotenv import load_dotenv

# 必须先加载环境变量，再导入模型相关库
load_dotenv(Path(__file__).parent / ".env", override=False)

from sentence_transformers import SentenceTransformer

from corpus import load_chunks

import chromadb

def save_to_chroma(
    chunks: list[dict],
    vectors: list[list[float]],
    model_name: str,
) -> int:
    """
    将片段编号、原文、来源信息和向量保存到本地 Chroma。

    返回类型：int。
    返回内容：保存后，集合中的记录总数。
    """
    # 读取数据库目录和集合名称
    persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME")

    if not persist_dir or not collection_name:
        raise ValueError(
            "请在 .env 中配置 CHROMA_PERSIST_DIR "
            "和 CHROMA_COLLECTION_NAME"
        )

    # 每个片段必须有一个对应的向量
    if len(chunks) != len(vectors):
        raise ValueError("片段数量与向量数量不一致")

    # 创建或打开本地数据库，数据会保存在指定目录中
    client = chromadb.PersistentClient(path=persist_dir)

    # 创建或打开集合；向量由我们提供，不让 Chroma 自动生成
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": model_name},
    )

    # 防止把不同模型生成的向量混入已有集合
    saved_model = (collection.metadata or {}).get("embedding_model")
    if saved_model != model_name:
        raise ValueError("集合使用的模型不同，请改用新的集合名称")

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        # 编号用作数据库中这条记录的唯一标识
        ids.append(chunk["chunk_id"])

        # 保存原文，不添加 passage: 前缀
        documents.append(chunk["text"])

        # 保存来源信息，略过值为 None 的字段
        metadata = {}
        for key in ("source", "page", "section", "source_url"):
            value = chunk.get(key)
            if value is not None:
                metadata[key] = value

        metadatas.append(metadata)

    # 编号不存在则新增，已存在则更新
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=vectors,
    )

    return collection.count()

def main() -> None:
    """
    读取知识库片段，生成向量，并保存到本地 Chroma。

    返回类型：None。
    返回内容：无；将数据写入数据库，在终端打印数量。
    """
    #读取昨天合并好的片段，保留正文、编号和来源信息
    file_path = Path(__file__).parent / "data" / "all_chunks.jsonl"
    chunks = load_chunks(file_path)

    if not chunks:
        raise ValueError("知识库没有片段，请检查 all_chunks.jsonl")

    print("读取片段数：", len(chunks))

    # 从环境变量读取模型配置
    model_name = os.getenv("EMBEDDING_MODEL_NAME")
    cache_dir = os.getenv("HF_HUB_CACHE")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")

    if not model_name or not cache_dir:
        raise ValueError(
            "请在 .env 中配置 EMBEDDING_MODEL_NAME 和 HF_HUB_CACHE"
        )

    print("正在加载模型……")
    model = SentenceTransformer(
        model_name,
        cache_folder=cache_dir,
        device=device,
    )

    # 提取每个片段的正文，并添加 E5 模型需要的前缀
    passage_texts = [
        "passage: " + chunk["text"]
        for chunk in chunks
    ]

    # 检查输入长度，避免正文超过模型限制后被自动截掉
    encoded = model.tokenizer(passage_texts, truncation=False)
    token_counts = [len(ids) for ids in encoded["input_ids"]]

    if max(token_counts) > model.max_seq_length:
        raise ValueError(
            f"最长片段为 {max(token_counts)} tokens，"
            f"超过模型限制 {model.max_seq_length}，需要调整切块"
        )

    # 为全部片段生成向量；输出顺序与 chunks 中的顺序一致
    print("正在生成片段向量……")
    passage_vectors = model.encode(
        passage_texts,
        normalize_embeddings=True,
    )
    print("片段向量形状：", passage_vectors.shape)

    # 将 NumPy 数组转换成普通 Python 嵌套列表，再保存
    stored_count = save_to_chroma(
        chunks=chunks,
        vectors=passage_vectors.tolist(),
        model_name=model_name,
    )

    print("本次保存片段数：", len(chunks))
    print("集合记录总数：", stored_count)
    print("数据库保存完成。")

if __name__ == "__main__":
    main()