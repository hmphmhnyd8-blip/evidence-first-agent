import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# 先加载配置，再导入模型相关库
load_dotenv(Path(__file__).parent / ".env", override=False)

from langchain_core.tools import tool

from vector_search import search_index


@lru_cache(maxsize=1)
def load_retriever() -> tuple:
    """
    打开已有 Chroma 集合，从本地缓存加载 E5 模型。

    返回类型：tuple。
    返回内容：(向量模型对象, 集合对象)。
    同一次程序运行中，成功加载后复用这两个对象。
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME")
    model_name = os.getenv("EMBEDDING_MODEL_NAME")
    cache_dir = os.getenv("HF_HUB_CACHE")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")

    if not all([
        persist_dir,
        collection_name,
        model_name,
        cache_dir,
    ]):
        raise ValueError("请检查 .env 中的模型和数据库配置")

    db_dir = Path(persist_dir).resolve()

    if not db_dir.is_dir():
        raise FileNotFoundError("向量数据库目录不存在，请检查配置")

    client = chromadb.PersistentClient(path=str(db_dir))

    collection = client.get_collection(
        name=collection_name,
        embedding_function=None,
    )

    saved_model = (collection.metadata or {}).get("embedding_model")

    if saved_model != model_name:
        raise ValueError("查询模型与建库模型不一致")

    if collection.count() == 0:
        raise ValueError("向量集合为空，请先完成建库")

    print("首次使用资料检索：正在从本地加载 E5……")

    embedding_model = SentenceTransformer(
        model_name,
        cache_folder=cache_dir,
        device=device,
        local_files_only=True,
    )

    return embedding_model, collection


@tool
def search_documents(question: str) -> dict:
    """
    检索本地项目说明、AIS FAQ 和数据字典中的相关原文。
    用户询问平均航速计算规则、缺失数据处理、
    AIS 字段含义或资料中的说明时使用。

    参数 question：含义完整、明确的检索问题，
    例如：本项目零航速是否参与平均航速计算？

    此工具返回候选资料，不查询具体船舶的航速统计。
    返回字典包含 question、matches 和 count。
    matches 中每个片段包含 chunk_id、text 和来源信息。
    返回片段不代表一定足以回答，需要检查原文是否相关。
    """
    question = question.strip()

    if not question:
        raise ValueError("检索问题不能为空")

    embedding_model, collection = load_retriever()

    matches = search_index(
        model=embedding_model,
        collection=collection,
        question=question,
        top_k=5,
    )

    return {
        "question": question,
        "matches": matches,
        "count": len(matches),
    }


def main() -> None:
    """直接测试资料检索工具，打印返回的候选片段。"""
    result = search_documents.invoke({
        "question": "本项目零航速是否参与平均航速计算？",
    })

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()