import json
from pathlib import Path

from langchain_core.tools import tool

from sqlite_repository import (
    list_ship_ids_sqlite,
    query_ship_speed_sqlite,
)


# 数据库位置由程序确定，不需要模型提供
DB_PATH = Path(__file__).parent / "data" / "ships_demo.db"


@tool
def list_ship_ids() -> dict:
    """
    列出本地演示数据库中所有可查询的船舶编号。
    用户询问有哪些船舶、有哪些编号可查询时使用。
    此工具不需要输入参数。

    返回字典：
    vessel_ids 为船舶编号列表；
    count 为不同船舶的数量。
    """
    vessel_ids = list_ship_ids_sqlite(DB_PATH)

    return {
        "vessel_ids": vessel_ids,
        "count": len(vessel_ids),
    }


@tool
def get_ship_speed(vessel_id: str) -> dict:
    """
    根据船舶编号查询本地演示数据库中的航速统计。
    用户询问某艘船的平均航速、最大航速或记录数量时使用。
    航速单位为节，数据不代表船舶实时状态。

    参数 vessel_id：船舶编号，例如 DEMO-002。

    返回字典，包含船舶编号、查询状态和统计信息。
    状态包括：
    ok：有有效航速；
    no_valid_speed：有记录但航速全部缺失；
    not_found：没有该船舶的记录。
    """
    return query_ship_speed_sqlite(
        db_path=DB_PATH,
        vessel_id=vessel_id,
    )


def main() -> None:
    """直接调用两个工具，在终端展示结果。"""
    print("--- 工具一：列出船舶编号 ---")

    ids_result = list_ship_ids.invoke({})

    print(json.dumps(ids_result, ensure_ascii=False, indent=2))

    print("\n--- 工具二：查询指定船舶 ---")

    speed_result = get_ship_speed.invoke(
        {"vessel_id": "DEMO-002"}
    )

    print(json.dumps(speed_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()