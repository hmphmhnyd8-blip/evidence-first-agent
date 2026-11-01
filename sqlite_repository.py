import json
import sqlite3
from pathlib import Path


def query_ship_speed_sqlite(
    db_path: Path,
    vessel_id: str,
) -> dict:
    """
    从 SQLite 查询指定船舶的航速统计。

    返回类型：dict。
    返回内容：船舶编号、查询状态和统计信息。
    状态包括 ok、no_valid_speed、not_found。
    """
    if not isinstance(vessel_id, str):
        raise TypeError("船舶编号必须是字符串")

    vessel_id = vessel_id.strip()

    if not vessel_id:
        raise ValueError("船舶编号不能为空")

    db_path = db_path.resolve()

    if not db_path.is_file():
        raise FileNotFoundError("数据库不存在，请先运行建库脚本")

    # 以只读方式打开，查询时不创建或修改数据库
    database_uri = db_path.as_uri() + "?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)

    try:
        row = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(sog),
                AVG(sog),
                MAX(sog)
            FROM ship_speed_records
            WHERE vessel_id = ?
            """,
            (vessel_id,),
        ).fetchone()

    finally:
        connection.close()

    # 查询结果是四个值组成的元组，按顺序取出来
    total_records, valid_records, average_speed, max_speed = row

    if total_records == 0:
        return {
            "vessel_id": vessel_id,
            "status": "not_found",
            "statistics": None,
        }

    statistics = {
        "total_records": total_records,
        "valid_records": valid_records,
        "average_speed": average_speed,
        "max_speed": max_speed,
    }

    if valid_records == 0:
        status = "no_valid_speed"
    else:
        status = "ok"

    return {
        "vessel_id": vessel_id,
        "status": status,
        "statistics": statistics,
    }


def list_ship_ids_sqlite(db_path: Path) -> list[str]:
    """
    查询数据库中所有不重复的船舶编号，并按编号升序返回。

    返回类型：list[str]。
    返回内容：船舶编号列表；表中没有记录时返回空列表。
    """
    db_path = db_path.resolve()

    if not db_path.is_file():
        raise FileNotFoundError("数据库不存在，请先运行建库脚本")

    database_uri = db_path.as_uri() + "?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)

    try:
        rows = connection.execute(
            """
            SELECT DISTINCT vessel_id
            FROM ship_speed_records
            ORDER BY vessel_id
            """
        ).fetchall()

    finally:
        connection.close()

    vessel_ids = [row[0] for row in rows]

    return vessel_ids

def main() -> None:
    """
    展示可查询的船舶编号，再查询用户指定船舶的航速统计。

    返回类型：None。
    返回内容：无；在终端展示编号列表和查询结果。
    """
    db_path = Path(__file__).parent / "data" / "ships_demo.db"

    vessel_ids = list_ship_ids_sqlite(db_path)

    print("可查询的船舶编号：", vessel_ids)

    if not vessel_ids:
        print("数据库中没有船舶记录。")
        return

    vessel_id = input("请输入船舶编号：")

    result = query_ship_speed_sqlite(
        db_path=db_path,
        vessel_id=vessel_id,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()