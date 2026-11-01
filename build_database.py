import sqlite3
from pathlib import Path

from ship_data import load_records


def build_database(records: list[dict], db_path: Path) -> int:
    """
    将船舶记录写入 SQLite 演示数据库。

    每次执行都用传入记录替换 ship_speed_records 表的数据，
    避免重复运行时重复导入。原始 CSV 不受影响。

    返回类型：int。
    返回内容：导入后表中的记录总数。
    """
    # 确保保存数据库的文件夹存在
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 将字典列表转换为待插入的元组列表
    rows = [
        (record["vessel_id"], record["sog"])
        for record in records
    ]

    # 数据库文件不存在时，connect 会创建它
    connection = sqlite3.connect(db_path)

    try:
        # 正常完成时提交数据修改，发生异常时回滚事务
        with connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS ship_speed_records (
                    id INTEGER PRIMARY KEY,
                    vessel_id TEXT NOT NULL,
                    sog REAL
                )
            """)

            # 这张表是由 CSV 生成的演示副本
            # 清空旧记录，再完整导入，防止重复累加
            connection.execute(
                "DELETE FROM ship_speed_records"
            )

            connection.executemany(
                """
                INSERT INTO ship_speed_records (vessel_id, sog)
                VALUES (?, ?)
                """,
                rows,
            )

            count_row = connection.execute(
                "SELECT COUNT(*) FROM ship_speed_records"
            ).fetchone()

            count = count_row[0]

    finally:
        # with connection 管理事务；连接仍需关闭
        connection.close()

    return count


def main() -> None:
    """
    读取现有 CSV，建立 SQLite 演示数据库并打印导入数量。

    返回类型：None。
    返回内容：无；在终端打印源记录数、数据库记录数和保存位置。
    """
    project_dir = Path(__file__).parent

    csv_path = project_dir / "ships.csv"
    db_path = project_dir / "data" / "ships_demo.db"

    records = load_records(csv_path)

    count = build_database(
        records=records,
        db_path=db_path,
    )

    print("CSV 记录数：", len(records))
    print("数据库记录数：", count)
    print("数据库保存位置：", db_path.resolve())


if __name__ == "__main__":
    main()