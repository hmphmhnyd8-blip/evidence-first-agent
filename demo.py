"""No API, no downloaded model: demonstrate real read-only SQLite contracts."""
import json
from pathlib import Path
from sqlite_repository import query_ship_speed_sqlite, list_ship_ids_sqlite

if __name__ == '__main__':
    db = Path(__file__).parent / 'data/ships_demo.db'
    print('演示数据，不是实时船位；此入口验证数据工具，不模拟 LLM 回答。')
    for vessel in [*list_ship_ids_sqlite(db), 'DEMO-999']:
        print(json.dumps(query_ship_speed_sqlite(db, vessel), ensure_ascii=False))
