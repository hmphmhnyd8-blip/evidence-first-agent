import csv

def load_records(file_path):
    with open(file_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        records = list(reader)

    for record in records:
        raw_speed = record["sog"]
        record["sog"] = float(raw_speed) if raw_speed != "" else None

    return records

def summarize_speed(records):
    valid_speeds = [
        record["sog"]
        for record in records
        if record["sog"] is not None
    ]

    result = {
        "total_records": len(records),
        "valid_records": len(valid_speeds),
        "average_speed": None,
        "max_speed": None,
    }

    # 有有效数据时才计算，避免除以零
    if valid_speeds:
        result["average_speed"] = sum(valid_speeds) / len(valid_speeds)
        result["max_speed"] = max(valid_speeds)

    return result

def get_ship_records(records, vessel_id):
    return [
        record
        for record in records
        if record["vessel_id"] == vessel_id
    ]

def query_ship_speed(records, vessel_id):
    ship_records = get_ship_records(records, vessel_id)

    if not ship_records:
        return {
            "vessel_id": vessel_id,
            "status": "not_found",
            "statistics": None,
        }

    statistics = summarize_speed(ship_records)

    if statistics["valid_records"] == 0:
        status = "no_valid_speed"
    else:
        status = "ok"

    return {
        "vessel_id": vessel_id,
        "status": status,
        "statistics": statistics,
    }
