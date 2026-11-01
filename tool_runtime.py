import json
import traceback
from datetime import datetime
from pathlib import Path

def save_tool_log(tool_name: str, tool_args: dict, tool_result: dict) -> None:
    """
    将一次工具调用记录追加到本地日志文件。

    返回类型：None。
    返回内容：无；将时间、工具名称、参数和结果写入文件。
    """
    # 在当前脚本旁创建 logs 文件夹；已存在时不报错
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # 组装本次调用的记录
    entry = {
        "time": datetime.now().astimezone().isoformat(),
        "tool_name": tool_name,
        "args": tool_args,
        "result": tool_result,
    }

    # JSONL 格式：每行保存一个完整的 JSON 对象
    log_path = log_dir / "tool_calls.jsonl"

    # "a" 表示追加写入，保留文件里已有的记录
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")

def execute_tool(tool_call: dict, tools_by_name: dict) -> dict:
    """
    执行一个工具请求。

    返回类型：dict。
    返回内容：正常时返回工具结果；
    执行失败时返回 status 为 tool_error 的错误信息。
    """
    tool_name = tool_call["name"]
    selected_tool = tools_by_name.get(tool_name)

    if selected_tool is None:
        return {
            "status": "tool_error",
            "error_type": "unknown_tool",
            "message": "请求的工具未注册，未执行任何操作。",
        }

    try:
        return selected_tool.invoke(tool_call["args"])

    except FileNotFoundError:
        traceback.print_exc()

        return {
            "status": "tool_error",
            "error_type": "data_file_missing",
            "message": "工具所需的本地文件或目录不存在，请检查配置。",
        }

    except Exception as exc:
        traceback.print_exc()

        return {
            "status": "tool_error",
            "error_type": type(exc).__name__,
            "message": (
                "工具执行失败，本次没有取得有效结果。"
                "请检查终端错误信息，不要反复重试同一失败操作。"
            ),
        }
