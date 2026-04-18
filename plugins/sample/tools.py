from langchain_core.tools import tool


@tool
def echo(text: str) -> str:
    """Echo back the input text. Useful for testing."""
    return f"[echo] {text}"


@tool
def current_timestamp() -> str:
    """Get the current local timestamp in ISO format (Asia/Shanghai, UTC+8)."""
    from datetime import datetime
    import zoneinfo

    tz = zoneinfo.ZoneInfo("Asia/Shanghai")
    now = datetime.now(tz)
    return now.isoformat()
