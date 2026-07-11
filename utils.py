"""
Small helpers shared between backend.py and frontend.py. Kept separate
since they're not "tools" (nothing here is bound to the LLM) and not
graph-orchestration logic either — just plain utility functions.
"""

import re


def get_text_content(content) -> str:
    """Normalizes content that might be a plain string or a list of
    {'type': 'text', 'text': ...} blocks (a shape some providers use)
    into a plain string. Used both when building the LLM's answer and
    when the frontend displays messages."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content) if content else ""


def extract_python_code(text: str):
    """Look for a ```python ... ``` fenced block in the LLM's reply text.
    Used by the text-based code-execution fallback in backend.py."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else None


# A locked-down set of builtins for the sandboxed exec() in backend.py's
# code_exec_node. Notably ABSENT: open, __import__, exec, eval, input,
# compile, getattr/setattr/delattr — so the LLM's code cannot touch the
# filesystem or escape the sandbox via reflection tricks.
SAFE_BUILTINS = {
    "print": print, "len": len, "range": range, "sum": sum, "min": min,
    "max": max, "sorted": sorted, "abs": abs, "round": round, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "enumerate": enumerate,
    "zip": zip, "str": str, "int": int, "float": float, "bool": bool,
    "type": type, "isinstance": isinstance,
}