"""Custom tools. With LangChain, a tool is just a typed, documented function
decorated with @tool: the name, docstring and type hints become the schema."""
import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo
from ddgs import DDGS

from langchain.tools import tool

import store

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _eval(node):
    """Safe evaluator: only numbers and arithmetic, never eval()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression such as '(12.5 * 4) / 3 ** 2'.
    Supports + - * / // % ** and parentheses."""
    return str(_eval(ast.parse(expression, mode="eval").body))


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time in an IANA timezone, e.g. 'Asia/Kolkata'."""
    return datetime.now(ZoneInfo(timezone)).strftime("%A, %Y-%m-%d %H:%M:%S %Z")


@tool
def save_note(text: str) -> str:
    """Save a note or fact for the user to the database so it can be recalled later."""
    return f"Saved note #{store.add_note(text)}"


@tool
def search_notes(query: str) -> str:
    """Search the user's saved notes by keyword."""
    rows = store.search_notes(query)
    if not rows:
        return "No matching notes."
    return "\n".join(f"#{r['id']} ({r['created_at']}): {r['text']}" for r in rows)

@tool
def web_search(query: str) -> str:
    """Search the web for current information. Returns titles, snippets and URLs."""
    results = DDGS().text(query, max_results=5)
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}\n{r['body']}\n{r['href']}" for r in results)

CUSTOM_TOOLS = [web_search, calculate, get_current_time, save_note, search_notes]
