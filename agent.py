
import argparse
import sqlite3

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.sqlite import SqliteSaver

import store
from tools import CUSTOM_TOOLS

MODEL = "gemini-3.5-flash-lite"

SYSTEM = (
    "You are a helpful assistant. Use web_search for current information, "
    "the calculator for any arithmetic, and save_note / search_notes when the "
    "user wants you to remember or recall something."
)


def build_agent(checkpointer, model=None):
    model = model or ChatGoogleGenerativeAI(model=MODEL, max_tokens=4096)
    return create_agent(
        model=model,
        tools=CUSTOM_TOOLS,
        system_prompt=SYSTEM,
        checkpointer=checkpointer,   
    )


def run_turn(agent, conv_id: int, user_text: str) -> str:
    # thread_id is the key the checkpointer saves/loads state under.
    config = {"configurable": {"thread_id": str(conv_id)}, "recursion_limit": 25}
    final = ""
    for update in agent.stream(
        {"messages": [{"role": "user", "content": user_text}]},
        config,
        stream_mode="updates",
    ):
        for node_output in update.values():
            for msg in (node_output or {}).get("messages", []):
                if msg.type == "ai":
                    for call in msg.tool_calls:
                        print(f"  [tool] {call['name']}({call['args']})")
                    if msg.text:
                        final = msg.text
    return final


def print_history(agent, conv_id: int) -> None:
    config = {"configurable": {"thread_id": str(conv_id)}}
    for msg in agent.get_state(config).values.get("messages", []):
        label = {"human": "you", "ai": "agent", "tool": "tool"}.get(msg.type, msg.type)
        text = msg.text or ", ".join(f"{c['name']}({c['args']})" for c in getattr(msg, "tool_calls", []))
        if text:
            print(f"{label}> {text}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--resume", type=int)
    parser.add_argument("--history", type=int)
    args = parser.parse_args()

    if args.list:
        for c in store.list_conversations():
            print(f"#{c['id']}  {c['created_at']}  {c['title'] or ''}")
        return

    # Same SQLite file as store.py: the checkpointer adds its own tables.
    conn = sqlite3.connect(store.DB_PATH, check_same_thread=False)
    agent = build_agent(SqliteSaver(conn))

    if args.history:
        print_history(agent, args.history)
        return

    conv_id = args.resume or store.create_conversation()
    print(f"{'Resuming' if args.resume else 'Started'} conversation #{conv_id}. Type 'exit' to quit.\n")
    first = not args.resume
    while True:
        user_text = input("you> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue
        if first:
            store.set_title(conv_id, user_text[:60])
            first = False
        print("\nagent>", run_turn(agent, conv_id, user_text), "\n")


if __name__ == "__main__":
    main()
