"""Manual single-turn smoke runner: python -m backend.app.rag.cli 'question'."""
import argparse
import asyncio
import sys
from backend.app.rag.pipeline import answer, startup, shutdown


async def run(question, program_code):
    try:
        await startup()
        async for event in answer(question, [], program_code):
            if event["type"] == "token":
                print(event["text"], end="", flush=True)
            elif event["type"] == "error":
                print(f"\n{event['code']}: {event['message']}", file=sys.stderr)
                return 1
        print()
        return 0
    finally:
        await shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--program-code")
    args = parser.parse_args()
    return asyncio.run(run(args.question, args.program_code))


if __name__ == "__main__":
    raise SystemExit(main())
