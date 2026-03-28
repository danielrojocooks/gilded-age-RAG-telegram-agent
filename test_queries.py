"""Test the query engine with 3 sample questions about Gilded Age presidential dinners."""
from query_engine import get_query_engine

QUESTIONS = [
    "What dishes were typically served at Grover Cleveland's state dinners, and how were they presented?",
    "How did French service differ from Russian service at elite presidential dinners of the Gilded Age, and which style did the White House prefer?",
    "What wines and beverages accompanied a formal Gilded Age presidential banquet, and what were the rules of etiquette around their service?",
]

def main():
    print("Loading query engine…")
    engine = get_query_engine()
    print("Ready.\n" + "="*70 + "\n")

    for i, question in enumerate(QUESTIONS, 1):
        print(f"Q{i}: {question}")
        print("-" * 60)
        response = engine.query(question)
        print(str(response))
        print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
