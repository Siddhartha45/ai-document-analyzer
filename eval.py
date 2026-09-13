eval_dataset = [
    {
        "question": "How much is the home office equipment stipend?",
        "expected_answer_contains": ["$800"],
        "expected_source": "IT Equipment Policy",
    },
    {
        "question": "How long do I have to return my company laptop after my last day?",
        "expected_answer_contains": ["5 business days"],
        "expected_source": "IT Equipment Policy",
    },
    {
        "question": "What happens if I don't submit my stipend request by June 30, 2025?",
        "expected_answer_contains": ["next fiscal year"],
        "expected_source": "IT Equipment Policy",
    },
    {
        "question": "What is the company's policy on maternity leave?",
        "expected_answer_contains": ["don't have enough information"],
        "expected_source": None,
    },
]


from rag import answer_question
from database import SessionLocal


def run_evaluation(eval_dataset):
    results = []

    for case in eval_dataset:
        db = SessionLocal()
        try:
            result = answer_question(case["question"], db)
        finally:
            db.close()

        if isinstance(result, str):
            actual_answer = result
            actual_sources = []
        else:
            actual_answer = result["answer"]
            actual_sources = result["sources"]

        answer_check = all(
            phrase in actual_answer for phrase in case["expected_answer_contains"]
        )

        if case["expected_source"] is None:
            source_check = len(actual_sources) == 0
        else:
            source_check = case["expected_source"] in actual_sources

        results.append(
            {
                "question": case["question"],
                "answer_correct": answer_check,
                "grounding_correct": source_check,
                "actual_answer": actual_answer,
            }
        )

    return results


print(run_evaluation(eval_dataset))
