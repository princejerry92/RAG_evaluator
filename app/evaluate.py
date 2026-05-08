import json
import os
import logging
from typing import List, Dict, Any
from app.retrieve import Retriever
from app.answer import AnswerGenerator

logger = logging.getLogger(__name__)


def judge_answer(generator: AnswerGenerator, question: str, answer: str, context: str) -> Dict[str, Any]:
    """
    Use the LLM as a judge to score answer quality.
    Returns relevance (1-5), faithfulness (1-5), and a short reason.
    """
    judge_prompt = """You are an evaluation judge. Score the following answer on two dimensions.
Return ONLY valid JSON with no extra text.

Dimensions:
- relevance (1-5): Does the answer directly and sufficiently answer the user question?
- faithfulness (1-5): Is the answer fully supported by the retrieved context?
- reason: One sentence explaining your scores.

Output format (strict JSON):
{"relevance": 4, "faithfulness": 5, "reason": "..."}"""

    user_content = f"""Question: {question}

Answer: {answer}

Retrieved Context: {context[:2000]}"""

    fallback = {"relevance": 3, "faithfulness": 3, "reason": "Judge scoring unavailable"}

    # Use provider abstraction from AnswerGenerator
    for provider in generator.providers:
        try:
            raw = provider.generate(judge_prompt, user_content)
            # Extract JSON from response (handle markdown fences)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(raw)
            return {
                "relevance": max(1, min(5, int(parsed.get("relevance", 3)))),
                "faithfulness": max(1, min(5, int(parsed.get("faithfulness", 3)))),
                "reason": str(parsed.get("reason", ""))[:200]
            }
        except Exception as e:
            logger.warning(f"Judge parse failed with {provider.get_name()}: {e}")
            continue

    return fallback


class Evaluator:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = AnswerGenerator()
        self.test_cases_path = "tests/eval_questions.json"

    def run_evaluation(self) -> Dict[str, Any]:
        if not os.path.exists(self.test_cases_path):
            return {"error": "Test cases file not found", "summary": {}, "details": []}

        with open(self.test_cases_path, "r") as f:
            test_cases = json.load(f)

        results = []
        total_p_at_3 = 0
        total_citation_accuracy = 0
        total_relevance = 0
        total_faithfulness = 0

        for case in test_cases:
            question = case["question"]

            # 1. Retrieval Precision@3
            retrieved_chunks = self.retriever.retrieve_context(question, k=3)
            p_at_3 = 1 if any(
                case["expected_url_pattern"] in chunk.get("source_url", "")
                for chunk in retrieved_chunks
            ) else 0
            total_p_at_3 += p_at_3

            # 2. Answer generation
            answer_result = self.generator.generate_answer(question, retrieved_chunks)

            # 3. Citation accuracy
            citation_correct = False
            if answer_result["citations"]:
                citation_correct = any(
                    case["expected_url_pattern"] in cit["url"]
                    for cit in answer_result["citations"]
                )
            if citation_correct:
                total_citation_accuracy += 1

            # 4. LLM-as-judge scoring
            context_text = "\n".join(c.get("content", "") for c in retrieved_chunks)
            judge_result = judge_answer(
                self.generator, question, answer_result["answer"], context_text
            )
            total_relevance += judge_result["relevance"]
            total_faithfulness += judge_result["faithfulness"]

            results.append({
                "question": question,
                "retrieval_success": bool(p_at_3),
                "citation_accuracy": citation_correct,
                "relevance_score": judge_result["relevance"],
                "faithfulness_score": judge_result["faithfulness"],
                "judge_reason": judge_result["reason"],
                "answer_preview": answer_result["answer"][:150] + "..."
            })

        count = len(test_cases)
        return {
            "summary": {
                "precision_at_3": round(total_p_at_3 / count, 2) if count > 0 else 0,
                "citation_accuracy": round(total_citation_accuracy / count, 2) if count > 0 else 0,
                "avg_relevance_score": round(total_relevance / count, 2) if count > 0 else 0,
                "avg_faithfulness_score": round(total_faithfulness / count, 2) if count > 0 else 0,
                "total_questions": count
            },
            "details": results
        }


if __name__ == "__main__":
    evaluator = Evaluator()
    print(json.dumps(evaluator.run_evaluation(), indent=2))
