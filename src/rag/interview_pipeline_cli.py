# src/rag/interview_pipeline_cli.py

from typing import Dict, Any
from src.rag.question_generator import QuestionGenerator
from src.rag.answer_evaluator import AnswerEvaluator
from src.database.postgres_handler import PostgresHandler
from langchain_openai import ChatOpenAI
import os


class InterviewPipelineCLI:
    def __init__(self, pg: PostgresHandler, llm):
        self.pg = pg
        self.llm = llm
        self.qg = QuestionGenerator(pg, llm)
        self.evaluator = AnswerEvaluator(llm)
        self.max_followups = 3
        self.score_threshold = 8.0
        self.conversation_history = []

    def run_once(self, num_docs: int = 5) -> Dict[str, Any]:
        """Run one interactive interview session in CLI."""

        # 1) Select topic
        topic = self.qg.select_topics(limit=1)
        if not topic:
            raise RuntimeError("No topic available.")

        print(f"\n Topic: {topic['topic_name']} (id={topic['topic_id']})")

        # 2) Fetch docs
        docs = self.qg.fetch_topic_documents(topic_id=topic["topic_id"], num_docs=num_docs)
        if not docs:
            raise RuntimeError("No documents found for topic.")

        # 3) Generate initial question
        questions = self.qg.generate_questions(docs=docs, num_questions=1)
        if not questions:
            raise RuntimeError("No question generated.")
        q = questions[0]
        rubric = q.get("rubric")
        context = " ".join(docs)

        print("\n=== INITIAL QUESTION ===")
        print(q["question"])
        print("\nRubric:", rubric)
        print("\n(Type your answer and press Enter)\n")

        final_evaluation = None

        # 4) Follow-up loop (up to max_followups)
        for attempt in range(self.max_followups):
            user_answer = input("Your answer: ")

            # store in conversation history
            self.conversation_history.append({
                "question": q["question"],
                "ideal_answer": q["ideal_answer"],
                "user_answer": user_answer
            })

            # Evaluate response
            evaluation = self.evaluator.evaluate(
                question=q["question"],
                user_answer=user_answer,
                ideal_answer=q["ideal_answer"],
                rubric=rubric,
                context=context
            )

            score = evaluation.get("score", 0)
            feedback = evaluation.get("improvement_feedback", "")
            follow_up_question = evaluation.get("follow_up_question")
            follow_up_ideal_answer = evaluation.get("follow_up_ideal_answer")

            print(f"\nEvaluation (Attempt {attempt+1}): Score = {score}")
            print("Feedback:", feedback)

            # Check if threshold met
            if score and score >= self.score_threshold:
                print("\nExcellent! You've met the expected proficiency level.")
                final_evaluation = evaluation
                break

            # If not met, and we have attempts left, generate a follow-up
            if follow_up_question:
                print("\n--- FOLLOW-UP QUESTION ---")
                print(follow_up_question)
                q["question"] = follow_up_question
                q["ideal_answer"] = follow_up_ideal_answer
            else:
                print("\nReached max attempts without meeting score threshold.")
                final_evaluation = evaluation

        # 5) Return results
        return {
            "topic": topic,
            "final_question": q["question"],
            "final_evaluation": final_evaluation,
            "conversation_history": self.conversation_history
        }


if __name__ == "__main__":
    import json
    from dotenv import load_dotenv

    load_dotenv()
    conn_params = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }

    pg = PostgresHandler(conn_params)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    pipeline = InterviewPipelineCLI(pg, llm)
    result = pipeline.run_once()
    print("\n==== FINAL RESULT ====")
    print(json.dumps(result, indent=2))

# Test this flow like this - 
# python -m src.rag.interview_pipeline_cli