 # src/rag/question_generator.py

import os
import random
import re
import json
from typing import List, Dict, Optional
from src.database.postgres_handler import PostgresHandler

class QuestionGenerator:
    def __init__(self, pg: PostgresHandler, llm):
        """
        db: PostgresHandler instance
        llm_client: Object with method .generate(prompt: str) -> str | dict
        """
        self.pg = pg
        self.llm = llm

    # -------------------------------
    #  1. Topic Selection
    # -------------------------------
    def select_topics(self, strategy: str = "random", limit: int = 1) -> Optional[Dict]:
        """
        Select topics from topic_metadata table.
        """
        if strategy == "random":
            query = f"SELECT id as topic_id, name as topic_name FROM topic_metadata ORDER BY RANDOM() LIMIT {limit};"
        elif strategy == "curriculum":
             query = f"SELECT id as topic_id, name as topic_name FROM topic_metadata ORDER BY RANDOM() LIMIT {limit};"
        else:
            raise ValueError("Invalid strategy. Use 'random' or 'curriculum'.")

        data = self.pg.fetch_all(query)
        if not data:
            print("No topics found.")
            return None
        topic = data[0]
        if not topic:
            print("No topics found.")
        return topic

    # -------------------------------
    #  2. Retrieve Context Docs
    # -------------------------------
    def fetch_topic_documents(self, topic_id: int, num_docs:int) -> str:
        """
        Fetch all document chunks for a topic and concatenate them.
        """
        query = """
        SELECT text FROM topic_documents WHERE topic_id = %s
        ORDER BY url, document_id limit %s;
        """
        docs = self.pg.fetch_all(query, (topic_id,num_docs,))
        if not docs:
            print(f" No documents found for topic_id={topic_id}")
            return ""

        return [row["text"] for row in docs]

    # -------------------------------
    # 3. Generate Initial Questions
    # -------------------------------
    def generate_questions(self, docs: list, num_questions: int = 5) -> List[Dict]:
        """
        Use LLM to generate questions based on context.
        Returns a JSON list of {question, type}.
        """

        QUESTION_GENERATION_PROMPT = """
            You are an expert technical interviewer.      
            Given the following topic material, generate {num_questions} insightful and conceptually clear interview question sets.

            Each question set must include:
            1. A single clear question that tests understanding of the topic.
            2. The difficulty level: easy, medium, or hard.
            3. A short rubric: 3-5 bullet points that describe the key ideas or components an ideal answer should include.
            4. An ideal answer: concise, factually correct, and based only on the provided context (no external assumptions).

            Each question set should be:
            - Relevant to the topic and context provided
            - Designed to test understanding of key ideas
            - Clear and self-contained
            - Increasing gradually in difficulty
            - If only one question is requested, choose an appropriate difficulty based on the topic depth

            If the context is long, focus on the most relevant portions to craft meaningful questions.

            Return the output in strict JSON format as a list of objects:
            [
                {{
                    "question": "<question text>",
                    "difficulty": "<easy|medium|hard>",
                    "rubric": ["<point1>", "<point2>", "<point3>"],
                    "ideal_answer": "<ideal answer text>"
                }}
            ]

            Do not include any explanations or commentary outside the JSON.

            Topic Material:
            {context}
        """
        # Step 1: Prepare context safely
        print(f"Context docs- {docs}")
        context = " ".join(docs)
        if len(context) > 12000:
            context = context[:12000] + "..."  # truncate if needed
            print(f"[INFO] Context length: {len(context)} characters")


        prompt = QUESTION_GENERATION_PROMPT.format(
            num_questions=num_questions,
            context=context
        )

        # Step 2: Invoke LLM
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response).strip()

        # Step 3: Attempt to parse LLM JSON output safely ----
        questions = []
        try:
            questions = json.loads(response_text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if match:
                try:
                    questions = json.loads(match.group(0))
                except json.JSONDecodeError:
                    print("[WARN] Partial JSON detected, falling back to minimal output.")
                    questions = [{"question": response_text[:300], "difficulty": "unknown"}]
            else:
                print("[ERROR] No valid JSON structure found in LLM response.")
                questions = [{"question": response_text[:300], "difficulty": "unknown"}]

        # Step 4: Validate and sanitize ----
        clean_questions = []
        for q in questions:
            clean_questions.append({
                "question": q.get("question", "").strip(),
                "difficulty": q.get("difficulty", "unknown"),
                "rubric": q.get("rubric", []),
                "ideal_answer": q.get("ideal_answer", "").strip()
            })

        print(f"Generated {len(clean_questions)} question(s).")
        return clean_questions
        

    # -------------------------------
    # Full Pipeline
    # -------------------------------
    def run(self, strategy="random", num_docs=5, num_questions=5):
        topic = self.select_topics(strategy="random", limit=1)
        if not topic:
            return None

        print(f"Selected Topic: topic id - {topic['topic_id']} topic name - ({topic['topic_name']})")

        docs = self.fetch_topic_documents(topic_id=topic["topic_id"], num_docs=num_docs)
        if not docs:
            return None

        questions = self.generate_questions(docs=docs, num_questions=num_questions)
        return {
            "topic": topic,
            "questions": questions
        }


if __name__ == "__main__":
    import os
    import time
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from src.database.postgres_handler import PostgresHandler
    load_dotenv()
    conn_params = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD")
    }
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    pg_handler = PostgresHandler(conn_params)
    llm = ChatOpenAI(model='gpt-4o')

    print("Starting process...")
    start_time = time.time()

    qg = QuestionGenerator(pg_handler, llm)
    result = qg.run(strategy="random", num_docs = 5, num_questions=1)

    print("------Result-------")
    print(result)

    elapsed = time.time() - start_time
    print(f"Process completed in {elapsed:.2f} seconds.")

# Test this flow like this - 
# python -m src.rag.question_generator