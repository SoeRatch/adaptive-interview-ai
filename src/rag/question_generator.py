 # src/rag/question_generator.py

import os
import random
import re
import json
from typing import List, Dict, Optional
from textwrap import shorten
from src.database.postgres_handler import PostgresHandler
from langchain_openai import ChatOpenAI

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
            You are an expert technical interviewer. Based on the following topic material, 
            generate {num_questions} insightful and conceptually clear interview questions.

            Each question should be:
            - Relevant to the topic and context provided
            - Designed to test understanding of key ideas
            - Clear and self-contained
            - Increasing gradually in difficulty

            Return the output in JSON format as a list of objects:
            [
            {{ "question": "<question text>", "difficulty": "<easy|medium|hard>" }}
            ]

            Topic Material:
            {context}
        """
        context = " ".join(docs)
        context = shorten(context, width=6000, placeholder="...")

        prompt = QUESTION_GENERATION_PROMPT.format(
            num_questions=num_questions,
            context=context
        )

        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        print(response_text)

        try:
            questions = json.loads(response)
        except Exception as e:
            print(f"Failed to parse LLM response as JSON: {e}")
            match = re.search(r"\[.*\]", response_text, re.DOTALL)
            questions = json.loads(match.group(0)) if match else [{"question": response_text, "difficulty": "unknown"}]

        print(f"Generated {len(questions)} questions")
        return questions

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
    result = qg.run(strategy="random", num_docs = 5, num_questions=5)
    print(result)

    elapsed = time.time() - start_time
    print(f"Process completed in {elapsed:.2f} seconds.")

# Test this flow like this - 
# python -m src.rag.question_generator