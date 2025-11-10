# src/rag/answer_evaluator.py
import time
import json
from typing import Dict, Any, Optional, List
from jsonschema import validate, ValidationError
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from src.rag.utils import (
    strip_code_fences, safe_json_parse, coerce_types
)

# -----------------------
# Schema & Structured parser
# -----------------------
schemas = [
    ResponseSchema(name="improvement_feedback", description="Feedback for improving the answer"),
    ResponseSchema(name="score", description="Score between 0 and 10"),
    ResponseSchema(name="next_action", description="'retry_same_topic' if score < 8, else 'next_topic_question'"),
    ResponseSchema(name="strengths", description="List of strong points"),
    ResponseSchema(name="weaknesses", description="List of weak points"),
    ResponseSchema(name="follow_up_question", description="Follow-up question if score < 8, else null"),
    ResponseSchema(name="follow_up_ideal_answer", description="Ideal answer for the follow-up question or null")
]
parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "improvement_feedback": {"type": "string"},
        "score": {"type": ["number", "null"]},
        "next_action": {"type": "string", "enum": ["retry_same_topic", "next_topic_question"]},
        "strengths": {"type": ["array", "null"], "items": {"type": "string"}},
        "weaknesses": {"type": ["array", "null"], "items": {"type": "string"}},
        "follow_up_question": {"type": ["string", "null"]},
        "follow_up_ideal_answer": {"type": ["string", "null"]}
    },
    # required fields we considered are important for downstream logic
    "required": ["improvement_feedback", "next_action"]
}


def parse_and_normalize_json(text: str) -> Dict[str, Any]:
    """Pipeline: clean fences -> parse -> coerce field types."""
    cleaned = strip_code_fences(text)
    parsed = safe_json_parse(cleaned)
    if not isinstance(parsed, dict):
        return {}
    return coerce_types(parsed)

class AnswerEvaluator:
    """
    Robust answer evaluator.
      - includes rubric/context in system message each invocation,
      - makes at most `max_retries` main LLM eval calls,
      - attempt a single targeted repair if parsing completely fails,
      - return validated structured output.
    """

    def __init__(self, llm, debug: bool = False, max_retries: int = 2):
        """
        Args:
            llm: LangChain-compatible LLM wrapper with .invoke or chain.invoke semantics.
            debug: print debug logs
            max_retries: number of times to re-request the evaluation prompt (not repair)
        """
        self.llm = llm
        self.debug = debug
        self.max_retries = max_retries

    def _build_prompt(self, system_instruction: SystemMessage) -> ChatPromptTemplate:
        """
        Returns a ChatPromptTemplate that expects fields:
          question, user_answer, ideal_answer, rubric, context, history (optional)
        """
        # Keep system message external (we pass it at runtime)
        user_template = (
            "Question: {question}\n"
            "User's Answer: {user_answer}\n"
            "Ideal Answer: {ideal_answer}\n\n"
            # Include format instructions produced by StructuredOutputParser
            "{format_instructions}\n"
            "Respond in strict JSON format only."
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", "{system_message}"),
                ("user", user_template)
            ]
        )

    def _failure_fallback(self) -> Dict[str, Any]:
        return {
            "improvement_feedback": "Evaluation failed due to parsing or model error.",
            "follow_up_question": None,
            "follow_up_ideal_answer": None,
            "score": None,
            "strengths": None,
            "weaknesses": None,
            "next_action": "retry_same_topic"
        }

    def _attempt_repair(self, bad_output: str) -> Dict[str, Any]:
        """
        One targeted repair call: ask the LLM to return valid JSON only.
        This is used sparingly (only when parsing fails).
        """
        if self.debug:
            print("[DEBUG] Attempting repair of malformed LLM output.")
        
        # Escape curly braces to avoid LangChain template parsing errors
        safe_schema = json.dumps(EVALUATION_SCHEMA, indent=2).replace("{", "{{").replace("}", "}}")

        repair_prompt_template = (
            "The following output is intended to be a JSON object matching this schema:\n"
            f"{safe_schema}\n\n"
            "Please FIX and RETURN ONLY valid JSON (no explanation):\n\n"
            "{bad_output}"
        )
        # We call the LLM directly with a short system instruction to fix JSON
        repair_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a strict JSON fixer. Return only valid JSON matching the schema."),
            ("user", repair_prompt_template)
            ])
        
        repair_chain = repair_prompt | self.llm

        try:
            repair_response = repair_chain.invoke({"bad_output": bad_output})
            repair_text = getattr(repair_response, "content", str(repair_response))
            repaired = parse_and_normalize_json(repair_text)

            validate(instance=repaired, schema=EVALUATION_SCHEMA)
            if self.debug:
                print("[DEBUG] Repair succeeded after coercion.")
            return repaired

        except ValidationError as ve:
            if self.debug:
                print(f"[WARN] Repaired output still invalid: {ve}")
            return {}
        except Exception as e:
            if self.debug:
                print(f"[ERROR] Repair process failed: {e}")
            return {}
        

    def evaluate(
        self,
        question: str,
        user_answer: str,
        ideal_answer: str,
        rubric: List[str],
        context: str,
        system_instruction_message: Optional[SystemMessage] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate user_answer. Returns a dict matching EVALUATION_SCHEMA (keys present, some may be None).
        - `system_instruction_message`: a SystemMessage that embeds rubric + context; if None, then build a simple one.
        - `history`: optional list of previous exchanges (not included in schema, but useful to pass to model if needed).
        """
        # ensure system instruction is always included per-call (avoids drift)
        if system_instruction_message is None:
            system_text = (
                "You are an expert technical interviewer evaluating a candidate's response.\n"
                f"Rubric: {json.dumps(rubric, ensure_ascii=False)}\n"
                f"Context: {context}\n\n"
                "Be objective and factual. Do not invent new facts outside the provided context.\n"
                "Return only valid JSON that matches the format instructions."
            )
            system_instruction_message = SystemMessage(system_text)

        prompt_template = self._build_prompt(system_instruction_message)

        # prepare template mapping
        template_vars = {
            "system_message": str(system_instruction_message),
            "format_instructions": format_instructions,
            "question": question,
            "user_answer": user_answer,
            "ideal_answer": ideal_answer
        }

        result: Dict[str, Any] = {}
        raw_output = ""

        for attempt in range(self.max_retries):
            try:
                # Build and execute chain: template | llm
                chain = prompt_template | self.llm
                response = chain.invoke(template_vars)
                raw_output = response.content if hasattr(response, "content") else str(response)
                if self.debug:
                    print(f"==== Attempt {attempt+1} RAW OUTPUT ====\n{raw_output}\n===================")

                # Parse & normalize
                parsed = parse_and_normalize_json(raw_output)
                if not parsed:
                    # nothing parsed, so attempt a single repair
                    repaired = self._attempt_repair(raw_output)
                    if repaired:
                        result = repaired
                        break
                    else:
                        # failed repair; continue to next iteration (re-ask main prompt)
                        if self.debug:
                            print("[WARN] Repair did not yield valid JSON; retrying main prompt.")
                        time.sleep(1)
                        continue
                
                validate(instance=parsed, schema=EVALUATION_SCHEMA)
                result = parsed
                break  # success

            except ValidationError as ve:
                # Validation failed for the parsed dict -> try a repair (one-shot)
                if self.debug:
                    print(f"[WARN] ValidationError at attempt {attempt+1}: {ve}. Attempting repair.")
                repaired = self._attempt_repair(raw_output)
                if repaired:
                    result = repaired
                    break
                else:
                    if self.debug:
                        print("[WARN] Repair failed. Retrying main evaluation prompt.")
                    time.sleep(1)
                    continue
            except Exception as e:
                if self.debug:
                    print(f"[ERROR] attempt {attempt+1} exception: {e}")
                # on last attempt, return fallback
                if attempt == self.max_retries - 1:
                    result = self._failure_fallback()
                else:
                    time.sleep(1)
                continue

        # Ensure all schema keys exist (avoid KeyError downstream)
        for k in EVALUATION_SCHEMA["properties"].keys():
            result.setdefault(k, None)

        return result


# Quick manual test
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage as LCSystemMessage

    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    evaluator = AnswerEvaluator(llm, debug=True, max_retries=2)

    rubric = [
        "Defines what an API Gateway is and its core functions.",
        "Explains the role of an API Gateway in microservices architecture.",
        "Describes benefits like centralized routing, security, and monitoring."
    ]

    docs = []
    context = " ".join(docs)[:12000]

    system_msg = LCSystemMessage(
        f"""
        You are an expert technical interviewer. 
        Evaluate the following answer strictly based on:
            - The rubric (key ideas expected)
            - The context
            - The provided question
            - The ideal answer
            - The previous answers
        """ + 
        "\nRubric: " + json.dumps(rubric) + 
        "\nContext: " + context +
        f"""
            Be objective and factual. Do not invent new facts outside the provided context.
            Your task:
            1. Compare the user's answer to the ideal answer and rubric.
            2. Assign a score between 0 and 10 based on accuracy, completeness, and depth.
            3. Highlight strengths and weaknesses as bullet points.
            4. Provide one concise improvement feedback (1-2 sentences).
            5. If the score is below 8, propose a single follow-up question that helps the user strengthen the weakest area.
                The follow-up question must stay relevant to the context.
            7. Also if the score is below 8, provide the follow-up ideal answer of the follow-up question.
                The follow-up ideal answer must stay relevant to the context.
            8. If the score is 8 or higher, set follow_up_question to null and next_action to "next_question".
            9. Also if the score is 8 or higher, set follow_up_ideal_answer to null.
        """ + 
        f"""
            Return only valid JSON that matches the given format instructions.
            Do not include markdown, explanations, or any text outside the JSON object.
        """
    )

    res = evaluator.evaluate(
        question="Explain the purpose of an API Gateway in system design and describe how it functions within a microservices architecture.",
        user_answer="API gateway routes requests and handles auth and rate-limiting.",
        ideal_answer="An API Gateway acts as the single entry point for client requests and handles routing, auth, rate-limiting, logging, and monitoring.",
        rubric=rubric,
        context=context,
        system_instruction_message=system_msg
    )

    print(json.dumps(res, indent=2))

# Test this flow like this - 
# python -m src.rag.answer_evaluator