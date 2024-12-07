from typing import List
import pandas as pd
from pydantic import BaseModel, Field
from .llm_factory import LLMFactory


class SynthesizedResponse(BaseModel):
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant had while synthesizing the answer"
    )
    sections: List[dict] = Field(
        description="List of sections, each containing a title and content",
        default_factory=list
    )
    key_points: List[str] = Field(
        description="List of key points extracted from the documents",
        default_factory=list
    )
    sources: List[str] = Field(
        description="List of document IDs used as sources",
        default_factory=list
    )
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )


# Define desired output structure of keyword checking 
class Keywords(BaseModel):
    content: list[str] = Field(description="List of keywords")


class Synthesizer:
    SYSTEM_PROMPT = """
    # Role and Purpose
    You are an AI assistant for a kommunalbanken document information retrieval system. Your task is to synthesize a structured 
    and clear answer based on the given question and relevant context from kommunalbanken documents.

    # Output Structure
    Your response should be organized as follows:
    1. Key Points: Bullet points of the most important information
    2. Detailed Sections: Break down the information into logical sections with clear headings
    3. Sources: Reference document IDs where the information came from. The id is the last part of the file name.
    

    # Guidelines:
    1. Provide a clear, concise, and informative answer to the question based on the documents.
    2. The context is retrieved based on relevance, so some information might be missing or less relevant.
    3. Use only the information from the relevant context to support your answer.
    4. Be transparent when there is insufficient information to fully answer the question.
    5. Do not make up or infer information not present in the provided documents.
    6. Maintain an objective and informative tone appropriate for financial advice.
    7. If there are multiple perspectives or conflicting information in the documents, present them objectively.
    
    Review the question from the user:
    """

    @staticmethod
    def generate_response(question: str, context: pd.DataFrame) -> SynthesizedResponse:
        """Generates a synthesized response based on the question and context from kommunalbanken documents.

        Args:
            question: The user's question.
            context: The relevant context retrieved from the kommunalbanken document database.

        Returns:
            A SynthesizedResponse containing thought process, answer, and context sufficiency.
        """
        context_str = Synthesizer.dataframe_to_json(
            context, columns_to_keep=["content"]
        )

        messages = [
            {"role": "system", "content": Synthesizer.SYSTEM_PROMPT},
            {"role": "user", "content": f"# User question:\n{question}"},
            {
                "role": "assistant",
                "content": f"# Retrieved information:\n{context_str}",
            },
        ]

        llm = LLMFactory("openai")
        return llm.create_completion(
            response_model=SynthesizedResponse,
            messages=messages,
        )

    @staticmethod
    def dataframe_to_json(
        context: pd.DataFrame,
        columns_to_keep: List[str],
    ) -> str:
        """
        Convert the context DataFrame to a JSON string.

        Args:
            context (pd.DataFrame): The context DataFrame.
            columns_to_keep (List[str]): The columns to include in the output.

        Returns:
            str: A JSON string representation of the selected columns.
        """
        return context[columns_to_keep].to_json(orient="records", indent=2)