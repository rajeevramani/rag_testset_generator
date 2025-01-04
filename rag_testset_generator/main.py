
import logging
from typing import List
import pandas as pd
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
import bs4
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress library-specific logs
logging.getLogger("httpx").setLevel(logging.WARNING)


class RAGTestGenerator:
    def __init__(self, chat_model_name: str = "llama3.2", embedding_model_name: str = "embedding2.1", base_url: str = "http://localhost:11434", chunk_size: int = 350, chunk_overlap: int = 50):
        """
        Initialize the RAG Test Generator with different models for chat and embeddings.

        Args:
            chat_model_name (str): Name of the Ollama model to use for chat.
            embedding_model_name (str): Name of the Ollama model to use for embeddings.
            base_url (str): Base URL for Ollama API.
            chunk_size (int): Size of text chunks.
            chunk_overlap (int): Overlap between text chunks.
        """
        try:
            self.llm = ChatOllama(model=chat_model_name)
            self.embeddings = OllamaEmbeddings(
                model=embedding_model_name, base_url=base_url)
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self.chat_model_name = chat_model_name
            self.embedding_model_name = embedding_model_name
            logger.info("Initialized chat and embedding models successfully.")
        except Exception as e:
            logger.error(f"Error initializing chat or embedding models: {e}")
            raise

    def fetch_and_process_url(self, url: str) -> List[Document]:
        """
        Fetch content from a URL and split it into chunks.

        Args:
            url (str): The URL to fetch content from.

        Returns:
            List[Document]: List of text chunks as Document objects.
        """
        try:
            loader = WebBaseLoader(
                url,
                bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                    class_=("page-content")))
            )
            docs = loader.load()
            logger.info(f"Fetched {len(docs)} documents from URL.")
        except Exception as e:
            logger.error(f"Error fetching content from URL: {e}")
            return []

        return self.text_splitter.split_documents(docs)

    async def judge_llm(self, context: str, question: str, answer: str):
        critique_prompt = """
                You will be given a question, answer, and a context.
                Your task is to provide a total rating using the additive point scoring system described below.

                Evaluation Criteria:
                - Groundedness: Can the question be answered from the given context? Add 1 point if the question can be answered from the context.
                - Stand-alone: Is the question understandable free of any context, for someone with domain knowledge/Internet access? Add 1 point if the question is independent and can stand alone.
                - Faithfulness: The answer should be grounded in the given context. Add 1 point if the answer can be derived from the context.
                - Answer Relevance: The generated answer should address the actual question that was provided. Add 1 point if the answer actually answers the question.

                Provide your answer as follows:

                Answer:::
                Evaluation: (your rationale for the rating, as a text)
                Total rating: (your rating, as a number between 0 and 4)

                Now here are the question, answer, and context.

                Question: {question}\n
                Answer: {answer}\n
                Context: {context}\n
                Answer::: """
        messages = [
            {"role": "system", "content": "You are a neutral judge."},
            {
                "role": "user",
                "content": critique_prompt.format(
                    question=question, answer=answer, context=context
                ),
            },
        ]

        try:
            ai_msg = self.llm.invoke(messages)  # Removed 'await' here
            return ai_msg.content
        except Exception as e:
            logger.error(f"Error in judge_llm: {e}")
            return "Evaluation failed due to an error."

    def qa_generator(self, context: str):
        generator_prompt = """Your task is to write a factoid question and an answer given a context.
                            Your factoid question should be answerable with a specific, concise piece of factual information from the context.
                            Your factoid question should be formulated in the same style as questions users could ask in a search engine.
                            This means that your factoid question MUST NOT mention something like \"according to the passage\" or \"context\".

                            Provide your answer as follows:

                            Output:::
                            Factoid question: (your factoid question)
                            Answer: (your answer to the factoid question)

                            Now here is the context.

                            Context: {context}\n
                            Output:::"""

        messages = [
            ("system", "You are a question-answer pair generator."),
            ("user", generator_prompt.format(context=context)),
        ]
        try:
            ai_msg = self.llm.invoke(messages)
            return ai_msg.content
        except Exception as e:
            logger.error(f"Error in qa_generator: {e}")
            return ""

    async def generate_test_cases(self, docs: List[Document], num_test_cases: int):
        outputs = []

        async def process_document(doc):
            qa_output = self.qa_generator(doc.page_content)
            question, answer = self.parse_qa_output(qa_output)
            if question and answer:
                return {
                    "context": doc.page_content,
                    "question": question,
                    "answer": answer,
                    "source_doc": doc.metadata.get("source", "unknown"),
                }

        tasks = [process_document(doc) for doc in docs]
        for completed in asyncio.as_completed(tasks):
            try:
                result = await completed
                if result:
                    outputs.append(result)
            except Exception as e:
                logger.error(f"Error processing document: {e}")

        return outputs

    @staticmethod
    def parse_qa_output(qa_output: str):
        try:
            match = re.search(
                r"Factoid question: (.*?)Answer: (.*)", qa_output, re.S)
            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                return question, answer
            raise ValueError("Invalid QA output format")
        except Exception as e:
            logger.error(f"Error parsing QA output: {e}")
            return None, None


def write_dataframe_to_csv(dataframe, file_path, index=False):
    """
    Writes a Pandas DataFrame to a CSV file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to write.
        file_path (str): The file path for the CSV file.
        index (bool): Whether to write row indices to the CSV. Default is False.

    Returns:
        None
    """
    try:
        dataframe.to_csv(file_path, index=index)
        logger.info(f"DataFrame successfully written to {file_path}")
    except Exception as e:
        logger.error(
            f"An error occurred while writing the DataFrame to CSV: {e}")


async def main(url: str, chat_model_name: str = "llama2", embedding_model_name: str = "embedding2.1", base_url: str = "http://localhost:11434", num_test_cases: int = 10, chunk_size: int = 300, chunk_overlap: int = 50, output_file: str = "dataset.csv"):
    """
    Main function to generate RAG test cases from a URL using Ollama.

    Args:
        url (str): URL to generate test cases from.
        chat_model_name (str): Name of the Ollama model to use for chat.
        embedding_model_name (str): Name of the Ollama model to use for embeddings.
        base_url (str): Base URL for Ollama API.
        num_test_cases (int): Number of test cases to generate.
        chunk_size (int): Size of text chunks.
        chunk_overlap (int): Overlap between text chunks.
        output_file (str): Output CSV file path.
    """
    try:
        generator = RAGTestGenerator(
            chat_model_name, embedding_model_name, base_url, chunk_size, chunk_overlap)
        docs = generator.fetch_and_process_url(url)
        outputs_test_cases = await generator.generate_test_cases(docs, num_test_cases)

        dataset = []
        for output in outputs_test_cases:
            evaluation = await generator.judge_llm(
                context=output["context"],
                question=output["question"],
                answer=output["answer"],
            )
            try:
                score = float(evaluation.split("Total rating: ")[-1].strip())
                eval_text_match = re.search(
                    r"Evaluation:\s*(.+?)Total rating:", evaluation, re.S)
                eval_text = eval_text_match.group(1).strip(
                ) if eval_text_match else "Evaluation not found"
                output.update({"score": score, "eval": eval_text,
                              "chat_model": chat_model_name, "embedding_model": embedding_model_name})
                if score >= 4:
                    dataset.append(output)
            except Exception as e:
                logger.error(f"Error processing evaluation: {e}")

        df = pd.DataFrame(dataset)
        write_dataframe_to_csv(df, output_file)

        logger.info(f"RAG test cases generation complete. Results saved to {
                    output_file}.")
        return df
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate RAG test cases from a URL using Ollama")
    parser.add_argument("url", help="URL to generate test cases from")
    parser.add_argument("--chat_model", default="llama2",
                        help="Ollama chat model name (default: llama2)")
    parser.add_argument("--embedding_model", default="embedding2.1",
                        help="Ollama embedding model name (default: embedding2.1)")
    parser.add_argument(
        "--base_url", default="http://localhost:11434", help="Ollama API base URL")
    parser.add_argument("--num_test_cases", type=int,
                        default=10, help="Number of test cases to generate")
    parser.add_argument("--chunk_size", type=int, default=300,
                        help="Chunk size for text splitting")
    parser.add_argument("--chunk_overlap", type=int, default=50,
                        help="Chunk overlap for text splitting")
    parser.add_argument("--output_file", default="dataset.csv",
                        help="Path to output CSV file")

    args = parser.parse_args()
    asyncio.run(main(args.url, args.chat_model, args.embedding_model, args.base_url,
                args.num_test_cases, args.chunk_size, args.chunk_overlap, args.output_file))
