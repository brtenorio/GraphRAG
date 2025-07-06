from dotenv import load_dotenv
import os
import json
import argparse
import ast
import re 
import logging
import sys

from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.schema import Document

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from typing import List, Dict, Any

sys.path.append("/Users/brncat/Downloads/NLP_practice/GraphRAG")
from relation_types import RELATION_TYPES, RelationType


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load from environment
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'


def output_parser(response: str, llm=None) -> str:
    model_name = getattr(llm, "model", "") if llm is not None else ""
    try:
        if model_name.lower() == "deepseek-r1:7b":
            response = re.sub(r"<think>.*?</think>\n?", "", response, flags=re.DOTALL)
        return ast.literal_eval(response)
    except Exception as e:
        logging.warning(f"output_parser: Failed to parse response: {e}\nResponse was: {response}")
        return None


class Neo4jSemanticRetriever:
    """Custom retriever that uses Neo4j semantic search"""
    
    def __init__(self, driver, model, cypher_query: str):
        self.driver = driver
        self.model = model
        self.cypher_query = cypher_query

    def _get_relevant_documents(self, query: str, k: int) -> List[Document]:
        """Retrieve relevant documents using Neo4j semantic search"""

        query_embedding = self.model.embed_query(query)
        
        with self.driver.session() as session:
            result = session.run(
                self.cypher_query,
                queryEmbedding=query_embedding,
                limit=k # number of documents to retrieve
            )
            
            documents = []
            for record in result:
                doc = Document(
                    page_content=record["summary"],
                    metadata={
                        "name": record["name"],
                        "score": record["score"],
                        "frequency": record["frequency"],
                        "document_id": record["document_id"],
                        "source": "neo4j_semantic_search"
                    }
                )
                if doc not in documents: # Ensure that each document will be unique
                    documents.append(doc)
                
        return documents


class Neo4jQAChat:
    """QA Chat system using Neo4j semantic search and LangChain"""
    
    def __init__(self, neo4j_driver, embedding_model, cypher_query: str, llm=None):
        self.driver = neo4j_driver
        self.model = embedding_model
        self.cypher_query = cypher_query
        self.llm = llm
        
        
        self.retriever = Neo4jSemanticRetriever(
            self.driver, 
            self.model,
            self.cypher_query,
        )

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Use the following pieces of context to answer the question at the end. 
            The context comes from a knowledge graph about a user's information.

            INSTRUCTIONS:
            - If context mentions confidence scores or relationship strengths, factor this into your certainty level.
            - Answer in a valid json format.
            - Answer to the questions objectively. 
            
            Context:
            {context}

            Format instructions:
            - Return your answer in the format List[str].
            - Do not return any additional text or thinking steps.
            - Output Example: ["item", "item", "item"]
            
            Question: {question}
            
            Answer: """
        )

    def ask(self, question: str, k: int) -> Dict[str, Any]:
        """Ask a question and get an answer with sources"""
        
        retrieved_docs = self.retriever._get_relevant_documents(question, k)
        context = "\n".join(["Document Source: " + doc.metadata["name"] + 
                            " Confidence Score: " + str(doc.metadata["score"]) + "\n" + 
                            doc.page_content for doc in retrieved_docs]
                            )
        prompt = self.prompt_template.format(context=context, question=question)
        chain = self.llm 
        answer = chain.invoke(prompt)
        answer = answer.content if hasattr(answer, 'content') else str(answer)
        answer = output_parser(answer, self.llm)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "name": doc.metadata["name"],
                    "score": doc.metadata["score"],
                    "frequency": doc.metadata["frequency"],
                    "document_id": doc.metadata["document_id"],
                    "summary": doc.page_content
                }
                for doc in retrieved_docs
            ]
        }

class RelationshipTypes:
    """
    Infers the relationship types from the question using an LLM.
    """
    def __init__(self, available_relation_types: List[RelationType], llm):
        self.llm = llm
        self.relation_types_list = "\n".join([f"- {rt.predicate}: {rt.description}" for rt in available_relation_types])

    def __call__(self, question: str):
        prompt_template = PromptTemplate(
        input_variables=["question", "relation_types"],
        template="""
        You are an expert Cypher query generator assistant.
        Your task is to identify the relevant relationship types from a predefined list to answer a user's question about their personal data.
        The user's question is: "{question}"

        Here is the list of available relationship types:
        {relation_types}

        Based on the user's question, which of these relationship types are most relevant to constructing a Cypher query to find the answer?
        
        Format instructions:
        - Return your answer in the format List[str].
        - Do not return any additional text or thinking steps.
        - Output Example: ["item", "item", "item"]

        Answer:
        """
        )

        prompt = prompt_template.format(question=question, relation_types=self.relation_types_list)
        result = self.llm.invoke(prompt)
        result = output_parser(result.content, self.llm)
        
        return result


def main(question: str, match_statment: str = None, k: int = 30, save_json: bool = False):
    """ inputs:
        question: a list os strings.
        match_statment: ex.: MATCH (n1)-[r]->(m)
        k: integrer for the maximum number of nodes = $limit
    """

    #llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    #llm = init_chat_model(model="claude-3-5-haiku-20241022", model_provider="anthropic", max_tokens=1024, temperature=0)
    llm = ChatOllama(model="deepseek-r1:7b", temperature=0)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=NEO4J_DATABASE)
    embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    # Infer relationship types from the first question
    if not match_statment:
        infer_relationship_types = RelationshipTypes(RELATION_TYPES, llm)
        inferred_relations = infer_relationship_types(question)
        print("Inferred relations: ", inferred_relations)
        if inferred_relations:
            match_statment = ":" + "|".join(inferred_relations)
        else:
            match_statment = ""

    
    cypher_query = f"""
        // Step 1: Get entities with degree >= 2 
        MATCH (e:Entity)
        WITH e, size([(e)--() | 1]) AS degree
        WHERE degree >= 2
        WITH collect(e) AS well_connected_entities

        // Step 2: Use only well-connected entities in main query
        MATCH (:USER)-[r{match_statment}]->(e)
        WHERE e IN well_connected_entities

        WITH type(r) AS relation_type, r.frequency as frequency, labels(e) AS entity_type, collect(e) AS matchedNodes, $queryEmbedding AS queryEmbedding
        ORDER BY frequency DESC
              CALL db.index.vector.queryNodes('entity_embedding', 1000, queryEmbedding)
              YIELD node AS matchedNode, score
        WHERE matchedNode IN matchedNodes AND score > 0.65
        RETURN matchedNode.name AS name, score, 
               matchedNode.extraction_result_summary AS summary, matchedNode.document_id AS document_id, frequency, relation_type, entity_type
        ORDER BY frequency * score DESC 
        LIMIT $limit
        """

    qa_chat = Neo4jQAChat(driver, embedding_model, cypher_query, llm=llm)

    dump_json = []
    result = qa_chat.ask(question, k)
    result["extracted relationships"] = match_statment
    dump_json.append(result)
    print(result['answer'])
    print(f"Sources used: {len(result['sources'])}")

    # Print sources for transparency
    for i, source in enumerate(result['sources'][:3]):  # Show top 3 sources
        print(f"  Source {i+1}: {source['name']} (score: {source['score']:.3f}, frequency: {source['frequency']})")
    
    if save_json:
        json_object = json.dumps(dump_json, indent=4, ensure_ascii=False)

        # Writing to sample.json
        with open("sample_Neo4jQA.json", "w", encoding="utf-8") as outfile:
            outfile.write(json_object)

    driver.close()

    return result

# CLI support
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Neo4j QA Chat with CLI support.")
    
    parser.add_argument(
        "-q", "--question",
        type=str,
        required=True, 
        help="List of questions to ask (space-separated)."
    )
    parser.add_argument(
        "-m", "--match_statment",
        type=str, 
        default="", 
        help="Cypher relation type, e.g., RELATED_TO (optional)."
    )
    parser.add_argument(
        "-k", "--k", 
        type=int, 
        default=30, 
        help="Maximum number of nodes to return (default: 30)."
    )
    parser.add_argument(
        "-s", "--save_json",
        action="store_true",
        help="Save a json file with the QA and metadata."
    )

    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(question=args.question, match_statment=args.match_statment, k=args.k, save_json=args.save_json)
