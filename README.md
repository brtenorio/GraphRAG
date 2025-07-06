# GraphRAG

GraphRAG is a Python project for question answering (QA) over a user's personal knowledge graph stored in Neo4j. It leverages semantic search, relationship inference, and large language models (LLMs) to answer natural language questions using structured and unstructured data.

## Features

- **Semantic Retrieval:** Uses vector embeddings and Neo4j's vector index for semantic document retrieval.
- **Relationship Inference:** Infers relevant relationship types from user questions using an LLM.
- **Flexible QA Pipeline:** Integrates with HuggingFace and Ollama/LLMs for context-aware question answering.
- **CLI Support:** Run QA queries and export results via the command line.
- **Customizable Ontology:** Easily extend entity and relationship types in [`relation_types.py`](relation_types.py).

## Directory Structure

```
.
├── __init__.py
├── .env
├── relation_types.py
├── requirements.txt
└── Neo4jQA.py
```

- [`Neo4jQA.py`](Neo4jQA.py): Main script for QA, semantic retrieval, and CLI interface.
- [`relation_types.py`](relation_types.py): Defines entity and relationship types for the knowledge graph.
- [`sample_Neo4jQA.json`](sample_Neo4jQA.json): Example output of a QA run, including answers and source metadata.
- `.env`: Environment variables for Neo4j connection.
- `__init__.py`: Marks the directory as a Python package.

## Requirements

- Python 3.10+
- Neo4j database (with vector index enabled)
- [langchain](https://python.langchain.com/), [langchain_community](https://github.com/langchain-ai/langchain), [langchain_huggingface](https://github.com/langchain-ai/langchain), [langchain_openai](https://github.com/langchain-ai/langchain), [langchain_ollama](https://github.com/langchain-ai/langchain)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [neo4j Python driver](https://pypi.org/project/neo4j/)

Install dependencies:
```sh
pip install -r requirements.txt
```

## Setup

1. **Configure Neo4j Connection:**

   Edit the `.env` file with your Neo4j credentials:
   ```
   NEO4J_USERNAME
   NEO4J_URI
   NEO4J_PASSWORD
   NEO4J_DATABASE
   ```

2. **Start Neo4j** with the required schema and vector index.

3. **Run QA from CLI:**
   ```sh
   python Neo4jQA.py -q "who are the user's friends?" -k 30 -s
   ```

   - `-q`: The question to ask.
   - `-k`: Number of results to return (default: 30).
   - `-s`: Save the output as `sample_Neo4jQA.json`.

## Example Output

See [`sample_Neo4jQA.json`](sample_Neo4jQA.json) for an example of the QA output, including answers, sources, and extracted relationships.

## License

MIT License (add a `LICENSE` file if needed).

---
