# rag_pipeline.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Load environment variables
load_dotenv()

# Step 2: Initialize LLM
llm_model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# Step 3: Retrieve Documents
def retrieve_docs(query, faiss_db):
    return faiss_db.similarity_search(query, k=4)

# Step 4: Combine Context
def get_context(documents):
    return "\n\n".join(doc.page_content for doc in documents)

# Step 5: Prompt Template
custom_prompt_template = """
You are an AI Legal Assistant.

Answer the user's question using only the provided PDF context.

If the answer is not available in the context, say:
"I don't know based on the provided document."

Do not make up information.

Question: {question}

Context:
{context}

Answer:
"""

# Step 6: Generate Answer
def answer_query(documents, model, query):
    context = get_context(documents)

    prompt = ChatPromptTemplate.from_template(custom_prompt_template)

    chain = prompt | model

    return chain.invoke({
        "question": query,
        "context": context
    })