import os
from dotenv import load_dotenv
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_embeddings():
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key:
        from langchain_openai import OpenAIEmbeddings
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    elif google_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)
    else:
        raise ValueError("Nenhuma API key configurada. Defina OPENAI_API_KEY ou GOOGLE_API_KEY no .env")


def get_llm():
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-5-nano")
    elif google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    else:
        raise ValueError("Nenhuma API key configurada. Defina OPENAI_API_KEY ou GOOGLE_API_KEY no .env")


def get_vectorstore():
    embeddings = get_embeddings()
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
    )
    return vectorstore


def search_prompt(question=None):
    vectorstore = get_vectorstore()
    llm = get_llm()

    if question is None:
        return vectorstore, llm

    results = vectorstore.similarity_search_with_score(question, k=10)

    contexto = "\n\n".join([doc.page_content for doc, score in results])

    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)

    response = llm.invoke(prompt)

    return response.content
