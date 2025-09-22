import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.output_parsers import StrOutputParser

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda de forma natural e conversacional com base no CONTEXTO.
- Formate sua resposta de maneira clara e amigável.
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

EXEMPLOS DE RESPOSTAS DESEJADAS:
Pergunta: "Qual o faturamento da empresa?"
Resposta: "O faturamento da empresa foi de R$ 85.675.568,77."

Pergunta: "Quantos funcionários a empresa tem?"
Resposta: "A empresa possui 150 funcionários."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO" DE FORMA NATURAL E CONVERSACIONAL:
"""


def search_prompt(question=None):
    load_dotenv()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL"), api_key=os.getenv("GOOGLE_API_KEY")
    )

    store = PGVector(
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        embeddings=embeddings,
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    results = store.similarity_search_with_score(question, k=1)

    # if results:
    # Get the top result (highest score - lowest distance)
    top_doc, top_score = results[0]

    # print("=" * 50)
    # print(f"Top Result (score: {top_score:.2f}):")
    # print("=" * 50)

    # print("\nTexto:\n")
    # print(top_doc.page_content.strip())

    # Initialize the LLM
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.5,
    )

    # Use the PROMPT_TEMPLATE with context and question
    formatted_prompt = PROMPT_TEMPLATE.format(
        contexto=top_doc.page_content.strip(), pergunta=question
    )

    # Get response from LLM
    response = model.invoke(formatted_prompt)

    return response.content

    # return None
