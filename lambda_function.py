from agno.agent import Agent
# from agno.db.sqlite import SqliteDb
from agno.db.postgres import PostgresDb
import os
from agno.models.google import Gemini
from agno.models.groq import Groq
from dotenv import load_dotenv
from qdrant_client import qdrant_client, models
from sentence_transformers import SentenceTransformer
# from agno.tools.tavily import TavilyTools
from agno.tools.calculator import CalculatorTools

load_dotenv()

api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
qdrant_client = qdrant_client.QdrantClient(
    url=qdrant_url, 
    api_key=api_key,
)

def query_database(query: str) -> str:
    """
    Tool for querying the knowledge database about agricultural topics when needed

    Args: 
    query (str): Query to search in the knowledge database
    """

    print("Querying knowledge database...")

    query_vector = model.encode(query).tolist()

    points = qdrant_client.query_points(
        collection_name="sb100",
        with_vectors=False,
        limit=4,
        query=query_vector,
        using="vetor_denso"
    )

    return points

# WEB_SEARCH = False
# if WEB_SEARCH:
#     tools = [TavilyTools(), CalculatorTools(), query_database]
# else:
tools = [query_database, CalculatorTools()]

# PostgreSQL connection configuration from environment variables
postgres_config = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "user": os.getenv("POSTGRES_USER", "agno_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "agno_password_123"),
    "database": os.getenv("POSTGRES_DB", "agno_db"),
}

# Alternative: use POSTGRES_URL if provided
postgres_url = os.getenv("POSTGRES_URL")
if postgres_url:
    db = PostgresDb(db_url=postgres_url, memory_table="my_memory_table")
else:
    db = PostgresDb(
        host=postgres_config["host"],
        port=postgres_config["port"],
        user=postgres_config["user"],
        password=postgres_config["password"],
        database=postgres_config["database"],
        memory_table="my_memory_table"
    )

# Create a fixed agent instance to maintain consistency across runs  
agent = Agent(
        model=Groq(),
        # model=Groq(id="openai/gpt-oss-120b"),
        # model=Gemini(),
        instructions="""
        You are a helpful assistants to help farmers with soil recommendations.
        Use the knowledge_base when necessary to answer questions about soil fertilization.
        If you don't know the answer, just say you don't know. Do not try to make up an answer.
        If the answer needs to show specific data create tables for explanation.

        Keep your answers conversational and easy to understand but concise.
        """,
        tools=tools,
        db=db,
        enable_user_memories=True,
        debug_mode=True
    )

def agent_run(user_id: str, session_id: str, message: str):

    response = agent.run(input=message, user_id=user_id, session_id=session_id)

    return response

if __name__ == "__main__":
    user_id = "user_123"
    session_id = "session_456"
    
    # Test 1: Make a first question
    print("=== Primeira pergunta ===")
    message1 = "Meu nome é junior"
    response1 = agent_run(user_id, session_id, message1)
    print("Response 1:", response1)
    
    # print("\n=== Segunda pergunta (testando memória) ===")
    # # Test 2: Ask about the previous question to test memory
    # message2 = "Qual foi minha pergunta anterior?"
    # response2 = agent_run(user_id, session_id, message2)
    # print("Response 2:", response2)
    
    # print("\n=== Terceira pergunta ===")
    # # Test 3: Ask a follow-up question
    # message3 = "Qual é o meu nome?"
    # response3 = agent_run(user_id, session_id, message3)
    # print("Response 3:", response3)