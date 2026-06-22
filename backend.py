import os
from dotenv import load_dotenv

from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# ==================================================
# ENV
# ==================================================
load_dotenv()


# ==================================================
# LLM
# ==================================================
def demo_chatbot():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=api_key
    )


# ==================================================
# MEMORY
# ==================================================
def demo_memory():
    llm = demo_chatbot()

    return ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1000,
        memory_key="history",
        input_key="input",
        return_messages=False
    )


# ==================================================
# EMBEDDINGS
# ==================================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ==================================================
# PROMPT
# ==================================================
RAG_PROMPT = ChatPromptTemplate.from_template(
"""
You are a helpful AI assistant.

Answer ONLY from the retrieved PDF context.

Conversation History:
{history}

Retrieved Context:
{context}

User Question:
{input}

Rules:
1. Use only the retrieved context.
2. Do not use outside knowledge.
3. If answer is not present, say:
   "I could not find this information in the uploaded PDF."

Answer:
"""
)


# ==================================================
# PDF PROCESSING (NEW - IMPORTANT)
# ==================================================
def process_pdf(uploaded_file):
    """
    Takes Streamlit uploaded PDF and builds FAISS vector DB
    """

    # Save temporarily
    temp_path = "temp_uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # Create vector DB
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


# ==================================================
# QUESTION ANSWERING (NEW ARCHITECTURE)
# ==================================================
def ask_question(vectorstore, input_text, memory=None):
    """
    Main RAG function for Streamlit UI
    """

    try:
        print("STEP A: Processing query")

        llm = demo_chatbot()

        print("STEP B: LLM created")

        # Retriever (IMPORTANT FIX: per-session)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        docs = retriever.invoke(input_text)

        print("STEP C: Documents retrieved")

        context = "\n\n".join([doc.page_content for doc in docs])

        print("STEP D: Context built")

        # Memory (optional but supported)
        history = ""
        if memory:
            history = memory.load_memory_variables({}).get("history", "")

        print("STEP E: History loaded")

        messages = RAG_PROMPT.format_messages(
            history=history,
            context=context,
            input=input_text
        )

        print("STEP F: Prompt created")

        response = llm.invoke(messages)

        print("STEP G: LLM responded")

        answer = response.content

        # Save memory if enabled
        if memory:
            memory.save_context(
                {"input": input_text},
                {"output": answer}
            )

        print("STEP H: Done")

        return answer

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"