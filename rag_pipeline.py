from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

CHUNK_SIZE=500
CHUNK_OVERLAP = 80
TOP_K_CHUNKS = 4

PROMPT_TEMPLATE = """You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I couldn't find this information in the document."

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def load_and_split(pdf_path: str) -> list:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n","\n","."," "]
    )

    chunks = splitter.split_documents(pages)
    return chunks

def build_vector_store(pdf_path: str, hf_token: str) -> FAISS:
    chunks = load_and_split(pdf_path)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True}
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def ask_question(question: str, vector_store: FAISS, hf_token: str) -> dict:
    retriever = vector_store.as_retriever(
        search_kwargs={"k": TOP_K_CHUNKS}
    )

    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    client = InferenceClient(
        model="Qwen/Qwen2.5-72B-Instruct",
        token=hf_token
    )
    
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )

    answer = response.choices[0].message.content
    sources = [doc.page_content for doc in docs]
    return {"answer": answer, "sources": sources}