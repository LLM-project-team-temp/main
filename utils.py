from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_upstage import UpstageEmbeddings, UpstageLayoutAnalysisLoader


def use_oracle_db(client):
    upstage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    vector_store = OracleVS(client=client, 
                            embedding_function=upstage_embeddings, 
                            table_name="text_embeddings2", 
                            distance_strategy=DistanceStrategy.DOT_PRODUCT)

    retriever = vector_store.as_retriever()
    return retriever


def load_context():
    context = ""
    
    # Let's load something big
    layzer = UpstageLayoutAnalysisLoader(
        "data/MBTI.pdf", output_type="html", use_ocr=True
    )
    # For improved memory efficiency, consider using the lazy_load method to load documents page by page.
    context = layzer.load()  # or layzer.load()
    return context

def rag(context):
    # RAG 1. load doc (done), 2. chunking, splits, 3. embeding - indexing, 4. retrieve

    # 2. Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(context)

    # 3. Embed & indexing
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=UpstageEmbeddings(model="solar-embedding-1-large")
    )
    
    # 4. retrive
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return retriever