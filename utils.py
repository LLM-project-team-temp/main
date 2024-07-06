from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_upstage import UpstageEmbeddings, UpstageLayoutAnalysisLoader


def chat(chain, input, history, context):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    response = chain.invoke({"input": input,
                             "history": history_langchain_format,
                             "context": context})

    return response

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
    print("Splits:", len(splits))

    # 3. Embed & indexing
    vectorstore = Chroma.from_documents(documents=splits, embedding=UpstageEmbeddings(model="solar-embedding-1-large"))
    
    # 4. retrive
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    result_docs = retriever.invoke("What is Bug Classification?")
    print(len(result_docs))
    print(result_docs[0].page_content[:100])
    
    return retriever