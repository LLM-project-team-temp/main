import warnings

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_upstage import ChatUpstage, UpstageLayoutAnalysisLoader

from utils import load_context, rag

warnings.filterwarnings("ignore")


# langchain, 1. llm define, 2. prompt define, 3. chain, 4. chain.invoke

# 1. define your favorate llm, solar
llm = ChatUpstage()

# 2. define chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (   "system",
            """
            You are a helpful assistant.
            너는 이제 mbti 전문가야 히히..
            아래에 줄 context를 참고해서 질문에 대답하렴!
            Use the following pieces of retrieved context to answer the question considering the history of the conversation.
            ---
            CONTEXT:
            {context}
            """
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

history = [
    HumanMessage("Do you know what MBTI I am? I am ISTJ!"),
    AIMessage("Great! I love their schedules!"),
]
latest_query = "What about ENFP?"
context = load_context()
retriever = rag(context)

# 3. define chain
chain = prompt | llm | StrOutputParser()

while (True):
    latest_query = input()
    if latest_query=='':
        break

    # 4. invoke the chain
    result_docs = retriever.invoke(latest_query)
    response = chain.invoke({"history": history,
                             "context": result_docs,
                             "input": latest_query})
    print(response, flush=True)

    history.append(HumanMessage(latest_query))
    history.append(AIMessage(response))
