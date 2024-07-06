from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_upstage import ChatUpstage, UpstageLayoutAnalysisLoader

from utils import load_context, rag


def chatmodel(add_to_system=None, add_history=None):
    # langchain, 1. llm define, 2. prompt define, 3. chain, 4. chain.invoke

    # 1. define your favorate llm, solar
    llm = ChatUpstage()

    # 2. define chat prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (   "system",
                f"""
                You are a helpful assistant.
                너는 이제 MBTI 전문가야.
                아래에 줄 context를 참고해서 질문에 대답하렴!
                Use the following pieces of retrieved context to answer the question considering the history of the conversation.
                {add_to_system}
                """
                +
                """
                ---
                CONTEXT:
                {context}
                """
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    history = []
    for human, ai in add_history:
        history.append(HumanMessage(content=human))
        history.append(AIMessage(content=ai))

    context = load_context()
    retriever = rag(context)

    # 3. define chain
    chain = prompt | llm | StrOutputParser()

    while True:
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
