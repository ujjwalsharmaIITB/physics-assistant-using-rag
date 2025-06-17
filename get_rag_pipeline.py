from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.messages import SystemMessage

from get_vector_store import retreiver
from get_chat_llm import chat_model




# set up the conversation memory for the chat
memory = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True,  k=3)


retriever = retreiver



# Set up the system message to guide the model's behavior

system_message = SystemMessage(
    content="""You are a helpful physics assistant.
Provide only the answer in markdown and elaborate sufficiently.
"""
)



# putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=chat_model, retriever=retreiver, memory=memory)

## Uncomment the line below if you want to see the output of the chain in the console
# conversation_chain = ConversationalRetrievalChain.from_llm(llm=chat_model, retriever=retreiver, memory=memory, callbacks=[StdOutCallbackHandler()])

conversation_chain.combine_docs_chain.llm_chain.prompt.messages.insert(0, system_message)

## Uncomment the line below if you want to see the system prompt messages used by the chain
# print(conversation_chain.combine_docs_chain.llm_chain.prompt.messages)