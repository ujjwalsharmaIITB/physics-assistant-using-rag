import gradio as gr
from get_rag_pipeline import conversation_chain


def chat(question, history):
    result = conversation_chain.invoke({
        "question": question,
    })['answer'].split("<|end_header_id|>")[-1].strip()
    return result


gr.ChatInterface(chat, title="⚛️ Physics Assistant",
                        description="Ask questions about physics based on the Feynman Lectures.",
                        examples=["What is the principle of least action?",
                                  "Explain the concept of entropy in thermodynamics.",
                                  "How does quantum mechanics differ from classical mechanics?"],
                        theme="default", type="messages").queue().launch()