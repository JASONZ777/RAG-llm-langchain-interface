from langchain.chat_models import ChatOpenAI  # use ChatOpenAI model
import gradio as gr


def prompts2response(prompts):
# send the argumented prompts to the llm model
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    response_text = model.predict(prompts)

    return response_text