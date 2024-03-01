from langchain.chat_models import ChatOpenAI  # use ChatOpenAI model
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
import torch


def prompts2response(prompts, md):
# send the argumented prompts to the llm model
    if md == 'gpt':
        model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
        response_text = model.predict(prompts)

    elif md == 'llama2':
        model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
 
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto",
            model_kwargs={"device": "cuda"}
        )
        generation_config = GenerationConfig.from_pretrained(model_id)
        generation_config.max_new_tokens = 1024
        generation_config.temperature = 0.1
        
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )
        
        llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.1})
        response_text = llm(prompts)

        
    return response_text