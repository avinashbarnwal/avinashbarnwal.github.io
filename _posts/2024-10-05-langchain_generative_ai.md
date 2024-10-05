---
title: "LangChain - Generative AI"
collection: posts
type: "Data Science"
permalink: /posts/langchain
date: 2024-10-05
---
**What it does** - It is a framework that allows developers to create applications using LLMs.      
**How it works** - It combines external components with LLMs. It can also add context and memory to existing LLMs, which can help them complete more complex tasks.      
**Features** - It includes centralized development environment, module-based approach, and ability to compare different prompts and foundation models.     
**Integration** - It integrates with other tools and frameworks, including Couchbase, a high-performance NoSQL database.     

**How to build applications**    

Lets Start building a simple prototype or use case of the Langchain      
1. Generic

```
import os
os.environ["OPENAI_API_KEY"] = "..."
Let’s build a basic chain — create a prompt and get a prediction
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
print(prompt.format(product="podcast player"))
```
```
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(
          model_name="text-davinci-003", # default model
          temperature=0.9) #temperature dictates how whacky the output should be
llmchain = LLMChain(llm=llm, prompt=prompt)
llmchain.run("podcast player")
```
```
from langchain.chat_models import ChatOpenAI
chatopenai = ChatOpenAI(
                model_name="gpt-3.5-turbo")
llmchain_chat = LLMChain(llm=chatopenai, prompt=prompt)
llmchain_chat.run("podcast player")
```

2. 
