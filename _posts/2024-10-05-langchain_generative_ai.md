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
1. Generic - A single chain is the simplest chain. It has two input parts:-
- Input Prompt
- Name of the LLM

It uses to generate the text. Let’s build a basic chain — create a prompt and get a prediction.

```
import os
from langchain.prompts import PromptTemplate
os.environ["OPENAI_API_KEY"] = "..."
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

Generix chains are used as building blocks for the Utility chains.

2. Utility:- It comprised of multiple chains to help solve the problems. One of the example is PalChain. PAL stands for Programme Aided Language Model. 

It reads  maths problem. --> Generates programs to solve the maths problem --> offloads the solution step to a runtime such as Python interpreter.

Again, it has two parts:-

- First uses a generic LLMChain to understand the query.
- Second uses Python REPL to solve the function/program outputted by the LLM.

from langchain.chains import PALChain
palchain = PALChain.from_math_prompt(llm=llm, verbose=True)
palchain.run("If my age is half of my dad's age and he is going to be 60 next year, what is my current age?")
```
# OUTPUT
# > Entering new PALChain chain...
# def solution():
#    """If my age is half of my dad's age and he is going to be 60 next year, what is my current age?"""
#    dad_age_next_year = 60
#    dad_age_now = dad_age_next_year - 1
#    my_age_now = dad_age_now / 2
#    result = my_age_now
#    return result
#
# > Finished chain.
# '29.5'
```
It has default prompt passed.
```
print(palchain.prompt.template)
```

**Building chains**
We have set up ourselves to build the chains. We need to build how we can pass output of one chain to another chain as input. We can use SimpleSequentialChain. One of the use case of Chains is to improve the OpenAI - ChatGPT as it is limited with recency of the information to build the models. 

We have similar concept to LLMs which is Agents. Agents have access to LLMs and suite of tools for example Google Search, Python REPL, math calculator, weather APIs, etc. One of the most common agent is zero-shot-react-description. This tool uses ReAct (Reason+Act) framework to pick the most usable tool. We need to initialize the agent and pass it the tools and LLM it needs. For example- we are using pal-math. We will pass the same LLM used before for initialization. 

```
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools

llm = OpenAI(temperature=0)
tools = load_tools(["pal-math"], llm=llm)

agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True)
```

Let’s test it out on the same example as above:
```
agent.run("If my age is half of my dad's age and he is going to be 60 next year, what is my current age?")
# OUTPUT
# > Entering new AgentExecutor chain...
# I need to figure out my dad's current age and then divide it by two.
# Action: PAL-MATH
# Action Input: What is my dad's current age if he is going to be 60 next year?
# Observation: 59
# Thought: I now know my dad's current age, so I can divide it by two to get my age.
# Action: Divide 59 by 2
# Action Input: 59/2
# Observation: Divide 59 by 2 is not a valid tool, try another one.
# Thought: I can use PAL-MATH to divide 59 by 2.
# Action: PAL-MATH
# Action Input: Divide 59 by 2
# Observation: 29.5
# Thought: I now know the final answer.
# Final Answer: My current age is 29.5 years old.

# > Finished chain.
# 'My current age is 29.5 years old.'
```
It takes one of the 3 steps to complete the tasks.
- Observation    
- Thought     
- Action    
This is mainly due to the ReAct framework and the associated prompt that the agent is using:
```
print(agent.agent.llm_chain.prompt.template)
# OUTPUT
# Answer the following questions as best you can. You have access to the following tools:
# PAL-MATH: A language model that is really good at solving complex word math problems. Input should be a fully worded hard word math problem.

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [PAL-MATH]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question
# Begin!
# Question: {input}
# Thought:{agent_scratchpad}
```

Agent lift is ability to use unknown chains which is generally not possible in the predetermined chain of calls to LLMs/other tools. For example, OpenAI has stale information as it is built using data till 2020. 

agent.run("My age is half of my dad's age. Next year he is going to be same age as Demi Moore. What is my current age?")

This can be easily fixed by including another tool —
tools = load_tools([“pal-math”, "serpapi"], llm=llm). serpapi is useful for answering questions about current events.

**Use case 1**     
Another example is podcast-api. 


tools = load_tools(["podcast-api"], llm=llm, listen_api_key="...")
agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True)

```
agent.run("Show me episodes for money saving tips.")

# OUTPUT
# > Entering new AgentExecutor chain...
# I should search for podcasts or episodes related to money saving
# Action: Podcast API
# Action Input: Money saving tips
# Observation:  The API call returned 3 podcasts related to money saving tips: The Money Nerds, The Rachel Cruze Show, and The Martin Lewis Podcast. These podcasts offer valuable money saving tips and advice to help people take control of their finances and create a life they love.
# Thought: I now have some options to choose from 
# Final Answer: The Money Nerds, The Rachel Cruze Show, and The Martin Lewis Podcast are great podcast options for money saving tips.

# > Finished chain.
```
It has two parts.
- First part is about creating the API URL based on our input instructions and making the API call. 
- Second part is about summerizing the response.

**Use case 2: Combine chains to create an age-appropriate gift generator**
We are going to create our own sequential chains based on our previous use case about age problems. We will combine them:-
- Chain #1:- The agent we just created that can solve age problems in math.
- Chain #2:- An LLM that takes the age of a person and suggests an appropriate gift for them.

```
# Chain1 - solve math problem, get the age
chain_one = agent

# Chain2 - suggest age-appropriate gift
template = """You are a gift recommender. Given a person's age,\n
 it is your job to suggest an appropriate gift for them.

Person Age:
{age}
Suggest gift:"""
prompt_template = PromptTemplate(input_variables=["age"], template=template)
chain_two = LLMChain(llm=llm, prompt=prompt_template) 
```
We can combine them using SimpleSequentialChain.
```
from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(
                  chains=[chain_one, chain_two],
                  verbose=True)
```

Below is the same math problem run again:-
```
question = "If my age is half of my dad's age and he is going to be 60 next year, what is my current age?"
overall_chain.run(question)

# OUTPUT
# > Entering new SimpleSequentialChain chain...


# > Entering new AgentExecutor chain...
# I need to figure out my dad's current age and then divide it by two.
# Action: PAL-MATH
# Action Input: What is my dad's current age if he is going to be 60 next year?
# Observation: 59
# Thought: I now know my dad's current age, so I can divide it by two to get my age.
# Action: Divide 59 by 2
# Action Input: 59/2
# Observation: Divide 59 by 2 is not a valid tool, try another one.
# Thought: I need to use PAL-MATH to divide 59 by 2.
# Action: PAL-MATH
# Action Input: Divide 59 by 2
# Observation: 29.5
# Thought: I now know the final answer.
# Final Answer: My current age is 29.5 years old.

# > Finished chain.
# My current age is 29.5 years old.

# Given your age, a great gift would be something that you can use and enjoy now like a nice bottle of wine, a luxury watch, a cookbook, or a gift card to a favorite store or restaurant. Or, you could get something that will last for years like a nice piece of jewelry or a quality leather wallet.

# > Finished chain.
```

We can also pass extra information to second chain using SimpleMemory. For example, lets add budget as the input variable. 

```
template = """You are a gift recommender. Given a person's age,\n
 it is your job to suggest an appropriate gift for them. If age is under 10,\n
 the gift should cost no more than {budget} otherwise it should cost atleast 10 times {budget}.

Person Age:
{output}
Suggest gift:"""
prompt_template = PromptTemplate(input_variables=["output", "budget"], template=template)
chain_two = LLMChain(llm=llm, prompt=prompt_template)
```

We need to careful about the output variable and it has been changed from age. 
```
print(agent.agent.llm_chain.output_keys)

# OUTPUT
["output"]
```

SimpleSequentialChain works with only single input and single output. Now we are using two inputs with budget as additional variable. Therefore, we need to use SequentialChain which can handle multiple multiple inputs and outputs.
```
overall_chain = SequentialChain(
                input_variables=["input"],
                memory=SimpleMemory(memories={"budget": "100 GBP"}),
                chains=[agent, chain_two],
                verbose=True)
```

We need to aware the input variable name used in the first agent, which can be found by inspecting from the code.

```
print(agent.agent.llm_chain.prompt.template)
# OUTPUT
#Answer the following questions as best you can. You have access to the following tools:
#PAL-MATH: A language model that is really good at solving complex word math problems. Input should be a fully worded hard word math problem.
#Use the following format:
#Question: the input question you must answer
#Thought: you should always think about what to do
#Action: the action to take, should be one of [PAL-MATH]
#Action Input: the input to the action
#Observation: the result of the action
#... (this Thought/Action/Action Input/Observation can repeat N times)
#Thought: I now know the final answer
#Final Answer: the final answer to the original input question
#Begin!
#Question: {input}
#Thought:{agent_scratchpad}
```
Finally, let’s run the new chain with the same prompt as before.

```
overall_chain.run("If my age is half of my dad's age and he is going to be 60 next year, what is my current age?")
# OUTPUT
#> Entering new SequentialChain chain...
#> Entering new AgentExecutor chain...
# I need to figure out my dad's current age and then divide it by two.
#Action: PAL-MATH
#Action Input: What is my dad's current age if he is going to be 60 next year?
#Observation: 59
#Thought: I now know my dad's current age, so I can divide it by two to get my age.
#Action: Divide 59 by 2
#Action Input: 59/2
#Observation: Divide 59 by 2 is not a valid tool, try another one.
#Thought: I can use PAL-MATH to divide 59 by 2.
#Action: PAL-MATH
#Action Input: Divide 59 by 2
#Observation: 29.5
#Thought: I now know the final answer.
#Final Answer: My current age is 29.5 years old.

#> Finished chain.
# For someone of your age, a good gift would be something that is both practical and meaningful. Consider something like a nice watch, a piece of jewelry, a nice leather bag, or a gift card to a favorite store or restaurant.\nIf you have a larger budget, you could consider something like a weekend getaway, a spa package, or a special experience.'}
#> Finished chain.
```
```
For someone of your age, a good gift would be something that is both practical and meaningful. Consider something like a nice watch, a piece of jewelry, a nice leather bag, or a gift card to a favorite store or restaurant.\nIf you have a larger budget, you could consider something like a weekend getaway, a spa package, or a special experience.'}
```

**Conclusion**
This is overview about LangChain can be used to build the applications using LLMs. We also shared the concept about agents and how it can used. There are lot more concepts about improving the applications using LangChain such as how to optimize memory so that we can be selective about the summaries of the conversations. I hope you had fun reading this article.

**References**
[1] https://towardsdatascience.com/a-gentle-intro-to-chaining-llms-agents-and-utils-via-langchain-16cd385fca81