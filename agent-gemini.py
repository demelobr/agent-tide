import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults


# Carregando as chaves de api: Gemini e Tavily.
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL")
MODEL_TEMPERATURE = os.getenv("MODEL_TEMPERATURE")


# Inicializando o Tavily
tavily_tool = TavilySearchResults(max_results=5, tavily_api_key=TAVILY_API_KEY)
tools = [tavily_tool]

# Inicializando o Gemini
llm = ChatGoogleGenerativeAI(
    model=GOOGLE_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=MODEL_TEMPERATURE
)

# System Prompt
template = """Você será um agente da Defesa Civil e será responsável por responder perguntas sobre a tábua de maré,
baseado na cidade e data fornecida pelo usuário.

Você tem acesso às seguintes ferramentas:

{tools}

Use o seguinte formato para as suas respostas:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# Criando o agente ReAct
agent = create_react_agent(llm, tools, prompt)

# Criando o agente executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Main
if __name__ == "__main__":
    print("Bem-vindo ao Agente da Defesa Civil para Tábua de Marés!")
    print("Digite 'sair' para encerrar.")

    while True:
        user_input = input("\nSua pergunta(informe a cidade e a data): ")
        if user_input.lower() == 'sair':
            break

        try:
            response = agent_executor.invoke({"input": user_input})
            print("\nResposta do Agente:")
            print(response["output"])
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            print("Por favor, verifique suas chaves de API e a conectividade.")