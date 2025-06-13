import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool # Importar a anotação 'tool'

# Carregando as chaves de api: Bedrock e Tavily.
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AWS_MODEL = os.getenv("AWS_MODEL")
AWS_INFERENCE_PROFILE_ID = os.getenv("AWS_INFERENCE_PROFILE_ID")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_SECURITY_TOKEN = os.getenv("AWS_SECURITY_TOKEN") # Note: aws_security_token é um alias para aws_session_token
AWS_REGION = os.getenv("AWS_REGION")
MODEL_TEMPERATURE_STR = os.getenv("MODEL_TEMPERATURE", "0.7") # Default para 0.7
MODEL_TEMPERATURE = float(MODEL_TEMPERATURE_STR)

# Ferramenta que retorna a data/hora atual
@tool
def get_current_datetime() -> str:
    """
    Retorna a data e hora atuais no fuso horário local.
    Útil para perguntas que dependem do "agora", "hoje", "neste momento", etc.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Inicializando o Tavily
tavily_tool = TavilySearchResults(max_results=5, tavily_api_key=TAVILY_API_KEY)

# Lista de todas as ferramentas disponíveis para o agente
tools = [tavily_tool, get_current_datetime]

# Inicializando o Bedrock
llm = ChatBedrock(
    model_id=AWS_MODEL,
    inference_profile_id=AWS_INFERENCE_PROFILE_ID,
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
    aws_security_token=AWS_SECURITY_TOKEN,
    temperature=MODEL_TEMPERATURE
)

# System Prompt
template = """Você será um agente da Defesa Civil e será responsável por responder perguntas sobre a tábua de maré,
baseado na cidade e data fornecida pelo usuário.

Você tem acesso às seguintes ferramentas:

{tools}

Use o seguinte formato para as suas respostas:

Question: the input question you must answer
Thought: você sempre deve pensar no que fazer.
Se a pergunta do usuário precisar da data ou hora atual (ex: "hoje", "agora", "neste momento"), **use a ferramenta `get_current_datetime` primeiro para obter a informação temporal precisa**.
Formate a url onde terá sua respsota, siga o padrão: "https://pt.tideschart.com/<país>/<estado>/<cidade>#_mares", onde <país> será o nome do país em inglês, <estado> é o estado onde a Cidade se encontra e <cidade> conterá o nome da cidade que o usuário pediu
Para acessar informações necessaŕias no link formado no passo anterior, use a ferramenta `tavily_search_results`.
Não use nehum outro link além deste formado.
Confira se realmente as informações são exatamente do dia pedido.
Retone sempre uma tabela com o nível da maré e o horário e os links das fontes usadas.

Action: a ação a ser tomada, deve ser uma de [{tool_names}]
Action Input: a entrada para a ação
Observation: o resultado da ação
... (este Thought/Action/Action Input/Observation pode se repetir N vezes)
Thought: Eu agora sei a resposta final
Final Answer: a resposta final para a pergunta original

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
        user_input = input("\nSua pergunta (Se possível, informe a cidade e a data): ")
        if user_input.lower() == 'sair':
            break

        try:
            response = agent_executor.invoke({"input": user_input})
            print("\nResposta do Agente:")
            print(response["output"])
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            print("Por favor, verifique suas chaves de API e a conectividade.")