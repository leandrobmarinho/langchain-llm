from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv, dotenv_values
from langchain_core.output_parsers import StrOutputParser

if __name__ == "__main__":
    load_dotenv()
    config = dotenv_values(".env")

    print("Hello, World!")

    information = """"
    William Henry Gates III (born October 28, 1955) is an American businessman and philanthropist best known for co-founding the software company Microsoft with his childhood friend Paul Allen. He later held the positions of chairman, chief executive officer (CEO), president, and chief software architect of the company. Gates was also its largest individual shareholder until May 2014.[a] He was a pioneer of the microcomputer revolution of the 1970s and 1980s.

    In June 2008, Gates transitioned into a part-time role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation he and his then-wife Melinda had established in 2000. He stepped down as chairman of the Microsoft board in February 2014 and assumed the role of technology adviser to support newly appointed CEO Satya Nadella. In March 2020, Gates left his board positions at Microsoft and Berkshire Hathaway to focus on his philanthropic efforts on climate change, global health and development, and education.

    Gates was born and raised in Seattle, Washington. In 1975, he and Allen founded Microsoft in Albuquerque, New Mexico. Gates led the company as its chairman and chief executive officer until stepping down as CEO in January 2000, succeeded by Steve Ballmer, but he remained chairman of the board of directors and became chief software architect. During the late 1990s, he was criticized for his business tactics, which were considered anti-competitive.

    Since 1987, Gates has been included in the Forbes list of the world's top billionaires. From 1995 to 2017, he held the title of the wealthiest person in the world every year except in 2008 and from 2010 to 2013. In 1999, he became the first ever centibillionaire when his net worth briefly surpassed US$100 billion. Since leaving day-to-day operations at Microsoft in 2008, Gates has pursued other business and philanthropic endeavors.

    He is the founder and chairman of several companies, including BEN, Cascade Investment, TerraPower, Gates Ventures, and Breakthrough Energy. He has donated to various charitable organizations and scientific research programs through the Bill & Melinda Gates Foundation, reported to be the world's largest private charity. Through the foundation, he led an early 21st century vaccination campaign that significantly contributed to the eradication of the wild poliovirus in Africa. In 2010, Gates and Warren Buffett founded the Giving Pledge, whereby they and other billionaires pledge to give at least half of their wealth towards philanthropy.
    """

    summary_template = """
    Given the information {information}, create:
    1. A short summary.
    2. Two interesting facts about the person.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # Obtém a chave da API
    api_key = config.get("OPENAI_API_KEY")

    # Inicializa o modelo da OpenAI
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=api_key)
    llm = ChatOllama(model="llama3.2:latest")

    chain = summary_prompt_template | llm | StrOutputParser()

    res = chain.invoke(input={"information": information})

    print(res)
