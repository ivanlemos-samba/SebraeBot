import os
from dotenv import load_dotenv
import requests
import json
import jq
import time
from requests_html import HTMLSession
#langchain imports
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


load_dotenv()

HEADERS = {"User-Agent": "Mozilla/5.0"}

def coletar_links_receitas(url_base, limite=100):
    """Coleta links de receitas das páginas do Receiteria até atingir o limite."""
    session = HTMLSession()
    links = []
    
    r = session.get(url_base)
   
    artigos = r.html.find("a.btn.shadow-sm")
    
    for art in artigos:
        if art.attrs["href"] not in links:
            links.append(art.attrs["href"])
            if len(links) >= limite:
                break
    
    return links[:limite]


def extrair_detalhes_receita(url):
    """Extrai informações detalhadas de uma receita usando requests_html (renderizando o JS)."""
    print(f"🍗 Extraindo detalhes de: {url}")

    session = HTMLSession()
    try:
        r = session.get(url)
        # Removido render() para evitar dependência do Chromium; a página é renderizada no servidor

        # Título
        titulo_tag = r.html.find("div.title h1", first=True)
        titulo = titulo_tag.text if titulo_tag else None
        print(titulo)
        # Tempo de preparo
        r = session.get(url)
        clock_icon = r.html.find("i.fa-solid.fa-clock", first=True)
        if clock_icon:
            parent = clock_icon.element.getparent()  # sobe para o <span>
            tempo_preparo = parent.text_content().strip()
            print(tempo_preparo)

        # Ingredientes (cada <li> na lista)
        ingredientes = [li.text for li in r.html.find("div.ingredientes ul li")]

        # Modo de preparo (geralmente em <div class="modo-de-preparo"> <p>)
        modo_preparo = [p.text for p in r.html.find("div.preparo ol li span")]

        # Tabela nutricional (se existir)
        # print("Table nutricional")
        informacoes_nutricionais = {}
        linhas_tabela = r.html.find("div.nuticao div.item")
        
        for linha in linhas_tabela:
            content = linha.text.split(" ")
            valor = content[0]
            chave = content[1]
            informacoes_nutricionais[chave] = valor
        #    print(f"{chave}:{valor}")
        
        linhas_tabela = r.html.find("table.table.table-borderless.mt-3 tbody tr")
        for linha in linhas_tabela:
            cols = linha.find("td")
            chave = cols[0].text.strip()
            valor = cols[1].text.strip()
            informacoes_nutricionais[chave] = valor
        #    print(f"{chave}:{valor}")

        return {
            "titulo": titulo,
            "tempo_preparo": tempo_preparo,
            "ingredientes": ingredientes or None,
            "modo_preparo": modo_preparo or None,
            "informacoes_nutricionais": informacoes_nutricionais or None
        }

    except Exception as e:
        print(f"Erro ao extrair {url}: {e}")
        return None
    finally:
        session.close()


def scraping_cook_data():
    url_base = "https://www.receiteria.com.br/receitas-com-frango"
    limite = 100

    links = coletar_links_receitas(url_base, limite)
    print(f"{len(links)} links coletados.")

    receitas = []
    for i, link in enumerate(links, 1):
        print(f"\n[{i}/{len(links)}] Processando receita...")
        detalhes = extrair_detalhes_receita(link)
        if detalhes:
            receitas.append(detalhes)
        time.sleep(1.5)  # evitar bloqueio

    with open("receitas_frango_detalhadas.json", "w", encoding="utf-8") as f:
        json.dump(receitas, f, ensure_ascii=False, indent=2)

    print(f"\n Extração concluída! {len(receitas)} receitas salvas em 'receitas_frango_detalhadas.json'.")

def convert_JSON_to_documents():
    loader = JSONLoader(
        file_path="receitas_frango_detalhadas.json",
        jq_schema=r"""
        .[] | {
            page_content: (
                "Título: " + .titulo + "\n" +
                "Tempo de preparo: " + .tempo_preparo + "\n\n" +
                "Ingredientes:\n" + (.ingredientes | join("\n")) + "\n\n" +
                "Modo de preparo:\n" + (.modo_preparo | join("\n")) + "\n\n" +
                "Informações nutricionais:\n" +
                (if .informacoes_nutricionais != null
                 then (.informacoes_nutricionais | to_entries | map(.key + ": " + .value) | join("\n"))
                 else "Não informado"
                 end)
            )
        }
        """,
        metadata_func=lambda x, y: {
            "titulo": x.get("titulo"),
            "tempo": x.get("tempo_preparo"),
            "ingredientes": x.get("ingredientes"),
            "info_nutricional": x.get("informacoes_nutricionais")
        },
        text_content=False,  # obrigatório com jq_schema
    )
    return loader.load()


def main():
    # Captura dados de receitas de frango em do site receitaria.com
    # scraping_cook_data()

    # 1 -  Carrega o JSON e transforma em documentos
    docs = convert_JSON_to_documents()
    print(f"{len(docs)} documentos carregados.")

    # 2 Criar Embeddings e indexa no Faiss - Cria um banco vetorial FAISS para buscas semânticas. 
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    # 3 Save FAISS - (do inglês Facebook AI Similarity Search) É um mecanismo para armazenar 
    # e buscar vetores (embeddings) de forma muito rápida, mesmo quando há milhões de itens.
    db.save_local("indice_receitas_faiss")
    print("Índice salvo em >>> ./indice_receitas_faiss")

    # -----------------------------------------
    # 3.5 Carrega o índice salvo (simulando outro script)
    # -----------------------------------------
    db2 = FAISS.load_local("indice_receitas_faiss", embeddings, allow_dangerous_deserialization=True)

    # 4 Criar o retriver - o retriver é o componente que vai buscar os documentos relevantes para a consulta.
    # Retriever class returns Documents given a text query.
    # It is more general than a vector store. A retriever does not need to be able to store documents, 
    # only to return (or retrieve) it. Vector stores can be used as the backbone of a retriever, but there 
    # are other types of retrievers as well.
    retriever = db2.as_retriever(search_type="similarity", search_kwargs={"k": 2})


    # 5 Criar a chain com RAG
    # llm args: temperature = controle da criatividade, timeout = tempo max. de resposta, 
    # max_tokens = limita o numero de tokens (palavras e pontuação) na resposta,
    # stop = especifica quando o modelo deve parar de gerar tokens [strings flags]
    # max_retries = numero de tentativas em caso de falha de rede
    # api_key
    # base_url = url da API API para onde as solicitações são enviadas (end-point)
    # rate_limiter = opcional para espaçar solicitações e evitar exceder os limites de taxa.

    # langchain API antiga run() e call()
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=ChatOpenAI(model="gpt-4o-mini"),
    #     retriever=retriever,
    #     chain_type="stuff"
    # )

    template = """
    Você é um chefe especializado em nutrição.
    Dada as receitas abaixo: 
    {receitas}
    
    Crie 2 novas:
    1. com muito carboidrato;
    2. com muita proteína.

    Apresente:
    - Título:
    - Informações Nutricionais: [calorias, proteínas, gorduras, carboidratos]
    - Ingredientes:
    - Modo de preparo:
    """
    prompt_template = PromptTemplate(
        input_variables=["receitas"],
        template=template
    )

    # LCEL (LangChain Expression Language) o LCEL é uma DSL (Domain-Specific Language) do LangChain para 
    # montar pipelines de LLMs de forma modular, declarativa e composicional — como montar blocos Lego.
    # Cada componente (prompt, modelo, parser, retriever etc.) é um “nó” que pode ser conectado com o operador |.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
    # Aqui, `RunnablePassthrough` apenas passa o valor para o próximo nó
    chain = (
        {"receitas": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs))}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    #query = "Qual receita é mais rica em proteínas?"
    #response = qa_chain.run(query) 
    # .run() é uma conveniência legada usada em chains herdadas, especialmente:
    # LLMChain
    # RetrievalQA
    # ConversationalRetrievalChain
    # AgentExecutor
    
    query = "receita para hipertrofia"
    response = chain.invoke(query)
    print("Pergunta:", query)
    print("Resposta:", response)


if __name__ == "__main__":
    main()
