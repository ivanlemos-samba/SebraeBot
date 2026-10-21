import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
import logging
import time
from pathlib import Path
import re


# ---------- Configura o logger ROOT (o que todo logging.info() usa) ----------
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Limpa handlers existentes para evitar duplicação
root_logger.handlers.clear()

# ---------- Console ----------
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# ---------- Arquivo .txt ----------
file_handler = logging.FileHandler("bot.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(console_formatter)
root_logger.addHandler(file_handler)

# ---------- Silenciar logs HTTP ----------
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

load_dotenv()

# Definições
EMBEDDING_MODEL = "embeddinggemma:300m"
LLM_MODEL = "lms-lmstudio-community-gemma-4-e4b-it-gguf-gemma-4-e4b-it-q4_k_m:latest" # Recomendo somente modelos com thinking
FAISS_PATH = "indice_cursos_faiss"
DOCUMENT_DIR = "./Material de treinamento"


def load_documents(caminho: str) -> list[Document]:
    """Carrega os documentos do curso em formato .md a partir de um diretório."""
    logging.info(f"Iniciando carregamento de documentos no caminho: {caminho}")
    docs = []

    if not Path(caminho).exists():
        logging.error(f"Diretório não encontrado: {caminho}. Por favor, crie o diretório e adicione os arquivos .md.")
        return docs
    for arquivo in Path(caminho).rglob("*.md"):
        try:
            with open(arquivo , "r" , encoding="utf-8") as f:
                conteudo = f.read()

            docs.append(
                Document(
                    page_content=conteudo,
                    metadata={
                        "source": str(arquivo),
                        "arquivo": arquivo.name
                    }
                )
            )
        except Exception as e:
            logging.warning(f"Erro ao ler o arquivo {arquivo}: {e}")

    return docs

def main():
    # Captura dados de receitas de frango em do site receitaria.com
    # scraping_cook_data()

    # # 1 -  Carrega o JSON e transforma em documentos
    # docs = load_documents(DOCUMENT_DIR)
    # print(f"{len(docs)} documentos carregados.")
    #
    # if not docs:
    #     logging.error("Não foi possível carregar nenhum documento. Encerrando o script.")
    #     return
    #
    # # 1.5 - Divide o texto em chunks (caso os textos sejam longos)
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=100,
    #     length_function=len
    # )
    # docs_chunked = splitter.split_documents(docs)
    # print(f"{len(docs_chunked)} chunks criados.")
    #
    # 2 Criar Embeddings e indexa no Faiss - Cria um banco vetorial FAISS para buscas semânticas.
    embeddings = OllamaEmbeddings(model="embeddinggemma:300m")
    # db = FAISS.from_documents(docs_chunked, embeddings)
    #
    # # 3 Save FAISS - (do inglês Facebook AI Similarity Search) É um mecanismo para armazenar
    # # e buscar vetores (embeddings) de forma muito rápida, mesmo quando há milhões de itens.
    # db.save_local(FAISS_PATH)
    # print("Índice salvo em >>> ./indice_cursos_faiss")


    # 3.5 Carrega o índice salvo (simulando outro script)
    try:
        # Carrega o índice (simulando outro script)
        db2 = FAISS.load_local(FAISS_PATH , embeddings , allow_dangerous_deserialization=True)

        # Cria o retriever - Componente que busca os documentos relevantes para a consulta.
        # Retriever class returns Documents given a text query.
        # It is more general than a vector store. A retriever does not need to be able to store documents,
        # only to return (or retrieve) it. Vector stores can be used as the backbone of a retriever, but there
        # are other types of retrievers as well.
        retriever = db2.as_retriever(search_type="similarity" ,search_kwargs={"k": 4})
    except Exception as e:
        logging.error(f"Erro ao carregar o índice FAISS. Verifique se a pasta '{FAISS_PATH}' existe. Erro: {e}")
        return

    # todo - add guarrails de segurança, evitar prompt injection
    # TODO - improv flux
    template = """
    
    [INSTRUÇÕES DE SISTEMA]    
    Você é Stela, assistente virtual do SebraePlay. Sua missão exclusiva é guiar o usuário através do catálogo de cursos disponíveis na plataforma, utilizando apenas as informações fornecidas em CONTEXTO.
    
    [PERSONA]
    Tom: Mentora profissional, acessível e encorajadora. Use linguagem voltada para MEIs e pequenos empreendedores.
    Objetivo: Inspirar confiança e fornecer caminhos de aprendizado através dos cursos do SebraePlay.
    
    [REGRAS]
    Você DEVE seguir estas regras em todas as interações:    
    1. Fonte Única: Baseie-se exclusivamente no conteúdo do CONTEXTO.
    2. Proibido Inventar: Jamais crie ou deduza dados, estatísticas, preços, títulos, links ou informações factuais que não estejam explicitamente presentes na base de conhecimento.
    3. Limites de Consultoria (Disclaimer Obrigatório): Se o usuário solicitar aconselhamento jurídico, contábil ou tributário, interrompa a resposta e use obrigatoriamente esta orientação: "Lembre-se que eu sou uma assistente virtual focada nos cursos da plataforma. Para questões jurídicas ou contábeis específicas do seu negócio, é fundamental buscar um especialista qualificado (contador, advogado) ou consultar diretamente o Portal Sebrae."
    4. Limites de Escopo: Caso a pergunta não tenha relação com gestão, empreendedorismo ou os cursos listados em CONTEXTO, redirecione gentilmente: "Sobre [x] especificamente, não localizei nenhum curso relacionado."
    5. Limites Operacionais: Reforce sempre que toda ajuda deve ser neste chat: "Tudo o que posso te ajudar será aqui mesmo, neste chat. Por questões de segurança e organização, não consigo enviar informações por WhatsApp, e-mail ou telefone."
    
    [FLUXO DE CONVERSAÇÃO] (PROCESSAMENTO DA DÚVIDA)
    1. Contexto Válido: Se a resposta estiver em CONTEXTO, forneça detalhes de valor agregado (ex: "Este curso é ótimo porque ensina X, ajuda a resolver Y e é ideal para quem está começando como Z") . 
    2. Fallback Inteligente (Resposta Não Encontrada): Se não souber responder, mas o tópico for relacionado ao negócio: Reconheça a limitação e sempre sugira um curso alternativo ou mais relevante que esteja no contexto para manter o foco na solução do catálogo. 
    3. Resposta de Recusa: Ao violar alguma regra, utilize o texto obrigatório da regra correspondente, sem tentar responder à pergunta original.
    
    CONTEXTO:
    {context}
    
    PERGUNTA:
    {question}
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    # Parser customizado para filtrar e extrair apenas a resposta final
    class FilterThoughtParser(StrOutputParser):
        def parse(self , raw_response: str) -> str:
            # Remove blocos de pensamento/reasoning se existirem
            match = re.search(r"Resposta:\s*(.*)" , raw_response , re.DOTALL)
            if match:
                return match.group(1).strip()
            match = re.search(r"Answer:\s*(.*)" , raw_response , re.DOTALL)
            if match:
                return match.group(1).strip()
            return raw_response


    # LCEL (LangChain Expression Language) o LCEL é uma DSL (Domain-Specific Language) do LangChain para 
    # montar pipelines de LLMs de forma modular, declarativa e composicional — como montar blocos Lego.
    # Cada componente (prompt, modelo, parser, retriever etc.) é um “nó” que pode ser conectado com o operador |.
    llm = ChatOllama(temperature = 0,validate_model_on_init=True, model=LLM_MODEL)
    # Aqui, `RunnablePassthrough` apenas passa o valor para o próximo nó
    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), # A pergunta do usuário busca no retriever e formata os docs
            "question": RunnablePassthrough(),  # Mantém a pergunta original disponível para o prompt
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    def executar_mock(nome_do_curso: str , perguntas: list[str] , chain):
        """Executa o mock de testes e imprime os resultados."""
        resultados = []

        print("\n" + "=" * 80)
        print(f"🚀 INICIANDO MOCK DE TESTES PARA O CURSO: '{nome_do_curso}'")
        print("=" * 80)

        for i , pergunta_template in enumerate(perguntas):
            # Formata a pergunta com o nome do curso
            query = pergunta_template.format(nome_do_curso=nome_do_curso)

            inicio = time.perf_counter()
            try:
                # Invoca a chain com a pergunta formatada
                response = chain.invoke(query)
                fim = time.perf_counter()
                latencia = round(fim - inicio , 3)

                logging.info(f" Pergunta {i + 1}/{len(perguntas)}: {query}")
                logging.info(f"Resposta: {response}")
                logging.info(f"Latência: {latencia}s")
                logging.info("-------------------------------------------------")  # Separador no log

                print("\n>>> Pergunta:" , query)
                print("<<< Resposta:" , response)
                print(f"   [Latência Calculada]: {latencia}s")
            except Exception as e:
                logging.error(f"Erro ao executar a pergunta '{query}': {e}")
                response = "ERRO NA RESPOSTA."
                latencia = 0

            resultados.append({
                "pergunta": query ,
                "resposta": response ,
                "latencia": latencia
            })

        return resultados

    # --- DEFINIÇÃO DOS CENÁRIOS DE TESTE (REFACTORADO) ---
    test_scenarios = [
        {
            'nome': 'Apresentações de Impacto' ,
            'perguntas': [
                "O curso {nome_do_curso} dura quantas horas?" ,
                "O curso {nome_do_curso} possui certificado?" ,
                "Qual o nível do curso {nome_do_curso}?" ,
                "Quantos módulos existem no curso {nome_do_curso}?" ,
                "Quais são os principais conhecimentos ensinados no curso {nome_do_curso}?" ,
                "O curso {nome_do_curso} aborda vídeos e GIFs?" ,
                "Qual é a duração da aula sobre o funcionamento do cérebro no curso {nome_do_curso}?" ,
                "Para quem o curso {nome_do_curso} é indicado?"
            ] ,
            'respostas': [
                "3h." ,
                "Sim, possui certificado (com download)." ,
                "Básico." ,
                "8 módulos." ,
                "Escolher cores, fontes, imagens e vídeos para slides; encantar o público; ser memorável; organizar a história; preparar os slides; técnicas de apresentação." ,
                "Sim — módulo 'Vídeos e gifs' (aula de 20 min) ensina a incorporá-los e as melhores plataformas para encontrá-los." ,
                "15 minutos ('Como o nosso cérebro funciona')." ,
                "Para quem precisa apresentar ideias, vender produtos ou atrair investidores; nível básico, sem exigência técnica."
            ]
        } ,
        {
            'nome': 'Criatividade para Negócios' ,
            'perguntas': [
                "Qual módulo do curso {nome_do_curso} fala sobre criatividade aplicada aos negócios?" ,
                "Quais ferramentas são apresentadas no curso {nome_do_curso}?" ,
                "Qual é o objetivo principal do curso {nome_do_curso}?" ,
                "Quais metodologias são citadas no curso {nome_do_curso}?"
            ] ,
            'respostas': [
                "Não há módulos declarados: o curso é uma única aula ao vivo chamada 'Criatividade para Negócios - Curso Ao Vivo'." ,
                "Técnicas de design thinking e os conceitos do livro 'Confiança Criativa'." ,
                "Despertar a criatividade para resolver problemas, gerar ideias e criar soluções eficientes para o negócio, superando bloqueios criativos." ,
                "Design thinking (e os conceitos do livro 'Confiança Criativa')."
            ]
        } ,
        {
            'nome': 'ChatGPT para pequenos negócios' ,
            'perguntas': [
                "O que o aluno aprende no curso {nome_do_curso}?" ,
                "Como o curso {nome_do_curso} ensina a utilizar o ChatGPT nos negócios?" ,
                "Quais benefícios o empreendedor pode obter ao aplicar os conhecimentos do curso {nome_do_curso}?" ,
                "O curso {nome_do_curso} aborda o uso de inteligência artificial para marketing e vendas?"
            ] ,
            'respostas': [
                "Criar e configurar a conta no ChatGPT; criar prompts claros e estratégicos; gerar ideias de conteúdo; estruturar funis de vendas automatizados; desenvolver discursos de vendas mais persuasivos; aplicar IA no dia a dia do negócio." ,
                "De forma prática em 5 aulas curtas (35 min): criação da conta, geração de ideias de conteúdo, montagem de funil de vendas com inbound marketing, criação de discursos de vendas e escrita de prompts melhores." ,
                "Mais produtividade, criatividade e eficiência, usando a IA como apoio estratégico no crescimento do negócio." ,
                "Sim — o curso aplica o ChatGPT em marketing, vendas e comunicação (funil de vendas e discursos de venda)."
            ]
        }
    ]

    # Executa todos os cenários em um loop único
    all_results = []
    for scenario in test_scenarios:
        resultados = executar_mock(
            nome_do_curso=scenario['nome'] ,
            perguntas=scenario['perguntas'] ,
            chain=chain
        )
        all_results.extend(resultados)

    logging.info("\n=============================================")
    logging.info("✅ MOCK DE TESTES CONCLUÍDO!")
    logging.info("=============================================")



if __name__ == "__main__":
    main()
