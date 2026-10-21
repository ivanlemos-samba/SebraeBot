# 🤖 Stela Chatbot: Sistema RAG para Catálogo de Cursos SebraePlay
Este projeto implementa um sistema avançado de Question Answering (Q&A) baseado em Retrieval-Augmented Generation (RAG). O objetivo é criar um assistente virtual, chamado Stela, que pode responder perguntas complexas sobre o catálogo de cursos do SebraePlay utilizando apenas as informações contidas nos documentos fornecidos.

## 📚 Funcionalidades Principais
Base de Conhecimento: Carrega e processa múltiplos arquivos em formato Markdown (.md) de um diretório específico (materiais do curso).
Vectorização & Indexação: Cria embeddings dos documentos usando o modelo embeddinggemma:300m e armazena-os em um índice vetorial FAISS, otimizado para buscas semânticas rápidas.
Arquitetura RAG: Utiliza a LangChain Expression Language (LCEL) para orquestrar o fluxo de consulta: busca relevante $\rightarrow$ formatação do contexto $\rightarrow$ geração da resposta.
Persona e Regras: Implementa um System Prompt altamente detalhado que define a persona ("Mentora profissional") e regras estritas de segurança e escopo (ex: Disclaimer obrigatório para questões jurídicas).
## ⚙️ Arquitetura Técnica
O sistema segue o fluxo padrão RAG, garantindo respostas factuais e restritas ao material fonte.

Entrada: Documentos em Markdown (materiais_curso/).
Pré-processamento (Chunking): Os documentos longos são divididos em pequenos pedaços de texto (chunks) para otimizar a busca.
Embedding: Cada chunk é transformado em um vetor numérico (embedding) usando OllamaEmbeddings.
Indexação: O FAISS armazena esses vetores, criando o índice local (indice_cursos_faiss).
Busca (Retrieval): Quando uma pergunta é feita, o sistema busca os $K$ chunks mais semanticamente relevantes no índice FAISS.
Geração (Generation): O contexto recuperado e a pergunta original são enviados ao LLM (ChatOllama), que gera a resposta seguindo rigorosamente as regras definidas no PromptTemplate.
## 🚀 Pré-requisitos
Antes de rodar o código, você precisa garantir que os seguintes itens estejam instalados e configurados:

Python: Versão 3.8+
Ollama Service: O serviço Ollama deve estar instalado e em execução localmente na sua máquina. (É necessário para carregar os modelos de embedding e LLM).
Estrutura de Dados: Crie um diretório chamado materiais_curso e coloque todos os arquivos .md do curso dentro dele.
## 🛠️ Instalação e Setup
Siga estes passos na ordem correta:

1. Clonar o Projeto e Instalar Dependências
git clone [URL_DO_REPO]
cd seu-projeto-chatbot
pip install -r requirements.txt # Se você criar um arquivo de requisitos
. Ou instale manualmente (baseado nas suas importações):
pip install langchain openai python-dotenv langchain-community faiss-cpu
2. Configurar o Ambiente (.env)
Crie um arquivo .env na raiz do projeto, caso precise configurar chaves API específicas ou variáveis de ambiente (embora para este exemplo usando Ollama localmente, ele pode ser opcional).

3. Baixar Modelos no Ollama
Certifique-se de que os modelos usados pelo código estejam baixados e prontos no seu serviço Ollama:

ollama pull embeddinggemma:300m
ollama pull [nome do modelo]
> Para resultados satisfatórios recomendo modelos de 4B+ por conta do contexto, é importante que o modelo tenha thinking.
