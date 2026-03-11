# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema de ingestão e busca semântica sobre documentos PDF utilizando LangChain, PostgreSQL com pgVector e OpenAI.

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Chave de API da OpenAI (ou Google Gemini)

## Configuração

1. Clone o repositório e crie o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Crie o arquivo `.env` a partir do template e preencha sua API key:

```bash
cp .env.example .env
```

Edite o `.env` e configure pelo menos a `OPENAI_API_KEY`:

```env
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=pdf_documents
PDF_PATH=document.pdf
```

3. Coloque o PDF que deseja ingerir na raiz do projeto como `document.pdf` (ou ajuste `PDF_PATH` no `.env`).

## Execução

### 1. Subir o banco de dados

```bash
docker compose up -d
```

Aguarde o container ficar saudável (a extensão `vector` será criada automaticamente).

### 2. Ingestão do PDF

```bash
source .venv/bin/activate
python src/ingest.py
```

O script carrega o PDF, divide em chunks de 1000 caracteres (com overlap de 150) e armazena os embeddings no PostgreSQL.

### 3. Chat via CLI

```bash
python src/chat.py
```

Faça perguntas sobre o conteúdo do PDF. Digite `sair` para encerrar.

Exemplo:

```
Faça sua pergunta:
Qual o faturamento da Empresa SuperTechIABrazil?

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.
```

## Estrutura do projeto

```
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py         # Ingestão do PDF no banco vetorial
│   ├── search.py         # Busca semântica e chamada à LLM
│   └── chat.py           # CLI interativo
├── document.pdf          # PDF para ingestão
└── README.md
```
