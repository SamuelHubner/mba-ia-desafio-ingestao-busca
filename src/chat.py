from search import search_prompt


def main():
    try:
        vectorstore, llm = search_prompt()
    except Exception as e:
        print(f"Não foi possível iniciar o chat. Erro: {e}")
        return

    print("Chat iniciado! Digite 'sair' para encerrar.\n")

    while True:
        question = input("Faça sua pergunta: \n\n")

        if question.strip().lower() == "sair":
            print("Encerrando chat. Até logo!")
            break

        try:
            answer = search_prompt(question)
            print(f"\nPERGUNTA: {question}")
            print(f"RESPOSTA: {answer}\n")
        except Exception as e:
            print(f"Erro ao processar a pergunta: {e}\n")


if __name__ == "__main__":
    main()
