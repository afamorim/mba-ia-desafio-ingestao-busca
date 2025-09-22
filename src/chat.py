from search import search_prompt


def main():
    print("=" * 50)
    print("Chat com Documentos PDF")
    print("=" * 50)
    print("Digite suas perguntas ou 'exit' para sair.")
    print("=" * 50)

    while True:
        # Get user input
        question = input("\nVocê: ").strip()

        # Check if user wants to exit
        if question.lower() == "exit":
            print("Obrigado por usar o chat! Até logo!")
            break

        # Skip empty questions
        if not question:
            continue

        try:
            # Use search_prompt to process the question
            result = search_prompt(question)
            # print(result)
            if result is None:
                print("Não tenho informações necessárias para responder sua pergunta.")
            else:
                print(result)

        except Exception as e:
            print(f"Erro ao processar sua pergunta: {str(e)}")


if __name__ == "__main__":
    main()
