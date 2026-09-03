from src.service import build_rag_service


def main():

    service = build_rag_service()

    while True:

        question = input(
            "\nUser: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:
            break

        try:

            result = service.ask(
                question
            )

        except ValueError as error:

            print(
                "\nError:",
                str(error),
            )

            continue

        print(
            "\nAssistant:"
        )

        print(
            result["answer"]
        )


if __name__ == "__main__":
    main()