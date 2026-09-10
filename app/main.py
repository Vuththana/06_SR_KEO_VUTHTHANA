from pipeline import pipeline


while True:
    prompt = input("Enter message (Type exit, quit to quit): ")

    if prompt.lower() in ["exit", "quit"]:
        break

    response = pipeline(prompt)

    # Print LLM's response and provided context
    print(response)