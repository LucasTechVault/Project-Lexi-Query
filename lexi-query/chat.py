from app.generation.answerer import RagAnswerer

def main():
    print("Waking up the RAG Pipeline (Connecting to Chroma and vLLM)...")
    try:
        app = RagAnswerer()
        print("System Ready! Type 'exit' to quit.")
        print("=" * 70)
    except Exception as e:
        print(f"Failed to start: {e}")
        print("Did you make sure your vLLM server is running in another terminal?")
        return

    while True:
        query = input("\n💡 Ask a financial question: ")
        if query.lower() in ['exit', 'quit']:
            print("Shutting down. Goodbye!")
            break

        print("🔍 Searching archives and synthesizing answer...")
        
        # Run the entire pipeline
        result = app.answer(query)

        print("\n" + "="*70)
        print("LLM ANSWER:")
        print("="*70)
        print(result["answer"])
        
        print("\n" + "-"*70)
        print("SOURCES CITED:")
        
        # Use a set to remove duplicate pages if multiple chunks came from the same page
        unique_sources = {f"{c['metadata']['doc_id']} (Page {c['metadata']['page_num']})" for c in result["chunks"]}
        for source in unique_sources:
            print(f" - {source}")
        print("-" * 70 + "\n")

if __name__ == "__main__":
    main()