from sentence_transformers import SentenceTransformer


def main():
    model_id = "sentence-transformers/all-mpnet-base-v2"
    output_dir = "model/all-mpnet-base-v2"
    SentenceTransformer(model_id).save(output_dir)
    print(f"Saved embedding model to: {output_dir}")


if __name__ == "__main__":
    main()
