# main.py
import numpy as np
from vector_search.search_engine import VectorSearchEngine
from vector_search.embeddings import SimpleEmbeddingGenerator
from vector_search.dimensionality import DimensionalityReducer, SVDReducer
from vector_search.visualization import SearchVisualizer


def main():
    print("=" * 60)
    print("Linear Algebra Project: Vector Search Engine")
    print("=" * 60)

    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Linear algebra is fundamental to machine learning",
        "Vectors and matrices are key concepts in linear algebra",
        "Natural language processing deals with human language",
        "Computer vision enables machines to see and understand images",
        "Reinforcement learning is about agents and environments",
        "Statistics provides foundations for data science",
        "Python is a popular programming language for data science",
        "Calculus is important for understanding optimization algorithms"
    ]

    # Initialize search engine
    search_engine = VectorSearchEngine(dimension=50)

    # Generate embeddings
    print("\n1. Generating embeddings for documents...")
    embedding_gen = SimpleEmbeddingGenerator(dimension=50)
    embedding_gen.build_vocab(documents)
    vectors = embedding_gen.documents_to_vectors(documents)

    print(f"   Created {len(documents)} document vectors of dimension {vectors.shape[1]}")

    # Add documents and build index
    search_engine.add_documents(documents)
    search_engine.build_index(vectors)

    # Demonstrate dimensionality reduction
    print("\n2. Applying dimensionality reduction (PCA)...")
    reducer = DimensionalityReducer(n_components=2)
    reduced_vectors = reducer.fit_transform(vectors)

    print(f"   Reduced from {vectors.shape[1]} to 2 dimensions")
    print(f"   Explained variance ratio: {reducer.explained_variance_ratio()}")

    # Visualize in 2D
    print("\n3. Visualizing documents in 2D space...")
    SearchVisualizer.plot_vectors_2d(
        reduced_vectors,
        labels=[f"Doc {i + 1}" for i in range(len(documents))],
        title="Document Vectors after PCA Reduction"
    )

    # Perform searches
    print("\n4. Performing vector searches...")

    # Example query 1
    query1 = "artificial intelligence and neural networks"
    query_vector1 = embedding_gen.document_to_vector(query1)

    results1 = search_engine.search(query_vector1, top_k=3, metric='cosine')

    print(f"\n   Query: '{query1}'")
    print("   Top 3 results:")
    for i, (doc_idx, similarity, metadata) in enumerate(results1):
        print(f"   {i + 1}. Doc {doc_idx + 1}: Similarity = {similarity:.4f}")
        print(f"      Text: {documents[doc_idx][:80]}...")

    # Example query 2
    query2 = "mathematics for data science"
    query_vector2 = embedding_gen.document_to_vector(query2)

    results2 = search_engine.search(query_vector2, top_k=3, metric='cosine')

    print(f"\n   Query: '{query2}'")
    print("   Top 3 results:")
    for i, (doc_idx, similarity, metadata) in enumerate(results2):
        print(f"   {i + 1}. Doc {doc_idx + 1}: Similarity = {similarity:.4f}")
        print(f"      Text: {documents[doc_idx][:80]}...")

    # Visualize search results
    print("\n5. Visualizing search results...")
    result_vectors = np.array([vectors[idx] for idx, _, _ in results1])
    SearchVisualizer.plot_search_results(
        query_vector1,
        result_vectors,
        title=f"Search Results for: '{query1}'"
    )

    # Compare similarity metrics
    print("\n6. Comparing similarity metrics...")
    print("\n   Cosine similarity vs Euclidean distance (inverse):")

    for i in range(min(3, len(documents))):
        for j in range(i + 1, min(4, len(documents))):
            cos_sim = search_engine.cosine_similarity(vectors[i], vectors[j])
            euc_dist = search_engine.euclidean_distance(vectors[i], vectors[j])
            euc_sim = 1 / (1 + euc_dist)

            print(f"   Doc {i + 1} & Doc {j + 1}:")
            print(f"     Cosine: {cos_sim:.4f}, Euclidean similarity: {euc_sim:.4f}")

    print("\n" + "=" * 60)
    print("Project completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()