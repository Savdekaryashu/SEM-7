import re
from collections import defaultdict

# Sample documents
documents = {
    1: "Data Science is an interdisciplinary field.",
    2: "Machine learning is a part of Data Science.",
    3: "Information Retrieval systems use inverted index.",
    4: "Search engines use indexing to retrieve documents efficiently.",
}


# Preprocessing: Tokenization and Normalization
def tokenize(text):
    # Remove punctuation and lowercase everything
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.lower().split()
    return tokens


# Step 1: Build Inverted Index
def build_inverted_index(docs):
    inverted_index = defaultdict(set)
    for doc_id, content in docs.items():
        tokens = tokenize(content)
        for word in tokens:
            inverted_index[word].add(doc_id)
    return inverted_index


# Step 2: Query Processor
def search(query, index):
    tokens = tokenize(query)
    result_sets = []
    for token in tokens:
        result_sets.append(index.get(token, set()))

    # Intersection of all word results for multi-word query
    if result_sets:
        return set.intersection(*result_sets)
    return set()


# Main
inverted_index = build_inverted_index(documents)

# Print the inverted index
print("\n🔎 Inverted Index:")
for word, doc_ids in sorted(inverted_index.items()):
    print(f"{word}: {sorted(doc_ids)}")

# Sample queries
while True:
    query = input("\nEnter a search term (or 'exit' to stop): ")
    if query.lower() == 'exit':
        break
    result = search(query, inverted_index)
    if result:
        print(f"Documents containing '{query}': {sorted(result)}")
    else:
        print(f"No documents found for '{query}'.")
