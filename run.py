from src.pipeline import index_document, query

# # Offline: parse + embed + index (run once per document)
# index_document("data/adobe-express-for-enterprise-security-overview.pdf")

result = query("Describe the flow of data when a user creates content in adobe express for enterprise.", doc_name="adobe-express-for-enterprise-security-overview"
               , top_k=10)
print(result["answer"]) # 从generation变成locate

for s in result["sources"]:
    print(f"  page {s['pages']} [{s['type']}] score={s['score']}")
    print(f"  {s['preview']}\n")

# preview qdrant
# from qdrant_client import QdrantClient
# from collections import Counter

# client = QdrantClient(path='./data/qdrant')
# collection = 'multimodal_rag'

# info = client.get_collection(collection)
# print(f'Total points: {info.points_count}')
# print()

# results = client.scroll(collection_name=collection, limit=100, with_payload=True, with_vectors=False)
# points = results[0]

# types = Counter(p.payload.get('type') for p in points)
# docs  = Counter(p.payload.get('doc_name') for p in points)

# print('By type:')
# for t, n in types.items():
#     print(f'  {t}: {n}')

# print()
# print('By document:')
# for d, n in docs.items():
#     print(f'  {d}: {n}')

# print()
# print('Chunks:')
# for p in points:
#     preview = p.payload.get('chunk_text', '')[:100].replace('\n', ' ')
#     print(f"  [{p.payload.get('type')}] p{p.payload.get('chunk_pages')} — {preview}...")
# client.close()

