from gov_service_agent.rag.engine.embeddings import embed_chunk_batch


def test_embed_chunk_batch_uses_provider_safe_batch_size():
    class LimitedBatchEmbeddingProvider:
        name = "limited"
        dimension = 1
        document_batch_size = 10

        def __init__(self):
            self.batches = []

        def embed_documents(self, texts: list[str]):
            if len(texts) > 10:
                raise ValueError("batch size is invalid")
            self.batches.append(list(texts))
            return [[float(len(text))] for text in texts]

    provider = LimitedBatchEmbeddingProvider()
    vectors = embed_chunk_batch(provider, [f"chunk-{index}" for index in range(25)], max_retries=0)

    assert [len(batch) for batch in provider.batches] == [10, 10, 5]
    assert len(vectors) == 25
