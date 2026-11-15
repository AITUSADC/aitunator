import json
import unittest

import httpx
import numpy as np

from backend.app.providers.embeddings.openai import OpenAIEmbedder, load_api_embedder
from backend.app.providers.embeddings.sentence_transformer import encode_documents, encode_query


def _client(requests):
    def handler(request):
        body = json.loads(request.content)
        requests.append((request, body))
        data = [{"index": i, "embedding": [3.0, 4.0] + [0.0] * (body["dimensions"] - 2)}
                for i in reversed(range(len(body["input"])))]
        return httpx.Response(200, json={"data": data})
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.test/v1/")


class OpenAIEmbedderTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.model = OpenAIEmbedder("text-embedding-3-large", client=_client(self.requests), batch_size=2)

    def test_documents_are_batched_ordered_and_normalized(self):
        vectors = encode_documents(self.model, ["a", "b", "c"], show_progress_bar=False)
        self.assertEqual(vectors.shape, (3, 1024))
        np.testing.assert_allclose(vectors[:, :2], [[0.6, 0.8]] * 3)
        self.assertEqual([body["input"] for _, body in self.requests], [["a", "b"], ["c"]])
        self.assertEqual(self.requests[0][1]["model"], "text-embedding-3-large")
        self.assertTrue(str(self.requests[0][0].url).endswith("/v1/embeddings"))

    def test_query_returns_single_vector(self):
        self.assertEqual(encode_query(self.model, "вопрос").shape, (1024,))

    def test_only_openai_prefixed_names_use_the_api(self):
        self.assertIsNone(load_api_embedder("BAAI/bge-m3"))
        with self.assertRaises(ValueError):
            load_api_embedder("openai/text-embedding-3-large", env={})
        embedder = load_api_embedder("openai/text-embedding-3-large", env={"OPENAI_API_KEY": "k"})
        self.assertEqual(embedder.model, "text-embedding-3-large")


if __name__ == "__main__":
    unittest.main()
