from pymilvus import Milvus, DataType, connections, IndexType, MetricType

class MilvusHelper(Milvus):
    def __init__(self, host='localhost', port='19530', collection_name='raven-mvp', dimension=768):
        super().__init__(host=host, port=port)
        self.collection_name = collection_name
        self.dimension = dimension

        # Create a collection if it does not exist.
        if not self.milvus.has_collection(collection_name)[1]:
            collection_param = {
                "fields": [
                    # Assuming that vector embeddings are floats.
                    {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": dimension}, "indexes": [{"metric_type": "L2"}]},
                    # You can add more fields if needed.
                ],
                "segment_row_limit": 10000,
                "auto_id": False
            }
            self.milvus.create_collection(collection_name, collection_param)
            
            index_params = {
                "metric_type": "L2",
                "index_type": IndexType.IVF_FLAT,
                "params": {"nlist": 1024}
            }
            self.milvus.create_index(self.collection_name, "embedding", index_params)

    def insert(self, records):
        # This function takes a list of records, each record is a tuple of (id, vector)
        ids, vectors = zip(*records)
        entities = [
            {"name": "embedding", "type": DataType.FLOAT_VECTOR, "values": vectors},
        ]
        self.milvus.insert(self.collection_name, entities, ids=ids)

    def query(self, vector, top_k):
        # This function takes a vector and returns top_k most similar vectors from the collection
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.milvus.search(self.collection_name, {"embedding": vector}, search_params, top_k, "embedding > 0.5")
        # You might need to adjust the structure of the returned results according to your needs
        return results

    def upsert(self, records):
        # This function inserts new vectors and updates existing ones
        self.insert(records)

# You can use the MilvusHelper class as follows:
# helper = MilvusHelper()
# helper.insert([('id1', vector1), ('id2', vector2), ...])
# results = helper.query(vector, top_k=10)
