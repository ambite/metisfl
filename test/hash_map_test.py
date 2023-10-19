from metisfl.controller.store.hash_map import HashMapModelStore
from metisfl.proto import model_pb2


def test_insert():
    pairs = [
        ("l1", model_pb2.Model()),
        ("l2", model_pb2.Model()),
    ]
    store = HashMapModelStore(0)
    store.insert(pairs)
    assert len(store.store_cache) == 2
    
def test_select():
    pairs = (
        ("l1", "model_pb2.Model()"),
        ("l2", "model_pb2.Model()"),
        ("l1", "model_pb2.Model()"),
        ("l2", "model_pb2.Model()"),
    )
    store = HashMapModelStore(0)
    store.insert(pairs)
    selected = store.select([("l1", 2)])

    assert len(selected["l1"]) == 2

def test_capacity():
    pairs = (
        ("l1", "model_pb2.Model()"),
        ("l1", "model_pb2.Model()"),
        ("l1", "model_pb2.Model()"),
        ("l1", "model_pb2.Model()"),
    )

    store = HashMapModelStore(2)
    store.insert(pairs)
    assert len(store.store_cache) == 2

