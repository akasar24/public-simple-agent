from memory.full_memory import FullHistoryMemory
from memory.window_memory import WindowMemory


# --- FullHistoryMemory ---

def test_full_history_stores_all_messages():
    memory = FullHistoryMemory()
    memory.add_messages([{"role": "user", "content": "My name is Alice"}])
    memory.add_messages([{"role": "assistant", "content": "Hi Alice!"}])
    assert len(memory.get_context()) == 2

def test_full_history_retains_early_messages():
    memory = FullHistoryMemory()
    memory.add_messages([{"role": "user", "content": "My name is Alice"}])
    for i in range(20):
        memory.add_messages([{"role": "user", "content": f"message {i}"}])
        memory.add_messages([{"role": "assistant", "content": f"response {i}"}])
    context = memory.get_context()
    assert any("Alice" in m["content"] for m in context)

def test_full_history_grows_unbounded():
    memory = FullHistoryMemory()
    for i in range(100):
        memory.add_messages([{"role": "user", "content": f"message {i}"}])
    assert len(memory.get_context()) == 100


# --- WindowMemory ---

def test_window_memory_limits_context():
    memory = WindowMemory(window_size=4)
    for i in range(10):
        memory.add_messages([{"role": "user", "content": f"message {i}"}])
        memory.add_messages([{"role": "assistant", "content": f"response {i}"}])
    assert len(memory.get_context()) <= 4

def test_window_memory_retains_recent():
    memory = WindowMemory(window_size=4)
    for i in range(10):
        memory.add_messages([{"role": "user", "content": f"message {i}"}])
        memory.add_messages([{"role": "assistant", "content": f"response {i}"}])
    context = memory.get_context()
    contents = [m["content"] for m in context]
    assert any("message 9" in c or "response 9" in c for c in contents)

def test_window_memory_starts_with_user():
    memory = WindowMemory(window_size=3)
    for i in range(5):
        memory.add_messages([{"role": "user", "content": f"message {i}"}])
        memory.add_messages([{"role": "assistant", "content": f"response {i}"}])
    assert memory.get_context()[0]["role"] == "user"

def test_window_memory_forgets_early_messages():
    memory = WindowMemory(window_size=4)
    memory.add_messages([{"role": "user", "content": "My name is Alice"}])
    for i in range(10):
        memory.add_messages([{"role": "user", "content": f"message {i}"}])
        memory.add_messages([{"role": "assistant", "content": f"response {i}"}])
    context = memory.get_context()
    assert not any("Alice" in m["content"] for m in context)