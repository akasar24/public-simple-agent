import time
import pytest
from dotenv import load_dotenv

from agent.core import make_agent
from memory.full_memory import FullHistoryMemory
from memory.summary_memory import SummaryMemory
from memory.window_memory import WindowMemory

load_dotenv()

N_RUNS = 5


@pytest.fixture
def agent():
    return make_agent()


def test_agent_responds(agent):
    """Agent should return a non-empty response to a simple question."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is 2 + 2?"}]}
    )
    assert len(result["messages"]) > 1
    ai_msg = result["messages"][-1]
    assert ai_msg.content
    assert "4" in ai_msg.content


def test_agent_multi_turn(agent):
    """Agent should handle multi-turn conversation."""
    r1 = agent.invoke(
        {"messages": [{"role": "user", "content": "My name is Alice."}]}
    )
    msgs = r1["messages"]
    msgs.append({"role": "user", "content": "What is my name?"})
    r2 = agent.invoke({"messages": msgs})
    ai_msg = r2["messages"][-1]
    assert "Alice" in ai_msg.content


# --- filler sets ---

SHORT_FILLER = [
    "What is 3 + 3?",
    "What is the capital of France?",
    "How many days are in a week?",
    "What color is the sky?",
    "What is 10 * 2?",
]

LONG_FILLER = SHORT_FILLER + [
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water?",
    "How many continents are there?",
    "What is the speed of light?",
    "What language is spoken in Brazil?",
    "What is the largest ocean?",
    "How many sides does a hexagon have?",
    "What is the square root of 144?",
    "What planet is closest to the sun?",
    "How many hours are in a day?",
]


def ai_msg_to_dict(msg) -> dict:
    return {"role": "assistant", "content": msg.content}


def timed_invoke(agent, memory):
    """Invoke agent and return (result, elapsed)."""
    start = time.time()
    result = agent.invoke({"messages": memory.get_context()})
    elapsed = time.time() - start
    return result, elapsed


def run_memory_harness(agent, memory, filler, setup_msgs, final_question, check_response, post_filler_msgs=None):
    """
    Generalized harness: send setup messages, filler, optional post-filler
    messages, then a final question.
    Returns (correct: bool, char_count: int, total_elapsed: float)
    """
    total_elapsed = 0.0

    for msg in setup_msgs:
        memory.add_messages([{"role": "user", "content": msg}])
        result, elapsed = timed_invoke(agent, memory)
        total_elapsed += elapsed
        memory.add_messages([ai_msg_to_dict(result["messages"][-1])])

    for msg in filler:
        memory.add_messages([{"role": "user", "content": msg}])
        result, elapsed = timed_invoke(agent, memory)
        total_elapsed += elapsed
        memory.add_messages([ai_msg_to_dict(result["messages"][-1])])

    for msg in (post_filler_msgs or []):
        memory.add_messages([{"role": "user", "content": msg}])
        result, elapsed = timed_invoke(agent, memory)
        total_elapsed += elapsed
        memory.add_messages([ai_msg_to_dict(result["messages"][-1])])

    memory.add_messages([{"role": "user", "content": final_question}])
    result, elapsed = timed_invoke(agent, memory)
    total_elapsed += elapsed
    memory.add_messages([ai_msg_to_dict(result["messages"][-1])])

    ai_response = result["messages"][-1].content
    correct = check_response(ai_response)
    char_count = sum(len(m["content"]) for m in memory.get_context())

    return correct, char_count, total_elapsed


def run_n_times(agent_fixture, memory_cls, memory_kwargs, scenario, n=N_RUNS):
    """Run the harness N times with fresh memory each time, return aggregated results."""
    correctness = []
    char_counts = []
    elapsed_times = []

    for _ in range(n):
        memory = memory_cls(**memory_kwargs)
        correct, char_count, total_elapsed = run_memory_harness(
            agent_fixture,
            memory,
            filler=scenario["filler"],
            setup_msgs=scenario["setup_msgs"],
            final_question=scenario["final_question"],
            check_response=scenario["check_response"],
            post_filler_msgs=scenario.get("post_filler_msgs"),
        )
        correctness.append(correct)
        char_counts.append(char_count)
        elapsed_times.append(total_elapsed)

    return {
        "correctness": f"{sum(correctness)}/{n}",
        "avg_chars": int(sum(char_counts) / n),
        "avg_total_elapsed": round(sum(elapsed_times) / n, 2),
    }


# --- scenarios ---

SCENARIOS = {
    "basic_recall_short": dict(
        filler=SHORT_FILLER,
        setup_msgs=["My name is Alice and I work in finance."],
        final_question="What is my name and what do I do for work?",
        check_response=lambda r: "Alice" in r and "finance" in r.lower(),
    ),
    "basic_recall_long": dict(
        filler=LONG_FILLER,
        setup_msgs=["My name is Alice and I work in finance."],
        final_question="What is my name and what do I do for work?",
        check_response=lambda r: "Alice" in r and "finance" in r.lower(),
    ),
    "contradiction": dict(
        filler=SHORT_FILLER,
        setup_msgs=["My name is Alice."],
        post_filler_msgs=["Actually my name is Bob, not Alice. I misspoke earlier."],
        final_question="What is my name?",
        check_response=lambda r: "Bob" in r and "Alice" not in r,
    ),
    "numerical_precision": dict(
        filler=LONG_FILLER,
        setup_msgs=["My budget is exactly $47,382."],
        final_question="What is my exact budget?",
        check_response=lambda r: "47,382" in r or "47382" in r,
    ),
    "preference_accumulation": dict(
        filler=SHORT_FILLER,
        setup_msgs=[
            "I prefer concise answers.",
            "I work in finance.",
            "I am a visual learner.",
        ],
        final_question="Given what you know about me, how should you explain complex topics to me?",
        check_response=lambda r: "concise" in r.lower() and "visual" in r.lower() and "finance" in r.lower(),
    ),
}

MEMORY_CONFIGS = [
    (FullHistoryMemory, {}),
    (WindowMemory, {"window_size": 4}),
    (SummaryMemory, {"threshold": 6, "keep_recent": 2}),
]


@pytest.mark.parametrize("scenario_name,scenario", SCENARIOS.items())
@pytest.mark.parametrize("memory_cls,memory_kwargs", MEMORY_CONFIGS)
def test_memory_harness(agent, memory_cls, memory_kwargs, scenario_name, scenario):
    results = run_n_times(agent, memory_cls, memory_kwargs, scenario)
    print(
        f"\n[{scenario_name}][{memory_cls.__name__}] "
        f"correctness={results['correctness']}, "
        f"avg_chars={results['avg_chars']}, "
        f"avg_total_elapsed={results['avg_total_elapsed']}s"
    )