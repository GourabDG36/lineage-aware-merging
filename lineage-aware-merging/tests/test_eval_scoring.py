"""Checks for the parts of eval_model.py that do not need a GPU, a model or a download."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_model as ev

ANSWER = "Janet sells 16 - 3 - 4 = 9 eggs.\nShe makes 9 * 2 = $18.\n#### 18"


def test_reference():
    assert ev.gsm8k_reference(ANSWER) == 18
    assert ev.gsm8k_reference("... #### 1,250") == 1250


def test_prediction_variants():
    assert ev.gsm8k_prediction("so 9*2 = 18\n#### 18") == 18
    assert ev.gsm8k_prediction("The answer is \\boxed{18}.") == 18
    assert ev.gsm8k_prediction("It costs $1,250.00 in total.") == 1250
    assert ev.gsm8k_prediction("first 5, then 7, final answer 12.") == 12
    assert ev.gsm8k_prediction("#### -3.5") == -3.5
    assert ev.gsm8k_prediction("I cannot solve this.") is None
    # the last '####' wins over earlier numbers and an earlier '####'
    assert ev.gsm8k_prediction("#### 5\nwait, recheck: 7 apples\n#### 7") == 7


def test_correct():
    assert ev.gsm8k_correct("work...\n#### 18", ANSWER)
    assert not ev.gsm8k_correct("work...\n#### 19", ANSWER)
    assert not ev.gsm8k_correct("no number here", ANSWER)


def test_extract_code():
    gen = "Here you go:\n```python\ndef f(x):\n    return x + 1\n```\nDone."
    assert ev.extract_code(gen).strip() == "def f(x):\n    return x + 1"
    assert ev.extract_code("```\ndef g():\n    return 2\n```").strip() == "def g():\n    return 2"
    # cut off by the token limit before the closing fence
    assert ev.extract_code("```python\ndef h():\n    return 3\n").strip() == "def h():\n    return 3"
    assert ev.extract_code("def k():\n    return 4") == "def k():\n    return 4"


def test_run_mbpp_tests():
    tests = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]
    assert ev.run_mbpp_tests("def add(a, b):\n    return a + b", "", tests)
    assert not ev.run_mbpp_tests("def add(a, b):\n    return a - b", "", tests)
    assert not ev.run_mbpp_tests("def add(a, b) return", "", tests)          # syntax error
    assert not ev.run_mbpp_tests("while True:\n    pass", "", tests, timeout=2)  # hangs
    assert ev.run_mbpp_tests("def sq(x):\n    return math.sqrt(x)", "import math",
                             ["assert sq(9) == 3"])                            # setup code


def test_score_dispatch():
    assert ev.score("gsm8k", {"answer": ANSWER}, "#### 18")
    item = {"tests": ["assert add(1, 2) == 3"], "setup": ""}
    assert ev.score("mbpp", item, "```python\ndef add(a, b):\n    return a + b\n```")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all passed")
