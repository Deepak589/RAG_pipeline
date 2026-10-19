"""Offline check of the spend guards (no API calls): python test_guards.py"""
import os
import tempfile
from argparse import Namespace

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import crag_pipeline as crag
from crag_eval import Judgement

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test")
api_calls = []


def fake_generate(self, messages, stop=None, run_manager=None, **kw):
    api_calls.append(1)
    usage = {"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000}
    if kw.get("tools"):
        msg = AIMessage(content="", usage_metadata=usage, tool_calls=[{
            "name": "Judgement", "id": "t1", "args": {"verdict": "correct", "reason": "ok"}}])
    else:
        msg = AIMessage(content="hi", usage_metadata=usage)
    return ChatResult(generations=[ChatGeneration(message=msg)])


ChatAnthropic._generate = fake_generate
with tempfile.TemporaryDirectory() as d:
    crag.enable_guards(Namespace(max_usd=2.5, llm_cache=f"{d}/c.sqlite"))
    llm = crag.make_chat_anthropic("claude-haiku-4-5", temperature=0)

    llm.invoke("q1")                                   # 1M in @ $1/MTok
    assert abs(crag.METER.usd - 1.0) < 1e-9, crag.METER.report()

    llm.invoke("q1")                                   # cache hit: free, no API call
    m = crag.METER.by_model["claude-haiku-4-5"]
    assert (m["calls"], m["cache_hits"], len(api_calls)) == (1, 1, 1), m

    j = llm.with_structured_output(Judgement).invoke("q2")   # structured path is metered too
    assert j.verdict == "correct" and abs(crag.METER.usd - 2.0) < 1e-9

    llm.invoke("q3")                                   # $3 > cap, but check runs before the call
    try:
        llm.invoke("q4")
        raise AssertionError("cap not enforced")
    except crag.BudgetExceeded:
        pass
    assert len(api_calls) == 3, "q4 must not reach the API"

    crag.METER.add("claude-unknown-9", {"input_tokens": 1_000_000})   # priced as most expensive
    assert abs(crag.METER.by_model["claude-unknown-9"]["usd"] - 10.0) < 1e-9

print("guards ok", crag.METER.report())
