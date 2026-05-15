"""Stage 2: Quality evaluation via round-robin 1v1 duels.

Each qualified miner generates answers to questions.
An LLM judge (OpenAI) picks the winner for each pair.
"""

import os
import tempfile
import torch
import tiktoken
import numpy as np
import bittensor as bt
import asyncio
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from .chronogpt_model import load_model
from .model_store import download_model, parse_repo, get_device

tokenizer = tiktoken.get_encoding("gpt2")


def generate_answer(model, device, question, max_new_tokens=128):
    """Generate an answer from the model."""
    tokens = torch.tensor(tokenizer.encode(question), dtype=torch.long).unsqueeze(0).to(device)
    xgen = tokens.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(xgen)[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            next_token = torch.gather(topk_indices, -1, torch.multinomial(topk_probs, 1))
            xgen = torch.cat([xgen, next_token], dim=1)
    return tokenizer.decode(xgen[0][tokens.shape[1]:].tolist())


JUDGE_SYSTEM_PROMPT = """You are a judge evaluating two AI-generated answers to a question.

Evaluate based on:
1. Factual accuracy
2. Relevance to the question
3. Coherence and clarity
4. Completeness

You MUST respond with exactly one of these three words: a, b, tie
Do not explain your reasoning. Just output the single word."""


class Judge:
    def __init__(self, model=None):
        self.model = model or os.environ.get("JUDGE_MODEL", "gpt-5.4-mini-2026-03-17")
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=5)

    async def judge_one(self, question, answer_a, answer_b):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                ChatCompletionSystemMessageParam(role="system", content=JUDGE_SYSTEM_PROMPT),
                ChatCompletionUserMessageParam(role="user", content=(
                    f"Question: {question}\n\n"
                    f"Answer A:\n{answer_a}\n\n"
                    f"Answer B:\n{answer_b}"
                )),
            ],
            max_completion_tokens=5,
            temperature=0,
        )

        result = response.choices[0].message.content.strip().lower()
        if result in ("a", "b", "tie"):
            return result
        if "a" in result and "b" not in result:
            return "a"
        if "b" in result and "a" not in result:
            return "b"
        return "tie"

    def __call__(self, question, answer_a, answer_b):
        return asyncio.run(self.judge_one(question, answer_a, answer_b))

    async def judge_batch(self, tasks):
        """Judge multiple (question, answer_a, answer_b) tuples in parallel."""
        return await asyncio.gather(*[self.judge_one(q, a, b) for q, a, b in tasks])


judge = Judge()


def duel(miner_answers, uid_a, uid_b, questions):
    """Run a duel between two miners. Returns winner uid or None for tie."""
    tasks = [(q, miner_answers[uid_a][i], miner_answers[uid_b][i]) for i, q in enumerate(questions)]
    results = asyncio.run(judge.judge_batch(tasks))

    wins_a = 0
    wins_b = 0
    for q_idx, winner in enumerate(results):
        if winner == "a":
            wins_a += 1
        elif winner == "b":
            wins_b += 1
        bt.logging.info(f"    Q{q_idx}: winner={winner}")

    bt.logging.info(f"  UID {uid_a} ({wins_a}) vs UID {uid_b} ({wins_b})")

    if wins_a > wins_b:
        return uid_a
    elif wins_b > wins_a:
        return uid_b
    return None


def run_quality_duels(qualified, submissions, questions, metagraph):
    """Round-robin 1v1 duels between qualified miners.

    Returns:
        np.array of win rates (indexed by uid, 0-1).
    """
    # Generate answers for each qualified miner
    miner_answers = {}
    for uid, _ in qualified:
        bt.logging.info(f"UID {uid}: generating answers")
        repo_str = list(submissions[uid].values())[0]
        repo_id, revision = parse_repo(repo_str)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = download_model(repo_id, tmpdir, revision=revision)
                model = load_model(path, get_device())
                answers = [generate_answer(model, get_device(), q) for q in questions]
                miner_answers[uid] = answers
                del model
        except Exception as e:
            bt.logging.error(f"UID {uid}: answer generation FAILED — {type(e).__name__}")
            miner_answers[uid] = [""] * len(questions)

    # Round-robin: every pair duels
    uids = [uid for uid, _ in qualified]
    wins = {uid: 0 for uid in uids}

    for i in range(len(uids)):
        for j in range(i + 1, len(uids)):
            uid_a, uid_b = uids[i], uids[j]
            bt.logging.info(f"Duel: UID {uid_a} vs UID {uid_b}")
            winner = duel(miner_answers, uid_a, uid_b, questions)

            if winner == uid_a:
                wins[uid_a] += 1
            elif winner == uid_b:
                wins[uid_b] += 1

    # Win rate = wins / total possible wins
    total_opponents = max(1, len(uids) - 1)
    win_rates = np.zeros(metagraph.n)
    for uid in uids:
        win_rates[uid] = wins[uid] / total_opponents
        bt.logging.info(f"UID {uid}: wins={wins[uid]}/{total_opponents} win_rate={win_rates[uid]:.4f}")

    return win_rates
