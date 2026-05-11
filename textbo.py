"""Streamlit app for prompt-search-based ad generation."""

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import mimetypes
import os
import random
import threading
import time
import traceback
from typing import Any

import streamlit as st
import tiktoken

from agent import (
    build_generation_prompt,
    chat,
    extract_fields,
    get_openai_client,
    try_parse_brief,
)
from generate import generate_image_gemini, generate_image_openai
from schema import STYLE_DIR, STYLE_PRESETS

REFLECTION_MODEL = "gpt-5-nano"
PROMPT_GENERATION_MODEL = "gpt-4o"
EVAL_MODEL = "gpt-4o"
REFLECTION_REASONING_EFFORT = "minimal"
DEFAULT_INITIAL_PROMPTS = 10
DEFAULT_LOWEST_PROMPTS = 5
DEFAULT_OPTIMIZATION_STEPS = 10
DEFAULT_EFFICIENT_CANDIDATES = 2
DEFAULT_TEXTBO_GRADIENT_STEPS = 2
HIDDEN_BASELINE_CANDIDATES = 1
HIDDEN_BASELINE_GRADIENT_STEPS = 1
DEFAULT_JUDGE_REPEATS = 1
DEFAULT_PERSONA_COUNT = 5
MAX_PERSONA_FILES = 5
MAX_PERSONA_TOKENS = 5000
PERSONA_DIR = os.path.join(os.path.dirname(__file__), "persona")
TOURNAMENT_COMPARISONS = 3
EVAL_CONCURRENCY_LIMIT = 5
EVAL_REQUEST_RETRIES = 3
EVAL_RETRY_BASE_SECONDS = 0.75
DEFAULT_INITIAL_PARALLELISM = 10
DEFAULT_BASE_CHAIN_PARALLELISM = 10
DEFAULT_EFFICIENT_TRAJECTORY_PARALLELISM = 10
DEFAULT_EFFICIENT_PRESCORE_PARALLELISM = 10
JUDGE_CALIBRATION = """Calibration:
Score relative to professional paid ads, not relative to ordinary AI outputs.
Treat 3 as the default ceiling for ads that are merely usable, attractive, or coherent.
Score 4 only when the ad is production-ready, campaign-specific, and has no major weakness.
Score 5 only for rare exceptional ads that are clearly stronger than most professional paid ads.
If uncertain between two scores, choose the lower score.
If the ad is attractive but generic, score at most 3.
If the product, message, audience fit, or CTA is unclear, score at most 3.
If requested text appears unreadable, distorted, or missing, score at most 2.
Penalize generic product staging, vague audience fit, weak CTA visibility, unreadable text, artificial-looking composition, or missing brand distinctiveness."""


_PERSONA_CACHE: list[dict[str, str]] | None = None
_EVAL_REQUEST_SEMAPHORE = threading.BoundedSemaphore(EVAL_CONCURRENCY_LIMIT)


def _supports_custom_sampling(model: str) -> bool:
    """Return whether the model accepts non-default sampling params."""
    return not model.startswith("gpt-5")


def _uses_reasoning_effort(model: str) -> bool:
    """Return whether the model accepts reasoning-effort controls."""
    return model.startswith("gpt-5")


def _text_request_kwargs(
    model: str,
    *,
    temperature: float,
    top_p: float | None,
    max_completion_tokens: int,
) -> dict[str, Any]:
    """Return sampling params supported by the configured text model."""
    params: dict[str, Any] = {"max_completion_tokens": max_completion_tokens}
    if _uses_reasoning_effort(model):
        params["reasoning_effort"] = REFLECTION_REASONING_EFFORT
    if _supports_custom_sampling(model):
        params["temperature"] = temperature
    if _supports_custom_sampling(model) and top_p is not None:
        params["top_p"] = top_p
    return params


def _is_output_limit_error(exc: BaseException) -> bool:
    """Detect OpenAI errors caused by an unfinished, overlong response."""
    message = str(exc)
    return (
        "max_tokens or model output limit" in message
        or "Could not finish the message" in message
    )


def _short_error(exc: BaseException) -> str:
    """Compact exception text for score diagnostics."""
    message = " ".join(str(exc).split())
    if len(message) > 500:
        message = message[:497].rstrip() + "..."
    return f"{type(exc).__name__}: {message}"


def _is_retryable_api_error(exc: BaseException) -> bool:
    """Return whether an evaluator API failure is likely transient."""
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    retry_markers = (
        "rate limit",
        "timeout",
        "timed out",
        "temporarily",
        "overloaded",
        "server error",
        "connection",
        "try again",
        "429",
        "500",
        "502",
        "503",
        "504",
    )
    return any(marker in message for marker in retry_markers)


def _request_eval_completion(client, **kwargs):
    """Run GPT-4o evaluator calls with bounded concurrency and retries."""
    last_exc: BaseException | None = None
    for attempt in range(EVAL_REQUEST_RETRIES):
        try:
            with _EVAL_REQUEST_SEMAPHORE:
                return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= EVAL_REQUEST_RETRIES - 1 or not _is_retryable_api_error(exc):
                raise
            delay = EVAL_RETRY_BASE_SECONDS * (2**attempt) + random.random() * 0.25
            time.sleep(delay)
    raise last_exc


def _failure_probs() -> dict[str, float]:
    """Use a pessimistic fallback when scoring/parsing fails."""
    return {str(i): (1.0 if i == 1 else 0.0) for i in range(1, 6)}


def _one_hot_probs(selected: str | None) -> dict[str, float]:
    """Create a one-hot probability distribution for 1-5."""
    if selected not in {"1", "2", "3", "4", "5"}:
        return _failure_probs()
    return {str(i): (1.0 if str(i) == selected else 0.0) for i in range(1, 6)}


def _parse_score_digit(text: str) -> str | None:
    """Extract a score digit from model text."""
    stripped = (text or "").strip()
    return stripped if stripped in {"1", "2", "3", "4", "5"} else None


def _parse_json_text(text: str) -> dict[str, Any]:
    """Parse JSON from a model response with simple fence stripping."""
    raw = (text or "").strip()
    if not raw:
        return {}

    if "```json" in raw:
        start = raw.index("```json") + 7
        end = raw.index("```", start)
        raw = raw[start:end].strip()
    elif "```" in raw:
        start = raw.index("```") + 3
        end = raw.index("```", start)
        raw = raw[start:end].strip()
    elif "{" in raw and "}" in raw:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        raw = raw[start:end]

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}


def _request_json_object(
    client,
    system_prompt: str,
    user_prompt: str | list[dict[str, Any]],
    *,
    model: str = REFLECTION_MODEL,
    temperature: float = 1.0,
    top_p: float | None = None,
    max_completion_tokens: int = 8192,
) -> dict[str, Any]:
    """Request a JSON object from a text model."""
    request_kwargs = _text_request_kwargs(
        model,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            **request_kwargs,
        )
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **request_kwargs,
        )
    return _parse_json_text(response.choices[0].message.content or "{}")


def _request_text(
    client,
    system_prompt: str,
    user_prompt: str | list[dict[str, Any]],
    *,
    model: str = REFLECTION_MODEL,
    temperature: float = 1.0,
    top_p: float | None = None,
    max_completion_tokens: int = 4096,
) -> str:
    """Request plain text from a text model."""
    token_limits = [max_completion_tokens]
    if max_completion_tokens < 8192:
        token_limits.append(8192)

    last_exc: BaseException | None = None
    for token_limit in token_limits:
        request_kwargs = _text_request_kwargs(
            model,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=token_limit,
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **request_kwargs,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            if not _is_output_limit_error(exc):
                raise

    raise last_exc


def _prompt_excerpt(prompt: str, max_chars: int = 180) -> str:
    """Short prompt preview for UI and meta-reflection."""
    compact = " ".join((prompt or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _normalize_prompt(prompt: str) -> str:
    """Normalize prompt text for dedupe and display."""
    return " ".join((prompt or "").split())


def _dedupe_prompts(prompts: list[str]) -> list[str]:
    """Deduplicate prompts while preserving order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for prompt in prompts:
        cleaned = _normalize_prompt(prompt)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _format_exception_traceback(exc: BaseException) -> str:
    """Format an exception traceback for the debug UI."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _record_debug_event(
    stage: str,
    message: str,
    *,
    error_type: str | None = None,
    traceback_text: str | None = None,
) -> None:
    """Persist a lightweight debug event across reruns."""
    events = list(st.session_state.get("debug_events", []))
    events.append(
        {
            "stage": stage,
            "message": message,
            "error_type": error_type,
            "traceback": traceback_text,
        }
    )
    st.session_state.debug_events = events[-100:]


def _path_to_data_url(path: str) -> str:
    """Convert a local image path into a data URL for multimodal judging."""
    mime, _ = mimetypes.guess_type(path)
    subtype = "png" if mime is None else mime.split("/")[-1]
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{subtype};base64,{encoded}"


def _load_personas() -> list[dict[str, str]]:
    """Load the fixed five local personas used for TextBO evaluation."""
    global _PERSONA_CACHE
    if _PERSONA_CACHE is not None:
        return _PERSONA_CACHE

    personas: list[dict[str, str]] = []
    if not os.path.isdir(PERSONA_DIR):
        _PERSONA_CACHE = []
        return _PERSONA_CACHE

    persona_files = sorted(
        os.path.join(PERSONA_DIR, filename)
        for filename in os.listdir(PERSONA_DIR)
        if filename.startswith("pid_") and filename.endswith("_mega_persona.txt")
    )
    for path in persona_files[:MAX_PERSONA_FILES]:
        filename = os.path.basename(path)
        parts = filename.split("_")
        persona_id = parts[1] if len(parts) > 1 else filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except OSError:
            continue
        if text:
            personas.append({"persona_id": persona_id, "text": text, "path": path})

    _PERSONA_CACHE = personas
    return _PERSONA_CACHE


def _persona_sample_text(persona: dict[str, str]) -> str:
    """Return the first persona tokens used in judge prompts."""
    text = persona.get("text", "")
    encoding = tiktoken.encoding_for_model(EVAL_MODEL)
    tokens = encoding.encode(text)
    if len(tokens) <= MAX_PERSONA_TOKENS:
        return text
    return encoding.decode(tokens[:MAX_PERSONA_TOKENS]).rstrip()


def _campaign_context_text(
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
) -> str:
    """Format the campaign context for generation and judging."""
    brief_json = json.dumps(creative_brief, indent=2, ensure_ascii=False)
    lines = [
        f"Product / Service: {session_data.get('product_name', '')}",
        f"Target Audience: {session_data.get('target_audience', '')}",
        f"Campaign Goal: {session_data.get('campaign_goal', '')}",
        f"Key Message / CTA: {session_data.get('key_message', '')}",
        f"Brand Tone: {session_data.get('brand_tone', '')}",
        f"Style Reference: {session_data.get('style_reference', '')}",
        f"Model: {sidebar_settings.get('model', '')}",
        f"Resolution: {sidebar_settings.get('resolution', '')}",
        f"Style Direction: {sidebar_settings.get('style_description', '')}",
        "",
        "Approved Creative Brief:",
        brief_json,
    ]
    return "\n".join(lines)


def _history_summary(
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 3,
    bottom_n: int = 3,
) -> str:
    """Summarize the best and worst prompts in a history list."""
    if not candidates:
        return "No prior history."

    sorted_candidates = sorted(candidates, key=lambda c: c.get("score", 0.0))
    bottom = sorted_candidates[:bottom_n]
    top = list(reversed(sorted_candidates[-top_n:]))

    lines = ["Highest-scoring prompts:"]
    if top:
        for entry in top:
            lines.append(
                f"- score={entry.get('score', 0.0):.3f} | {_prompt_excerpt(entry.get('prompt', ''))}"
            )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Lowest-scoring prompts:")
    if bottom:
        for entry in bottom:
            lines.append(
                f"- score={entry.get('score', 0.0):.3f} | {_prompt_excerpt(entry.get('prompt', ''))}"
            )
    else:
        lines.append("- none")

    return "\n".join(lines)


def _reflection_examples(
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 5,
    bottom_n: int = 5,
) -> list[dict[str, Any]]:
    """Select ranked high/low examples for multimodal reflection."""
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda c: c.get("score", 0.0))
    ranked_candidates = list(enumerate(sorted_candidates, start=1))
    total = len(ranked_candidates)
    examples: list[dict[str, Any]] = []

    if total < top_n + bottom_n:
        midpoint = total // 2
        lower_half = ranked_candidates[:midpoint]
        upper_half = ranked_candidates[midpoint:]

        for overall_rank, entry in lower_half:
            examples.append(
                {
                    "entry": entry,
                    "overall_rank": overall_rank,
                    "total": total,
                    "performance_label": "LOWER HALF",
                }
            )
        for overall_rank, entry in upper_half:
            examples.append(
                {
                    "entry": entry,
                    "overall_rank": overall_rank,
                    "total": total,
                    "performance_label": "UPPER HALF",
                }
            )
        return examples

    bottom_examples = ranked_candidates[:bottom_n]
    top_examples = list(reversed(ranked_candidates[-top_n:]))

    for cohort_rank, (overall_rank, entry) in enumerate(bottom_examples, start=1):
        examples.append(
            {
                "entry": entry,
                "overall_rank": overall_rank,
                "total": total,
                "performance_label": f"{cohort_rank} WORST PERFORMING",
            }
        )

    for cohort_rank, (overall_rank, entry) in enumerate(top_examples, start=1):
        examples.append(
            {
                "entry": entry,
                "overall_rank": overall_rank,
                "total": total,
                "performance_label": f"{cohort_rank} BEST PERFORMING",
            }
        )

    return examples


def _build_multimodal_reflection_content(
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 5,
    bottom_n: int = 5,
) -> list[dict[str, Any]]:
    """Build reference-style visual pattern analysis content from history."""
    examples = _reflection_examples(candidates, top_n=top_n, bottom_n=bottom_n)
    if not examples:
        return []

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are an expert at analyzing visual patterns in advertising performance.\n\n"
                "VISUAL ANALYSIS TASK: I will show you images from the lowest-scoring and "
                "highest-scoring ad iterations. Identify specific visual patterns that "
                "distinguish effective from ineffective ads.\n\n"
                "VISUAL EXAMPLES - BEST VS WORST PERFORMING:"
            ),
        }
    ]

    for example in examples:
        entry = example["entry"]
        output_path = str(entry.get("output_path") or "").strip()
        example_text = (
            f"{example['performance_label']} "
            f"(Rank #{example['overall_rank']}/{example['total']}, "
            f"Score: {entry.get('score', 0.0):.3f})\n"
            f"Prompt excerpt: {_prompt_excerpt(entry.get('prompt', ''), max_chars=240)}"
        )
        content.append({"type": "text", "text": example_text})

        if output_path and os.path.exists(output_path):
            try:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _path_to_data_url(output_path)},
                    }
                )
            except Exception:
                content.append(
                    {
                        "type": "text",
                        "text": "[Image could not be loaded for this example.]",
                    }
                )
        else:
            content.append(
                {
                    "type": "text",
                    "text": "[Rendered image not available for this example.]",
                }
            )

    return content


def _score_from_probs(probs: dict[str, float]) -> float:
    """Convert 1-5 probabilities into an expected score."""
    return sum(int(k) * probs[k] for k in sorted(probs.keys()))


def _extract_digit_probs_from_completion(response) -> dict[str, float]:
    """Extract 1-5 token probabilities from OpenAI chat logprobs."""
    probs = {str(i): 0.0 for i in range(1, 6)}
    choice = response.choices[0]
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs else None

    if content:
        token_info = content[0]
        for top in getattr(token_info, "top_logprobs", []) or []:
            token = (top.token or "").strip()
            if token in probs:
                probs[token] = math.exp(top.logprob)
        chosen_token = (getattr(token_info, "token", "") or "").strip()
        chosen_logprob = getattr(token_info, "logprob", None)
        if chosen_token in probs and probs[chosen_token] == 0.0 and chosen_logprob is not None:
            probs[chosen_token] = math.exp(chosen_logprob)

    total = sum(probs.values())
    if total <= 0:
        content_text = (choice.message.content or "").strip()
        if content_text in probs:
            probs[content_text] = 1.0
            total = 1.0

    if total <= 0:
        return _failure_probs()

    return {key: value / total for key, value in probs.items()}


def _score_prompt_with_logprobs(
    client,
    prompt: str,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    *,
    seed: int | None = None,
    persona: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Text-only pre-score for prompt selection."""
    persona_id = persona.get("persona_id") if persona else None
    if persona:
        judge_prompt = f"""PERSONA DATA:
{_persona_sample_text(persona)}

Campaign context:
{_campaign_context_text(session_data, sidebar_settings, creative_brief)}

Candidate prompt:
{prompt}

TASK:
Return only one item from ["1","2","3","4","5"] for ad effectiveness.

Effective Score Scale Definition:
1: Extremely Unlikely. The persona would actively ignore or be annoyed by this ad.
2: Unlikely. The persona would likely scroll past without a second thought.
3: Mediocre. It is hard to decide whether the persona would click or not click.
4: Likely. The persona is intrigued and has a good chance of clicking to learn more.
5: Extremely Likely. The persona is the ideal target; a click is almost certain.

No explanation. Just the score."""
    else:
        judge_prompt = f"""You are rating an image-generation prompt for a single advertising campaign.

Campaign context:
{_campaign_context_text(session_data, sidebar_settings, creative_brief)}

Candidate prompt:
{prompt}

Return exactly one token from ["1","2","3","4","5"].

Scale:
1 = very unlikely to produce an effective ad for this campaign
2 = somewhat unlikely to produce an effective ad
3 = acceptable but ordinary, or not yet clearly strong for this campaign
4 = strong, with clear evidence across most dimensions and no obvious weakness
5 = exceptional, near-production-ready, and rare for this campaign

{JUDGE_CALIBRATION}

Judge based on likely audience fit, clarity, visual specificity, brand-tone alignment, CTA support, and style adherence.
Use 3 for prompts that seem serviceable but generic, incomplete, or not yet clearly compelling.
Use 4 only when the prompt shows clear strength across most dimensions without an obvious weakness.
Use 5 only for unusually strong prompts that are highly specific, cohesive, campaign-aligned, and likely to yield standout output.
Do not explain your answer."""

    try:
        response = _request_eval_completion(
            client,
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": "Return exactly one token: 1, 2, 3, 4, or 5."},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
            max_completion_tokens=1,
            logprobs=True,
            top_logprobs=5,
            seed=seed,
        )
        probs = _extract_digit_probs_from_completion(response)
        mode = "prompt-persona-logprobs" if persona else "prompt-logprobs"
        error = None
    except Exception as logprob_exc:
        try:
            response = _request_eval_completion(
                client,
                model=EVAL_MODEL,
                messages=[
                    {"role": "system", "content": "Return exactly one token: 1, 2, 3, 4, or 5."},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0,
                max_completion_tokens=8,
                seed=seed,
            )
            probs = _one_hot_probs(_parse_score_digit(response.choices[0].message.content or ""))
            mode = "prompt-persona-plain" if persona else "prompt-plain"
            error = _short_error(logprob_exc)
        except Exception as plain_exc:
            probs = _failure_probs()
            mode = "prompt-persona-fallback" if persona else "prompt-fallback"
            error = f"logprobs failed: {_short_error(logprob_exc)} | plain failed: {_short_error(plain_exc)}"
    return {
        "score": _score_from_probs(probs),
        "probs": probs,
        "mode": mode,
        "persona_id": persona_id,
        "error": error,
    }


def _score_image_with_logprobs(
    client,
    image_path: str,
    prompt: str,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    *,
    seed: int | None = None,
    persona: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Multimodal score for a rendered advertisement."""
    persona_id = persona.get("persona_id") if persona else None
    if persona:
        judge_text = f"""PERSONA DATA:
{_persona_sample_text(persona)}

TASK:
Return only one item from ["1","2","3","4","5"] for ad effectiveness.

Effective Score Scale Definition:
1: Extremely Unlikely. The persona would actively ignore or be annoyed by this ad.
2: Unlikely. The persona would likely scroll past without a second thought.
3: Mediocre. It is hard to decide whether the persona would click or not click.
4: Likely. The persona is intrigued and has a good chance of clicking to learn more.
5: Extremely Likely. The persona is the ideal target; a click is almost certain.

No explanation. Just the score."""
    else:
        judge_text = f"""Evaluate this advertisement candidate for the following campaign.

Campaign context:
{_campaign_context_text(session_data, sidebar_settings, creative_brief)}

Rendering prompt:
{prompt}

Return exactly one token from ["1","2","3","4","5"].

Scale:
1 = ineffective and poorly aligned with the campaign
2 = weak and unlikely to engage the target audience
3 = acceptable but ordinary, or not yet clearly strong for this campaign
4 = strong, with clear evidence across most dimensions and no obvious weakness
5 = exceptional, near-production-ready, and rare for this campaign

{JUDGE_CALIBRATION}

Judge based on overall ad effectiveness, audience fit, visual clarity, message delivery, CTA support, and style consistency.
Use 3 for ads that are serviceable but generic, uneven, or missing clear strength.
Use 4 only when the ad shows clear strength across most dimensions without an obvious weakness.
Use 5 only for unusually strong ads that look standout, campaign-aligned, and close to production-ready.
Do not explain your answer."""

    try:
        image_url = _path_to_data_url(image_path)
    except Exception as image_file_exc:
        try:
            fallback = _score_prompt_with_logprobs(
                client,
                prompt,
                session_data,
                sidebar_settings,
                creative_brief,
                seed=seed,
                persona=persona,
            )
            return {
                "score": fallback["score"],
                "probs": fallback["probs"],
                "mode": f"prompt-fallback:{fallback.get('mode', 'unknown')}",
                "persona_id": persona_id,
                "error": f"image load failed: {_short_error(image_file_exc)}",
            }
        except Exception as prompt_exc:
            probs = _failure_probs()
            return {
                "score": _score_from_probs(probs),
                "probs": probs,
                "mode": "image-persona-fallback" if persona else "image-fallback",
                "persona_id": persona_id,
                "error": (
                    f"image load failed: {_short_error(image_file_exc)} | "
                    f"prompt fallback failed: {_short_error(prompt_exc)}"
                ),
            }

    try:
        response = _request_eval_completion(
            client,
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": "Return exactly one token: 1, 2, 3, 4, or 5."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": judge_text},
                    ],
                },
            ],
            temperature=0,
            max_completion_tokens=1,
            logprobs=True,
            top_logprobs=5,
            seed=seed,
        )
        probs = _extract_digit_probs_from_completion(response)
        mode = "image-persona-logprobs" if persona else "image-logprobs"
        error = None
    except Exception as logprob_exc:
        try:
            response = _request_eval_completion(
                client,
                model=EVAL_MODEL,
                messages=[
                    {"role": "system", "content": "Return exactly one token: 1, 2, 3, 4, or 5."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": judge_text},
                        ],
                    },
                ],
                temperature=0,
                max_completion_tokens=8,
                seed=seed,
            )
            probs = _one_hot_probs(_parse_score_digit(response.choices[0].message.content or ""))
            mode = "image-persona-plain" if persona else "image-plain"
            error = _short_error(logprob_exc)
        except Exception as image_exc:
            try:
                fallback = _score_prompt_with_logprobs(
                    client,
                    prompt,
                    session_data,
                    sidebar_settings,
                    creative_brief,
                    seed=seed,
                    persona=persona,
                )
                probs = fallback["probs"]
                mode = f"prompt-fallback:{fallback.get('mode', 'unknown')}"
                error = (
                    f"image logprobs failed: {_short_error(logprob_exc)} | "
                    f"image plain failed: {_short_error(image_exc)}"
                )
                if fallback.get("error"):
                    error += f" | prompt fallback: {fallback['error']}"
            except Exception as prompt_exc:
                probs = _failure_probs()
                mode = "image-persona-fallback" if persona else "image-fallback"
                error = (
                    f"image logprobs failed: {_short_error(logprob_exc)} | "
                    f"image plain failed: {_short_error(image_exc)} | "
                    f"prompt fallback failed: {_short_error(prompt_exc)}"
                )
    return {
        "score": _score_from_probs(probs),
        "probs": probs,
        "mode": mode,
        "persona_id": persona_id,
        "error": error,
    }


def _aggregate_soft_scores(
    scorer,
    *,
    repeats: int,
    seed_base: int | None = None,
) -> dict[str, Any]:
    """Average judge distributions but use a pessimistic score across repeats."""
    all_probs: list[dict[str, float]] = []
    repeat_scores: list[float] = []
    mode = None
    for idx in range(max(1, repeats)):
        try:
            result = scorer(None if seed_base is None else seed_base + idx)
        except Exception:
            result = {"probs": _failure_probs(), "mode": "fallback"}
        all_probs.append(result["probs"])
        repeat_scores.append(_score_from_probs(result["probs"]))
        mode = result.get("mode")

    averaged = {
        key: sum(prob[key] for prob in all_probs) / len(all_probs)
        for key in all_probs[0].keys()
    }
    mean_score = _score_from_probs(averaged)
    if len(repeat_scores) > 1:
        variance = sum((score - mean_score) ** 2 for score in repeat_scores) / len(repeat_scores)
        conservative_score = max(1.0, mean_score - math.sqrt(variance))
        aggregation_mode = "mean-minus-std"
    else:
        conservative_score = mean_score
        aggregation_mode = "single"

    return {
        "score": conservative_score,
        "probs": averaged,
        "mode": f"{mode or 'unknown'}:{aggregation_mode}",
        "mean_score": mean_score,
        "repeat_scores": repeat_scores,
        "score_aggregation": aggregation_mode,
    }


def _aggregate_persona_scores(
    scorer,
    *,
    seed_base: int | None = None,
    fallback_scorer=None,
    fallback_repeats: int = 1,
) -> dict[str, Any]:
    """Aggregate exactly the local persona score distributions."""
    available_personas = _load_personas()[:MAX_PERSONA_FILES]
    if available_personas and DEFAULT_PERSONA_COUNT < len(available_personas):
        rng = random.Random((seed_base or 0) + 1000)
        personas = rng.sample(available_personas, DEFAULT_PERSONA_COUNT)
    else:
        personas = available_personas
    if not personas:
        if fallback_scorer is None:
            return {
                "score": 1.0,
                "probs": _failure_probs(),
                "mode": "persona-missing",
                "mean_score": 1.0,
                "repeat_scores": [1.0],
                "score_aggregation": "persona-missing",
                "persona_scores": {},
                "persona_count": 0,
                "persona_errors": {},
            }
        fallback = _aggregate_soft_scores(
            fallback_scorer,
            repeats=fallback_repeats,
            seed_base=seed_base,
        )
        fallback["mode"] = f"{fallback.get('mode', 'unknown')}:no-persona-fallback"
        fallback["persona_scores"] = {}
        fallback["persona_count"] = 0
        fallback["persona_errors"] = {}
        return fallback

    all_probs: list[dict[str, float]] = []
    persona_scores: dict[str, float] = {}
    persona_errors: dict[str, str] = {}
    mode = None
    for idx, persona in enumerate(personas):
        seed = None if seed_base is None else seed_base + idx
        persona_id = persona.get("persona_id", str(idx + 1))
        try:
            result = scorer(persona, seed)
        except Exception as exc:
            result = {
                "probs": _failure_probs(),
                "mode": "persona-fallback",
                "persona_id": persona_id,
                "error": _short_error(exc),
            }

        probs = result.get("probs") or _failure_probs()
        all_probs.append(probs)
        persona_scores[persona_id] = _score_from_probs(probs)
        if result.get("error"):
            persona_errors[persona_id] = str(result["error"])
        mode = result.get("mode") or mode

    averaged = {
        key: sum(prob[key] for prob in all_probs) / len(all_probs)
        for key in all_probs[0].keys()
    }
    mean_score = _score_from_probs(averaged)
    individual_scores = list(persona_scores.values())
    aggregation_mode = f"persona-mean:n={len(personas)}"

    return {
        "score": mean_score,
        "probs": averaged,
        "mode": f"{mode or 'unknown'}:{aggregation_mode}",
        "mean_score": mean_score,
        "repeat_scores": individual_scores,
        "score_aggregation": aggregation_mode,
        "persona_scores": persona_scores,
        "persona_count": len(personas),
        "persona_errors": persona_errors,
    }


def _generate_initial_prompt_variants(
    client,
    base_prompt: str,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    *,
    total_count: int,
) -> list[str]:
    """Generate the starting set of prompt variants."""
    if total_count <= 1:
        return [base_prompt]

    campaign_context = _campaign_context_text(session_data, sidebar_settings, creative_brief)
    system_prompt = (
        "You generate polished image-generation prompts for advertising creatives. "
        "Return JSON only."
    )
    prompts = [base_prompt]
    seen_prompt_text = {_normalize_prompt(base_prompt).lower()}
    creative_routes = [
        "product hero close-up with distinctive staging and tactile detail",
        "adult lifestyle moment with a clear action and emotional hook",
        "environment-led scene where the setting tells the campaign story",
        "bold graphic/social-feed layout with strong negative space for copy",
        "dynamic motion or before-after contrast that dramatizes the benefit",
        "premium editorial composition with distinctive lighting and camera angle",
        "social proof scene with adults interacting around the product or outcome",
        "unexpected metaphorical visual that still makes the product and CTA clear",
        "minimal studio product shot with unusual prop, surface, or color strategy",
        "immersive point-of-view scene that puts the viewer inside the use case",
    ]

    for variant_idx in range(2, total_count + 1):
        route = creative_routes[(variant_idx - 2) % len(creative_routes)]
        generated_prompt = ""
        for attempt in range(2):
            prior_prompt_text = "\n".join(
                f"- {_prompt_excerpt(prompt, max_chars=180)}"
                for prompt in prompts
            )
            retry_note = (
                ""
                if attempt == 0
                else "\nThe previous attempt was too similar. Choose a more different scene, camera language, and subject action."
            )
            user_prompt = f"""Create one highly distinct image-generation prompt for ad candidate #{variant_idx}.

Locked campaign context:
{campaign_context}

Approved base prompt:
{base_prompt}

Recently generated prompts to avoid repeating:
{prior_prompt_text}

Creative route for this candidate:
{route}

Requirements:
- Keep the same product, audience, campaign goal, core message, tone, style direction, and aspect ratio.
- Make this candidate clearly different from every prior prompt above.
- Use a different scene, setting, subject action, camera framing, visual hierarchy, and copy-placement strategy from the prior prompts whenever possible.
- Focus on one strong, coherent ad idea rather than small wording changes.
- If humans appear, use adults only.
- The prompt must be self-contained and ready for image generation.
- Keep the prompt concise and under 220 words.
- Return JSON with one key: "prompt".
{retry_note}
"""

            try:
                payload = _request_json_object(
                    client,
                    system_prompt,
                    user_prompt,
                    model=PROMPT_GENERATION_MODEL,
                    temperature=1.2,
                    top_p=0.95,
                    max_completion_tokens=2048,
                )
                prompt = _normalize_prompt(str(payload.get("prompt", "")).strip())
            except Exception:
                prompt = ""

            prompt_key = prompt.lower()
            if prompt and prompt_key not in seen_prompt_text:
                generated_prompt = prompt
                break

        if not generated_prompt:
            generated_prompt = _normalize_prompt(
                f"{base_prompt} Creative variation: {route}. Use a clearly different scene, "
                "setting, subject action, camera framing, visual hierarchy, and copy-placement "
                "strategy from the approved base prompt while preserving the same campaign."
            )

        if generated_prompt:
            prompts.append(generated_prompt)
            seen_prompt_text.add(generated_prompt.lower())

    prompts = _dedupe_prompts(prompts)
    while len(prompts) < total_count:
        prompts.append(base_prompt)
    return prompts[:total_count]


def _generate_base_revision(
    client,
    current_prompt: str,
    current_score: float,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    chain_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate one prompt revision for the base optimizer."""
    history_text = _history_summary(chain_history, top_n=2, bottom_n=2)
    system_prompt = "You improve image-generation prompts for ad performance. Return JSON only."
    user_prompt = f"""Revise the current ad prompt for a better result.

Campaign context:
{_campaign_context_text(session_data, sidebar_settings, creative_brief)}

Current score: {current_score:.3f}
Current prompt:
{current_prompt}

Recent local history:
{history_text}

Task:
- Produce exactly one revised prompt that preserves the campaign intent.
- Make a focused change rather than rewriting the campaign from scratch.
- Keep the prompt coherent, vivid, and under 220 words.
- Return JSON with keys "prompt" and "strategy".
"""

    payload = _request_json_object(
        client,
        system_prompt,
        user_prompt,
        model=PROMPT_GENERATION_MODEL,
        temperature=0.1,
        top_p=0.95,
        max_completion_tokens=4096,
    )
    prompt = _normalize_prompt(str(payload.get("prompt", "")).strip()) or current_prompt
    strategy = str(payload.get("strategy", "")).strip()
    return {"prompt": prompt, "strategy": strategy}


def _generate_shared_reflection(
    client,
    shared_history: list[dict[str, Any]],
) -> str:
    """Summarize shared prompt patterns for the efficient optimizer."""
    fallback_reflection = (
        "Favor clear product focus, readable CTA space, campaign-specific context, "
        "and simpler compositions over generic staging."
    )
    if len(shared_history) < 2:
        return fallback_reflection

    def _text_only_reflection() -> str:
        system_prompt = "You analyze ad-performance patterns from prompt history."
        user_prompt = f"""You are an expert at analyzing visual patterns in advertising performance.

Review the prompt history below as a fallback when rendered images are unavailable. Infer visual patterns that distinguish lower-scoring from higher-scoring ad iterations.

{_history_summary(shared_history, top_n=5, bottom_n=5)}

RESPONSE FORMAT:
Provide a structured analysis of visual patterns observed, focusing on what distinguishes high-performing from low-performing ads.
Cover composition, lighting, color palette, subject positioning, brand integration, and environmental elements. Focus on concrete changes future prompts should make.
"""

        try:
            reflection = _request_text(
                client,
                system_prompt,
                user_prompt,
                model=REFLECTION_MODEL,
                temperature=0.3,
                top_p=0.9,
                max_completion_tokens=4096,
            )
        except Exception:
            return fallback_reflection
        return reflection or fallback_reflection

    multimodal_content = _build_multimodal_reflection_content(
        shared_history,
        top_n=5,
        bottom_n=5,
    )
    if not multimodal_content:
        return _text_only_reflection()

    multimodal_content.append(
        {
            "type": "text",
            "text": (
                "Based on your visual analysis, identify patterns that correlate with higher "
                "effectiveness scores:\n"
                "1. Visual composition and framing differences\n"
                "2. Lighting conditions and mood variations\n"
                "3. Color palettes and visual tone patterns\n"
                "4. Subject positioning and action effectiveness\n"
                "5. Brand integration approaches\n"
                "6. Environmental and atmospheric elements\n\n"
                "RESPONSE FORMAT:\n"
                "Provide a structured analysis of visual patterns observed, focusing on "
                "what distinguishes high-performing from low-performing ads."
            ),
        }
    )

    system_prompt = "You analyze ad-performance patterns from prompt/image history."
    try:
        reflection = _request_text(
            client,
            system_prompt,
            multimodal_content,
            model=REFLECTION_MODEL,
            temperature=0.3,
            top_p=0.9,
            max_completion_tokens=4096,
        )
    except Exception:
        return _text_only_reflection()

    return reflection or _text_only_reflection()


def _generate_textual_gradient(
    client,
    current_prompt: str,
    shared_history: list[dict[str, Any]],
    shared_reflection: str | None,
) -> str:
    """Generate reference-style textual improvement suggestions."""
    if shared_reflection:
        user_prompt = f"""You are an expert at optimizing image generation prompts for advertising effectiveness.

CURRENT PROMPT TO IMPROVE:
{current_prompt}

PERFORMANCE ANALYSIS FROM PREVIOUS ITERATIONS:
{shared_reflection}

TASK:
Based on the performance analysis above, generate specific, actionable improvements to make the current prompt more effective.

Focus on implementing the successful visual patterns identified in the analysis while avoiding the ineffective elements.

Provide 3-5 specific, implementable suggestions for improvement. At least two suggestions must create a meaningful visual change, such as changing the scene, setting, subject action, camera framing, CTA/text placement, product staging, or emotional hook.
It can be addition of a new prompt part, deletion of an existing prompt part, or rewriting a prompt part.
Each suggestion should reference insights from the performance analysis.
Preserve only the campaign invariants: product/service, audience, campaign goal, key message/CTA, brand tone, and style direction. Do not preserve the original scene or composition unless the analysis clearly supports it.

RESPONSE FORMAT:
1. [Specific improvement suggestion based on performance analysis]
2. [Specific improvement suggestion based on performance analysis]
3. [Specific improvement suggestion based on performance analysis]

Be concrete and actionable. Prefer changes large enough to produce visibly different image candidates, not minor wording polish."""
    else:
        user_prompt = f"""You are an expert at optimizing image generation prompts for advertising effectiveness.

CURRENT PROMPT TO IMPROVE:
{current_prompt}

TASK:
Generate specific, actionable improvements to make this ad prompt more effective.
Focus on elements that will increase engagement and appeal to the target audience.

Consider improvements in:
1. Visual composition and framing
2. Emotional appeal and messaging
3. Color palette and lighting
4. Subject positioning and action
5. Brand integration and logo placement
6. Text overlay space and readability

Provide 3-5 specific, implementable suggestions for improvement. At least two suggestions must create a meaningful visual change, such as changing the scene, setting, subject action, camera framing, CTA/text placement, product staging, or emotional hook.
Each suggestion should be concrete and actionable.
Preserve only the campaign invariants: product/service, audience, campaign goal, key message/CTA, brand tone, and style direction. Do not preserve the original scene or composition by default.

RESPONSE FORMAT:
1. [Specific improvement suggestion]
2. [Specific improvement suggestion]
3. [Specific improvement suggestion]

Be concise but specific. Prefer changes large enough to produce visibly different image candidates, not minor wording polish."""

    try:
        return _request_text(
            client,
            "You generate textual gradients for advertising prompt optimization.",
        user_prompt,
        model=PROMPT_GENERATION_MODEL,
        temperature=1.0,
        top_p=0.9,
        max_completion_tokens=2048,
    )
    except Exception:
        return "Improve visual appeal and emotional connection with the target audience."


def _apply_textual_gradient(
    client,
    current_prompt: str,
    gradient: str,
) -> str:
    """Apply textual improvement suggestions to produce one revised prompt."""
    user_prompt = f"""You are an expert at revising image generation prompts based on improvement suggestions.

ORIGINAL PROMPT:
{current_prompt}

IMPROVEMENT SUGGESTIONS:
{gradient}

TASK:
Create a meaningfully different revised prompt that incorporates the improvement suggestions.

REQUIREMENTS:
- Preserve the campaign invariants: product/service, audience, campaign goal, key message/CTA, brand tone, style direction, and aspect ratio.
- Do not preserve the original wording, order, scene, or composition unless it is clearly the strongest choice.
- Apply at least one substantial visual change to scene, setting, subject action, camera framing, CTA/text placement, product staging, or emotional hook.
- Integrate the improvement suggestions naturally into a self-contained image-generation prompt.
- Maintain coherence and readability.
- Ensure the prompt is optimized for image generation.

Return ONLY the revised prompt, no explanations or additional text."""
    try:
        revised = _request_text(
            client,
            "You apply textual gradients to image-generation prompts.",
            user_prompt,
            model=PROMPT_GENERATION_MODEL,
            temperature=0.1,
            top_p=0.95,
            max_completion_tokens=2048,
        )
        return _normalize_prompt(revised) or current_prompt
    except Exception:
        return current_prompt


def _generate_efficient_candidates(
    client,
    current_prompt: str,
    current_score: float,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    shared_history: list[dict[str, Any]],
    shared_reflection: str | None,
    *,
    candidate_count: int,
) -> dict[str, Any]:
    """Generate multiple gradient-applied prompt candidates for TextBO."""
    prompts: list[str] = []
    gradients: list[str] = []
    for _ in range(candidate_count):
        gradient = _generate_textual_gradient(
            client,
            current_prompt,
            shared_history,
            shared_reflection,
        )
        gradients.append(gradient)
        prompts.append(_apply_textual_gradient(client, current_prompt, gradient))

    prompts = _dedupe_prompts(prompts)
    if not prompts:
        prompts = [current_prompt]

    while len(prompts) < candidate_count:
        prompts.append(prompts[-1])

    return {
        "strategy": "\n\n".join(gradients[:3]),
        "prompts": prompts[:candidate_count],
    }


def _generate_image_for_prompt(
    *,
    client,
    gemini_key: str,
    model_key: str,
    prompt: str,
    aspect_ratio: str,
    gpt_image_quality: str,
    style_image_bytes: bytes | None,
) -> tuple[str | None, str | None]:
    """Generate an image using the selected backend."""
    if model_key == "gemini":
        output_path, error = generate_image_gemini(
            gemini_key,
            prompt,
            style_image_bytes=style_image_bytes,
        )
    else:
        size_map = {
            "1536x1024 (landscape)": "1536x1024",
            "1024x1024 (square)": "1024x1024",
            "1024x1536 (portrait)": "1024x1536",
        }
        output_path, error = generate_image_openai(
            client,
            prompt,
            size=size_map.get(aspect_ratio, "1536x1024"),
            quality=gpt_image_quality,
        )

    return (str(output_path), error) if output_path else (None, error)


def _render_textbo_tournament_candidate(
    *,
    candidate_prompt: str,
    candidate_idx: int,
    openai_client,
    gemini_key: str,
    model_key: str,
    aspect_ratio: str,
    gpt_image_quality: str,
    style_image_bytes: bytes | None,
) -> dict[str, Any]:
    """Render one Best-of-N tournament candidate."""
    output_path, error = _generate_image_for_prompt(
        client=openai_client,
        gemini_key=gemini_key,
        model_key=model_key,
        prompt=candidate_prompt,
        aspect_ratio=aspect_ratio,
        gpt_image_quality=gpt_image_quality,
        style_image_bytes=style_image_bytes,
    )
    return {
        "candidate_idx": candidate_idx,
        "prompt": candidate_prompt,
        "output_path": output_path,
        "error": error,
        "selected": False,
    }


def _parse_binary_choice(text: str) -> int | None:
    """Parse an exact 1/2 tournament choice."""
    stripped = (text or "").strip()
    return int(stripped) if stripped in {"1", "2"} else None


def _pairwise_compare_tournament_candidates(
    *,
    client,
    candidate1: dict[str, Any],
    candidate2: dict[str, Any],
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    rng: random.Random,
    comparisons: int = TOURNAMENT_COMPARISONS,
) -> tuple[int, str]:
    """Compare two rendered candidates; returns 0 when candidate1 wins, 1 otherwise."""
    path1 = candidate1.get("output_path")
    path2 = candidate2.get("output_path")
    if not path1 or not os.path.exists(path1):
        return 1, "candidate1-missing-image"
    if not path2 or not os.path.exists(path2):
        return 0, "candidate2-missing-image"

    comparison_prompt = """You are evaluating two advertisement images for mobile Instagram ads.
Which image would be more effective at engaging users and driving clicks?

CRITICAL: Return exactly 1 or 2 with no other text.
- Return 1 if the first image is more effective.
- Return 2 if the second image is more effective.

Your response must be exactly one character: either 1 or 2."""

    votes_candidate1 = 0
    votes_candidate2 = 0
    fallback_votes = 0
    image1_url = _path_to_data_url(path1)
    image2_url = _path_to_data_url(path2)
    for _ in range(max(1, comparisons)):
        try:
            response = _request_eval_completion(
                client,
                model=EVAL_MODEL,
                messages=[
                    {"role": "system", "content": "Return exactly one token: 1 or 2."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image1_url}},
                            {"type": "image_url", "image_url": {"url": image2_url}},
                            {"type": "text", "text": comparison_prompt},
                        ],
                    },
                ],
                temperature=0,
                max_completion_tokens=1,
            )
            choice = _parse_binary_choice(response.choices[0].message.content or "")
            if choice == 1:
                votes_candidate1 += 1
                continue
            if choice == 2:
                votes_candidate2 += 1
                continue
        except Exception:
            pass

        fallback_votes += 1
        if rng.randint(0, 1) == 0:
            votes_candidate1 += 1
        else:
            votes_candidate2 += 1

    winner_idx = 0 if votes_candidate1 > votes_candidate2 else 1
    mode = f"pairwise-majority:k={max(1, comparisons)}"
    if fallback_votes:
        mode += f":fallback_votes={fallback_votes}"
    return winner_idx, mode


def _tournament_select_candidate(
    *,
    client,
    candidates: list[dict[str, Any]],
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    tournament_seed: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select a Best-of-N winner using the reference-style knockout tournament."""
    if not candidates:
        raise ValueError("Tournament requires at least one candidate.")

    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("output_path") and os.path.exists(candidate["output_path"])
    ]
    if not valid_candidates:
        return candidates[0], []
    if len(valid_candidates) == 1:
        return valid_candidates[0], []

    rng = random.Random(tournament_seed)
    current_round = [dict(candidate) for candidate in valid_candidates]
    rng.shuffle(current_round)
    tournament_log: list[dict[str, Any]] = []
    round_num = 1

    while len(current_round) > 1:
        next_round: list[dict[str, Any]] = []
        for match_idx in range(0, len(current_round) - 1, 2):
            candidate1 = current_round[match_idx]
            candidate2 = current_round[match_idx + 1]
            winner_idx, mode = _pairwise_compare_tournament_candidates(
                client=client,
                candidate1=candidate1,
                candidate2=candidate2,
                session_data=session_data,
                sidebar_settings=sidebar_settings,
                creative_brief=creative_brief,
                rng=rng,
            )
            winner = candidate1 if winner_idx == 0 else candidate2
            next_round.append(winner)
            tournament_log.append(
                {
                    "round": round_num,
                    "match": match_idx // 2 + 1,
                    "candidate_1_idx": candidate1.get("candidate_idx"),
                    "candidate_2_idx": candidate2.get("candidate_idx"),
                    "winner_idx": winner.get("candidate_idx"),
                    "mode": mode,
                }
            )

        if len(current_round) % 2 == 1:
            bye_candidate = current_round[-1]
            next_round.append(bye_candidate)
            tournament_log.append(
                {
                    "round": round_num,
                    "match": len(current_round) // 2 + 1,
                    "candidate_1_idx": bye_candidate.get("candidate_idx"),
                    "candidate_2_idx": None,
                    "winner_idx": bye_candidate.get("candidate_idx"),
                    "mode": "bye",
                }
            )

        current_round = next_round
        round_num += 1

    return current_round[0], tournament_log


def _run_textbo_gradient_tournament(
    *,
    openai_client,
    gemini_key: str,
    model_key: str,
    aspect_ratio: str,
    gpt_image_quality: str,
    style_image_bytes: bytes | None,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    trajectory: dict[str, Any],
    step: int,
    gradient_step: int,
    efficient_candidates: int,
    shared_history_snapshot: list[dict[str, Any]],
    shared_reflection: str | None,
) -> dict[str, Any]:
    """Run one TextBO gradient step: generate N prompts, render, then tournament-select."""
    current = trajectory["current"]
    proposals = _generate_efficient_candidates(
        openai_client,
        current["prompt"],
        current["score"],
        session_data,
        sidebar_settings,
        creative_brief,
        shared_history_snapshot,
        shared_reflection,
        candidate_count=efficient_candidates,
    )

    rendered_candidates: list[dict[str, Any]] = []
    render_parallelism = min(
        DEFAULT_EFFICIENT_PRESCORE_PARALLELISM,
        max(1, len(proposals["prompts"])),
    )
    with ThreadPoolExecutor(max_workers=render_parallelism) as executor:
        future_to_idx = {
            executor.submit(
                _render_textbo_tournament_candidate,
                candidate_idx=cand_idx,
                candidate_prompt=candidate_prompt,
                openai_client=openai_client,
                gemini_key=gemini_key,
                model_key=model_key,
                aspect_ratio=aspect_ratio,
                gpt_image_quality=gpt_image_quality,
                style_image_bytes=style_image_bytes,
            ): cand_idx
            for cand_idx, candidate_prompt in enumerate(proposals["prompts"], start=1)
        }

        for future in as_completed(future_to_idx):
            cand_idx = future_to_idx[future]
            try:
                candidate = future.result()
            except Exception as exc:
                candidate = {
                    "candidate_idx": cand_idx,
                    "prompt": proposals["prompts"][cand_idx - 1],
                    "output_path": None,
                    "error": f"Tournament candidate failed: {exc}",
                    "selected": False,
                }
            candidate["gradient_step"] = gradient_step
            rendered_candidates.append(candidate)

    rendered_candidates.sort(key=lambda item: item["candidate_idx"])
    tournament_seed = step * 30000 + trajectory["trajectory_id"] * 100 + gradient_step * 10
    selected_candidate, tournament_log = _tournament_select_candidate(
        client=openai_client,
        candidates=rendered_candidates,
        session_data=session_data,
        sidebar_settings=sidebar_settings,
        creative_brief=creative_brief,
        tournament_seed=tournament_seed,
    )
    selected_idx = selected_candidate.get("candidate_idx")
    for candidate in rendered_candidates:
        candidate["selected"] = candidate.get("candidate_idx") == selected_idx

    for match in tournament_log:
        match["gradient_step"] = gradient_step

    return {
        "gradient_step": gradient_step,
        "selected_candidate": selected_candidate,
        "rendered_candidates": rendered_candidates,
        "tournament_log": tournament_log,
        "tournament_seed": tournament_seed,
        "strategy": proposals.get("strategy"),
    }


def _build_candidate_record(
    *,
    candidate_id: str,
    prompt: str,
    output_path: str | None,
    score: float,
    probs: dict[str, float],
    source: str,
    error: str | None = None,
    prompt_prescore: float | None = None,
    prompt_prescore_probs: dict[str, float] | None = None,
    score_details: dict[str, Any] | None = None,
    prompt_prescore_details: dict[str, Any] | None = None,
    score_mode: str | None = None,
    prompt_prescore_mode: str | None = None,
    strategy: str | None = None,
    accepted: bool | None = None,
    start_seed_id: str | None = None,
    start_seed_rank: int | None = None,
    step: int | None = None,
    chain_id: int | None = None,
    trajectory_id: int | None = None,
    debug_traceback: str | None = None,
) -> dict[str, Any]:
    """Create a consistent candidate record."""
    return {
        "candidate_id": candidate_id,
        "prompt": prompt,
        "output_path": output_path,
        "score": score,
        "probs": probs,
        "source": source,
        "error": error,
        "prompt_prescore": prompt_prescore,
        "prompt_prescore_probs": prompt_prescore_probs,
        "score_details": score_details,
        "prompt_prescore_details": prompt_prescore_details,
        "score_mode": score_mode,
        "prompt_prescore_mode": prompt_prescore_mode,
        "strategy": strategy,
        "accepted": accepted,
        "start_seed_id": start_seed_id,
        "start_seed_rank": start_seed_rank,
        "step": step,
        "chain_id": chain_id,
        "trajectory_id": trajectory_id,
        "debug_traceback": debug_traceback,
    }


def _evaluate_initial_candidate(
    *,
    candidate_idx: int,
    total_candidates: int,
    prompt: str,
    openai_client,
    gemini_key: str,
    model_key: str,
    aspect_ratio: str,
    gpt_image_quality: str,
    style_image_bytes: bytes | None,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    judge_repeats: int,
) -> dict[str, Any]:
    """Generate and score one initial candidate."""
    output_path, error = _generate_image_for_prompt(
        client=openai_client,
        gemini_key=gemini_key,
        model_key=model_key,
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        gpt_image_quality=gpt_image_quality,
        style_image_bytes=style_image_bytes,
    )

    if output_path:
        scored = _aggregate_persona_scores(
            lambda persona, seed_value: _score_image_with_logprobs(
                openai_client,
                output_path,
                prompt,
                session_data,
                sidebar_settings,
                creative_brief,
                seed=seed_value,
                persona=persona,
            ),
            seed_base=candidate_idx * 100,
            fallback_scorer=lambda seed_value: _score_image_with_logprobs(
                openai_client,
                output_path,
                prompt,
                session_data,
                sidebar_settings,
                creative_brief,
                seed=seed_value,
            ),
            fallback_repeats=judge_repeats,
        )
        score = scored["score"]
        probs = scored["probs"]
        score_mode = scored.get("mode")
        score_details = {
            "mean_score": scored.get("mean_score"),
            "persona_scores": scored.get("persona_scores"),
            "persona_count": scored.get("persona_count"),
            "score_aggregation": scored.get("score_aggregation"),
            "persona_errors": scored.get("persona_errors"),
        }
    else:
        score = 1.0
        probs = {str(i): (1.0 if i == 1 else 0.0) for i in range(1, 6)}
        score_mode = "image-generation-failed"
        score_details = None

    return {
        "candidate_idx": candidate_idx,
        "total_candidates": total_candidates,
        "record": _build_candidate_record(
            candidate_id=f"initial_{candidate_idx:02d}",
            prompt=prompt,
            output_path=output_path,
            score=score,
            probs=probs,
            source="initial",
            error=error,
            score_details=score_details,
            score_mode=score_mode,
        ),
    }


def _prescore_efficient_candidate(
    *,
    candidate_prompt: str,
    candidate_idx: int,
    trajectory_id: int,
    step: int,
    openai_client,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    judge_repeats: int,
) -> dict[str, Any]:
    """Pre-score one efficient-search prompt candidate."""
    prompt_scored = _aggregate_persona_scores(
        lambda persona, seed_value: _score_prompt_with_logprobs(
            openai_client,
            candidate_prompt,
            session_data,
            sidebar_settings,
            creative_brief,
            seed=seed_value,
            persona=persona,
        ),
        seed_base=step * 10000 + trajectory_id * 100 + candidate_idx,
        fallback_scorer=lambda seed_value: _score_prompt_with_logprobs(
            openai_client,
            candidate_prompt,
            session_data,
            sidebar_settings,
            creative_brief,
            seed=seed_value,
        ),
        fallback_repeats=judge_repeats,
    )
    return {
        "prompt": candidate_prompt,
        "score": prompt_scored["score"],
        "probs": prompt_scored["probs"],
        "mode": prompt_scored.get("mode"),
        "details": {
            "mean_score": prompt_scored.get("mean_score"),
            "persona_scores": prompt_scored.get("persona_scores"),
            "persona_count": prompt_scored.get("persona_count"),
            "score_aggregation": prompt_scored.get("score_aggregation"),
            "persona_errors": prompt_scored.get("persona_errors"),
        },
    }


def _run_efficient_trajectory_step(
    *,
    openai_client,
    gemini_key: str,
    model_key: str,
    aspect_ratio: str,
    gpt_image_quality: str,
    style_image_bytes: bytes | None,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    trajectory: dict[str, Any],
    step: int,
    efficient_candidates: int,
    gradient_steps: int,
    judge_repeats: int,
    shared_history_snapshot: list[dict[str, Any]],
    shared_reflection: str | None,
    source: str = "efficient",
    candidate_id_prefix: str = "efficient",
) -> dict[str, Any]:
    """Run one efficient-search trajectory step against a shared snapshot."""
    current = trajectory["current"]
    selected_candidate: dict[str, Any] = {
        "candidate_idx": 0,
        "gradient_step": 0,
        "prompt": current["prompt"],
        "output_path": current.get("output_path"),
        "error": current.get("error"),
        "selected": True,
    }
    rendered_candidates: list[dict[str, Any]] = []
    tournament_log: list[dict[str, Any]] = []
    tournament_seeds: list[int] = []
    strategies: list[str] = []

    for gradient_step in range(1, max(1, gradient_steps) + 1):
        gradient_result = _run_textbo_gradient_tournament(
            openai_client=openai_client,
            gemini_key=gemini_key,
            model_key=model_key,
            aspect_ratio=aspect_ratio,
            gpt_image_quality=gpt_image_quality,
            style_image_bytes=style_image_bytes,
            session_data=session_data,
            sidebar_settings=sidebar_settings,
            creative_brief=creative_brief,
            trajectory=trajectory,
            step=step,
            gradient_step=gradient_step,
            efficient_candidates=efficient_candidates,
            shared_history_snapshot=shared_history_snapshot,
            shared_reflection=shared_reflection,
        )
        rendered_candidates.extend(gradient_result["rendered_candidates"])
        tournament_log.extend(gradient_result["tournament_log"])
        tournament_seeds.append(gradient_result["tournament_seed"])
        if gradient_result.get("strategy"):
            strategies.append(str(gradient_result["strategy"]))
        candidate = gradient_result["selected_candidate"]
        if candidate.get("output_path") and os.path.exists(candidate["output_path"]):
            selected_candidate = candidate

    final_selected_id = (
        selected_candidate.get("gradient_step"),
        selected_candidate.get("candidate_idx"),
    )
    for candidate in rendered_candidates:
        candidate["selected"] = (
            candidate.get("gradient_step"),
            candidate.get("candidate_idx"),
        ) == final_selected_id

    selected_prompt = selected_candidate["prompt"]
    output_path = selected_candidate.get("output_path")
    error = selected_candidate.get("error")
    if output_path:
        image_scored = _aggregate_persona_scores(
            lambda persona, seed_value: _score_image_with_logprobs(
                openai_client,
                output_path,
                selected_prompt,
                session_data,
                sidebar_settings,
                creative_brief,
                seed=seed_value,
                persona=persona,
            ),
            seed_base=step * 20000 + trajectory["trajectory_id"] * 100,
            fallback_scorer=lambda seed_value: _score_image_with_logprobs(
                openai_client,
                output_path,
                selected_prompt,
                session_data,
                sidebar_settings,
                creative_brief,
                seed=seed_value,
            ),
            fallback_repeats=judge_repeats,
        )
        final_score = image_scored["score"]
        final_probs = image_scored["probs"]
        final_score_mode = image_scored.get("mode")
        final_score_details = {
            "mean_score": image_scored.get("mean_score"),
            "persona_scores": image_scored.get("persona_scores"),
            "persona_count": image_scored.get("persona_count"),
            "score_aggregation": image_scored.get("score_aggregation"),
            "persona_errors": image_scored.get("persona_errors"),
        }
    else:
        final_score = 3.0
        final_probs = {str(i): (1.0 if i == 3 else 0.0) for i in range(1, 6)}
        final_score_mode = "image-generation-failed"
        final_score_details = None

    previous_score = current["score"]
    accepted = final_score > previous_score
    step_entry = _build_candidate_record(
        candidate_id=f"{candidate_id_prefix}_traj{trajectory['trajectory_id']:02d}_step{step:02d}",
        prompt=selected_prompt,
        output_path=output_path,
        score=final_score,
        probs=final_probs,
        source=source,
        error=error,
        prompt_prescore=None,
        prompt_prescore_probs=None,
        score_details=final_score_details,
        prompt_prescore_details=None,
        score_mode=final_score_mode,
        prompt_prescore_mode="image-tournament",
        strategy="\n\n".join(strategies[:3]),
        accepted=accepted,
        start_seed_id=trajectory["seed"]["candidate_id"],
        start_seed_rank=trajectory["seed"].get("worst_rank"),
        step=step,
        trajectory_id=trajectory["trajectory_id"],
    )
    step_entry["previous_score"] = previous_score
    step_entry["score_delta"] = final_score - previous_score
    step_entry["candidate_prompts"] = rendered_candidates
    step_entry["shared_reflection"] = shared_reflection
    step_entry["selection_mode"] = f"image-tournament:G={max(1, gradient_steps)}:N={efficient_candidates}"
    step_entry["tournament_log"] = tournament_log
    step_entry["tournament_seed"] = tournament_seeds[-1] if tournament_seeds else None
    step_entry["tournament_seeds"] = tournament_seeds
    step_entry["tournament_candidate_count"] = len(rendered_candidates)
    step_entry["gradient_steps"] = max(1, gradient_steps)

    return {
        "trajectory_id": trajectory["trajectory_id"],
        "step_entry": step_entry,
        "accepted": accepted,
    }


def _rank_candidates_desc(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort candidates by score descending."""
    return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)


def _shared_history_boundary_seeds(seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Initialize shared reflection history with worst and kth-worst boundary seeds."""
    if not seeds:
        return []
    if len(seeds) == 1:
        return [dict(seeds[0])]
    return [dict(seeds[0]), dict(seeds[-1])]


def _merge_shared_history(
    initial_history: list[dict[str, Any]] | None,
    boundary_seeds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Initialize shared history from initial evaluations while preserving boundary seeds."""
    shared_history: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for entry in initial_history or []:
        candidate_id = str(entry.get("candidate_id") or "")
        if candidate_id and candidate_id in seen_ids:
            continue
        shared_history.append(dict(entry))
        if candidate_id:
            seen_ids.add(candidate_id)

    for seed in boundary_seeds:
        candidate_id = str(seed.get("candidate_id") or "")
        if candidate_id and candidate_id in seen_ids:
            continue
        shared_history.append(dict(seed))
        if candidate_id:
            seen_ids.add(candidate_id)

    return shared_history


def _run_base_optimizer_chain(
    *,
    openai_client,
    gemini_key: str,
    model_key: str,
    aspect_ratio: str,
    gpt_image_quality: str,
    style_image_bytes: bytes | None,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    seed: dict[str, Any],
    chain_idx: int,
    optimization_steps: int,
    judge_repeats: int,
) -> dict[str, Any]:
    """Run one base optimization chain from a single starting seed."""
    current = dict(seed)
    best = dict(seed)
    chain_history = [dict(seed)]
    step_entries: list[dict[str, Any]] = []

    for step in range(1, optimization_steps + 1):
        revision = _generate_base_revision(
            openai_client,
            current["prompt"],
            current["score"],
            session_data,
            sidebar_settings,
            creative_brief,
            chain_history,
        )
        new_prompt = revision["prompt"]
        output_path, error = _generate_image_for_prompt(
            client=openai_client,
            gemini_key=gemini_key,
            model_key=model_key,
            prompt=new_prompt,
            aspect_ratio=aspect_ratio,
            gpt_image_quality=gpt_image_quality,
            style_image_bytes=style_image_bytes,
        )

        if output_path:
            scored = _aggregate_persona_scores(
                lambda persona, seed_value: _score_image_with_logprobs(
                    openai_client,
                    output_path,
                    new_prompt,
                    session_data,
                    sidebar_settings,
                    creative_brief,
                    seed=seed_value,
                    persona=persona,
                ),
                seed_base=chain_idx * 1000 + step * 10,
                fallback_scorer=lambda seed_value: _score_image_with_logprobs(
                    openai_client,
                    output_path,
                    new_prompt,
                    session_data,
                    sidebar_settings,
                    creative_brief,
                    seed=seed_value,
                ),
                fallback_repeats=judge_repeats,
            )
            score = scored["score"]
            probs = scored["probs"]
            score_mode = scored.get("mode")
            score_details = {
                "mean_score": scored.get("mean_score"),
                "persona_scores": scored.get("persona_scores"),
                "persona_count": scored.get("persona_count"),
                "score_aggregation": scored.get("score_aggregation"),
                "persona_errors": scored.get("persona_errors"),
            }
        else:
            score = 1.0
            probs = {str(i): (1.0 if i == 1 else 0.0) for i in range(1, 6)}
            score_mode = "image-generation-failed"
            score_details = None

        accepted = score > current["score"]
        step_entry = _build_candidate_record(
            candidate_id=f"base_chain{chain_idx:02d}_step{step:02d}",
            prompt=new_prompt,
            output_path=output_path,
            score=score,
            probs=probs,
            source="base",
            error=error,
            score_details=score_details,
            score_mode=score_mode,
            strategy=revision.get("strategy"),
            accepted=accepted,
            start_seed_id=seed["candidate_id"],
            start_seed_rank=seed.get("worst_rank"),
            step=step,
            chain_id=chain_idx,
        )
        step_entries.append(step_entry)
        chain_history.append(step_entry)

        if accepted:
            current = dict(step_entry)
        if score > best["score"]:
            best = dict(step_entry)

    return {
        "chain_id": chain_idx,
        "seed": seed,
        "steps": step_entries,
        "final": current,
        "best": best,
    }


def _run_base_optimizer(
    *,
    openai_client,
    gemini_key: str,
    model_key: str,
    aspect_ratio: str,
    gpt_image_quality: str,
    style_image_bytes: bytes | None,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    worst_seeds: list[dict[str, Any]],
    optimization_steps: int,
    judge_repeats: int,
    report,
) -> dict[str, Any]:
    """Run the base optimization chains independently from the worst seeds."""
    chains: list[dict[str, Any]] = []

    chain_parallelism = min(DEFAULT_BASE_CHAIN_PARALLELISM, max(1, len(worst_seeds)))
    report(
        f"Base optimizer running {len(worst_seeds)} chains with up to "
        f"{chain_parallelism} parallel workers"
    )

    with ThreadPoolExecutor(max_workers=chain_parallelism) as executor:
        future_to_chain = {
            executor.submit(
                _run_base_optimizer_chain,
                openai_client=openai_client,
                gemini_key=gemini_key,
                model_key=model_key,
                aspect_ratio=aspect_ratio,
                gpt_image_quality=gpt_image_quality,
                style_image_bytes=style_image_bytes,
                session_data=session_data,
                sidebar_settings=sidebar_settings,
                creative_brief=creative_brief,
                seed=seed,
                chain_idx=chain_idx,
                optimization_steps=optimization_steps,
                judge_repeats=judge_repeats,
            ): chain_idx
            for chain_idx, seed in enumerate(worst_seeds, start=1)
        }

        for future in as_completed(future_to_chain):
            chain_idx = future_to_chain[future]
            try:
                chain = future.result()
            except Exception as exc:
                traceback_text = _format_exception_traceback(exc)
                _record_debug_event(
                    "base-chain",
                    f"Base chain {chain_idx}/{len(worst_seeds)} failed: {type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__,
                    traceback_text=traceback_text,
                )
                raise RuntimeError(f"Base chain {chain_idx} failed: {exc}") from exc

            chains.append(chain)
            report(
                f"Base chain {chain_idx}/{len(worst_seeds)} complete: best score "
                f"{_format_score(chain['best']['score'])}"
            )

    chains.sort(key=lambda chain: chain["chain_id"])
    all_evaluated = [
        step_entry
        for chain in chains
        for step_entry in chain["steps"]
    ]

    best_overall = max([chain["best"] for chain in chains], key=lambda item: item["score"])
    return {
        "chains": chains,
        "all_evaluated": all_evaluated,
        "best": best_overall,
    }


def _run_efficient_optimizer(
    *,
    openai_client,
    gemini_key: str,
    model_key: str,
    aspect_ratio: str,
    gpt_image_quality: str,
    style_image_bytes: bytes | None,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    worst_seeds: list[dict[str, Any]],
    optimization_steps: int,
    efficient_candidates: int,
    gradient_steps: int,
    judge_repeats: int,
    report,
    report_image=None,
    optimizer_label: str = "TextBO",
    source: str = "efficient",
    candidate_id_prefix: str = "efficient",
    initial_shared_history: list[dict[str, Any]] | None = None,
    initial_shared_reflection: str | None = None,
) -> dict[str, Any]:
    """Run the shared-history efficient optimizer from the same worst seeds."""
    trajectories: list[dict[str, Any]] = []
    boundary_seeds = _shared_history_boundary_seeds(worst_seeds)
    shared_history = _merge_shared_history(initial_shared_history, boundary_seeds)
    shared_reflection: str | None = initial_shared_reflection
    all_evaluated: list[dict[str, Any]] = []
    step_winners: list[dict[str, Any]] = []
    shared_reflection_history = [
        {
            "step": 0,
            "reflection": shared_reflection or "No meta-reflection yet.",
            "history_size": len(shared_history),
        }
    ]

    for trajectory_idx, seed in enumerate(worst_seeds, start=1):
        trajectories.append(
            {
                "trajectory_id": trajectory_idx,
                "seed": seed,
                "current": dict(seed),
                "best": dict(seed),
                "steps": [],
            }
        )

    trajectory_parallelism = min(
        DEFAULT_EFFICIENT_TRAJECTORY_PARALLELISM,
        max(1, len(trajectories)),
    )
    report(
        f"{optimizer_label} running {len(trajectories)} trajectories with up to "
        f"{trajectory_parallelism} parallel workers per step | G={gradient_steps}, N={efficient_candidates}"
    )

    for step in range(1, optimization_steps + 1):
        report(
            f"{optimizer_label} step {step}/{optimization_steps}: using shared history with "
            f"{len(shared_history)} evaluated prompts"
        )

        shared_history_snapshot = [dict(item) for item in shared_history]
        trajectory_by_id = {trajectory["trajectory_id"]: trajectory for trajectory in trajectories}
        with ThreadPoolExecutor(max_workers=trajectory_parallelism) as executor:
            future_to_trajectory = {
                executor.submit(
                    _run_efficient_trajectory_step,
                    openai_client=openai_client,
                    gemini_key=gemini_key,
                    model_key=model_key,
                    aspect_ratio=aspect_ratio,
                    gpt_image_quality=gpt_image_quality,
                    style_image_bytes=style_image_bytes,
                    session_data=session_data,
                    sidebar_settings=sidebar_settings,
                    creative_brief=creative_brief,
                    trajectory={
                        "trajectory_id": trajectory["trajectory_id"],
                        "seed": trajectory["seed"],
                        "current": dict(trajectory["current"]),
                    },
                    step=step,
                    efficient_candidates=efficient_candidates,
                    gradient_steps=gradient_steps,
                    judge_repeats=judge_repeats,
                    shared_history_snapshot=shared_history_snapshot,
                    shared_reflection=shared_reflection,
                    source=source,
                    candidate_id_prefix=candidate_id_prefix,
                ): trajectory["trajectory_id"]
                for trajectory in trajectories
            }

            step_results: list[dict[str, Any]] = []
            for future in as_completed(future_to_trajectory):
                step_results.append(future.result())

        step_results.sort(key=lambda item: item["trajectory_id"])
        for result in step_results:
            trajectory = trajectory_by_id[result["trajectory_id"]]
            step_entry = result["step_entry"]
            trajectory["steps"].append(step_entry)
            shared_history.append(step_entry)
            all_evaluated.append(step_entry)

            if result["accepted"]:
                trajectory["current"] = dict(step_entry)
            if step_entry["score"] > trajectory["best"]["score"]:
                trajectory["best"] = dict(step_entry)

        step_winner_result = max(step_results, key=lambda item: item["step_entry"]["score"])
        step_winner_entry = step_winner_result["step_entry"]
        accepted_count = sum(1 for item in step_results if item["accepted"])
        step_winner = {
            "step": step,
            "trajectory_id": step_winner_result["trajectory_id"],
            "score": step_winner_entry["score"],
            "previous_score": step_winner_entry.get("previous_score"),
            "score_delta": step_winner_entry.get("score_delta"),
            "accepted": step_winner_result["accepted"],
            "prompt_prescore": step_winner_entry.get("prompt_prescore"),
            "probs": step_winner_entry.get("probs"),
            "score_details": step_winner_entry.get("score_details"),
            "prompt_prescore_details": step_winner_entry.get("prompt_prescore_details"),
            "output_path": step_winner_entry.get("output_path"),
            "error": step_winner_entry.get("error"),
            "prompt": step_winner_entry["prompt"],
            "strategy": step_winner_entry.get("strategy"),
            "candidate_id": step_winner_entry["candidate_id"],
            "score_mode": step_winner_entry.get("score_mode"),
            "prompt_prescore_mode": step_winner_entry.get("prompt_prescore_mode"),
            "debug_traceback": step_winner_entry.get("debug_traceback"),
            "source": source,
            "selection_mode": step_winner_entry.get("selection_mode"),
            "tournament_candidate_count": step_winner_entry.get("tournament_candidate_count"),
            "gradient_steps": step_winner_entry.get("gradient_steps"),
        }
        step_winners.append(step_winner)
        report(
            f"{optimizer_label} step {step}/{optimization_steps} tournament winner: trajectory "
            f"{step_winner['trajectory_id']} | G={step_winner.get('gradient_steps') or gradient_steps} | "
            f"candidates {step_winner.get('tournament_candidate_count') or 0} | "
            f"final score {_format_score(step_winner['score'])} "
            f"({step_winner.get('score_mode') or 'unknown'}) | "
            f"{accepted_count}/{len(step_results)} trajectories accepted | "
            f"winner {'accepted' if step_winner['accepted'] else 'rejected'} "
            f"({step_winner.get('score_delta', 0.0):+.6f})"
        )
        if report_image and step_winner.get("output_path"):
            report_image(
                step_winner["output_path"],
                caption=(
                    f"{optimizer_label} step {step} winner | trajectory {step_winner['trajectory_id']} | "
                    f"score {_format_score(step_winner['score'])} "
                    f"({step_winner.get('score_mode') or 'unknown'})"
                ),
            )

        if len(shared_history) >= 3:
            recent_history = shared_history[-min(10, len(shared_history)) :]
            shared_reflection = _generate_shared_reflection(openai_client, recent_history)
            shared_reflection_history.append(
                {
                    "step": step,
                    "reflection": shared_reflection,
                    "history_size": len(shared_history),
                }
            )

    best_overall = max([trajectory["best"] for trajectory in trajectories], key=lambda item: item["score"])
    return {
        "trajectories": trajectories,
        "all_evaluated": all_evaluated,
        "best": best_overall,
        "step_winners": step_winners,
        "shared_reflection_history": shared_reflection_history,
    }


def _noop_report(*_args, **_kwargs) -> None:
    """Swallow background optimizer progress updates."""


def _build_textbo_baseline_comparison(
    textbo_results: dict[str, Any],
    baseline_results: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a compact final comparison between TextBO and Base N=1, G=1."""
    if not baseline_results:
        return None

    textbo_best = textbo_results["best"]
    baseline_best = baseline_results["best"]
    score_delta = textbo_best["score"] - baseline_best["score"]
    if score_delta > 0:
        winner = "textbo"
    elif score_delta < 0:
        winner = "hidden_baseline"
    else:
        winner = "tie"

    return {
        "textbo_best": textbo_best,
        "hidden_baseline_best": baseline_best,
        "score_delta": score_delta,
        "winner": winner,
        "hidden_baseline_config": {
            "best_of_n": HIDDEN_BASELINE_CANDIDATES,
            "gradient_steps_per_iteration": HIDDEN_BASELINE_GRADIENT_STEPS,
        },
    }


def _run_search_pipeline(
    *,
    openai_client,
    gemini_key: str,
    model_key: str,
    aspect_ratio: str,
    gpt_image_quality: str,
    style_image_bytes: bytes | None,
    session_data: dict[str, Any],
    sidebar_settings: dict[str, Any],
    creative_brief: dict[str, Any],
    initial_prompt_count: int,
    lowest_prompt_count: int,
    optimization_steps: int,
    efficient_candidates: int,
    textbo_gradient_steps: int,
    judge_repeats: int,
    report,
    report_image=None,
) -> dict[str, Any]:
    """Run the full prompt-search pipeline."""
    base_prompt = build_generation_prompt(
        creative_brief,
        session_data,
        sidebar_settings=sidebar_settings,
        style_description=sidebar_settings.get("style_description", ""),
        has_style_image=(model_key == "gemini" and style_image_bytes is not None),
    )
    report(f"Base prompt ready ({len(base_prompt.split())} words)")
    personas = _load_personas()
    report(
        f"Persona evaluator loaded {len(personas)} local personas; using "
        f"{min(DEFAULT_PERSONA_COUNT, len(personas))} per evaluation"
    )

    prompt_variants = _generate_initial_prompt_variants(
        openai_client,
        base_prompt,
        session_data,
        sidebar_settings,
        creative_brief,
        total_count=initial_prompt_count,
    )
    report(f"Prepared {len(prompt_variants)} initial prompt candidates")

    initial_candidates: list[dict[str, Any]] = []
    initial_parallelism = min(
        DEFAULT_INITIAL_PARALLELISM,
        max(1, len(prompt_variants)),
    )
    report(
        f"Running {len(prompt_variants)} initial candidates with up to "
        f"{initial_parallelism} parallel workers"
    )

    with ThreadPoolExecutor(max_workers=initial_parallelism) as executor:
        future_to_idx = {
            executor.submit(
                _evaluate_initial_candidate,
                candidate_idx=idx,
                total_candidates=len(prompt_variants),
                prompt=prompt,
                openai_client=openai_client,
                gemini_key=gemini_key,
                model_key=model_key,
                aspect_ratio=aspect_ratio,
                gpt_image_quality=gpt_image_quality,
                style_image_bytes=style_image_bytes,
                session_data=session_data,
                sidebar_settings=sidebar_settings,
                creative_brief=creative_brief,
                judge_repeats=judge_repeats,
            ): idx
            for idx, prompt in enumerate(prompt_variants, start=1)
        }

        completed_candidates: list[dict[str, Any]] = []
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
            except Exception as exc:
                traceback_text = _format_exception_traceback(exc)
                _record_debug_event(
                    "initial-worker",
                    (
                        f"Initial candidate {idx}/{len(prompt_variants)} failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    error_type=type(exc).__name__,
                    traceback_text=traceback_text,
                )
                result = {
                    "candidate_idx": idx,
                    "total_candidates": len(prompt_variants),
                    "record": _build_candidate_record(
                        candidate_id=f"initial_{idx:02d}",
                        prompt=prompt_variants[idx - 1],
                        output_path=None,
                        score=1.0,
                        probs={str(i): (1.0 if i == 1 else 0.0) for i in range(1, 6)},
                        source="initial",
                        error=f"Initial candidate failed: {exc}",
                        score_mode="worker-exception",
                        debug_traceback=traceback_text,
                    ),
                }

            completed_candidates.append(result)
            report(
                f"Initial candidate {result['candidate_idx']}/{result['total_candidates']}: complete"
            )

    completed_candidates.sort(key=lambda item: item["candidate_idx"])
    initial_candidates = [item["record"] for item in completed_candidates]

    ranked_initial = _rank_candidates_desc(initial_candidates)
    successful_initial = [item for item in initial_candidates if item.get("output_path")]
    if not successful_initial:
        failed_initial = [item for item in initial_candidates if item.get("error")]
        for entry in failed_initial[:10]:
            _record_debug_event(
                "initial-candidate",
                f"{entry['candidate_id']} failed: {entry['error']}",
                traceback_text=entry.get("debug_traceback"),
            )
        if len(failed_initial) > 10:
            _record_debug_event(
                "initial-candidate",
                f"{len(failed_initial) - 10} additional initial candidate failures were omitted.",
            )
        raise RuntimeError("All initial image generations failed.")

    worst_pool = sorted(successful_initial, key=lambda item: item["score"])[:lowest_prompt_count]
    for rank, entry in enumerate(worst_pool, start=1):
        entry["worst_rank"] = rank

    report(
        f"Selected {len(worst_pool)} lowest-scoring starting points for all optimizers"
    )

    initial_shared_history = [dict(item) for item in successful_initial]
    initial_shared_reflection = None
    if len(initial_shared_history) >= 2:
        report(
            "Building initial reflection from evaluated initial ads "
            f"({len(initial_shared_history)} examples)"
        )
        initial_shared_reflection = _generate_shared_reflection(
            openai_client,
            initial_shared_history,
        )

    report(
        "Base N=1, G=1 baseline started from the same starting points "
        f"for {optimization_steps} iterations"
    )
    baseline_executor = ThreadPoolExecutor(max_workers=1)
    baseline_future = baseline_executor.submit(
        _run_efficient_optimizer,
        openai_client=openai_client,
        gemini_key=gemini_key,
        model_key=model_key,
        aspect_ratio=aspect_ratio,
        gpt_image_quality=gpt_image_quality,
        style_image_bytes=style_image_bytes,
        session_data=session_data,
        sidebar_settings=sidebar_settings,
        creative_brief=creative_brief,
        worst_seeds=[dict(seed) for seed in worst_pool],
        optimization_steps=optimization_steps,
        efficient_candidates=HIDDEN_BASELINE_CANDIDATES,
        gradient_steps=HIDDEN_BASELINE_GRADIENT_STEPS,
        judge_repeats=judge_repeats,
        report=_noop_report,
        report_image=None,
        optimizer_label="Base N=1, G=1",
        source="base",
        candidate_id_prefix="base",
        initial_shared_history=[dict(item) for item in initial_shared_history],
        initial_shared_reflection=initial_shared_reflection,
    )
    hidden_baseline_results: dict[str, Any] | None = None
    hidden_baseline_error: dict[str, str] | None = None

    try:
        efficient_results = _run_efficient_optimizer(
            openai_client=openai_client,
            gemini_key=gemini_key,
            model_key=model_key,
            aspect_ratio=aspect_ratio,
            gpt_image_quality=gpt_image_quality,
            style_image_bytes=style_image_bytes,
            session_data=session_data,
            sidebar_settings=sidebar_settings,
            creative_brief=creative_brief,
            worst_seeds=worst_pool,
            optimization_steps=optimization_steps,
            efficient_candidates=efficient_candidates,
            gradient_steps=textbo_gradient_steps,
            judge_repeats=judge_repeats,
            report=report,
            report_image=report_image,
            optimizer_label="TextBO",
            source="textbo",
            candidate_id_prefix="textbo",
            initial_shared_history=[dict(item) for item in initial_shared_history],
            initial_shared_reflection=initial_shared_reflection,
        )
    except Exception:
        if not baseline_future.done():
            baseline_future.cancel()
        baseline_executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        if not baseline_future.done():
            report("TextBO complete; waiting for Base N=1, G=1 baseline")
        baseline_executor.shutdown(wait=True)

    try:
        hidden_baseline_results = baseline_future.result()
        report(
            "Base N=1, G=1 baseline complete: best score "
            f"{_format_score(hidden_baseline_results['best']['score'])}"
        )
    except Exception as exc:
        traceback_text = _format_exception_traceback(exc)
        hidden_baseline_error = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback_text,
        }
        _record_debug_event(
            "hidden-baseline",
            f"Base N=1, G=1 baseline failed: {type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
            traceback_text=traceback_text,
        )
        report(f"Base N=1, G=1 baseline failed: {exc}")

    textbo_baseline_comparison = _build_textbo_baseline_comparison(
        efficient_results,
        hidden_baseline_results,
    )

    best_initial_overall = max(ranked_initial, key=lambda item: item["score"])
    best_starting_initial = max(worst_pool, key=lambda item: item["score"])
    overall_candidates = [
        best_starting_initial,
        efficient_results["best"],
    ]
    if hidden_baseline_results:
        overall_candidates.append(hidden_baseline_results["best"])
    overall_best = max(overall_candidates, key=lambda item: item["score"])

    return {
        "base_prompt": base_prompt,
        "creative_brief": creative_brief,
        "initial_candidates": initial_candidates,
        "initial_ranked": ranked_initial,
        "worst_pool": worst_pool,
        "best_initial_overall": best_initial_overall,
        "best_starting_initial": best_starting_initial,
        "base": hidden_baseline_results,
        "base_error": hidden_baseline_error,
        "efficient": efficient_results,
        "hidden_baseline": hidden_baseline_results,
        "hidden_baseline_error": hidden_baseline_error,
        "textbo_baseline_comparison": textbo_baseline_comparison,
        "overall_best": overall_best,
        "config": {
            "initial_prompt_count": initial_prompt_count,
            "lowest_prompt_count": lowest_prompt_count,
            "optimization_steps": optimization_steps,
            "efficient_candidates": efficient_candidates,
            "textbo_gradient_steps": textbo_gradient_steps,
            "hidden_baseline_candidates": HIDDEN_BASELINE_CANDIDATES,
            "hidden_baseline_gradient_steps": HIDDEN_BASELINE_GRADIENT_STEPS,
            "persona_count": min(DEFAULT_PERSONA_COUNT, len(_load_personas())),
            "available_persona_count": len(_load_personas()),
            "persona_dir": PERSONA_DIR,
            "judge_repeats": judge_repeats,
            "model": sidebar_settings.get("model"),
            "resolution": sidebar_settings.get("resolution"),
        },
    }


def _render_probs(probs: dict[str, float]) -> str:
    """Render a compact probability summary."""
    return " | ".join(f"{k}: {probs.get(k, 0.0):.2f}" for k in ["1", "2", "3", "4", "5"])


def _render_persona_score_details(details: dict[str, Any] | None, label: str = "Persona scores") -> None:
    """Render compact per-persona scores when present."""
    if not details or not details.get("persona_scores"):
        return
    persona_scores = details["persona_scores"]
    score_text = " | ".join(
        f"pid {persona_id}: {_format_score(score)}"
        for persona_id, score in sorted(persona_scores.items())
    )
    mean_score = details.get("mean_score")
    if mean_score is not None:
        score_text = f"mean {_format_score(mean_score)} | {score_text}"
    st.caption(f"{label}: {score_text}")
    persona_errors = details.get("persona_errors") or {}
    if st.session_state.get("debug_mode") and persona_errors:
        with st.expander(f"{label} scorer fallbacks", expanded=False):
            for persona_id, error_text in sorted(persona_errors.items()):
                st.code(f"pid {persona_id}: {error_text}", language="text")


def _format_score(score: float) -> str:
    """Render enough precision to distinguish near-integer scores."""
    return f"{score:.6f}"


def _render_candidate_gallery(
    title: str,
    candidates: list[dict[str, Any]],
    *,
    key_prefix: str,
    sort_desc: bool = True,
) -> None:
    """Render a gallery of candidates with images, prompts, and scores."""
    st.subheader(title)
    items = _rank_candidates_desc(candidates) if sort_desc else list(candidates)
    if not items:
        st.info("No candidates available.")
        return

    for idx in range(0, len(items), 2):
        cols = st.columns(2, gap="large")
        for col, candidate in zip(cols, items[idx : idx + 2]):
            with col:
                st.markdown('<div class="output-card">', unsafe_allow_html=True)
                st.markdown(
                    f"**Score:** {_format_score(candidate['score'])}  \n"
                    f"**Score Mode:** {candidate.get('score_mode', 'unknown')}  \n"
                    f"**Source:** {candidate['source']}"
                )
                if candidate.get("prompt_prescore") is not None:
                    st.caption(
                        "Prompt pre-score: "
                        f"{_format_score(candidate['prompt_prescore'])} "
                        f"({candidate.get('prompt_prescore_mode', 'unknown')})"
                    )
                if candidate.get("output_path"):
                    st.image(candidate["output_path"], use_container_width=True)
                else:
                    st.warning(candidate.get("error") or "Image generation failed.")
                    if st.session_state.get("debug_mode") and candidate.get("debug_traceback"):
                        with st.expander("Traceback", expanded=False):
                            st.code(candidate["debug_traceback"], language="python")
                with st.expander("Prompt", expanded=False):
                    st.code(candidate["prompt"], language="text")
                st.caption(_render_probs(candidate["probs"]))
                _render_persona_score_details(candidate.get("score_details"))
                _render_persona_score_details(
                    candidate.get("prompt_prescore_details"),
                    label="Prompt persona pre-scores",
                )
                if candidate.get("strategy"):
                    st.markdown(f"**Strategy:** {candidate['strategy']}")
                st.markdown("</div>", unsafe_allow_html=True)


def _render_base_results(results: dict[str, Any]) -> None:
    """Render base optimizer results."""
    st.subheader("Base Search")
    best = results["best"]
    st.markdown(
        f"Best base result: **{_format_score(best['score'])}** "
        f"({best.get('score_mode', 'unknown')}) from seed `{best.get('start_seed_id', 'n/a')}`."
    )

    for chain in results["chains"]:
        seed = chain["seed"]
        best_chain = chain["best"]
        with st.expander(
            f"Chain {chain['chain_id']} | seed score {_format_score(seed['score'])} | "
            f"best {_format_score(best_chain['score'])}",
            expanded=False,
        ):
            st.markdown(f"**Seed prompt:** `{seed['candidate_id']}`")
            if seed.get("output_path"):
                st.image(seed["output_path"], use_container_width=True)
            elif seed.get("error"):
                st.warning(seed["error"])
                if st.session_state.get("debug_mode") and seed.get("debug_traceback"):
                    with st.expander("Traceback", expanded=False):
                        st.code(seed["debug_traceback"], language="python")
            st.code(seed["prompt"], language="text")
            if chain["steps"]:
                _render_candidate_gallery(
                    "Step Results",
                    chain["steps"],
                    key_prefix=f"base_chain_{chain['chain_id']}",
                    sort_desc=False,
                )


def _render_textbo_iteration_winners(winners: list[dict[str, Any]], label: str = "TextBO") -> None:
    """Render the best candidate at each parallel iteration."""
    st.subheader(f"{label} Middle Outcomes")
    st.caption(f"Best rendered candidate at each iteration across parallel {label} trajectories.")
    if not winners:
        st.info("No middle outcomes available.")
        return

    for idx in range(0, len(winners), 2):
        cols = st.columns(2, gap="large")
        for col, winner in zip(cols, winners[idx : idx + 2]):
            with col:
                st.markdown('<div class="output-card">', unsafe_allow_html=True)
                st.markdown(
                    f"**Iteration {winner['step']}** | trajectory {winner['trajectory_id']}  \n"
                    f"**Score:** {_format_score(winner['score'])} "
                    f"({winner.get('score_mode', 'unknown')})  \n"
                    f"**Selection:** {winner.get('selection_mode') or winner.get('prompt_prescore_mode', 'unknown')}"
                )
                if winner.get("gradient_steps"):
                    st.caption(
                        f"G={winner['gradient_steps']} | "
                        f"rendered candidates={winner.get('tournament_candidate_count', 0)}"
                    )
                if winner.get("prompt_prescore") is not None:
                    st.caption(
                        "Prompt pre-score: "
                        f"{_format_score(winner['prompt_prescore'])} "
                        f"({winner.get('prompt_prescore_mode', 'unknown')})"
                    )
                if winner.get("output_path"):
                    st.image(winner["output_path"], use_container_width=True)
                elif winner.get("error"):
                    st.warning(winner["error"])
                    if st.session_state.get("debug_mode") and winner.get("debug_traceback"):
                        with st.expander("Traceback", expanded=False):
                            st.code(winner["debug_traceback"], language="python")
                with st.expander(f"Winning Prompt For Iteration {winner['step']}", expanded=False):
                    st.code(winner["prompt"], language="text")
                if winner.get("probs"):
                    st.caption(_render_probs(winner["probs"]))
                _render_persona_score_details(winner.get("score_details"))
                _render_persona_score_details(
                    winner.get("prompt_prescore_details"),
                    label="Prompt persona pre-scores",
                )
                if winner.get("strategy"):
                    st.markdown(f"**Strategy:** {winner['strategy']}")
                st.markdown("</div>", unsafe_allow_html=True)


def _render_efficient_results(results: dict[str, Any], label: str = "TextBO") -> None:
    """Render TextBO-style optimizer results."""
    st.subheader(f"{label} Search")
    best = results["best"]
    st.markdown(
        f"Best {label} result: **{_format_score(best['score'])}** "
        f"({best.get('score_mode', 'unknown')}) from seed `{best.get('start_seed_id', 'n/a')}`."
    )

    if results.get("step_winners"):
        _render_textbo_iteration_winners(results["step_winners"], label=label)

    with st.expander("Shared Reflections", expanded=False):
        for item in results["shared_reflection_history"]:
            st.markdown(
                f"**Step {item['step']}**  \n"
                f"History size: {item['history_size']}  \n"
                f"{item['reflection']}"
            )

    for trajectory in results["trajectories"]:
        seed = trajectory["seed"]
        best_traj = trajectory["best"]
        with st.expander(
            f"Trajectory {trajectory['trajectory_id']} | seed score {_format_score(seed['score'])} | "
            f"best {_format_score(best_traj['score'])}",
            expanded=False,
        ):
            st.markdown(f"**Seed prompt:** `{seed['candidate_id']}`")
            if seed.get("output_path"):
                st.image(seed["output_path"], use_container_width=True)
            elif seed.get("error"):
                st.warning(seed["error"])
                if st.session_state.get("debug_mode") and seed.get("debug_traceback"):
                    with st.expander("Traceback", expanded=False):
                        st.code(seed["debug_traceback"], language="python")
            st.code(seed["prompt"], language="text")

            for step_entry in trajectory["steps"]:
                if step_entry.get("prompt_prescore") is not None:
                    selection_text = (
                        f"pre-score {_format_score(step_entry.get('prompt_prescore', 0.0))} "
                        f"({step_entry.get('prompt_prescore_mode', 'unknown')})"
                    )
                else:
                    selection_text = (
                        f"{step_entry.get('selection_mode', 'image-tournament')} "
                        f"over {step_entry.get('tournament_candidate_count', 0)} candidates"
                    )
                st.markdown(
                    f"**Step {step_entry['step']}** | "
                    f"{selection_text} | "
                    f"final score {_format_score(step_entry['score'])} "
                    f"({step_entry.get('score_mode', 'unknown')})"
                )
                if step_entry.get("output_path"):
                    st.image(step_entry["output_path"], use_container_width=True)
                elif step_entry.get("error"):
                    st.warning(step_entry["error"])
                    if st.session_state.get("debug_mode") and step_entry.get("debug_traceback"):
                        with st.expander("Traceback", expanded=False):
                            st.code(step_entry["debug_traceback"], language="python")
                _render_persona_score_details(step_entry.get("score_details"))
                if step_entry.get("candidate_prompts"):
                    with st.expander(
                        f"Best-of-N Candidates For Step {step_entry['step']}",
                        expanded=False,
                    ):
                        for cand in step_entry["candidate_prompts"]:
                            if cand.get("score") is not None:
                                st.markdown(
                                    f"- pre-score {_format_score(cand['score'])} "
                                    f"({cand.get('mode', 'unknown')}) | "
                                    f"{_prompt_excerpt(cand['prompt'], 260)}"
                                )
                                _render_persona_score_details(
                                    cand.get("details"),
                                    label="Prompt persona pre-scores",
                                )
                            else:
                                selected_label = " | tournament winner" if cand.get("selected") else ""
                                image_label = "image ready" if cand.get("output_path") else "image failed"
                                st.markdown(
                                    f"- G{cand.get('gradient_step', '?')} candidate {cand.get('candidate_idx')} | {image_label}"
                                    f"{selected_label} | {_prompt_excerpt(cand.get('prompt', ''), 260)}"
                                )
                                if cand.get("error"):
                                    st.caption(cand["error"])
                st.code(step_entry["prompt"], language="text")
                if step_entry.get("strategy"):
                    st.caption(step_entry["strategy"])


def _render_textbo_baseline_comparison(results: dict[str, Any]) -> None:
    """Render final TextBO versus Base N=1, G=1 comparison."""
    comparison = results.get("textbo_baseline_comparison")
    baseline_error = results.get("hidden_baseline_error")

    st.subheader("TextBO vs Base")
    st.caption(
        "Base uses the same starting prompts and iteration count, with "
        "Best of N = 1 and one revision step per iteration."
    )

    if baseline_error:
        st.warning(
            "Base comparison is unavailable: "
            f"{baseline_error.get('error_type', 'Error')}: {baseline_error.get('message', '')}"
        )
        if st.session_state.get("debug_mode") and baseline_error.get("traceback"):
            with st.expander("Base Traceback", expanded=False):
                st.code(baseline_error["traceback"], language="python")
        return

    if not comparison:
        st.info("Base comparison is unavailable for this run.")
        return

    textbo_best = comparison["textbo_best"]
    baseline_best = comparison["hidden_baseline_best"]
    score_delta = comparison["score_delta"]
    winner_label = {
        "textbo": "TextBO",
        "hidden_baseline": "Base",
        "tie": "Tie",
    }.get(comparison["winner"], comparison["winner"])

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("TextBO Best", _format_score(textbo_best["score"]))
    with metric_cols[1]:
        st.metric("Base Best", _format_score(baseline_best["score"]))
        st.caption("Base = hidden N=1, G=1 optimizer")
    with metric_cols[2]:
        st.metric("TextBO Delta", f"{score_delta:+.6f}")
    st.markdown(f"Final comparison winner: **{winner_label}**")

    outcome_cols = st.columns(2, gap="large")
    for col, title, candidate in [
        (outcome_cols[0], "TextBO Final Outcome", textbo_best),
        (outcome_cols[1], "Base Final Outcome", baseline_best),
    ]:
        with col:
            st.markdown(f"**{title}**")
            st.markdown(
                f"Score: **{_format_score(candidate['score'])}** "
                f"({candidate.get('score_mode', 'unknown')})  \n"
                f"Source: `{candidate.get('source', 'unknown')}`"
            )
            if candidate.get("output_path"):
                st.image(candidate["output_path"], use_container_width=True)
            else:
                st.warning(candidate.get("error") or "Image generation failed.")
            with st.expander(f"{title} Prompt", expanded=False):
                st.code(candidate["prompt"], language="text")
            _render_persona_score_details(candidate.get("score_details"))
            if candidate.get("strategy"):
                st.caption(candidate["strategy"])


def _render_run_diagnostics() -> None:
    """Render persistent diagnostics for the latest run."""
    last_error = st.session_state.get("last_error")
    debug_mode = bool(st.session_state.get("debug_mode", False))
    debug_events = st.session_state.get("debug_events", [])

    if not last_error and not (debug_mode and debug_events):
        return

    st.divider()
    st.subheader("Run Diagnostics")

    if last_error:
        st.error(
            "Last run failed during "
            f"`{last_error.get('stage', 'unknown')}`: "
            f"{last_error.get('error_type', 'Error')}: {last_error.get('message', 'Unknown error')}"
        )
        if debug_mode and last_error.get("traceback"):
            with st.expander("Latest Traceback", expanded=True):
                st.code(last_error["traceback"], language="python")

    if debug_mode and debug_events:
        with st.expander("Debug Event Log", expanded=last_error is None):
            for event in debug_events:
                label = event["stage"]
                if event.get("error_type"):
                    label = f"{label} | {event['error_type']}"
                st.markdown(f"**{label}**: {event['message']}")
                if event.get("traceback"):
                    st.code(event["traceback"], language="python")


# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Ad Campaign Agent Optimizer",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    .stChatMessage {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .style-selected-ring {
        border: 2.5px solid #6366f1;
        border-radius: 10px;
        padding: 2px;
    }
    .style-default-ring {
        border: 2.5px solid transparent;
        border-radius: 10px;
        padding: 2px;
    }
    .phase-step {
        display: flex; align-items: center; gap: 0.5rem;
        padding: 0.3rem 0; font-size: 0.85rem; color: #94a3b8;
    }
    .phase-step.active { color: #6366f1; font-weight: 600; }
    .phase-step.done   { color: #22c55e; }
    .phase-dot {
        width: 10px; height: 10px; border-radius: 50%;
        background: #cbd5e1; flex-shrink: 0;
    }
    .phase-step.active .phase-dot { background: #6366f1; box-shadow: 0 0 0 3px rgba(99,102,241,0.25); }
    .phase-step.done .phase-dot { background: #22c55e; }
    .output-card {
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.25rem;
        background: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin: 0.75rem 0;
    }
    .welcome-hero {
        background: linear-gradient(135deg, #0f766e 0%, #0ea5e9 50%, #1d4ed8 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .welcome-hero h2 { color: white; margin: 0 0 0.5rem 0; font-size: 1.5rem; }
    .welcome-hero p  { color: rgba(255,255,255,0.85); margin: 0; font-size: 0.95rem; line-height: 1.5; }
    .approve-bar {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 0.75rem 1.25rem;
        margin: 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .approve-bar p {
        margin: 0;
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.4;
    }
    .model-note {
        font-size: 0.78rem;
        color: #64748b;
        line-height: 1.4;
        margin-top: 0.3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Ad Campaign Agent Optimizer")
st.caption("GPT-4o prompt generation + GPT-5-nano reflection + image generation")

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("API Keys")

    openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key_input")
    gemini_key = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        key="gemini_key_input",
    )

    if not openai_key:
        st.warning("OpenAI API key required.")
        st.stop()

    st.divider()
    st.header("Generation Settings")

    image_models = {"GPT Image 1.5": "gpt"}
    if gemini_key:
        image_models["Gemini 2.5 Flash"] = "gemini"

    model_label = st.selectbox("Model", list(image_models.keys()))
    model_key = image_models[model_label]

    gpt_quality = "medium"
    if model_key == "gemini":
        st.markdown(
            '<div class="model-note">Multimodal LLM with native image output. '
            'Uses the style reference image directly for visual matching.</div>',
            unsafe_allow_html=True,
        )
        aspect_ratio = "auto"
    else:
        st.markdown(
            '<div class="model-note">OpenAI dedicated image generation API. '
            'Style is applied via text prompt only.</div>',
            unsafe_allow_html=True,
        )
        gpt_res = st.selectbox(
            "Resolution",
            ["1536x1024 (landscape)", "1024x1024 (square)", "1024x1536 (portrait)"],
        )
        aspect_ratio = gpt_res
        gpt_quality = st.selectbox(
            "Quality",
            ["low", "medium", "high"],
            index=1,
            format_func=lambda q: q.capitalize(),
        )

    gpt_image_quality = gpt_quality

    st.divider()
    st.header("Style Reference")

    style_keys_list = list(STYLE_PRESETS.keys())
    if "selected_style" not in st.session_state:
        st.session_state.selected_style = style_keys_list[0]

    cols_per_row = 2
    for row_start in range(0, len(style_keys_list), cols_per_row):
        cols = st.columns(cols_per_row, gap="small")
        for col_idx, col in enumerate(cols):
            key_idx = row_start + col_idx
            if key_idx >= len(style_keys_list):
                break
            skey = style_keys_list[key_idx]
            preset = STYLE_PRESETS[skey]
            style_path = STYLE_DIR / f"{skey}.png"
            is_selected = st.session_state.selected_style == skey
            with col:
                ring_class = "style-selected-ring" if is_selected else "style-default-ring"
                if style_path.exists():
                    st.markdown(f'<div class="{ring_class}">', unsafe_allow_html=True)
                    st.image(str(style_path), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                btn_type = "primary" if is_selected else "secondary"
                if st.button(
                    preset["label"],
                    key=f"style_btn_{skey}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    st.session_state.selected_style = skey
                    st.rerun()

    selected_style_key = st.session_state.selected_style
    style_path = STYLE_DIR / f"{selected_style_key}.png"
    style_image_bytes = style_path.read_bytes() if style_path.exists() else None
    style_description = STYLE_PRESETS[selected_style_key]["description"]

    st.divider()
    st.header("Search Settings")
    initial_prompt_count = st.number_input(
        "Initial prompts",
        min_value=2,
        max_value=20,
        value=DEFAULT_INITIAL_PROMPTS,
        step=1,
    )
    lowest_prompt_count = st.number_input(
        "Lowest prompts to optimize",
        min_value=1,
        max_value=10,
        value=DEFAULT_LOWEST_PROMPTS,
        step=1,
    )
    optimization_steps = st.number_input(
        "Optimization steps",
        min_value=1,
        max_value=20,
        value=DEFAULT_OPTIMIZATION_STEPS,
        step=1,
    )
    efficient_candidates = st.number_input(
        "TextBO best-of-N candidates / gradient step",
        min_value=1,
        max_value=5,
        value=DEFAULT_EFFICIENT_CANDIDATES,
        step=1,
    )
    textbo_gradient_steps = st.number_input(
        "TextBO gradient steps / iteration",
        min_value=2,
        max_value=10,
        value=DEFAULT_TEXTBO_GRADIENT_STEPS,
        step=1,
        help="Visible TextBO uses G > 1. Base remains fixed at N=1 and G=1.",
    )

    st.divider()
    st.header("Diagnostics")
    debug_mode = st.checkbox(
        "Debug mode",
        value=bool(st.session_state.get("debug_mode", False)),
        help="Show detailed errors and tracebacks in the UI when a run fails.",
    )
    st.session_state.debug_mode = debug_mode

    sidebar_settings = {
        "model": model_label,
        "resolution": aspect_ratio if aspect_ratio != "auto" else "auto",
        "style_description": style_description,
        "debug_mode": debug_mode,
    }

# ─── Session State ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_data" not in st.session_state:
    st.session_state.session_data = {
        "product_name": "",
        "target_audience": "",
        "campaign_goal": "",
        "key_message": "",
        "brand_tone": "",
        "style_reference": "",
    }

if "phase" not in st.session_state:
    st.session_state.phase = "collecting"

if "creative_brief" not in st.session_state:
    st.session_state.creative_brief = None

if "optimization_results" not in st.session_state:
    st.session_state.optimization_results = None

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

if "last_error" not in st.session_state:
    st.session_state.last_error = None

if "debug_events" not in st.session_state:
    st.session_state.debug_events = []

if style_image_bytes:
    st.session_state.session_data["style_reference"] = STYLE_PRESETS[selected_style_key]["label"]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

openai_client = get_openai_client(openai_key)

if not st.session_state.messages:
    st.markdown(
        """
    <div class="welcome-hero">
        <h2>Welcome to Ad Campaign Agent Optimizer</h2>
        <p>Choose a model and visual style in the sidebar, then describe your campaign.
        After you approve the brief, the app will seed multiple prompts, score them,
        and run the Base N=1/G=1 and TextBO prompt-search loops.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    greeting = (
        "Hi! I'm your **Ad Campaign Agent Optimizer**. I'll help you build and search over ad prompts.\n\n"
        "To get started, tell me about your campaign. I'll need these details:\n\n"
        "- **Product / Service Name** — what you're advertising\n"
        "- **Target Audience** — who you're reaching\n"
        "- **Campaign Goal** — awareness, consideration, conversion, or launch\n"
        "- **Key Message / CTA** — the main takeaway and call to action\n"
        "- **Brand Tone** — the emotional feel (e.g. bold, calm, playful)\n\n"
        "You can share everything at once or one at a time — I'll guide you through it."
    )
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    with st.chat_message("assistant"):
        st.markdown(greeting)

trigger_generation = st.session_state.pop("trigger_generation", False)

if (
    st.session_state.phase == "reviewing"
    and st.session_state.creative_brief
    and not trigger_generation
):
    st.markdown(
        '<div class="approve-bar">'
        '<p>Brief is ready for review. Type feedback below to request changes, '
        'or click the button to run the prompt search.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_btn, col_spacer = st.columns([1, 3])
    with col_btn:
        if st.button("Approve & Run", type="primary", use_container_width=True):
            st.session_state.trigger_generation = True
            st.rerun()


def _run_optimization() -> None:
    """Execute the prompt-search pipeline and persist results."""
    st.session_state.phase = "generating"
    st.session_state.optimization_results = None
    st.session_state.last_error = None
    st.session_state.debug_events = []

    try:
        with st.chat_message("assistant"):
            with st.status("Running prompt search...", expanded=True) as status:
                def _status_report_image(image_path: str, caption: str | None = None) -> None:
                    if image_path:
                        st.image(image_path, caption=caption, use_container_width=True)

                results = _run_search_pipeline(
                    openai_client=openai_client,
                    gemini_key=gemini_key,
                    model_key=model_key,
                    aspect_ratio=aspect_ratio,
                    gpt_image_quality=gpt_image_quality,
                    style_image_bytes=style_image_bytes,
                    session_data=st.session_state.session_data,
                    sidebar_settings=sidebar_settings,
                    creative_brief=st.session_state.creative_brief,
                    initial_prompt_count=int(initial_prompt_count),
                    lowest_prompt_count=int(lowest_prompt_count),
                    optimization_steps=int(optimization_steps),
                    efficient_candidates=int(efficient_candidates),
                    textbo_gradient_steps=int(textbo_gradient_steps),
                    judge_repeats=DEFAULT_JUDGE_REPEATS,
                    report=st.write,
                    report_image=_status_report_image,
                )
                status.update(label="Prompt search complete!", state="complete", expanded=False)

        st.session_state.optimization_results = results
        st.session_state.last_error = None
        st.session_state.phase = "done"
        response_text = (
            "Search finished. The initial prompts, the lowest-performing seeds, and the "
            "Base N=1/G=1 and TextBO comparison results are shown below."
        )
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()
    except Exception as exc:
        traceback_text = _format_exception_traceback(exc)
        st.session_state.last_error = {
            "stage": "optimization",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback_text,
        }
        _record_debug_event(
            "optimization",
            f"Search failed: {type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
            traceback_text=traceback_text,
        )
        st.session_state.phase = "reviewing"
        error_text = f"Search failed: {exc}"
        st.session_state.messages.append({"role": "assistant", "content": error_text})
        st.error(error_text)


if trigger_generation and st.session_state.creative_brief:
    approval_msg = "Approved — run the prompt search."
    st.session_state.messages.append({"role": "user", "content": approval_msg})
    with st.chat_message("user"):
        st.markdown(approval_msg)
    _run_optimization()

phase = st.session_state.get("phase", "collecting")
placeholders = {
    "collecting": "Describe your product and campaign...",
    "reviewing": "Type feedback to revise the brief...",
    "done": "Start a new campaign (reset in sidebar)",
}
user_input = st.chat_input(placeholders.get(phase, "Describe your campaign idea..."))

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.phase == "done":
        done_text = "Reset the session from the sidebar to start a new campaign."
        st.session_state.messages.append({"role": "assistant", "content": done_text})
        with st.chat_message("assistant"):
            st.markdown(done_text)
        st.stop()

    approve_words = [
        "approve",
        "approved",
        "looks good",
        "let's go",
        "run",
        "yes",
        "proceed",
        "go ahead",
        "perfect",
        "love it",
        "lgtm",
        "do it",
        "go for it",
        "ship it",
        "make it",
        "create it",
        "好",
        "好的",
        "可以",
        "没问题",
        "开始",
        "生成",
        "确认",
        "批准",
        "通过",
        "同意",
        "行",
        "就这样",
        "没意见",
        "ok",
    ]
    is_approval = (
        st.session_state.phase == "reviewing"
        and any(word in user_input.lower().strip() for word in approve_words)
    )

    if is_approval and st.session_state.creative_brief:
        _run_optimization()
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = chat(
                    openai_client,
                    st.session_state.messages,
                    st.session_state.session_data,
                    sidebar_settings=sidebar_settings,
                )

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

            brief = try_parse_brief(response_text)
            if brief and (
                "composition" in brief or "headline" in brief or "visual_style" in brief
            ):
                st.session_state.creative_brief = brief
                st.session_state.phase = "reviewing"
                st.rerun()

            if st.session_state.phase == "collecting":
                try:
                    extracted = extract_fields(openai_client, st.session_state.messages)
                    for key, value in extracted.items():
                        if value and key in st.session_state.session_data:
                            st.session_state.session_data[key] = value
                except Exception:
                    pass

if st.session_state.optimization_results:
    results = st.session_state.optimization_results

    st.divider()
    st.header("Search Results")

    metric_cols = st.columns(3)
    with metric_cols[0]:
        best_initial = results.get("best_starting_initial") or max(
            results["worst_pool"],
            key=lambda item: item["score"],
        )
        st.metric("Best Initial", _format_score(best_initial["score"]))
        st.caption(
            f"Best among the {len(results['worst_pool'])} lowest starting prompts"
        )
    with metric_cols[1]:
        base_best = results["base"]["best"] if results.get("base") else None
        if results.get("base"):
            st.metric("Best Base", _format_score(base_best["score"]))
            st.caption("Base = hidden N=1, G=1 optimizer")
        else:
            st.metric("Best Base", "n/a")
    with metric_cols[2]:
        st.metric("Best TextBO", _format_score(results["efficient"]["best"]["score"]))

    display_overall_candidates = [best_initial, results["efficient"]["best"]]
    if base_best:
        display_overall_candidates.append(base_best)
    overall_best = max(display_overall_candidates, key=lambda item: item["score"])

    st.markdown(
        f"**Overall best score:** {_format_score(overall_best['score'])} "
        f"({overall_best.get('score_mode', 'unknown')}) from `{overall_best['source']}`."
    )
    if overall_best.get("output_path"):
        st.image(overall_best["output_path"], use_container_width=True)
    with st.expander("Overall Best Prompt", expanded=False):
        st.code(overall_best["prompt"], language="text")

    _render_textbo_baseline_comparison(results)

    download_payload = json.dumps(results, indent=2, ensure_ascii=False)
    st.download_button(
        "Download Results JSON",
        data=download_payload,
        file_name="ad_campaign_optimizer_results.json",
        mime="application/json",
        use_container_width=False,
    )

    tabs = st.tabs(
        [
            "Initial",
            f"Lowest {len(results['worst_pool'])}",
            "Base",
            "TextBO",
        ]
    )
    with tabs[0]:
        _render_candidate_gallery(
            "Initial Prompt Candidates",
            results["initial_candidates"],
            key_prefix="initial_candidates",
        )
    with tabs[1]:
        _render_candidate_gallery(
            "Lowest-Scoring Starting Points",
            results["worst_pool"],
            key_prefix="worst_pool",
            sort_desc=False,
        )
    with tabs[2]:
        if results.get("base"):
            _render_efficient_results(results["base"], label="Base")
        else:
            base_error = results.get("base_error") or results.get("hidden_baseline_error")
            st.subheader("Base")
            st.warning(
                "Base N=1, G=1 result is unavailable"
                + (
                    f": {base_error.get('error_type', 'Error')}: {base_error.get('message', '')}"
                    if base_error
                    else "."
                )
            )
            if (
                st.session_state.get("debug_mode")
                and base_error
                and base_error.get("traceback")
            ):
                with st.expander("Base Traceback", expanded=False):
                    st.code(base_error["traceback"], language="python")
    with tabs[3]:
        _render_efficient_results(results["efficient"])

_render_run_diagnostics()

with st.sidebar:
    st.divider()
    st.header("Session Status")

    current_phase = st.session_state.phase
    steps = [
        ("collecting", "Collect info"),
        ("reviewing", "Review brief"),
        ("generating", "Run search"),
        ("done", "Done"),
    ]
    phase_order = [item[0] for item in steps]
    current_idx = phase_order.index(current_phase) if current_phase in phase_order else 0

    stepper_html = ""
    for i, (_, phase_label) in enumerate(steps):
        if i < current_idx:
            cls = "phase-step done"
        elif i == current_idx:
            cls = "phase-step active"
        else:
            cls = "phase-step"
        stepper_html += f'<div class="{cls}"><span class="phase-dot"></span>{phase_label}</div>'

    st.markdown(stepper_html, unsafe_allow_html=True)

    if st.button("Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
