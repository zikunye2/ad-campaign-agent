"""GPT-5-nano agent for the Ad Campaign pipeline."""

import json

from openai import OpenAI

from prompts import SYSTEM_PROMPT
from schema import validate_fields, format_fields_for_prompt

TEXT_MODEL = "gpt-5-nano"
TEXT_REASONING_EFFORT = "minimal"


def get_openai_client(api_key: str) -> OpenAI:
    """Create an OpenAI client."""
    return OpenAI(api_key=api_key)


def _build_messages(
    conversation_history: list[dict],
    session_data: dict,
    sidebar_settings: dict | None = None,
) -> list[dict]:
    """Build the messages array for the OpenAI API call."""
    missing = validate_fields(session_data)
    fields_status = format_fields_for_prompt(session_data)

    system_content = SYSTEM_PROMPT + f"""

## Current Session State

**Collected fields:**
{fields_status}

**Missing required fields:** {', '.join(missing) if missing else 'NONE — all required fields collected'}

**Phase:** {"Collecting info (Loop 1)" if missing else "Ready to generate brief or revising (Loop 2)"}
"""

    if sidebar_settings:
        parts = ["\n**Sidebar Settings (use these EXACT values in the creative brief — do NOT invent your own):**"]
        parts.append(f"- Model: {sidebar_settings.get('model', 'Image')}")
        parts.append(f"- Resolution: {sidebar_settings.get('resolution', '1920x1080')}")
        style_desc = sidebar_settings.get('style_description', '')
        if style_desc:
            parts.append(f"- Style Direction: {style_desc}")
        system_content += "\n".join(parts) + "\n"

    messages = [{"role": "system", "content": system_content}]

    for msg in conversation_history:
        messages.append(msg)

    return messages


def chat(
    client: OpenAI,
    conversation_history: list[dict],
    session_data: dict,
    sidebar_settings: dict | None = None,
) -> str:
    """Send the conversation to GPT-5-nano and get a response."""
    messages = _build_messages(conversation_history, session_data, sidebar_settings)

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        temperature=1,
        max_completion_tokens=65536,
        reasoning_effort=TEXT_REASONING_EFFORT,
    )

    return response.choices[0].message.content or ""


def build_generation_prompt(
    brief: dict,
    session_data: dict,
    sidebar_settings: dict | None = None,
    style_description: str | None = None,
    has_style_image: bool = False,
) -> str:
    """Build a generation prompt directly from the creative brief (no LLM call).

    Assembles composition, visual style, and style direction into a single
    cinematic paragraph optimized for image generation models.

    When has_style_image is False (e.g. GPT Image), the style description is
    reinforced with extra emphasis so text-only models match the intended aesthetic.
    """
    parts = []

    # Style-first framing for text-only models (no reference image)
    if style_description and not has_style_image:
        parts.append(
            f"STYLE DIRECTIVE — follow this aesthetic strictly: {style_description}. "
            "Every element (lighting, color, texture, composition) must reflect this style."
        )

    # Product context
    product = session_data.get("product_name", "product")
    parts.append(f"Cinematic shot of {product}.")

    # Style direction (from sidebar selection)
    if style_description and has_style_image:
        parts.append(f"Visual style direction: {style_description}.")

    # Style direction from brief (LLM-generated)
    style_dir = brief.get("style_direction")
    if style_dir:
        parts.append(f"Style: {style_dir}.")

    # Composition
    comp = brief.get("composition", {})
    if comp.get("description"):
        parts.append(comp["description"])
    if comp.get("camera_angle"):
        parts.append(f"Camera angle: {comp['camera_angle']}.")
    if comp.get("product_placement"):
        parts.append(f"Product placement: {comp['product_placement']}.")

    # Visual style
    vs = brief.get("visual_style", {})
    if vs.get("lighting"):
        parts.append(f"Lighting: {vs['lighting']}.")
    if vs.get("color_palette"):
        colors = ", ".join(vs["color_palette"])
        parts.append(f"Color palette: {colors}.")
    if vs.get("aesthetic"):
        parts.append(f"Aesthetic: {vs['aesthetic']}.")

    # Product notes
    if brief.get("product_notes"):
        parts.append(brief["product_notes"])

    # Headline and CTA as text overlays
    if brief.get("headline"):
        parts.append(f"Text overlay headline: '{brief['headline']}'.")
    cta = brief.get("cta", {})
    if cta.get("text"):
        placement = cta.get("placement", "bottom-right")
        parts.append(f"CTA text: '{cta['text']}', placed {placement}.")

    # Tone
    tone = session_data.get("brand_tone")
    if tone:
        parts.append(f"Overall mood: {tone}.")

    # Format from sidebar
    if sidebar_settings:
        resolution = sidebar_settings.get("resolution", "")
        if resolution:
            parts.append(f"Output: {resolution}.")

    return " ".join(parts)


def extract_fields(
    client: OpenAI,
    conversation_history: list[dict],
) -> dict:
    """Extract structured field values from the conversation so far.

    Returns a dict with only the fields that were explicitly mentioned.
    Keys match session_data keys. Empty/unknown fields are omitted.
    """
    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in conversation_history
        if isinstance(m.get("content"), str)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Extract campaign fields from the conversation. Return ONLY a JSON object "
                "with these EXACT keys (omit keys where the value was not mentioned):\n\n"
                '  "product_name": string — the product or service being advertised\n'
                '  "target_audience": string — demographics, interests, psychographics\n'
                '  "campaign_goal": string — MUST be one of: awareness, consideration, conversion, launch\n'
                '  "key_message": string — core message and call to action\n'
                '  "brand_tone": string — emotional tone and personality\n\n'
                "Output ONLY valid JSON. No markdown fences. No explanation.\n"
                'Example: {"product_name": "Nike Air Max", "target_audience": "Gen Z, 18-25", '
                '"campaign_goal": "launch", "key_message": "Step into the future — Shop now", '
                '"brand_tone": "bold, energetic"}'
            ),
        },
        {"role": "user", "content": convo_text[-4000:]},
    ]

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        temperature=1,
        max_completion_tokens=16384,
        reasoning_effort=TEXT_REASONING_EFFORT,
        response_format={"type": "json_object"},
    )

    text = response.choices[0].message.content or "{}"
    try:
        if "```" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            text = text[start:end]
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}


def try_parse_brief(text: str) -> dict | None:
    """Try to extract a JSON creative brief from agent response text."""
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        json_str = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        json_str = text[start:end].strip()
    elif "{" in text and "}" in text:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
    else:
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None
