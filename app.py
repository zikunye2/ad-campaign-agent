"""Ad Campaign Agent — Streamlit App."""

import json
import streamlit as st

from agent import get_openai_client, chat, build_generation_prompt, extract_fields, try_parse_brief
from generate import generate_image_openai, generate_image_gemini
from schema import validate_fields, STYLE_PRESETS, STYLE_DIR

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Ad Campaign Agent",
    page_icon="🎨",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }

    /* ── Chat messages ── */
    .stChatMessage {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s ease;
    }

    /* ── Style grid: selected state ── */
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

    /* ── Phase stepper ── */
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
    .phase-step.done   .phase-dot { background: #22c55e; }

    /* ── Output card ── */
    .output-card {
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        background: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin: 1rem 0;
    }
    .output-card img { border-radius: 12px; }

    /* ── Welcome hero ── */
    .welcome-hero {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .welcome-hero h2 { color: white; margin: 0 0 0.5rem 0; font-size: 1.5rem; }
    .welcome-hero p  { color: rgba(255,255,255,0.85); margin: 0; font-size: 0.95rem; line-height: 1.5; }

    /* ── Approve bar ── */
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

    /* ── Model description ── */
    .model-note {
        font-size: 0.78rem;
        color: #64748b;
        line-height: 1.4;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Ad Campaign Agent")
st.caption("GPT-5-nano + Image Generation")

# ─── API Keys ──────────────────────────────────────────────────

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

    # ─── Generation Settings ───────────────────────────────────
    st.divider()
    st.header("Generation Settings")

    image_models = {"GPT Image 1.5": "gpt"}
    if gemini_key:
        image_models["Gemini 2.5 Flash"] = "gemini"

    model_label = st.selectbox("Model", list(image_models.keys()))
    model_key = image_models[model_label]

    # Model-specific notes & settings
    gpt_quality = "medium"  # default, overridden below if GPT selected
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

        # Quality selector with pricing
        gpt_quality_options = ["low", "medium", "high"]
        gpt_quality = st.selectbox(
            "Quality",
            gpt_quality_options,
            index=1,  # default to medium
            format_func=lambda q: q.capitalize(),
        )

        # Show price for selected resolution + quality
        is_square = "1024x1024" in gpt_res
        _prices = {
            "low":    {"square": 0.009, "rect": 0.013},
            "medium": {"square": 0.034, "rect": 0.051},
            "high":   {"square": 0.133, "rect": 0.200},
        }
        price = _prices[gpt_quality]["square" if is_square else "rect"]
        st.markdown(
            f'<div class="model-note">Estimated cost: <b>${price:.3f}</b> / image</div>',
            unsafe_allow_html=True,
        )

    gpt_image_quality = gpt_quality

    # ─── Style Reference ──────────────────────────────────────────
    st.divider()
    st.header("Style Reference")

    style_keys_list = list(STYLE_PRESETS.keys())

    if "selected_style" not in st.session_state:
        st.session_state.selected_style = style_keys_list[0]

    # 2-column thumbnail grid
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
                    st.markdown('</div>', unsafe_allow_html=True)
                btn_type = "primary" if is_selected else "secondary"
                if st.button(preset["label"], key=f"style_btn_{skey}",
                             use_container_width=True, type=btn_type):
                    st.session_state.selected_style = skey
                    st.rerun()

    selected_style_key = st.session_state.selected_style

    style_image_bytes = None
    style_description = ""

    style_path = STYLE_DIR / f"{selected_style_key}.png"
    if style_path.exists():
        with open(style_path, "rb") as f:
            style_image_bytes = f.read()
    style_description = STYLE_PRESETS[selected_style_key]["description"]

    sidebar_settings = {
        "model": model_label,
        "resolution": aspect_ratio if aspect_ratio != "auto" else "auto",
        "style_description": style_description,
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

if "output_path" not in st.session_state:
    st.session_state.output_path = None

# Track style selection
if style_image_bytes:
    st.session_state.session_data["style_reference"] = (
        STYLE_PRESETS[selected_style_key]["label"]
        if selected_style_key != "custom"
        else "Custom upload"
    )

# ─── Display Chat History ─────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Show Output if Done ──────────────────────────────────────
if st.session_state.output_path:
    st.markdown('<div class="output-card">', unsafe_allow_html=True)
    st.image(str(st.session_state.output_path), use_container_width=True)
    col_dl, col_spacer = st.columns([1, 3])
    with col_dl:
        with open(st.session_state.output_path, "rb") as vf:
            st.download_button(
                "Download Image",
                data=vf.read(),
                file_name="ad_campaign.png",
                mime="image/png",
                use_container_width=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Chat ──────────────────────────────────────────────────────
openai_client = get_openai_client(openai_key)

# Welcome hero
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-hero">
        <h2>Welcome to Ad Campaign Agent</h2>
        <p>Choose a model and visual style in the sidebar, then tell me about your campaign.
        Your product doesn't need to exist yet — just describe the concept.</p>
    </div>
    """, unsafe_allow_html=True)

    greeting = (
        "Hi! I'm your **Ad Campaign Agent**. I'll help you create a polished ad image.\n\n"
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

# ─── Approve / Feedback bar (reviewing phase) ─────────────────
trigger_generation = st.session_state.pop("trigger_generation", False)

if (st.session_state.phase == "reviewing"
        and st.session_state.creative_brief
        and not trigger_generation):
    st.markdown(
        '<div class="approve-bar">'
        '<p>Brief is ready for review. Type feedback below to request changes, '
        'or click the button to start generating.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_btn, col_spacer = st.columns([1, 3])
    with col_btn:
        if st.button("Approve & Generate", type="primary", use_container_width=True):
            st.session_state.trigger_generation = True
            st.rerun()

# ─── Generation helper ────────────────────────────────────────


def _run_generation():
    """Execute image generation and update session state."""
    st.session_state.phase = "generating"

    with st.chat_message("assistant"):
        with st.status(f"Generating with {model_label}...", expanded=True) as status:
            st.write("Analyzing creative brief...")
            # Gemini receives the style image directly; GPT only gets text
            _has_style_img = model_key == "gemini" and style_image_bytes is not None
            gen_prompt = build_generation_prompt(
                st.session_state.creative_brief,
                st.session_state.session_data,
                sidebar_settings=sidebar_settings,
                style_description=style_description,
                has_style_image=_has_style_img,
            )
            st.write(f"Prompt ready ({len(gen_prompt.split())} words)")

            st.write(f"Sending to {model_label} — this may take a moment...")
            output_path = None
            error = None

            if model_key == "gemini":
                output_path, error = generate_image_gemini(
                    gemini_key, gen_prompt,
                    style_image_bytes=style_image_bytes,
                )
            elif model_key == "gpt":
                size_map = {
                    "1536x1024 (landscape)": "1536x1024",
                    "1024x1024 (square)": "1024x1024",
                    "1024x1536 (portrait)": "1024x1536",
                }
                output_path, error = generate_image_openai(
                    openai_client, gen_prompt,
                    size=size_map.get(aspect_ratio, "1536x1024"),
                    quality=gpt_image_quality,
                )

            if output_path:
                status.update(label="Generation complete!", state="complete", expanded=False)
                st.session_state.output_path = output_path
                st.session_state.phase = "done"
                response_text = "Your ad creative is ready! Scroll up to see it."
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.markdown(response_text)
                st.rerun()
            else:
                status.update(label="Generation failed", state="error")
                error_detail = error or "Unknown error"
                response_text = (
                    f"**Generation failed:** {error_detail}\n\n"
                    "Try a different model in the sidebar, then click **Approve & Generate** again."
                )
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.phase = "reviewing"
                st.error(f"Generation failed: {error_detail}")


# ─── Handle button-triggered generation ───────────────────────
if trigger_generation and st.session_state.creative_brief:
    approval_msg = "Approved — generate the image."
    st.session_state.messages.append({"role": "user", "content": approval_msg})
    with st.chat_message("user"):
        st.markdown(approval_msg)
    _run_generation()

# ─── Chat Input ───────────────────────────────────────────────
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

    phase = st.session_state.phase

    # Approval detection — English + Chinese keywords
    approve_words = [
        # English
        "approve", "approved", "looks good", "let's go", "generate",
        "yes", "proceed", "go ahead", "perfect", "love it",
        "lgtm", "do it", "go for it", "ship it", "make it", "create it",
        # Chinese
        "好", "好的", "可以", "没问题", "开始", "生成", "确认", "批准",
        "通过", "同意", "行", "就这样", "没意见", "ok",
    ]
    input_lower = user_input.lower().strip()
    is_approval = (
        phase == "reviewing"
        and any(w in input_lower for w in approve_words)
    )

    if is_approval and st.session_state.creative_brief:
        _run_generation()
    else:
        # Normal conversation
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

            # Brief detection FIRST — instant parse, rerun immediately to show button
            brief = try_parse_brief(response_text)
            if brief and ("composition" in brief or "headline" in brief or "visual_style" in brief):
                st.session_state.creative_brief = brief
                st.session_state.phase = "reviewing"
                st.rerun()  # button appears instantly — no extra API call

            # Extract fields only during collecting phase (skip once we have a brief)
            if st.session_state.phase == "collecting":
                try:
                    extracted = extract_fields(openai_client, st.session_state.messages)
                    for key, val in extracted.items():
                        if val and key in st.session_state.session_data:
                            st.session_state.session_data[key] = val
                except Exception:
                    pass

# ─── Sidebar: Session Status ──────────────────────────────────
with st.sidebar:
    st.divider()
    st.header("Session Status")

    # Phase stepper
    current_phase = st.session_state.phase
    steps = [
        ("collecting", "Collect info"),
        ("reviewing", "Review brief"),
        ("generating", "Generate"),
        ("done", "Done"),
    ]
    phase_order = [s[0] for s in steps]
    current_idx = phase_order.index(current_phase) if current_phase in phase_order else 0

    stepper_html = ""
    for i, (phase_key, phase_label) in enumerate(steps):
        if i < current_idx:
            cls = "phase-step done"
        elif i == current_idx:
            cls = "phase-step active"
        else:
            cls = "phase-step"
        stepper_html += f'<div class="{cls}"><span class="phase-dot"></span>{phase_label}</div>'

    st.markdown(stepper_html, unsafe_allow_html=True)

    st.markdown("")
    if st.button("Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
