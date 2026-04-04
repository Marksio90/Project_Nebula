"""
services/orchestrator/crew/crew.py
─────────────────────────────────────────────────────────────────────────────
CrewAI Crew factories for Project Nebula.

Three purpose-built crews, each with self-reflection capabilities:

  build_strategy_crew(...)   → CSO determines BPM/subgenre/arc
  build_audio_prompt_crew(…) → Audio PE generates batched MusicGen prompts
  build_visual_prompt_crew(…)→ Visual PE generates image/video prompts
  build_seo_crew(…)          → Polish SEO agent generates all metadata

All crews use GPT-4o as the LLM backbone via the OpenAI API.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
from pathlib import Path

import yaml
from crewai import Agent, Crew, Process, Task
from crewai import LLM

from shared.config import get_settings
from services.orchestrator.crew.tools import BpmRegistryTool, YoutubeAnalyticsTool

log = logging.getLogger("nebula.crew")
settings = get_settings()

# ── Config paths ──────────────────────────────────────────────────────────────
_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_yaml(filename: str) -> dict:
    with open(_CONFIG_DIR / filename, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── LLM singleton ─────────────────────────────────────────────────────────────

def _get_llm() -> LLM:
    """
    Creative LLM for prompt generation tasks.
    Model controlled by LLM_MODEL env var (default: gpt-4o-mini).
    16 384 output tokens — sufficient for 90+ stem prompts in one response.
    """
    return LLM(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.7,
        max_tokens=16_384,
    )


def _get_precise_llm() -> LLM:
    """
    Structured-output LLM for CSO strategy, QA, and SEO tasks.
    Model controlled by LLM_PRECISE_MODEL env var (default: gpt-4o-mini).
    Low temperature + high token budget for reliable JSON + Polish copy.
    """
    return LLM(
        model=settings.llm_precise_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
        max_tokens=16_384,
    )


# ── Agent factory ─────────────────────────────────────────────────────────────

def _make_agent(key: str, tools: list | None = None, llm=None) -> Agent:
    cfg = _load_yaml("agents.yaml")[key]
    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        verbose=settings.crewai_verbose,
        allow_delegation=cfg.get("allow_delegation", False),
        max_iter=cfg.get("max_iter", 5),
        tools=tools or [],
        llm=llm or _get_llm(),
    )


# ── Crew 1: Strategy ──────────────────────────────────────────────────────────

def build_strategy_crew(
    mix_id: str,
    genre: str,
    genre_bpm_range: tuple[int, int],
    requested_duration_minutes: int,
    used_combinations_json: str,
) -> Crew:
    """
    CSO crew: determines the unique BPM, subgenre, key, and arc for the mix.
    Genre is the only user input; the agent decides all other parameters.
    Includes QA self-reflection to verify the chosen combo is genuinely novel.
    """
    task_cfg = _load_yaml("tasks.yaml")["cso_strategy_task"]

    cso = _make_agent(
        "cso_agent",
        tools=[BpmRegistryTool(), YoutubeAnalyticsTool()],
        llm=_get_precise_llm(),
    )
    qa  = _make_agent("qa_agent", llm=_get_precise_llm())

    strategy_task = Task(
        description=task_cfg["description"].format(
            mix_id=mix_id,
            genre=genre,
            genre_bpm_min=genre_bpm_range[0],
            genre_bpm_max=genre_bpm_range[1],
            requested_duration_minutes=requested_duration_minutes,
            used_combinations=used_combinations_json,
        ),
        expected_output=task_cfg["expected_output"],
        agent=cso,
    )

    # Self-reflection: QA agent verifies the combo is novel and JSON is valid
    qa_task = Task(
        description=(
            f"Review the CSO's strategy output for mix_id='{mix_id}'. "
            "CRITICAL: Your ENTIRE response must be ONLY a raw JSON object — "
            "no explanation, no prose, no markdown fences. Start with {{ end with }}. "
            "Verify: (1) the JSON is valid, (2) the bpm/subgenre/key combination "
            f"does NOT appear in this used list: {used_combinations_json}, "
            "(3) stem_count equals floor(requested_duration_minutes * 2) capped at 150. "
            "If any check fails, output the corrected JSON. If all checks pass, "
            "output the original JSON unchanged — ONLY JSON, nothing else."
        ),
        expected_output=(
            "ONLY a raw JSON object with no surrounding text. "
            "No explanation, no confirmation, no markdown fences. "
            "Response must begin with { and end with }."
        ),
        agent=qa,
        context=[strategy_task],
    )

    return Crew(
        agents=[cso, qa],
        tasks=[strategy_task, qa_task],
        process=Process.sequential,
        verbose=settings.crewai_verbose,
    )


# ── Crew 2: Audio Prompts ─────────────────────────────────────────────────────

def build_audio_prompt_crew(
    mix_id: str,
    bpm: float,
    subgenre: str,
    key_signature: str,
    style_description: str,
    transition_arc: str,
    total_stems: int,
    position_start: int,
    position_end_inclusive: int,
    batch_num: int,
    total_batches: int,
) -> Crew:
    """
    Audio PE crew: generates one batch of audio prompts.

    Single-agent design (no QA): removing the QA agent cuts output token
    usage by ~50% (QA had to reproduce all N prompts to "validate" them).
    All validation (count, transition_type, intensity arc, word count) is
    handled in Python inside run_audio_prompt_engineer() — faster, cheaper,
    and more reliable than a second LLM call.
    """
    task_cfg   = _load_yaml("tasks.yaml")["audio_prompt_task"]
    batch_size = position_end_inclusive - position_start + 1

    arc_intro_end   = max(0, int(total_stems * 0.15) - 1)
    arc_peak_start  = int(total_stems * 0.45)
    arc_peak_end    = int(total_stems * 0.70)
    arc_outro_start = int(total_stems * 0.85)

    audio_pe = _make_agent("audio_prompt_engineer", llm=_get_llm())

    prompt_task = Task(
        description=task_cfg["description"].format(
            mix_id=mix_id,
            bpm=bpm,
            subgenre=subgenre,
            key_signature=key_signature,
            style_description=style_description,
            transition_arc=transition_arc,
            batch_num=batch_num,
            total_batches=total_batches,
            total_stems=total_stems,
            batch_size=batch_size,
            position_start=position_start,
            position_end_inclusive=position_end_inclusive,
            arc_intro_end=arc_intro_end,
            arc_peak_start=arc_peak_start,
            arc_peak_end=arc_peak_end,
            arc_outro_start=arc_outro_start,
            total_stems_minus_1=total_stems - 1,
            batch_size_minus_1=batch_size - 1,
            batch_size_plus_1=batch_size + 1,
        ),
        expected_output=(
            f"ONLY a raw JSON object — no prose, no markdown fences. "
            f"Must contain EXACTLY {batch_size} entries in 'prompts', "
            f"positions {position_start} to {position_end_inclusive}. "
            "Each entry: position (int), prompt_en (50-80 word English string with "
            "BPM, drum type, bassline, atmosphere, energy phrase), "
            "transition_type (intro|build|drop|breakdown|peak|outro), "
            "intensity (float 0.0–1.0 following the arc). "
            "Response begins with { and ends with }."
        ),
        agent=audio_pe,
    )

    return Crew(
        agents=[audio_pe],
        tasks=[prompt_task],
        process=Process.sequential,
        verbose=settings.crewai_verbose,
    )


# ── Crew 3: Visual Prompts ────────────────────────────────────────────────────

def build_visual_prompt_crew(
    mix_id: str,
    subgenre: str,
    bpm: float,
    style_description: str,
) -> Crew:
    task_cfg = _load_yaml("tasks.yaml")["visual_prompt_task"]

    visual_pe = _make_agent("visual_prompt_engineer", llm=_get_llm())

    visual_task = Task(
        description=task_cfg["description"].format(
            subgenre=subgenre,
            bpm=bpm,
            style_description=style_description,
        ),
        expected_output=task_cfg["expected_output"],
        agent=visual_pe,
    )

    return Crew(
        agents=[visual_pe],
        tasks=[visual_task],
        process=Process.sequential,
        verbose=settings.crewai_verbose,
    )


# ── Crew 4: Polish SEO ────────────────────────────────────────────────────────

def build_seo_crew(
    mix_id: str,
    bpm: float,
    subgenre: str,
    style_description: str,
    transition_arc: str,
    actual_duration_seconds: float,
    style_hint: str | None,
) -> Crew:
    """
    Polish SEO crew: generates ALL user-facing Polish metadata.
    QA agent verifies that the output is genuinely in Polish.
    """
    task_cfg  = _load_yaml("tasks.yaml")["polish_seo_task"]

    duration_minutes = int(actual_duration_seconds // 60)
    # Estimate chapters: 1 per 5 minutes, minimum 5
    chapter_count = max(5, duration_minutes // 5)

    seo = _make_agent("polish_seo_agent", llm=_get_llm())
    qa  = _make_agent("qa_agent", llm=_get_precise_llm())

    seo_task = Task(
        description=task_cfg["description"].format(
            subgenre=subgenre,
            bpm=bpm,
            duration_seconds=actual_duration_seconds,
            duration_minutes=duration_minutes,
            style_description=style_description,
            transition_arc=transition_arc,
            chapter_count=chapter_count,
        ),
        expected_output=task_cfg["expected_output"],
        agent=seo,
    )

    qa_task = Task(
        description=(
            "Review the Polish SEO metadata output. "
            "CRITICAL: Your ENTIRE response must be ONLY a raw JSON object — "
            "no explanation, no prose, no markdown fences. Start with {{ end with }}. "
            "Checks: "
            "(1) title_pl, description_pl, tags_pl, chapters_pl, shorts_titles_pl "
            "are ALL in Polish — reject any English sentences in these fields, "
            "(2) title_pl is ≤ 100 characters, "
            "(3) exactly 3 items in shorts_titles_pl, "
            "(4) chapters_pl contains at least 5 entries with valid time_str format 'MM:SS', "
            "(5) the JSON structure matches the required schema exactly. "
            "If valid: output original JSON unchanged. If corrected: output ONLY the corrected JSON."
        ),
        expected_output=(
            "ONLY a raw JSON object with no surrounding text. "
            "No explanation, no confirmation, no markdown fences. "
            "Response must begin with { and end with }."
        ),
        agent=qa,
        context=[seo_task],
    )

    return Crew(
        agents=[seo, qa],
        tasks=[seo_task, qa_task],
        process=Process.sequential,
        verbose=settings.crewai_verbose,
    )
