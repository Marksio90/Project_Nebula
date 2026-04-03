"""
services/orchestrator/crew/crew.py
─────────────────────────────────────────────────────────────────────────────
CrewAI Crew factories for Project Nebula.

Three purpose-built crews, each with self-reflection capabilities:

  build_strategy_crew(...)   → CSO determines BPM/subgenre/arc
  build_audio_prompt_crew(…) → Audio PE generates 90+ Lyria 3 prompts
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
    return LLM(
        model="gpt-4o",
        api_key=settings.openai_api_key,
        temperature=0.7,
        max_tokens=4096,
    )


def _get_precise_llm() -> LLM:
    """Lower temperature for structured JSON outputs."""
    return LLM(
        model="gpt-4o",
        api_key=settings.openai_api_key,
        temperature=0.2,
        max_tokens=8192,
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
            "Verify: (1) the JSON is valid, (2) the bpm/subgenre/key combination "
            f"does NOT appear in this used list: {used_combinations_json}, "
            "(3) stem_count equals floor(requested_duration_minutes * 2) capped at 96. "
            "If any check fails, provide the corrected JSON. If all checks pass, "
            "output the original JSON unchanged."
        ),
        expected_output=(
            "The final, validated JSON strategy object — identical to input if valid, "
            "or corrected if any issue was found."
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
    stem_count: int,
) -> Crew:
    """
    Audio PE crew: generates stem_count ordered Lyria 3 prompts in English.
    QA agent reviews for prompt quality, arc coherence, and intensity curve.
    """
    task_cfg = _load_yaml("tasks.yaml")["audio_prompt_task"]

    audio_pe = _make_agent("audio_prompt_engineer", llm=_get_llm())
    qa       = _make_agent("qa_agent", llm=_get_precise_llm())

    prompt_task = Task(
        description=task_cfg["description"].format(
            bpm=bpm,
            subgenre=subgenre,
            key_signature=key_signature,
            style_description=style_description,
            transition_arc=transition_arc,
            stem_count=stem_count,
        ),
        expected_output=task_cfg["expected_output"],
        agent=audio_pe,
    )

    qa_task = Task(
        description=(
            f"Review the {stem_count} audio prompts generated for mix_id='{mix_id}'. "
            "Check: (1) exactly {stem_count} prompts exist, (2) intensity values form "
            "a plausible arc (not flat at 1.0 throughout), (3) each prompt is 40-80 words "
            "and includes BPM, drum type, bass type, and atmosphere descriptors, "
            "(4) transition_type values are valid enum values. "
            "Return the corrected JSON if issues found, or original JSON if valid."
        ).format(stem_count=stem_count),
        expected_output="The final, validated JSON AudioPromptBatch.",
        agent=qa,
        context=[prompt_task],
    )

    return Crew(
        agents=[audio_pe, qa],
        tasks=[prompt_task, qa_task],
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
            "Review the Polish SEO metadata output. CRITICAL CHECKS: "
            "(1) title_pl, description_pl, tags_pl, chapters_pl, shorts_titles_pl "
            "are ALL in Polish — reject any English sentences in these fields, "
            "(2) title_pl is ≤ 100 characters, "
            "(3) exactly 3 items in shorts_titles_pl, "
            "(4) chapters_pl contains at least 5 entries with valid time_str format 'MM:SS', "
            "(5) the JSON structure matches the required schema exactly. "
            "Output the corrected JSON or the original if valid."
        ),
        expected_output="The final, validated JSON PolishSEOMetadata object.",
        agent=qa,
        context=[seo_task],
    )

    return Crew(
        agents=[seo, qa],
        tasks=[seo_task, qa_task],
        process=Process.sequential,
        verbose=settings.crewai_verbose,
    )
