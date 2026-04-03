"""
shared/genres.py
─────────────────────────────────────────────────────────────────────────────
Music genre registry for Project Nebula.

Each genre carries:
  bpm_range       — (min, max) BPM — CSO picks the exact value within range
  duration_range  — (min, max) mix duration in minutes — picked by the
                    intelligent duration selector, never the same twice
  energy          — "low" | "medium" | "high" | "extreme"

Used by:
  1. pick_intelligent_duration()   — orchestrate_mix_pipeline task
  2. GET /genres                   — frontend dropdown
  3. run_cso_agent()               — hands genre profile to the CSO crew
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import random
from typing import TypedDict


class GenreProfile(TypedDict):
    bpm_range:      tuple[int, int]
    duration_range: tuple[int, int]   # minutes
    energy:         str


# ── 120-genre registry ────────────────────────────────────────────────────────

GENRES: dict[str, GenreProfile] = {

    # ── Drum and Bass / Jungle ─────────────────────────────────────────────
    "Drum and Bass":        {"bpm_range": (160, 180), "duration_range": (30, 90),  "energy": "high"},
    "Neurofunk":            {"bpm_range": (170, 180), "duration_range": (30, 75),  "energy": "high"},
    "Liquid DnB":           {"bpm_range": (160, 175), "duration_range": (30, 90),  "energy": "high"},
    "Jump Up":              {"bpm_range": (170, 180), "duration_range": (25, 60),  "energy": "high"},
    "Minimal DnB":          {"bpm_range": (160, 175), "duration_range": (30, 75),  "energy": "high"},
    "Techstep":             {"bpm_range": (165, 175), "duration_range": (30, 75),  "energy": "high"},
    "Darkstep":             {"bpm_range": (165, 180), "duration_range": (25, 60),  "energy": "high"},
    "Atmospheric DnB":      {"bpm_range": (160, 174), "duration_range": (30, 90),  "energy": "high"},
    "Rollers":              {"bpm_range": (170, 178), "duration_range": (30, 75),  "energy": "high"},
    "Jungle":               {"bpm_range": (160, 175), "duration_range": (30, 75),  "energy": "high"},
    "Raggajungle":          {"bpm_range": (160, 175), "duration_range": (25, 60),  "energy": "high"},

    # ── Techno ─────────────────────────────────────────────────────────────
    "Techno":               {"bpm_range": (130, 150), "duration_range": (45, 120), "energy": "high"},
    "Minimal Techno":       {"bpm_range": (128, 138), "duration_range": (60, 120), "energy": "medium"},
    "Acid Techno":          {"bpm_range": (130, 145), "duration_range": (45, 90),  "energy": "high"},
    "Industrial Techno":    {"bpm_range": (130, 150), "duration_range": (45, 90),  "energy": "high"},
    "Detroit Techno":       {"bpm_range": (130, 145), "duration_range": (45, 90),  "energy": "high"},
    "Berlin Techno":        {"bpm_range": (132, 148), "duration_range": (45, 120), "energy": "high"},
    "Hypnotic Techno":      {"bpm_range": (130, 140), "duration_range": (60, 120), "energy": "medium"},
    "Dub Techno":           {"bpm_range": (128, 138), "duration_range": (45, 90),  "energy": "medium"},
    "Melodic Techno":       {"bpm_range": (130, 140), "duration_range": (45, 90),  "energy": "medium"},
    "Tribal Techno":        {"bpm_range": (132, 142), "duration_range": (45, 90),  "energy": "high"},
    "Afro Techno":          {"bpm_range": (122, 135), "duration_range": (30, 75),  "energy": "high"},

    # ── House ──────────────────────────────────────────────────────────────
    "House":                {"bpm_range": (120, 130), "duration_range": (30, 90),  "energy": "medium"},
    "Deep House":           {"bpm_range": (118, 126), "duration_range": (30, 90),  "energy": "medium"},
    "Tech House":           {"bpm_range": (124, 132), "duration_range": (30, 90),  "energy": "medium"},
    "Progressive House":    {"bpm_range": (126, 132), "duration_range": (45, 90),  "energy": "medium"},
    "Acid House":           {"bpm_range": (120, 130), "duration_range": (30, 75),  "energy": "medium"},
    "Chicago House":        {"bpm_range": (120, 128), "duration_range": (30, 75),  "energy": "medium"},
    "Tropical House":       {"bpm_range": (116, 124), "duration_range": (20, 60),  "energy": "low"},
    "Organic House":        {"bpm_range": (120, 125), "duration_range": (30, 75),  "energy": "low"},
    "Afro House":           {"bpm_range": (122, 128), "duration_range": (30, 75),  "energy": "medium"},
    "Bass House":           {"bpm_range": (124, 132), "duration_range": (25, 60),  "energy": "high"},
    "Future House":         {"bpm_range": (126, 132), "duration_range": (25, 60),  "energy": "high"},
    "Electro House":        {"bpm_range": (128, 135), "duration_range": (25, 60),  "energy": "high"},
    "Tribal House":         {"bpm_range": (122, 130), "duration_range": (30, 75),  "energy": "medium"},

    # ── Trance ─────────────────────────────────────────────────────────────
    "Trance":               {"bpm_range": (135, 145), "duration_range": (45, 90),  "energy": "high"},
    "Progressive Trance":   {"bpm_range": (130, 140), "duration_range": (45, 90),  "energy": "medium"},
    "Psytrance":            {"bpm_range": (140, 148), "duration_range": (30, 75),  "energy": "high"},
    "Goa Trance":           {"bpm_range": (140, 148), "duration_range": (30, 75),  "energy": "high"},
    "Full On Psytrance":    {"bpm_range": (142, 150), "duration_range": (30, 75),  "energy": "high"},
    "Dark Psy":             {"bpm_range": (148, 162), "duration_range": (25, 60),  "energy": "extreme"},
    "Forest Psytrance":     {"bpm_range": (145, 155), "duration_range": (25, 60),  "energy": "high"},
    "Hi-Tech":              {"bpm_range": (160, 200), "duration_range": (20, 50),  "energy": "extreme"},
    "Ambient Trance":       {"bpm_range": (130, 140), "duration_range": (45, 90),  "energy": "low"},
    "Suomisaundi":          {"bpm_range": (144, 152), "duration_range": (25, 60),  "energy": "high"},

    # ── Hardstyle / Hardcore ────────────────────────────────────────────────
    "Hardstyle":            {"bpm_range": (150, 160), "duration_range": (30, 75),  "energy": "extreme"},
    "Euphoric Hardstyle":   {"bpm_range": (150, 158), "duration_range": (30, 75),  "energy": "extreme"},
    "Raw Hardstyle":        {"bpm_range": (150, 160), "duration_range": (25, 60),  "energy": "extreme"},
    "Hardcore":             {"bpm_range": (160, 200), "duration_range": (20, 60),  "energy": "extreme"},
    "Happy Hardcore":       {"bpm_range": (160, 185), "duration_range": (20, 60),  "energy": "extreme"},
    "UK Hardcore":          {"bpm_range": (160, 175), "duration_range": (20, 60),  "energy": "extreme"},
    "Gabber":               {"bpm_range": (180, 250), "duration_range": (15, 45),  "energy": "extreme"},
    "Frenchcore":           {"bpm_range": (175, 200), "duration_range": (15, 45),  "energy": "extreme"},
    "Speedcore":            {"bpm_range": (200, 300), "duration_range": (10, 30),  "energy": "extreme"},
    "Industrial Hardcore":  {"bpm_range": (165, 185), "duration_range": (20, 50),  "energy": "extreme"},

    # ── Dubstep / Bass ─────────────────────────────────────────────────────
    "Dubstep":              {"bpm_range": (138, 145), "duration_range": (25, 60),  "energy": "high"},
    "Brostep":              {"bpm_range": (138, 145), "duration_range": (25, 60),  "energy": "high"},
    "Riddim":               {"bpm_range": (140, 145), "duration_range": (20, 50),  "energy": "high"},
    "Melodic Dubstep":      {"bpm_range": (138, 148), "duration_range": (25, 60),  "energy": "medium"},
    "Future Bass":          {"bpm_range": (130, 150), "duration_range": (20, 50),  "energy": "high"},

    # ── Trap ───────────────────────────────────────────────────────────────
    "Trap":                 {"bpm_range": (130, 150), "duration_range": (20, 50),  "energy": "high"},
    "UK Trap":              {"bpm_range": (130, 145), "duration_range": (20, 45),  "energy": "high"},
    "Phonk":                {"bpm_range": (130, 145), "duration_range": (20, 45),  "energy": "high"},
    "Dark Trap":            {"bpm_range": (130, 148), "duration_range": (20, 45),  "energy": "high"},

    # ── Ambient / Downtempo ────────────────────────────────────────────────
    "Ambient":              {"bpm_range": (60, 90),   "duration_range": (30, 120), "energy": "low"},
    "Dark Ambient":         {"bpm_range": (55, 85),   "duration_range": (30, 120), "energy": "low"},
    "Drone":                {"bpm_range": (40, 70),   "duration_range": (20, 90),  "energy": "low"},
    "Space Ambient":        {"bpm_range": (60, 80),   "duration_range": (30, 120), "energy": "low"},
    "Binaural":             {"bpm_range": (60, 80),   "duration_range": (30, 90),  "energy": "low"},
    "Healing Frequencies":  {"bpm_range": (60, 80),   "duration_range": (30, 90),  "energy": "low"},
    "Psybient":             {"bpm_range": (75, 95),   "duration_range": (30, 90),  "energy": "low"},
    "Meditation":           {"bpm_range": (60, 80),   "duration_range": (20, 60),  "energy": "low"},
    "Downtempo":            {"bpm_range": (80, 100),  "duration_range": (30, 75),  "energy": "low"},
    "Chillout":             {"bpm_range": (85, 105),  "duration_range": (25, 75),  "energy": "low"},
    "Trip-Hop":             {"bpm_range": (70, 90),   "duration_range": (25, 60),  "energy": "low"},
    "Lo-Fi Hip Hop":        {"bpm_range": (75, 90),   "duration_range": (30, 90),  "energy": "low"},
    "Chillwave":            {"bpm_range": (80, 100),  "duration_range": (20, 60),  "energy": "low"},
    "Vaporwave":            {"bpm_range": (75, 90),   "duration_range": (20, 60),  "energy": "low"},

    # ── Synthwave / Retrowave ──────────────────────────────────────────────
    "Synthwave":            {"bpm_range": (95, 120),  "duration_range": (25, 75),  "energy": "medium"},
    "Outrun":               {"bpm_range": (95, 120),  "duration_range": (25, 60),  "energy": "medium"},
    "Retrowave":            {"bpm_range": (95, 120),  "duration_range": (25, 60),  "energy": "medium"},
    "Dreamwave":            {"bpm_range": (90, 115),  "duration_range": (25, 60),  "energy": "low"},

    # ── IDM / Experimental ─────────────────────────────────────────────────
    "IDM":                  {"bpm_range": (100, 160), "duration_range": (20, 60),  "energy": "medium"},
    "Glitch":               {"bpm_range": (90, 160),  "duration_range": (20, 45),  "energy": "medium"},
    "Experimental Electronic": {"bpm_range": (80, 160), "duration_range": (15, 60), "energy": "medium"},

    # ── Breaks / Electro ───────────────────────────────────────────────────
    "Breakbeat":            {"bpm_range": (128, 150), "duration_range": (25, 75),  "energy": "high"},
    "Nu-Skool Breaks":      {"bpm_range": (130, 150), "duration_range": (25, 60),  "energy": "high"},
    "Big Beat":             {"bpm_range": (120, 140), "duration_range": (25, 60),  "energy": "high"},
    "Electro":              {"bpm_range": (110, 130), "duration_range": (25, 60),  "energy": "medium"},
    "Electro-Funk":         {"bpm_range": (110, 130), "duration_range": (25, 60),  "energy": "medium"},
    "Miami Bass":           {"bpm_range": (120, 135), "duration_range": (20, 50),  "energy": "high"},
    "Footwork":             {"bpm_range": (155, 165), "duration_range": (20, 45),  "energy": "high"},
    "Juke":                 {"bpm_range": (155, 165), "duration_range": (20, 45),  "energy": "high"},

    # ── UK Bass / Garage / Grime ───────────────────────────────────────────
    "UK Garage":            {"bpm_range": (130, 140), "duration_range": (25, 60),  "energy": "high"},
    "2-Step":               {"bpm_range": (130, 140), "duration_range": (25, 60),  "energy": "high"},
    "Speed Garage":         {"bpm_range": (130, 142), "duration_range": (25, 60),  "energy": "high"},
    "Grime":                {"bpm_range": (140, 145), "duration_range": (20, 45),  "energy": "high"},
    "UK Bass":              {"bpm_range": (130, 140), "duration_range": (25, 60),  "energy": "high"},
    "Future Garage":        {"bpm_range": (130, 140), "duration_range": (25, 60),  "energy": "medium"},

    # ── Afrobeats / Latin / World ──────────────────────────────────────────
    "Afrobeats":            {"bpm_range": (95, 115),  "duration_range": (20, 50),  "energy": "medium"},
    "Afro Tech":            {"bpm_range": (122, 130), "duration_range": (25, 60),  "energy": "high"},
    "Kuduro":               {"bpm_range": (118, 128), "duration_range": (20, 45),  "energy": "high"},
    "Baile Funk":           {"bpm_range": (130, 145), "duration_range": (20, 45),  "energy": "high"},
    "Reggaeton":            {"bpm_range": (90, 100),  "duration_range": (20, 45),  "energy": "medium"},
    "Dancehall":            {"bpm_range": (70, 100),  "duration_range": (20, 45),  "energy": "medium"},
    "Cumbia":               {"bpm_range": (100, 120), "duration_range": (20, 45),  "energy": "medium"},
    "Latin House":          {"bpm_range": (120, 130), "duration_range": (25, 60),  "energy": "medium"},
    "Moombahton":           {"bpm_range": (108, 115), "duration_range": (20, 45),  "energy": "medium"},

    # ── Club / Jersey / Ballroom ───────────────────────────────────────────
    "Club":                 {"bpm_range": (124, 132), "duration_range": (25, 75),  "energy": "high"},
    "Jersey Club":          {"bpm_range": (120, 132), "duration_range": (20, 60),  "energy": "high"},
    "Ballroom / Vogue":     {"bpm_range": (130, 140), "duration_range": (20, 45),  "energy": "high"},
    "Hyperpop":             {"bpm_range": (130, 155), "duration_range": (15, 45),  "energy": "extreme"},

    # ── Disco / Nudisco ────────────────────────────────────────────────────
    "Nu-Disco":             {"bpm_range": (115, 125), "duration_range": (25, 75),  "energy": "medium"},
    "Italo Disco":          {"bpm_range": (118, 130), "duration_range": (25, 60),  "energy": "medium"},
    "Eurodance":            {"bpm_range": (126, 145), "duration_range": (20, 60),  "energy": "high"},
    "Eurobeat":             {"bpm_range": (145, 165), "duration_range": (20, 45),  "energy": "high"},

    # ── Industrial / EBM / Wave ────────────────────────────────────────────
    "Industrial":           {"bpm_range": (80, 130),  "duration_range": (25, 60),  "energy": "high"},
    "EBM":                  {"bpm_range": (120, 140), "duration_range": (25, 60),  "energy": "high"},
    "Aggrotech":            {"bpm_range": (130, 145), "duration_range": (20, 45),  "energy": "extreme"},
    "Dark Wave":            {"bpm_range": (80, 110),  "duration_range": (20, 60),  "energy": "low"},
    "Wave":                 {"bpm_range": (80, 110),  "duration_range": (20, 60),  "energy": "low"},
    "Post-Industrial":      {"bpm_range": (90, 130),  "duration_range": (25, 60),  "energy": "medium"},

    # ── Jazz / Soul / R&B (electronic fusion) ─────────────────────────────
    "Jazz Fusion Electronic": {"bpm_range": (90, 130), "duration_range": (25, 60), "energy": "medium"},
    "Nu-Jazz":              {"bpm_range": (85, 115),  "duration_range": (25, 60),  "energy": "low"},
    "Future Soul":          {"bpm_range": (80, 100),  "duration_range": (25, 60),  "energy": "low"},
    "R&B Electronic":       {"bpm_range": (85, 100),  "duration_range": (20, 50),  "energy": "low"},
    "Soul Electronic":      {"bpm_range": (85, 100),  "duration_range": (20, 50),  "energy": "low"},
    "Nu-Metal Electronic":  {"bpm_range": (100, 140), "duration_range": (25, 45),  "energy": "high"},
}

# Sorted genre names for the frontend dropdown
GENRE_NAMES: list[str] = sorted(GENRES.keys())


# ── Intelligent duration picker ───────────────────────────────────────────────

def pick_intelligent_duration(genre: str, recent_durations: list[int]) -> int:
    """
    Pick a mix duration with full freedom.

    Rules:
      - Global range: 7–75 minutes (1-minute steps = 69 possible values)
      - Genre provides a *preferred zone* (soft weight ×2), not a hard cap
      - Durations used in the last 15 mixes get weight 0.01 (near-zero chance)
      - Everything else is uniformly random — no centrality, no clustering

    recent_durations: requested_duration_minutes from the most recent N mixes
                      (any status, most recent first)
    """
    # Full candidate space — 7 to 75 minutes, every minute
    candidates = list(range(7, 76))

    profile = GENRES.get(genre)
    preferred_min = profile["duration_range"][0] if profile else 20
    preferred_max = profile["duration_range"][1] if profile else 60

    # Durations used in the last 15 mixes are strongly penalised
    recent_set = set(recent_durations[:15])

    weights: list[float] = []
    for d in candidates:
        if d in recent_set:
            w = 0.01            # near-zero: almost never repeat a recent duration
        elif preferred_min <= d <= preferred_max:
            w = 2.0             # genre sweet-spot: twice as likely, but not mandatory
        else:
            w = 1.0             # outside sweet-spot: fully valid, equally weighted
        weights.append(w)

    return random.choices(candidates, weights=weights, k=1)[0]
