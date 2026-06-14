# director.py — Gemma-powered scene breakdown + prompt writing via Ollama

import json
import logging
import ollama_client

from config import (
    OLLAMA_MODEL_CREATIVE, OUTPUT_LANGUAGE,
    SCENE_MIN_SEC, SCENE_MAX_SEC, SCENE_SWEET_SPOT_SEC,
)

log = logging.getLogger(__name__)


def _language_requirement() -> str:
    """Return the configured creative output language requirement."""
    if OUTPUT_LANGUAGE == "simplified_chinese":
        return (
            "FINAL FILM LANGUAGE REQUIREMENT: The final film's spoken dialogue and narration "
            "must be natural Simplified Chinese. The user's brief or script may be written in "
            "any language, but the words characters speak in the final video must be Simplified "
            "Chinese. If source dialogue is in another language, translate or adapt it into "
            "natural Simplified Chinese while preserving the meaning and character intent. "
            "Keep all JSON keys exactly as specified in English. Creative values such as "
            "character descriptions, voice descriptions, scene descriptions, dialogue, action, "
            "audio, settings, lighting, mood, and continuity notes should be written in natural "
            "Simplified Chinese. This overrides any instruction to preserve source-language "
            "dialogue exactly."
        )
    if OUTPUT_LANGUAGE == "english":
        return (
            "FINAL FILM LANGUAGE REQUIREMENT: The final film's spoken dialogue and narration "
            "must be natural English. The user's brief or script may be written in any language, "
            "but the words characters speak in the final video must be English. If source "
            "dialogue is in another language, translate or adapt it into natural spoken English "
            "while preserving the meaning and character intent. Keep all JSON keys exactly as "
            "specified in English. Creative values such as character descriptions, voice "
            "descriptions, scene descriptions, dialogue, action, audio, settings, lighting, "
            "mood, and continuity notes should be written in English. This overrides any "
            "instruction to preserve source-language dialogue exactly."
        )
    return ""


def _language_messages() -> list[dict]:
    requirement = _language_requirement()
    if not requirement:
        return []
    return [{"role": "system", "content": requirement}]


def _dialogue_instruction() -> str:
    if OUTPUT_LANGUAGE == "simplified_chinese":
        return (
            "make the spoken words natural Simplified Chinese. If the source dialogue below "
            "is not Chinese, translate or adapt it into Chinese; do not include the "
            "source-language line in the final prompt"
        )
    if OUTPUT_LANGUAGE == "english":
        return (
            "make the spoken words natural English. If the source dialogue below is not "
            "English, translate or adapt it into English; do not include the source-language "
            "line in the final prompt"
        )
    return "include these EXACT words in quotes — do NOT summarize or shorten"

# ── System prompts ─────────────────────────────────────────────────────────

BREAKDOWN_SYSTEM = f"""You are a film director AI planning scenes for an AI video generator that creates SYNCHRONIZED AUDIO AND VIDEO together.

FIRST, before listing any scenes, output a "characters" object that defines EVERY character in the video. Each character needs a THOROUGH physical description that will be copy-pasted into every video prompt. The video model has NO memory — it needs the full description every time.

For EACH character, describe:
- Full name or identifier
- Age range, ethnicity, build, height impression
- Face: shape, skin tone, eyes, nose, mouth, facial hair, distinguishing marks
- Hair: color, style, length, texture
- Clothing: EXACT outfit — colors, materials, fit, accessories
- Mannerisms: posture, typical gestures, how they hold themselves
- If it's a well-known public figure: describe their REAL, RECOGNIZABLE appearance accurately

You MUST also provide a "voices" object that defines HOW each character SOUNDS. The video model generates speech audio from text, and it needs voice anchors to produce consistent, recognizable voices. For each character describe:
- Pitch: high, medium, low, deep, bass
- Timbre: gravelly, smooth, nasal, breathy, raspy, clear, warm, sharp, booming
- Accent/dialect: specific regional accent, speech pattern, pronunciation quirks
- Speaking style: cadence, rhythm, pace, emphatic patterns, verbal tics
- Delivery: confident, hesitant, theatrical, deadpan, aggressive, gentle, sarcastic
- If it's a well-known public figure: describe their REAL, RECOGNIZABLE voice accurately

You MUST also provide a "style" string that captures the overall visual style and aesthetic of the production. This is a STYLE LOCK — a short (20-40 word) description of the rendering style, color palette, visual tone, and artistic direction that will be injected into EVERY scene prompt for consistency. Examples:
- "3D Pixar-style animation with soft, rounded character models, vibrant saturated colors, warm lighting, and expressive cartoon faces"
- "Gritty handheld documentary style, desaturated colors, natural lighting, 16mm film grain, shallow depth of field"
- "Noir-influenced cinematography with high contrast black and white, dramatic shadows, low-key lighting, and Dutch angles"
Derive this from the user's brief. If they mention a style (e.g. "Pixar", "anime", "documentary"), describe that look. If no style is mentioned, default to "Photorealistic cinematic style, natural lighting, shallow depth of field, film-quality color grading."

Output format:
{{
  "characters": {{
    "character_id": "Full detailed physical description here, 50-80 words minimum..."
  }},
  "voices": {{
    "character_id": "Full voice description here — pitch, timbre, accent, cadence, delivery style, 30-50 words..."
  }},
  "style": "Visual style lock description here — rendering style, color palette, lighting approach, artistic direction, 20-40 words...",
  "scenes": [ ... ]
}}

Each scene needs:
- scene_number
- description (visual action + what is heard)
- characters_in_scene (list of character_ids from the characters object above)
- dialogue (THE ACTUAL WORDS TO BE SPOKEN in quotes — see DIALOGUE RULES below)
- action_description (non-dialogue visual action: "walks to desk", "picks up coffee", "gestures broadly")
- action_seconds (estimated time for physical actions NOT covered by dialogue, e.g. 2 for a quick gesture, 5 for walking across room)
- shot_type (wide, medium, close-up, POV, tracking, over-the-shoulder, etc.)
- mood
- audio_description (sound effects, ambient sounds — NO background music)
- setting_description (FULL description of the environment: room type, walls, floor, background objects, props, furniture — be thorough, this gets reused verbatim)
- lighting_description (primary light source, color temperature, shadows, atmosphere)
- continuity_notes (what must match prev scene)

The system will AUTO-CALCULATE duration from your dialogue + action:
- Dialogue: counted at character-appropriate words-per-minute (fast talker ~170wpm, normal ~140wpm, slow/dramatic ~110wpm)
- Action: your action_seconds added on top of dialogue time
- You do NOT set duration_seconds — it is calculated for you

DIALOGUE RULES — THIS IS CRITICAL:
- You MUST write the ACTUAL WORDS the character says, not a summary or description
- WRONG: "Trump talks about how dishwashers don't work anymore"
- WRONG: "He complains about modern appliances in his signature style"
- RIGHT: "Let me tell you about dishwashers. They used to be fantastic, the best. You press the button, beautiful clean dishes. Now? You gotta run it three times. Three times, folks! It's a disgrace."
- Write dialogue that sounds like the character — their vocabulary, rhythm, verbal tics, catchphrases
- Each scene should have real sentences the character speaks, written out word for word
- If a scene has no dialogue (pure visual/action), set dialogue to ""

AUDIO CAPABILITIES — the video generator creates real audio from your text:
- Dialogue: the model will generate speech from your exact quoted words
- Sound effects: footsteps, door creaks, glass breaking, engine rumble, typing
- Ambient: rain, wind, crowd murmur, birds, traffic, room tone
- Voice tone: whispering, shouting, calm narration, excited
- Do NOT request background music or soundtrack — it cannot generate that cleanly

IMPORTANT: If the user requests a specific total duration (e.g. "5 minute video"), you MUST plan enough scenes so the content fills that time. For a 5-minute video, write substantial dialogue and actions across ~20+ scenes. Do NOT undershoot — write more scenes with real dialogue rather than fewer long silent ones.

There is NO limit on how many scenes you can create. Be as cinematic as you want:
- Quick 2-3 second cutaway shots for reactions, inserts, or visual emphasis
- 5-10 second beats to let moments breathe or establish atmosphere
- 15-20 second dialogue or action scenes
- Up to {SCENE_MAX_SEC} second long takes for extended sequences
Mix it up like a real film editor would. Vary shot length, camera angles, and pacing to keep the audience engaged.

MULTI-CHARACTER DIALOGUE — use cinematic shot/reverse-shot coverage:
- When two or more characters are talking, cut between close-ups of each speaker
- Show the LISTENER's reaction in close-up while the other character speaks (reaction shots)
- Use over-the-shoulder shots to establish spatial relationships
- Wide/medium establishing shots to remind the viewer where everyone is
- Each character speaking gets their own scene with only THEIR dialogue line
- Don't cram multiple characters' dialogue into one scene — split them into individual shots
Respond ONLY with a valid JSON array. No preamble. No markdown fences."""

PROMPT_WRITER_SYSTEM = """You write prompts for LTX Video 2.3, an AI that generates VIDEO WITH SYNCHRONIZED AUDIO from a single text prompt.

EACH PROMPT IS COMPLETELY INDEPENDENT. The video model has ZERO memory between scenes. It has never seen any other prompt. Every single prompt must rebuild the ENTIRE WORLD from scratch — the character, the setting, the lighting, the props, everything — as if you are describing a photograph to a blind person who has never seen anything before.

WORLD RECONSTRUCTION — EVERY PROMPT MUST CONTAIN ALL OF THIS:

1. CHARACTER (full physical description every time):
   - Full name, age range, ethnicity, build, height impression
   - Face: shape, skin tone/texture, eyes, nose, mouth, facial hair, expression
   - Hair: color, style, length, texture
   - Clothing: EXACT outfit description — suit color, tie color/pattern, shirt, shoes, accessories
   - Body language: posture, hand position, gestures, mannerisms
   - NEVER write "he remains" or "Trump continues" or "the same man" — FULLY describe them again
   - If the character was described to you, copy that description VERBATIM into every prompt

2. SETTING (full environment description every time):
   - Location type: what room/space, indoors/outdoors
   - Background: what's behind the character — walls, objects, decorations, depth
   - Props: microphone, desk, chair, glass of water — everything visible
   - Floor/surface: stage, carpet, concrete, grass
   - NEVER write "same stage" or "the comedy club" without fully describing it again

3. LIGHTING (every time):
   - Primary light source: direction, color temperature, intensity
   - Secondary/fill lights, shadows, contrast
   - Atmosphere: haze, smoke, dust particles, clean air
   - Time of day if applicable

4. CAMERA (every time):
   - Shot type: wide, medium, close-up, extreme close-up
   - Angle: eye level, low angle, high angle, Dutch angle
   - Movement: static, slow dolly, tracking, handheld
   - Lens impression: telephoto compression, wide-angle distortion

5. DIALOGUE (word for word):
   - Include ALL provided dialogue lines EXACTLY as written, in quotes
   - Embed naturally in action: 'The man leans forward and says "..."'
   - Never summarize, shorten, or paraphrase

6. VOICE/SPEECH (when characters speak):
   - Describe HOW the character speaks, not just WHAT they say
   - Include voice quality: pitch, timbre, accent, cadence, delivery
   - Example: 'says in a deep, booming voice with a New York accent, emphatic and staccato'
   - If voice anchors are provided, weave them naturally into the dialogue description
   - This is critical — the model generates speech audio and needs voice cues for recognizable delivery

7. AUDIO/SOUND:
   - Ambient sounds: room tone, crowd murmur, HVAC hum, outdoor noise
   - Sound effects tied to actions: footsteps, mic feedback, glass clink
   - Do NOT request background music or soundtrack

8. VISUAL STYLE (when a style anchor is provided):
   - Weave the style description naturally into the scene
   - Apply it to characters, environment, lighting, and color palette
   - Example: if style is "Pixar animation", describe rounded 3D character models, vibrant colors, etc.
   - The style anchor must be reflected consistently in every prompt — it is a STYLE LOCK

FORMAT:
- Present tense, one continuous moment
- No quality tags ("masterpiece", "8k", "cinematic")
- 300-500 words — be EXHAUSTIVELY descriptive
- Respond with ONLY the prompt text"""


DIRECTOR_REWRITE_SYSTEM = """You are a senior film director and editor doing a story-tightening pass BEFORE storyboard generation.

Your job is to make the scene plan more cinematic, more coherent, and more efficient without changing the user's core idea.

You are optimizing for:
- story momentum: every shot must reveal, escalate, or resolve something
- compactness: remove repeated beats and redundant establishing shots
- continuity: character identity, wardrobe, location, lighting, and props stay consistent
- shot variety: avoid repeating the same camera distance or angle unless intentionally rhythmic
- production realism: each scene is one clear filmable moment for an AI video model

Rules:
- Keep all JSON keys in English.
- Preserve the character IDs already used in characters_in_scene.
- Preserve the same schema for every scene.
- Do not add impossible multi-action scenes. One scene = one continuous shot.
- If the source is a script, preserve the meaning of every important dialogue beat. You may split, translate, or tighten dialogue only when required by the final film language instruction.
- If a target duration exists, keep the revised total close to that duration.

Respond ONLY with valid JSON in this exact shape:
{
  "director_notes": {
    "story_grade": "tight|okay|loose",
    "logline": "one sentence summary of the tightened film",
    "problems_fixed": ["specific issue fixed", "..."],
    "pacing_notes": "how the revised scene order improves momentum",
    "continuity_strategy": "what must stay consistent across scenes"
  },
  "scenes": [
    {
      "scene_number": 1,
      "description": "...",
      "characters_in_scene": ["..."],
      "dialogue": "...",
      "action_description": "...",
      "action_seconds": 2,
      "shot_type": "...",
      "mood": "...",
      "audio_description": "...",
      "setting_description": "...",
      "lighting_description": "...",
      "continuity_notes": "..."
    }
  ]
}"""


# ── Words-Per-Minute Profiles ──────────────────────────────────────────────

WPM_PROFILES = {
    "fast":    170,   # Fast talkers, auctioneers, excited speech
    "normal":  140,   # Average conversational pace
    "slow":    110,   # Dramatic, deliberate, elderly
    "default": 140,
}

# Known character speaking rates (add more as needed)
CHARACTER_WPM = {
    "trump":     170,  # Rapid-fire, repetitive, emphatic
    "obama":     130,  # Measured, deliberate pauses
    "morgan freeman": 110,  # Slow, deep narration
    "auctioneer": 200,
}

# Known voice anchors for recognizable public figures
# These are injected as fallbacks if the LLM doesn't generate good voice descriptions
VOICE_ANCHORS = {
    "trump": "Deep, booming male voice with a distinctive New York accent. Emphatic, staccato cadence with dramatic pauses for effect. Repetitive phrasing, superlatives ('tremendous', 'the best', 'nobody'), trailing off mid-sentence then circling back. Confident, bombastic delivery.",
    "obama": "Smooth, measured baritone with a calm, professorial delivery. Slight pause between phrases, deliberate pacing. Occasional drawn-out vowels. Warm but authoritative tone, rising inflection when making a point. Clear Midwestern-meets-coastal accent.",
    "hillary clinton": "Clear, composed alto voice with a controlled Midwestern accent. Precise articulation, measured cadence. Firm and polished delivery with occasional warmth. Steady rhythm, diplomatic tone, slight rise at the end of declarative statements.",
    "morgan freeman": "Deep, resonant bass voice with a warm Southern drawl. Slow, deliberate pacing with gravity in every word. Rich, velvety timbre. Calm, wise delivery that makes everything sound like profound narration.",
    "biden": "Gravelly, warm baritone with a distinctive Scranton, Pennsylvania working-class accent. Folksy cadence, whispered asides, sudden emphatic outbursts. Frequent use of 'look' and 'here's the deal'. Earnest, sometimes halting delivery.",
}

def _estimate_wpm(brief: str) -> int:
    """Guess speaking rate from the brief content."""
    brief_lower = brief.lower()
    for name, wpm in CHARACTER_WPM.items():
        if name in brief_lower:
            return wpm
    return WPM_PROFILES["default"]


def calc_scene_duration(scene: dict, wpm: int) -> int:
    """Calculate scene duration from dialogue word count + action time.

    Returns duration in seconds, clamped to SCENE_MIN/MAX.
    """
    # Count dialogue words
    dialogue = scene.get("dialogue", "")
    # Strip quote marks and stage directions in parentheses
    import re
    clean_dialogue = re.sub(r'\([^)]*\)', '', dialogue)
    clean_dialogue = clean_dialogue.replace('"', '').replace("'", "")
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', clean_dialogue))
    word_count = len(clean_dialogue.split()) if clean_dialogue.strip() else 0

    # Dialogue time
    if chinese_chars > 0:
        dialogue_seconds = chinese_chars / 4
    else:
        dialogue_seconds = (word_count / wpm) * 60 if word_count > 0 else 0

    # Action time (non-dialogue physical actions)
    action_seconds = scene.get("action_seconds", 2)

    # Total: dialogue + action, with a minimum beat of 1s between them
    if word_count > 0 and action_seconds > 0:
        total = dialogue_seconds + 1 + action_seconds
    elif word_count > 0:
        total = dialogue_seconds + 2  # Breathing room
    else:
        total = action_seconds + 2  # Pure action scene

    duration = int(round(total))
    duration = max(SCENE_MIN_SEC, min(SCENE_MAX_SEC, duration))
    return duration


# ── Script Detection & Parsing ─────────────────────────────────────────────

SCRIPT_PARSE_SYSTEM = """You are a script-to-scene converter for an AI video generator that creates SYNCHRONIZED AUDIO AND VIDEO.

You will receive a screenplay, script, or structured text. Your job is to break it into individual scenes suitable for AI video generation. Each scene should be one continuous shot (5-20 seconds).

FIRST output a "characters" object describing EVERY character's full physical appearance. This is critical because the video model generates each scene independently with NO memory. Each character needs: name, age, ethnicity, build, face details, hair, EXACT clothing, accessories, mannerisms. For well-known public figures describe their real recognizable appearance accurately. 50-80 words minimum per character.

ALSO output a "voices" object describing HOW each character SOUNDS. The video model generates speech audio and needs voice anchors for consistent, recognizable voices. For each character describe: pitch (high/medium/low/deep), timbre (gravelly/smooth/nasal/breathy/raspy/warm/booming), accent/dialect, speaking style and cadence, verbal tics, delivery style. For well-known public figures describe their real recognizable voice accurately. 30-50 words per character.

ALSO output a "style" string — a 20-40 word visual style lock describing the rendering style, color palette, lighting approach, and artistic direction. Derive from the script's genre/tone. This gets injected into every scene prompt for visual consistency.

Output format:
{"characters": {"character_id": "full physical description..."}, "voices": {"character_id": "full voice description..."}, "style": "visual style description...", "scenes": [...]}

For each scene output:
- scene_number
- characters_in_scene (list of character_ids)
- description (visual action — what the camera sees)
- dialogue (the EXACT lines spoken, word for word from the script, in quotes. If no dialogue, use "")
- action_description (physical actions: "walks to desk", "picks up phone")
- action_seconds (time for physical actions not covered by dialogue)
- shot_type (wide, medium, close-up, POV, tracking, over-the-shoulder — choose what fits the moment)
- mood (emotional tone of the moment)
- audio_description (sound effects, ambient sounds — NO background music)
- setting_description (FULL description of the environment: room, walls, floor, props, furniture, background)
- lighting_description (light sources, color temperature, shadows, atmosphere)
- continuity_notes (what must match prev scene)

RULES:
- Preserve ALL dialogue from the script — do NOT summarize, skip, or paraphrase any lines
- Split long monologues across multiple scenes with different camera angles
- One continuous action per scene — no time jumps within a scene
- If the script has stage directions, convert them to description + action_description
- If the script specifies camera angles or shots, use them
- setting_description and lighting_description must be THOROUGH — they get copied into every video prompt

Respond ONLY with valid JSON. No preamble. No markdown fences."""


def _is_script(text: str) -> bool:
    """Detect if the input looks like a screenplay/script rather than a brief."""
    import re
    lines = text.strip().split("\n")
    if len(lines) < 5:
        return False

    script_indicators = 0

    # Scene headings: INT. / EXT. / INTERIOR / EXTERIOR
    if re.search(r'(?:INT\.|EXT\.|INTERIOR|EXTERIOR)\s', text, re.IGNORECASE):
        script_indicators += 3

    # Character names in ALL CAPS followed by dialogue (standard screenplay format)
    # Pattern: line that's all caps, followed by a line of dialogue
    caps_lines = sum(1 for line in lines if line.strip().isupper() and 2 <= len(line.strip().split()) <= 5)
    if caps_lines >= 2:
        script_indicators += 2

    # Parenthetical stage directions: (beat), (smiling), (to camera)
    parens = len(re.findall(r'\([^)]{2,30}\)', text))
    if parens >= 2:
        script_indicators += 1

    # Dialogue patterns: lines in quotes with character attribution
    quoted_lines = len(re.findall(r'"[^"]{10,}"', text))
    if quoted_lines >= 3:
        script_indicators += 1

    # FADE IN, CUT TO, FADE OUT
    if re.search(r'(?:FADE IN|CUT TO|FADE OUT|DISSOLVE TO|SMASH CUT)', text, re.IGNORECASE):
        script_indicators += 2

    # Scene/Act markers
    if re.search(r'(?:SCENE \d|ACT \d|Scene:)', text, re.IGNORECASE):
        script_indicators += 2

    return script_indicators >= 3


def _chat_with_auto_tokens(model: str, messages: list, base_options: dict,
                           start_tokens: int = 8192, max_tokens: int = 32768) -> str:
    """Call ollama.chat with automatic token scaling on truncation.

    Detects truncated JSON output (unbalanced brackets) and retries with
    doubled token limit until it fits or hits max_tokens.
    """
    num_predict = start_tokens
    while num_predict <= max_tokens:
        opts = {**base_options, "num_predict": num_predict, "num_ctx": max(num_predict, 32768)}
        log.info("  LLM call: %s, num_predict=%d", model, num_predict)

        response = ollama_client.chat(model=model, messages=messages, options=opts)
        raw = response["message"]["content"].strip()

        if not raw:
            log.warning("  Empty response, retrying with more tokens...")
            num_predict *= 2
            continue

        # Check if output looks truncated (unbalanced brackets)
        open_braces = raw.count('{') - raw.count('}')
        open_brackets = raw.count('[') - raw.count(']')

        if open_braces > 0 or open_brackets > 0:
            log.warning("  Output truncated (unbalanced: %d braces, %d brackets, %d chars). "
                        "Retrying with %d tokens...",
                        open_braces, open_brackets, len(raw), num_predict * 2)
            num_predict *= 2
            continue

        return raw

    # Hit max — return what we have, let the bracket fixer in _fix_json handle it
    log.warning("  Hit max tokens (%d), returning potentially truncated output", max_tokens)
    return raw


def parse_script(script_text: str) -> list[dict]:
    """Parse a screenplay/script into scenes using Gemma."""
    log.info("Detected script format — parsing into scenes...")
    wpm = _estimate_wpm(script_text)
    log.info("Estimated speaking rate: %d WPM", wpm)

    raw = _chat_with_auto_tokens(
        model=OLLAMA_MODEL_CREATIVE,
        messages=[
            {"role": "system", "content": SCRIPT_PARSE_SYSTEM},
            *_language_messages(),
            {"role": "user", "content": f"Parse this script into scenes:\n\n{script_text}"},
        ],
        base_options={"temperature": 0.4},
    )
    scenes = _parse_json(raw, retries=2, brief=script_text)

    scenes = _apply_scene_timing(scenes, wpm, "")
    scenes = _director_rewrite_pass(script_text, scenes, wpm, None, "script")

    total_dur = sum(s["duration_seconds"] for s in scenes)
    log.info("Script parsed: %d scenes, total ~%ds (%.1f min)", len(scenes), total_dur, total_dur / 60)
    return scenes


# ── Scene Breakdown ────────────────────────────────────────────────────────

def _extract_target_duration(brief: str) -> int | None:
    """Extract target duration in seconds from user brief, if specified."""
    import re
    # Match patterns like "5 minute", "10 min", "3-minute", "120 seconds", "2 hour"
    m = re.search(r'(\d+)\s*[-]?\s*(minute|min|second|sec|hour|hr)s?', brief, re.IGNORECASE)
    if not m:
        m = re.search(r'(\d+)\s*(秒|分钟|分|小时)', brief)
    if not m:
        return None
    val = int(m.group(1))
    unit = m.group(2).lower()
    if unit in ("hour", "hr"):
        return val * 3600
    elif unit in ("minute", "min"):
        return val * 60
    elif unit == "小时":
        return val * 3600
    elif unit in ("分钟", "分"):
        return val * 60
    else:
        return val


def breakdown(brief: str, force_script: bool = False) -> list[dict]:
    """Take a user brief or script and return a list of scene dicts.

    Auto-detects if the input is a screenplay/script and parses accordingly.
    Set force_script=True to skip detection and treat input as a script.
    """
    # Auto-detect script vs brief
    if force_script or _is_script(brief):
        return parse_script(brief)

    log.info("Breaking down brief into scenes...")
    wpm = _estimate_wpm(brief)
    log.info("Estimated speaking rate: %d WPM", wpm)
    target_dur = _extract_target_duration(brief)
    if target_dur:
        log.info("Target duration from brief: %ds (%.1f min)", target_dur, target_dur / 60)

    # Phase 1: Deep planning pass — let the model think hard
    log.info("Phase 1: Deep planning (high token output, thinking mode)...")
    planning_prompt = f"""USER BRIEF: {brief}

{"TARGET DURATION: " + str(target_dur) + " seconds (" + f"{target_dur/60:.1f}" + " minutes). You MUST hit this target. Count your scenes and their dialogue carefully." if target_dur else ""}

Think step by step:
1. What is the overall narrative arc? Beginning, middle, end.
2. How many scenes do you need? Calculate: for each scene with dialogue, count the words and estimate seconds at {wpm} WPM. For action-only scenes, estimate action time. Sum all scene durations — does it hit the target?
3. For each scene, write the FULL dialogue word-for-word in the character's voice.
4. Plan camera angles for visual variety — don't repeat the same shot type consecutively.
5. Write continuity anchors (wardrobe, set, lighting, props) so every scene is visually consistent.
6. Double-check: add up all your estimated scene durations. If you're under the target, ADD MORE SCENES with more dialogue until you hit it.

After your thinking, output ONLY the final JSON array of scenes."""

    raw = _chat_with_auto_tokens(
        model=OLLAMA_MODEL_CREATIVE,
        messages=[
            {"role": "system", "content": BREAKDOWN_SYSTEM},
            *_language_messages(),
            {"role": "user", "content": planning_prompt},
        ],
        base_options={"temperature": 0.7},
    )
    scenes = _parse_json(raw, retries=2, brief=brief)

    # Phase 2: Calculate duration from dialogue + action
    scenes = _apply_scene_timing(scenes, wpm, "")

    total_dur_planned = sum(s["duration_seconds"] for s in scenes)
    log.info("Planned %d scenes, total ~%ds (%.1f min)", len(scenes), total_dur_planned, total_dur_planned / 60)

    # Phase 3: If undershooting target, have the model REWRITE the entire plan (not append)
    max_rewrites = 2
    for rewrite_attempt in range(max_rewrites):
        if not target_dur or total_dur_planned >= target_dur * 0.9:
            break

        shortfall = target_dur - total_dur_planned
        log.warning("Undershoot! Planned %ds but need %ds (short by %ds). Rewrite %d/%d...",
                     total_dur_planned, target_dur, shortfall,
                     rewrite_attempt + 1, max_rewrites)

        # Build a per-scene breakdown so the model sees exactly what it wrote and what each timed to
        scene_lines = []
        for s in scenes:
            dlg = s.get("dialogue", "")
            wc = len(dlg.replace('"', '').split()) if dlg else 0
            scene_lines.append(f"  Scene {s['scene_number']}: {s['duration_seconds']}s ({wc} words dialogue)")
        scene_breakdown = "\n".join(scene_lines)

        rewrite_prompt = f"""Your scene plan is TOO SHORT. Here's what happened:

{scene_breakdown}
TOTAL: {total_dur_planned} seconds — but the target is {target_dur} seconds ({target_dur/60:.1f} minutes).
You are SHORT by {shortfall} seconds.

REWRITE THE ENTIRE PLAN from scratch. Do NOT just append scenes at the end — that creates bad pacing.
Instead:
- Add more dialogue to existing scenes (longer monologues, more back-and-forth)
- Add new scenes in the MIDDLE of the narrative, not just at the end
- Expand the story arc — add more topics, more tangents, more reactions
- Every scene with dialogue: write MORE words so each scene runs longer
- Remember: dialogue duration = word_count / {wpm} WPM × 60 seconds

You need roughly {target_dur // SCENE_SWEET_SPOT_SEC} scenes averaging {SCENE_SWEET_SPOT_SEC}s each to hit {target_dur}s.

Output the COMPLETE rewritten JSON array of ALL scenes. No preamble."""

        raw = _chat_with_auto_tokens(
            model=OLLAMA_MODEL_CREATIVE,
            messages=[
                {"role": "system", "content": BREAKDOWN_SYSTEM},
                *_language_messages(),
                {"role": "user", "content": planning_prompt},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": rewrite_prompt},
            ],
            base_options={"temperature": 0.7},
        )
        try:
            scenes = _parse_json(raw, retries=1, brief=brief)
            scenes = _apply_scene_timing(scenes, wpm, " (rewrite)")

            total_dur_planned = sum(s["duration_seconds"] for s in scenes)
            log.info("After rewrite: %d scenes, total ~%ds (%.1f min)",
                     len(scenes), total_dur_planned, total_dur_planned / 60)
        except Exception as e:
            log.warning("Rewrite parse failed, keeping previous plan: %s", e)
            break

    scenes = _director_rewrite_pass(brief, scenes, wpm, target_dur, "brief")
    return scenes


# Module-level storage for character descriptions from the current breakdown
_current_characters = {}
_current_voices = {}
_current_style = ""
_current_director_notes = {}


def get_character_descriptions() -> dict:
    """Get the character descriptions from the most recent breakdown."""
    return _current_characters


def get_voice_descriptions() -> dict:
    """Get the voice descriptions from the most recent breakdown."""
    return _current_voices


def get_style_anchor() -> str:
    """Get the style anchor from the most recent breakdown."""
    return _current_style


def get_director_notes() -> dict:
    """Get director rewrite notes from the most recent breakdown."""
    return _current_director_notes


def _sanitize_scene(s: dict) -> dict:
    """Coerce scene fields to correct types. LLMs sometimes output numbers as strings."""
    # Integer fields
    for key in ("scene_number", "action_seconds", "duration_seconds"):
        if key in s:
            try:
                s[key] = int(float(str(s[key])))
            except (ValueError, TypeError):
                s[key] = 0

    # dialogue might be a list of dicts from some models — fix BEFORE string coercion
    if isinstance(s.get("dialogue"), list):
        parts = []
        for item in s["dialogue"]:
            if isinstance(item, dict):
                parts.append(item.get("text", item.get("line", str(item))))
            else:
                parts.append(str(item))
        s["dialogue"] = " ".join(parts)

    # String fields that might come as other types
    for key in ("description", "dialogue", "action_description", "shot_type",
                "mood", "audio_description", "setting_description",
                "lighting_description", "continuity_notes"):
        if key in s and not isinstance(s[key], str):
            s[key] = str(s[key]) if s[key] is not None else ""

    # characters_in_scene should be a list
    if "characters_in_scene" in s and not isinstance(s["characters_in_scene"], list):
        s["characters_in_scene"] = [str(s["characters_in_scene"])]

    # Ensure scene_number exists
    if "scene_number" not in s:
        s["scene_number"] = 0

    return s


def _fix_json(raw: str) -> str:
    """Fix common LLM JSON issues. Handles all known failure patterns from Gemma."""
    import re

    # 1. Flatten all whitespace (newlines inside keys/values break JSON)
    raw = raw.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')

    # 2. Collapse multiple spaces
    raw = re.sub(r'  +', ' ', raw)

    # 3. Remove single-line comments (// ...)
    raw = re.sub(r'//.*?(?=[,}\]\n])', '', raw)

    # 4. Remove trailing commas before } or ]
    raw = re.sub(r',\s*([}\]])', r'\1', raw)

    # 5. Fix unquoted string values after colon
    #    Catches: "key":_wide  "key":value  "key": value_here
    #    But NOT: "key": 123  "key": true  "key": false  "key": null  "key": [  "key": {  "key": "
    raw = re.sub(
        r':\s*(?!")(?![\d\[\{tfn-])([a-zA-Z_][a-zA-Z0-9_ ]*?)(?=\s*[,}\]])',
        r': "\1"',
        raw
    )

    # 6. Fix unquoted keys before colon
    #    Catches: {key: "value"}  , key: "value"
    raw = re.sub(
        r'([{\[,])\s*(?!")([a-zA-Z_][a-zA-Z0-9_]*?)\s*:',
        r'\1 "\2":',
        raw
    )

    # 7. Fix single-quoted strings -> double quotes
    #    Catches: 'value' -> "value" (but careful not to break apostrophes in text)
    #    Only do this for key positions (before colons)
    raw = re.sub(r"'([^']{1,50}?)'\s*:", r'"\1":', raw)

    # 8. Fix missing comma between objects inside arrays: } {  or } "
    #    Only inside arrays (after a ]), not at top level
    raw = re.sub(r'}\s*\{', '}, {', raw)
    # Don't fix } " at top level as it breaks {characters}{scenes} -> wrap detection

    # 9. Remove any BOM or zero-width characters
    raw = raw.replace('\ufeff', '').replace('\u200b', '')

    # 10. Fix truncated JSON — balance unclosed brackets/braces
    open_braces = raw.count('{') - raw.count('}')
    open_brackets = raw.count('[') - raw.count(']')
    if open_brackets > 0 or open_braces > 0:
        # Truncated output — close everything
        raw = raw.rstrip().rstrip(',')  # Remove trailing comma if any
        raw += ']' * open_brackets + '}' * open_braces

    return raw


def _try_parse_json(raw: str):
    """Try to parse JSON with progressively more aggressive fixes."""
    # Attempt 1: raw parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix trailing commas and comments
    fixed = _fix_json(raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 3: replace single quotes with double quotes
    fixed2 = fixed.replace("'", '"')
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass

    repaired = _repair_premature_root_close(fixed)
    if repaired != fixed:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Give up — raise the original error
    return json.loads(raw)


def _repair_premature_root_close(raw: str) -> str:
    decoder = json.JSONDecoder()
    candidate = raw
    for _ in range(3):
        try:
            _, end_index = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            return candidate

        tail = candidate[end_index:].lstrip()
        if not tail.startswith(","):
            return candidate

        close_index = end_index - 1
        while close_index >= 0 and candidate[close_index].isspace():
            close_index -= 1
        if close_index < 0 or candidate[close_index] != "}":
            return candidate

        candidate = candidate[:close_index] + candidate[close_index + 1:]
    return candidate


def _extract_json_text(raw: str) -> str:
    if "```" in raw:
        lines = raw.split("\n")
        inside = False
        cleaned = []
        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
                continue
            if inside:
                cleaned.append(line)
        raw = "\n".join(cleaned)

    first_brace = raw.find("{")
    first_bracket = raw.find("[")
    if first_brace >= 0 and (first_bracket < 0 or first_brace < first_bracket):
        raw = raw[first_brace:]
        end = raw.rfind("}") + 1
        if end > 0:
            raw = raw[:end]
    elif first_bracket >= 0:
        raw = raw[first_bracket:]
        end = raw.rfind("]") + 1
        if end > 0:
            raw = raw[:end]
    return raw


def _apply_scene_timing(scenes: list[dict], wpm: int, label: str) -> list[dict]:
    for s in scenes:
        s["duration_seconds"] = calc_scene_duration(s, wpm)
        s["status"] = "pending"

        dialogue = s.get("dialogue", "")
        word_count = len(dialogue.replace('"', '').split()) if dialogue else 0
        action_sec = s.get("action_seconds", 0)
        log.info("  Scene %d%s: %d words dialogue + %ds action = %ds",
                 s["scene_number"], label, word_count, action_sec, s["duration_seconds"])
    return scenes


def _parse_director_rewrite(raw: str) -> tuple[list[dict], dict]:
    parsed = _try_parse_json(_extract_json_text(raw))
    if not isinstance(parsed, dict):
        raise ValueError("Director rewrite must return a JSON object")
    scenes = parsed.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("Director rewrite response has no scenes list")
    notes = parsed.get("director_notes", {})
    if not isinstance(notes, dict):
        notes = {"pacing_notes": str(notes)}
    return [_sanitize_scene(scene) for scene in scenes], notes


def _director_rewrite_pass(
    brief: str,
    scenes: list[dict],
    wpm: int,
    target_dur: int | None,
    source_kind: str,
) -> list[dict]:
    """Run one story-tightening pass and store director notes."""
    global _current_director_notes

    scene_json = json.dumps(scenes, ensure_ascii=False, indent=2)
    target_text = (
        f"Target duration: {target_dur} seconds. Keep the revised total close to this."
        if target_dur else
        "No fixed target duration. Prefer tight, complete storytelling over adding length."
    )
    rewrite_prompt = f"""SOURCE KIND: {source_kind}
USER BRIEF OR SCRIPT:
{brief}

{target_text}

CURRENT SCENE PLAN:
{scene_json}

Perform one director rewrite pass. Remove or merge redundant beats, strengthen cause-and-effect, improve shot progression, and preserve continuity anchors. Return the revised JSON object only."""

    try:
        log.info("Director rewrite pass: tightening story and continuity...")
        raw = _chat_with_auto_tokens(
            model=OLLAMA_MODEL_CREATIVE,
            messages=[
                {"role": "system", "content": DIRECTOR_REWRITE_SYSTEM},
                *_language_messages(),
                {"role": "user", "content": rewrite_prompt},
            ],
            base_options={"temperature": 0.45},
            start_tokens=8192,
            max_tokens=32768,
        )
        revised_scenes, notes = _parse_director_rewrite(raw)
        revised_scenes = _apply_scene_timing(revised_scenes, wpm, " (director pass)")
        _current_director_notes = notes
        log.info("Director rewrite complete: %s", notes.get("story_grade", "ungraded"))
        return revised_scenes
    except Exception as exc:
        log.warning("Director rewrite failed; keeping original scene plan: %s", exc)
        _current_director_notes = {
            "story_grade": "unchecked",
            "logline": "",
            "problems_fixed": [],
            "pacing_notes": f"Director rewrite failed: {exc}",
            "continuity_strategy": "Use original continuity notes.",
        }
        return scenes


def _parse_json(raw: str, retries: int = 2, brief: str = "") -> list[dict]:
    """Parse JSON from Gemma output, retrying on failure.

    Handles both formats:
    - Plain array: [scene1, scene2, ...]
    - Dict with characters: {"characters": {...}, "scenes": [...]}
    """
    global _current_characters, _current_voices, _current_style

    raw = _extract_json_text(raw)

    try:
        parsed = _try_parse_json(raw)

        # Handle {characters, scenes} dict format
        scenes = None

        if isinstance(parsed, dict):
            if "characters" in parsed:
                _current_characters = parsed["characters"]
                log.info("Loaded %d character descriptions", len(_current_characters))
            if "voices" in parsed:
                _current_voices = parsed["voices"]
                log.info("Loaded %d voice descriptions", len(_current_voices))
            else:
                # Fallback: use known voice anchors for recognized characters
                for char_id in _current_characters:
                    char_lower = char_id.lower()
                    for name, anchor in VOICE_ANCHORS.items():
                        if name in char_lower:
                            _current_voices[char_id] = anchor
                            break
                if _current_voices:
                    log.info("Applied %d fallback voice anchors", len(_current_voices))
            if "style" in parsed and isinstance(parsed["style"], str):
                _current_style = parsed["style"]
                log.info("Loaded style anchor: %s", _current_style[:80])
            elif not _current_style:
                _current_style = "Photorealistic cinematic style, natural lighting, shallow depth of field, film-quality color grading."
                log.info("Using default style anchor")
            if "scenes" in parsed:
                scenes = parsed["scenes"]
            else:
                # Fallback: maybe it's a weird format, look for any list value
                for v in parsed.values():
                    if isinstance(v, list):
                        scenes = v
                        break
                if scenes is None:
                    raise ValueError("Dict has no 'scenes' key")
        elif isinstance(parsed, list):
            scenes = parsed
        else:
            raise ValueError(f"Unexpected JSON type: {type(parsed)}")

        # Sanitize all scenes — coerce types, fix dialogue format, etc.
        return [_sanitize_scene(s) for s in scenes]

    except (json.JSONDecodeError, ValueError) as e:
        if retries <= 0:
            raise ValueError(f"LLM returned invalid JSON after retries: {e}\nRaw: {raw[:500]}")
        # Log the area around the error for debugging
        if isinstance(e, json.JSONDecodeError):
            pos = e.pos if hasattr(e, 'pos') else 0
            start = max(0, pos - 100)
            end = min(len(raw), pos + 100)
            log.warning("JSON parse failed at position %d: %s", pos, e.msg)
            log.warning("Context around error: ...%s<<<ERROR HERE>>>%s...",
                        raw[start:pos], raw[pos:end])
        else:
            log.warning("JSON parse failed: %s", e)
        log.info("Retrying LLM call...")
        response = ollama_client.chat(
            model=OLLAMA_MODEL_CREATIVE,
            messages=[
                {"role": "system", "content": BREAKDOWN_SYSTEM},
                *_language_messages(),
                {"role": "user", "content": brief},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": 'That was not valid JSON. Respond with ONLY the JSON object: {"characters": {...}, "scenes": [...]}. No markdown fences.'},
            ],
            options={"temperature": 0.3},
        )
        return _parse_json(response["message"]["content"].strip(), retries - 1, brief)


# ── Prompt Writing ─────────────────────────────────────────────────────────

def write_prompt(scene: dict, prev_scene: dict = None, brief: str = "") -> str:
    """Write an LTX 2.3 prompt for a single scene."""

    # Get character descriptions from the breakdown
    characters = get_character_descriptions()
    voices = get_voice_descriptions()
    chars_in_scene = scene.get("characters_in_scene", [])

    context = f"=== SCENE {scene['scene_number']} ===\n\n"

    # STYLE ANCHOR — visual style lock for consistency across all scenes
    style = get_style_anchor()
    if style:
        context += f"VISUAL STYLE (apply this style consistently to the ENTIRE scene):\n  {style}\n\n"

    # CHARACTER(S) — full descriptions every time
    if chars_in_scene and characters:
        context += "CHARACTERS IN THIS SCENE (include these FULL descriptions VERBATIM in your prompt):\n"
        for char_id in chars_in_scene:
            desc = characters.get(char_id, "")
            if desc:
                context += f"  {char_id}: {desc}\n"
            else:
                context += f"  {char_id}: (no description found — describe them fully yourself)\n"
        context += "\n"
    elif characters:
        # No explicit list but we have characters — include all of them
        context += "CHARACTERS (include FULL descriptions VERBATIM for any that appear):\n"
        for char_id, desc in characters.items():
            context += f"  {char_id}: {desc}\n"
        context += "\n"
    else:
        context += "CHARACTERS: No pre-written descriptions available. You MUST write a detailed, specific physical description of every person in the scene (face, body, clothing, age, skin, hair, build) and include it in full.\n\n"

    # VOICE ANCHORS — how each character sounds
    scene_voices = {}
    for char_id in (chars_in_scene or characters.keys()):
        v = voices.get(char_id, "")
        if v:
            scene_voices[char_id] = v
        else:
            # Try fallback voice anchors for known figures
            char_lower = char_id.lower()
            for name, anchor in VOICE_ANCHORS.items():
                if name in char_lower:
                    scene_voices[char_id] = anchor
                    break
    if scene_voices:
        context += "VOICE ANCHORS (describe HOW each character speaks — the video model generates speech audio):\n"
        for char_id, voice_desc in scene_voices.items():
            context += f"  {char_id}: {voice_desc}\n"
        context += "Include voice characteristics in your prompt when describing dialogue (e.g. 'says in a deep, gravelly New York accent')\n\n"

    # SETTING — full world description every time
    setting = scene.get('setting_description', '')
    continuity = scene.get('continuity_notes', '')
    lighting = scene.get('lighting_description', '')
    context += "SETTING/ENVIRONMENT (describe ALL of this in your prompt):\n"
    if setting:
        context += f"  Environment: {setting}\n"
    if lighting:
        context += f"  Lighting: {lighting}\n"
    if continuity:
        context += f"  Continuity anchors: {continuity}\n"
    context += f"  Scene action: {scene['description']}\n\n"

    # CAMERA
    context += f"CAMERA: {scene['shot_type']}\n"
    context += f"MOOD/ATMOSPHERE: {scene['mood']}\n\n"

    # ACTION
    action = scene.get('action_description', '')
    if action:
        context += f"PHYSICAL ACTION/BODY LANGUAGE: {action}\n\n"

    # DIALOGUE — exact words
    dialogue = scene.get("dialogue", "")
    if dialogue:
        context += f"DIALOGUE ({_dialogue_instruction()}):\n{dialogue}\n\n"
    else:
        context += "DIALOGUE: None — this is a visual/ambient scene.\n\n"

    # AUDIO
    audio = scene.get('audio_description', scene.get('audio_notes', ''))
    if audio:
        context += f"SOUND/AUDIO: {audio}\n\n"

    context += f"DURATION: {scene['duration_seconds']} seconds\n\n"

    context += """CRITICAL REMINDERS:
- Rebuild the ENTIRE world from scratch — character, setting, lighting, props, camera
- The video model has NEVER seen any other scene. Describe EVERYTHING.
- Include the FULL character description — never use "he", "she", or "the same person"
- Include the FULL setting — never write "same room" or "same stage"
- Include the visual style anchor — weave the style description into the scene naturally
- Include voice characteristics when characters speak
- Embed all dialogue word-for-word in quotes within the action
- 300-500 words. Be EXHAUSTIVELY descriptive. Lazy/short prompts make bad video."""

    response = ollama_client.chat(
        model=OLLAMA_MODEL_CREATIVE,
        messages=[
            {"role": "system", "content": PROMPT_WRITER_SYSTEM},
            *_language_messages(),
            {"role": "user", "content": context},
        ],
        options={
            "temperature": 0.7,
            "num_predict": 3072,   # Plenty of room for rich descriptions with style + voice
            "num_ctx": 8192,
        },
    )
    prompt = response["message"]["content"].strip()
    log.info("Wrote prompt for scene %d (%d words): %s...",
             scene["scene_number"], len(prompt.split()), prompt[:100])
    return prompt


# ── Retry Prompt ───────────────────────────────────────────────────────────

def write_retry_prompt(scene: dict, eval_result: dict, attempt: int) -> str:
    """Rewrite a prompt based on evaluation failure."""
    fail_reason = eval_result.get("fail_reason", "unknown")
    suggestion = eval_result.get("retry_suggestion", "try a different approach")

    context = f"""A video generation attempt failed evaluation.

Original scene: {scene['description']}
Original prompt: {scene.get('ltx_prompt', 'none')}
Failure reason: {fail_reason}
Retry suggestion: {suggestion}
Attempt number: {attempt} of 3

Write a NEW LTX 2.3 prompt for this scene that addresses the failure.
Respond with ONLY the new prompt text."""

    response = ollama_client.chat(
        model=OLLAMA_MODEL_CREATIVE,
        messages=[
            {"role": "system", "content": PROMPT_WRITER_SYSTEM},
            {"role": "user", "content": context},
        ],
        options={"temperature": 0.5 + (attempt * 0.1)},  # Increase creativity on later retries
    )
    prompt = response["message"]["content"].strip()
    log.info("Retry prompt (attempt %d) for scene %d: %s...", attempt, scene["scene_number"], prompt[:80])
    return prompt
