"""
draft_prompts.py -- the ChatGPT half of the pipeline.

Drafting stays manual on purpose. ChatGPT has no API here, no seed control, and
no way to guarantee two runs match; pretending otherwise would add a dependency
and a failure mode for no gain. So the app does the part it is good at -- writing
a precise, consistent prompt -- and the director does the part that needs a
browser: paste it in, pick the best result, save it, bring it back.

Everything downstream anchors to that image, so the prompts here are opinionated
about the things that matter later:

  * a plain, even background, because IP-Adapter reads a busy one as identity
  * flat studio lighting, so the character can later be lit by the scene
  * multiple angles on one sheet, because Minimax REF2VA wants the profiles
  * no text, no watermarks, nothing that will be baked into every later frame
"""

CHARACTER_TEMPLATE = """Character concept sheet for an animated film.

SUBJECT: {subject}

{details}
LAYOUT: One image, a clean turnaround sheet with the same character repeated in
these views, evenly spaced on a single horizontal band:
  1. full body, front
  2. full body, three-quarter
  3. full body, side profile
  4. full body, from behind
Keep the character exactly the same in every view -- same proportions, same
costume, same colours. This sheet is a reference, so consistency across the four
views matters more than any single pose looking dramatic.

STYLE: {style}
LIGHTING: Flat, even studio lighting. No dramatic shadows, no rim light, no
coloured gels. The character will be re-lit later, so light it neutrally now.
BACKGROUND: Plain, solid, uniform light grey. Nothing else in the frame.

DO NOT INCLUDE: text, labels, arrows, measurements, watermarks, signatures,
multiple characters, background scenery, props not listed above, dramatic
lighting, motion blur, or any border or frame."""

PROP_TEMPLATE = """Prop concept sheet for an animated film.

OBJECT: {subject}

{details}
LAYOUT: One image, the same object repeated in these views, evenly spaced:
  1. front
  2. three-quarter
  3. side
  4. from behind
Identical object in every view -- same proportions, same materials, same wear.

STYLE: {style}
LIGHTING: Flat, even studio lighting so the materials read clearly.
BACKGROUND: Plain, solid, uniform light grey. Nothing else in the frame.
SCALE: {scale}

DO NOT INCLUDE: text, labels, dimensions, watermarks, hands or people holding
the object, background scenery, dramatic lighting, or any border."""

SCENE_TEMPLATE = """Environment concept art for an animated film.

LOCATION: {subject}

{details}
SHOT: Wide establishing shot. Show the whole space and how its parts relate, as
though this were the first time an audience sees it.

STYLE: {style}
LIGHTING: {lighting}
MOOD: {mood}

This image sets the colour palette and art direction for every shot that happens
here, so the palette should be deliberate and limited rather than busy.

DO NOT INCLUDE: text, watermarks, people or characters, close-up detail that
crowds out the space, lens flare, or any border or frame."""

ERA_LINE = "PERIOD: {era}. Every object, material, garment and piece of technology must belong to this period -- nothing anachronistic.\n"
CONTINUITY_LINE = "CONTINUITY: {note}\n"
PALETTE_LINE = "PALETTE: {palette}\n"

DEFAULT_STYLE = ("Clean, appealing animated-feature concept art. Confident line, "
                 "readable silhouette, painterly but not photographic.")


def _details(description: str = "", era: str = "", continuity: str = "",
             palette: str = "") -> str:
    parts = []
    if description:
        parts.append(f"DESCRIPTION: {description}\n")
    if era:
        parts.append(ERA_LINE.format(era=era))
    if continuity:
        parts.append(CONTINUITY_LINE.format(note=continuity))
    if palette:
        parts.append(PALETTE_LINE.format(palette=palette))
    return "".join(parts)


def character_sheet(subject: str, description: str = "", era: str = "",
                    continuity: str = "", palette: str = "",
                    style: str = "") -> str:
    return CHARACTER_TEMPLATE.format(
        subject=subject,
        details=_details(description, era, continuity, palette),
        style=style or DEFAULT_STYLE,
    )


def prop_sheet(subject: str, description: str = "", era: str = "",
               continuity: str = "", palette: str = "", style: str = "",
               scale: str = "") -> str:
    return PROP_TEMPLATE.format(
        subject=subject,
        details=_details(description, era, continuity, palette),
        style=style or DEFAULT_STYLE,
        scale=scale or "Show the object at a consistent, believable scale.",
    )


def scene_concept(subject: str, description: str = "", era: str = "",
                  continuity: str = "", palette: str = "", style: str = "",
                  lighting: str = "", mood: str = "") -> str:
    return SCENE_TEMPLATE.format(
        subject=subject,
        details=_details(description, era, continuity, palette),
        style=style or DEFAULT_STYLE,
        lighting=lighting or "Natural, motivated light appropriate to the location and time of day.",
        mood=mood or "Grounded and believable.",
    )


BUILDERS = {
    "CHARACTER": character_sheet,
    "PROP": prop_sheet,
    "BACKDROP": scene_concept,
}


def build(entry_type: str, subject: str, **kwargs) -> str:
    """Write the ChatGPT prompt for one asset.

    SHOT entries deliberately have no draft prompt: a shot is composed in FLUX
    from assets that already exist, not drafted from scratch, and drafting one
    would produce a character who doesn't match the sheet.
    """
    entry_type = (entry_type or "").upper()
    builder = BUILDERS.get(entry_type)
    if builder is None:
        if entry_type == "SHOT":
            raise ValueError(
                "Shots aren't drafted in ChatGPT -- they're composed in FLUX from the "
                "character and backdrop you've already locked. Use stage_keyframe instead.")
        raise ValueError(f"No draft prompt for '{entry_type}'. Use CHARACTER, PROP or BACKDROP.")
    accepted = builder.__code__.co_varnames[:builder.__code__.co_argcount]
    return builder(subject, **{k: v for k, v in kwargs.items() if k in accepted})


HANDOFF_STEPS = [
    "Copy the prompt above.",
    "Paste it into ChatGPT and let it draw.",
    "If it drifts, say what's wrong and ask for a fix rather than starting over -- "
    "the sheet only has to be right once.",
    "Save the image you're happy with.",
    "Bring it back here and attach it below.",
]
