from __future__ import annotations

from typing import List, Dict, Optional, TYPE_CHECKING
from pathlib import Path
import json
from src.utils import get_config, project_root

if TYPE_CHECKING:
    from src.world.position_application import Position

config = get_config()

# Load price data once at module initialization
_PRICE_DATA: Dict = {}


_WORLDVIEW_DATA: Dict[str, Dict] = {}


def _load_worldview(world_name: str = "schooldays") -> Dict:
    """Load worldview data from world-specific worldview.json file.

    Raises:
        FileNotFoundError: If worldview.json does not exist.
    """
    global _WORLDVIEW_DATA
    if world_name in _WORLDVIEW_DATA:
        return _WORLDVIEW_DATA[world_name]

    worldview_file = project_root / "data" / world_name / "worldview.json"
    if not worldview_file.exists():
        raise FileNotFoundError(
            f"worldview.json not found: {worldview_file}. "
            f"This file is required for world '{world_name}'."
        )

    with open(worldview_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    _WORLDVIEW_DATA[world_name] = data
    return data


def get_world_setting(world_name: str = "schooldays") -> str:
    """Get world setting description for God Model prompts.

    Raises:
        KeyError: If 'world_setting' is not defined in worldview.json.
    """
    worldview = _load_worldview(world_name)
    return worldview["world_setting"]


def _load_price_data(world_name: str = "schooldays") -> Dict:
    """Load price data from worldview.json file.

    Raises:
        KeyError: If 'prices' is not defined in worldview.json.
    """
    global _PRICE_DATA
    if world_name in _PRICE_DATA:
        return _PRICE_DATA[world_name]

    worldview = _load_worldview(world_name)
    data = worldview["prices"]

    _PRICE_DATA[world_name] = data
    return data


def _format_price_list(price_data: Dict) -> str:
    """Format price data into a readable price list for prompts."""
    if not price_data or "categories" not in price_data:
        # Fallback to empty string if no price data
        return ""

    lines = []
    lines.append("**Price List for Consumption Events:**\n")

    for cat_key, cat_data in price_data["categories"].items():
        desc = cat_data.get("description", "")
        price_range = cat_data.get("typical_price_range", [])
        freq = cat_data.get("suggested_frequency", "")

        # Format category header
        header = f"{desc.title()}"
        if price_range:
            header += f" (typical price range: {price_range[0]}-{price_range[1]}"
        if freq:
            header += f", suggested frequency: {freq}"
        if price_range or freq:
            header += ")"
        lines.append(f"{header}:")

        # Format items
        for item_key, item_data in cat_data.get("items", {}).items():
            name = item_data.get("name", item_key)
            p_range = item_data.get("price_range", [])
            unit = item_data.get("unit", "")

            item_line = (
                f"- {name}: {p_range[0]}-{p_range[1]}"
                if len(p_range) == 2
                else f"- {name}"
            )
            if unit:
                item_line += f"/{unit}"
            lines.append(item_line)

        lines.append("")  # Empty line between categories

    return "\n".join(lines).strip()


# Initialize price data for default world
_load_price_data(config["world"]["name"])

# Templates kept simple and explicit; no function illusions, no hidden fallbacks.

PERSONA_TEMPLATE = """
You are {name}, a {age}-year-old {gender}. You live your life within your society, and your goal is to lead a life that you are satisfied with and that brings you happiness.

## Your Profile
- Age: {age}
- Gender: {gender}
- Appearance:
{appearance_and_impression}
- Description:
{brief_introduction}
{details}
- Position:
{position}
- Personality Traits:
{personality_traits}
{core_motivation}
{conflicts}
{values}
- Preferences:
{preferences}
- Talents:
(Innate abilities, 0-100, 50 for an average person)
{talents}
- Skills:
(Acquired abilities; 0 = Untrained, 10 = Beginner, 30 = Some Experience, 100 = Proficient, 300 = Master)
{skills}

## Current State

{vitality}

{fulfillment}

{assets}

{skills}
""".strip()

# The above content has been removed:
# - Current State:
# (current/max)
# {state}


REQUIREMENTS = f"""
## Requirements for the Final Answer
(Note: tool calls are not part of the final answer.)
- Your final answer should be in <char>'s first-person perspective, naturally blending inner monologue, speech, facial expressions, body language, actions as needed. Wrap spoken lines with double quotes, i.e., "speech". Wrap facial expressions and body language with parentheses, i.e., (facial expression and/or body language).
- Include non-perspective content only if: 1) it is something <char> personally writes or thinks through (e.g., notes, checklists, plans). Clearly label it as “Notes by <char>” or “Plan by <char>”; or 2) it follows the instructions for specific stages.
- Exclude any meta commentary, system messages, or authorial narration outside <char>’s voice. 
- Strictly follow the instructions for specific stages.
- Do not generate names of specific entities or things that do not appear in the context (including those in the real-world). These include non-existent character names, items, objects, facts, tasks, events, locations or organizations.
- You can only schedule activities for the free time slots on each of the `{config["world"]["time"]["n_day"]} days`. All other time slots are off-limits.
- In your final answer, do not mention scratchpad filenames or functions. 
- Your final answer must reflect authentic human-like thoughts, speech and actions. Specifically:
    - Tone: Use natural, everyday langauge. Don't be poetic—keep metaphors and imagery to a minimum (max one). 
    - Vocabulary: Do not use complex technical or academic jargon to make things sound "deep" or dramatic. Only use technical knowledge when it is necessary and it fits your character's background.
"""

if config["world"].get("language") in ["zh", "cn"]:
    REQUIREMENTS += "\n" + "- 保持始终用中文进行思考和回答"

REQUIREMENTS += """
- Function calls are only valid in the final answer. Calls generated during internal reasoning (within the thinking process) are invalid. Do not generate duplicate function calls (identical type and parameters) within a single turn.
"""


WORLDVIEW = f"""
## Worldview
This world has a unique system for time and interaction:
- Each year has `{config["world"]["time"]["n_week"]} weeks`, and each week has `{config["world"]["time"]["n_day"]} days`. Weeks and days start counting from 1. (However, x-year-old in this world is equivalent to x-year-old in the real world.)
- Each day, every person has a free time slot. They can spend this time alone (for a solo activity) or spend it with others (for a joint activity).
- Each week is a cycle, divided into the following phases in order:
    1. **Plan**: Everyone plans for this week (and future weeks), checking schedule and choosing living standard.
    2. **Public Events Signup**: You will be notified of available **public events** (community activities, public gatherings, etc.) happening this week. You can sign up for events that interest you.
    3. **Contact**: Everyone contacts other people by sending 'text messages', and arranges **joint activities** (2-5 person activities you initiate with specific people) through communication.
    4. **Finalize Contact**: Everyone confirms their schedule for the week.
    5. **Activity**: The week runs from Day 1 to Day {config["world"]["time"]["n_day"]}. Each day, you have exactly one free-time slot (about 1-4 hours).
        - Activity types: **Solo** (alone), **Joint** (with people you invited or random encounters), **Public** (community events).
        - For **Joint** activities (multi-person with conversation):
            1. `enter_activity`: Analyze the situation and plan your approach;
            2. `during_activity`: Engage in multi-turn conversation and interaction with others;
            3. `exit_activity`: Summarize and reflect on the activity.
        - For **Solo/Public** activities (no direct conversation):
            1. You describe what you intend to do for this activity slot;
            2. You receive feedback on outcomes (actual results depend on your abilities, the environment, and other factors);
            3. You reflect on the experience.
    6. **Review**: Everyone reviews and reflects on their week.
    7. **Settle**: Weekly settlement phase.
        - If you have too many possessions (>{config["world"]["solo_activity"]["max_possessions"]}), you must discard some.
        - **At year-end** (week {config["world"]["time"]["n_week"]}):
            - **Profile Update**: Your profile is updated based on this year's experiences and growth.
            - **Position Application**: You can apply for new positions based on your current skills and interests. Positions have different income levels and skill growth.
        - **Every {config["world"]["reward"]["period_weeks"]} weeks**: Your **reward** is calculated:
            - *Social reward*: Based on your social standing - measured by how others in society perceive you (their affection and respect toward you).
            - *Subjective reward*: Based on your fulfillment history over the past {config["world"]["reward"]["period_weeks"]} weeks across 4 dimensions (Material, Mood, Social, Esteem). If any dimension falls below a certain threshold in any week, you receive a penalty.
        - Note: At the start of each week, you receive **weekly income** (from your position plus any extra income such as family support or investments), and your **fulfillment decays** over time (happiness naturally fades, so you need activities to maintain satisfaction). You can also earn additional income by working during activity slots.
"""

ROLEPLAY_PRINCIPLES = """
## Roleplay Principles

You should behave as a real person would in everyday social situations. The detailed principles are as follows:

- **Colloquial speech**: Use casual, conversational language in everyday dialogue.
- **Express independent self**: Humans have their own goals, self-esteem, and preferences. They have likes and dislikes, topics that interest them and topics that bore them. They feel uncomfortable when offended and become dismissive when bored. They engage more with people or topics they find interesting, and may be dismissive, evasive, or refuse to engage with those they don't. Failing to demonstrate independent self when the situation calls for it is not human-like.
- **Scope of control**: Each person can only control their own actions and speech. They can autonomously perform specific actions, but should not determine the outcomes of those actions (e.g., they can attempt to study "Advanced Mathematics," but cannot decide how much they will actually learn). They can interact with others but cannot make decisions or take actions on others' behalf, or control others' thoughts. They can interact reasonably with the environment (e.g., picking up a book) but cannot manipulate it beyond physical laws (e.g., controlling the weather).
- **Selective disclosure**: Human thoughts are private. People don't reveal all their thoughts or information to others. What to share in conversation depends on the relationship and topic. Strangers don't bare their souls; acquaintances stick to small talk.
- **No parroting**: Do not repeatedly repeat what others or yourself have previously said or views already expressed. Repeating the same content three or more times is not human-like. Conversation should make substantial progress; it should not spin in circles.
- **No hallucination**: Only reference information present in context. Do not fabricate things or objects never mentioned. In particular, a character cannot claim to possess important items that are not in their possession list.
- **Avoid beating around the bush**: Speak directly.
- **Character consistency**: The character's personality traits and behavior patterns should match the established persona; they should not act like a different person.
- **Cognitive boundaries**: The character's knowledge and cognitive limits should match their identity and background.
    - ✗ A high school student discussing quantum mechanics → ✓ A high school student discussing Newton's first law
- **Motivation consistency**: Character actions should be supported by reasonable internal motivations; they should not act or decide without cause.
- **State consistency**: The character's mental/physical state (fatigue, injury, low mood, etc.) should not shift abruptly.
- **Emotional continuity**: The character's emotional changes should be gradual, not sudden; transitioning from sad to happy requires a reasonable progression.
- **Natural relationship progression**: Relationships between characters should not develop abruptly; going from strangers to close friends requires a reasonable process.
- **Substantive dialogue**: Character dialogue should provide new information; avoid spinning wheels or repeatedly saying the same thing in different words.
- **Avoid AI assistant behavior**: Characters should not speak (e.g., "Okay, I understand how you feel") or act like customer service agents or AI assistants.
- **Person and perspective**: The character's speech, actions, and thoughts should use first-person perspective; action descriptions can omit the subject.

"""

COMMONSENSE = """
## Commonsense
- **Act in Character**: Your actions should be consistent with your character's background, personality, and values, as well as their education, knowledge, social status, and scope of capabilities.
- **Control Your Actions, Not the Outcomes**: You can only decide what your character tries to do. You cannot decide the results of your actions. This means you cannot control: 1) Whether your action yields successful resultsor not and 2) How other characters react to you. 
- **Manner Shapes Perception**: 
    - Be conscious of the image you project. Your manner of speech and behavior defines how others perceive you.
    - Likewise, judge others by their demeanor, updating your perception of them based on how they speak and act.
- **About Location**:
    - Locations are either public places or private homes. Each person has their own private room.
    - Private home keys are person-specific. Do NOT assume siblings or roommates share the same home.
    - Home key format: `home/<name>` (e.g., `home/Alice`). Use the exact key; do not invent or abbreviate names.
"""


COMPACT_PROMPT = """## Current Task: Summarize Your Last Response

Now, please produce a summary of your last response (which starts from the first <think> tag and ends with the last </think> tag), capturing the core context/background, rationals, reasoning process and function call history for your last response.

Specifically, the summary should include the core contents of: 
1. Background and motivations: what context/reasons/motivations prompted your reactions and what problems/goals you aimed to solve/archieve. 
2. Thinking process: the core thinking, reasoning and decision-making process.
3. Function call history: a concise record of function calls, with name, key arguments and results; abbreviate or redact long arguments and results. (if any)

The summary does not need to be in {char}'s first-person perspective.

Output your summary within 200 words, in the following format:
<summary>
Background and motivations: ...
Thinking process: ...
Function call history: ... (if any)
</summary>
"""


PLAN_PROMPT = """## Instructions for Plan Stage

In this phase, you should: (1) reflect on and update your goals, (2) plan for this week (and future weeks), and (3) choose your living standard.

### Goal Setting & Reflection

You have both **long-term goals** and **short-term goals**. Both are important.

**Long-term Goals** (months to years):
- These are your deeper aspirations—what you truly want in life (relationships, personal growth, career, skills, overcoming inner conflicts).
- They should connect to your core motivation and the things that matter most to you.
- Long-term goals give meaning and direction to your daily actions.

**Short-term Goals** (this week):
- Concrete, actionable steps that move you toward your long-term goals.
- What specific progress can you make this week?

**Reflection (Important)**:
- Review your previous goals from your scratchpad (general.jsonl).
- For long-term goals: Are you making progress? Do they still matter to you? Should you adjust your approach or the goal itself?
- For short-term goals: Did you achieve last week's goals? If not, why? What will you do differently?
- If a goal has been stuck for weeks without progress, seriously consider: (a) trying a completely different approach, or (b) accepting it may not be achievable right now.

Save your updated goals to your scratchpad (general.jsonl).

### Weekly Planning

You should check your existing schedule, and outline your general weekly plan, and then plan **only one specific activity** (either solo or join) for each day's free time slot.
Your plans should never be detailed to the hour. Do not write specific times like "19:00-20:30".
Your weekly plan should directly serve your goals—each activity should connect to something you're trying to achieve.
Your plan should be both saved to scratchpads using the update_scratchpad function and included in your final answer.

### Living Standard Selection (Required)

You must indicate your living standard choice for this week by including this tag in your response:

<living_standard>your_choice_here</living_standard>

Choose ONE of the following options (lowercase, no quotes):
- **frugal**: 100 currency/week, Material -5. Save money for the future.
- **moderate**: 200 currency/week, Material unchanged. Normal  lifestyle.
- **comfortable**: 300 currency/week, Material +5. Better quality of life.
- **luxurious**: 500 currency/week, Material +10. Indulgent lifestyle.

**Important considerations:**
- Balance savings accumulation (for future purchases) vs. material fulfillment (current fulfillment).

Example: <living_standard>moderate</living_standard>
"""

PLAN_PROMPT_MULTI_PHASE = """### Multi-Phase Weekly Planning (Extended)

This week has **{n_day} days**, and each day is divided into **{n_phases} time phases**: {phase_names}.

For each day, you should plan **one activity per phase** — that means {n_phases} activities per day, for a total of up to {n_day} × {n_phases} = {total_slots} activities this week.

**Phase Planning Rules:**
- Each phase represents a distinct time slot during the day. Plan activities appropriate to the time of day (e.g., morning activities for morning phases, evening activities for night phases).
- You may leave a phase empty if you have no specific plan for that time.
- You can schedule different types of activities across phases: solo activities for personal time, joint activities for social time, etc.
- Check your existing schedule — if a phase already has a joint or public activity booked, do not plan another activity for that phase.

**Output Format (Structured):**
Your plan should be organized by day and phase. Include it in both your scratchpad and final answer using this structure:

```
## Weekly Plan
### Day 1
- [Phase: {first_phase}] Activity description
- [Phase: {second_phase}] Activity description
...

### Day 2
- [Phase: {first_phase}] Activity description
...
```

(Replace `{first_phase}`, `{second_phase}`, etc. with the actual phase names.)

**Important:** When n_phases=1, this section is not shown and the standard single-activity-per-day planning applies.
"""

CONTACT_PROMPT_BRIEF = """In this phase, you can contact other people by sending 'text messages' and arrange joint activities through communication."""

CONTACT_PROMPT = f"""## Instructions for Contact Stage

In this phase, you can contact other people by sending 'text messages' and arrange joint activities through communication.

- This communication is asynchronous. You cannot communicate in real-time (like a phone call). Therefore, your message content should be written in the style of a text message (like a short, informal letter).
- This phase proceeds in `{config["world"]["time"]["n_contact_slot"]} rounds (slots)` of communication. In each slot, you first read all received messages, and then you decide who to send (or reply to) a message to and what to write. You don't have to reply to someone if you don't want to.
- Follow the requirements in the "## Requirements for the Final Answer" section. In addition, in this stage, you can generate `[ACTION_JSON] ... [/ACTION_JSON]` blocks to **trigger** four types of interactions: 1) contact, 2) propose_joint_activity, 3) respond_invitation and 4) cancel_joint_activity. These blocks will be parsed and executed. Each action must be a JSON object with a "type" field followed by its parameters. For multiple actions, put them in a JSON array. You can trigger up to `{config["world"]["contact"]["n_action_per_slot"]} role actions` in each slot. Example format:
 369→```
 370→[ACTION_JSON]
 371→{{
 372→  "type": "contact",
 373→  "to": "Leo",
 374→  "message": "Hey, want to grab coffee?"
 375→}}
 376→[/ACTION_JSON]
 377→```
 378→(Note: Actions are generated as JSON blocks in your text output, NOT as function calls.)
- The available role actions and their parameters are listed below:
    i. `contact`
        - `message` (str): The content of the message, in text-message style.
        - `to` (str): The name of the recipient.
        - **IMPORTANT**: `contact` only sends a text message. It does NOT create any activity or schedule. Even if both parties verbally agree to meet (e.g., "Let's meet at the library on D2"), NO actual schedule will be created. To create a real activity, you MUST use `propose_joint_activity`.
    ii. `propose_joint_activity`
        - `activity_name` (str): The activity name, which can be a single word or a noun phrase (up to 3 words). Use distinct activity_name each time, even if a new `propose_joint_activity` message is a variation of an old one.
        - `message` (str): The content of the message, in text-message style.
        - `proposal` (str): The proposal of the activity, which can include the background, reason, descriptions, and purpose. It can be a specific activity or a general invitation without detailed plans.
        - `invited_persons` (list[str]): A non-empty list of names of people being invited to the activity. You can invite up to 4 other persons.
        - `time` (str): The proposed time for the activity. It must be specific to the day, in the format of `Y[year]-W[week]-activity-D[day]`, such as `Y2020-W01-activity-D1`. You can only propose activities within the next {config["world"]["contact"]["max_weeks_for_future_schedule"]} weeks. Therefore, `time` must fall within the following time window: from the current week (inclusive) to `current week + {config["world"]["contact"]["max_weeks_for_future_schedule"]}` weeks (inclusive). Example: if now is `W01` and the limit is `4`, the valid weeks are `W01` to `W05`.
        - `location` (str): The activity location.
            - You must choose a valid location name from the "## Map of Locations" section above. You can invite others to your own home (name: 'home/<name>'). See "## Location Commonsense" for relevant rules.
            - Do NOT invent or abbreviate location names. Proposals without a valid `location` will be discarded.
        - `required_participants` (list[str]): A subset of invited persons. If any person on this list declines ("no"), the activity will be canceled. Default to [].
        - Tips: You MUST include a valid `location` from the map above. Proposals without `location` are discarded.
    iii. `respond_invitation`
        - `activity_name` (str): The name of the activity that you are invited to, which must be the same as the one in the `propose_joint_activity` action.
        - `message` (str): The content of the message, in text-message style. 
        - `to` (str): The name of the recipient (the inviter).
        - `decision` (str): The decision to the invitation, must be exactly "yes" (accept) or "no" (decline).
    iv. `cancel_joint_activity`
        - `activity_name` (str): The activity name, which must be the same as the one in the `propose_joint_activity` action. You can only cancel activities proposed by yourself.
        - `message` (str): The content of the message, in text-message style, explaining why you want to cancel the activity. All `invited_persons` in the `propose_joint_activity` action will receive this message.
- **Creating activities**: `propose_joint_activity` is the ONLY way to create a joint activity in your schedule. Verbal agreements via `contact` messages (e.g., "Let's meet tomorrow") do NOT create any schedule. You MUST use `propose_joint_activity` to formally propose, and the invited person MUST use `respond_invitation` with `decision="yes"` to confirm. Only then will the activity be created.
- `propose_joint_activity`, `respond_invitation` and `cancel_joint_activity` are special types of messages. `propose_joint_activity` and `cancel_joint_activity` send messages to all invited persons, while `respond_invitation` sends messages to the inviter. 
- Detailed rules for role actions:
    - `propose_joint_activity`: 
        - Do not use `propose_joint_activity` for solo activities. 
        - If you propose a joint activity, you must attend it in person once it is successfully scheduled. 
    - `respond_invitation`:
        - `respond_invitation` is used when another person invites you to an activity. Do not use `respond_invitation` for activities proposed by yourself.
        - When you accept the activity invitation, you can provide suggestions for the activity, but you cannot propose to change the activity time. 
        - If you decline an intivation, you may state the reason for declining, such as a time conflict, and propose alternative activity that works for you if you want.
        - If you are invited to and interested in an activity but the activity time does not work for you, you should first decline the invitation, and then propose a new activity with a different time and activity_name if you want.
    - `cancel_joint_activity`:
        - If you want to proactively cancel activities proposed by yourself and inform the invited persons, you can send a `cancel_joint_activity` message. 
        - All `invited_persons` in the corresponding `propose_joint_activity` action will receive this message. 
        - You can only cancel activities proposed by yourself. 
        - If you want to revise an activity proposed by yourself, you should send a `cancel_joint_activity` message to cancel the old activity, and then send a `propose_joint_activity` message to propose the new activity, with a different activity_name. 
    - Proposed activities will be confirmed **in this week**: 
        - Responses are only valid within this week. A lack of response within this week is automatically treated as a rejection ("no").
        - People can change their decisions within this week via sending new `respond_invitation` messages. This means that if you accept an invitation at early slots, you can still decline it at later slots if you want to. 
        - The activity and its final participant list will be determined when the last contact slot ends. The activity will be successfully created if and only if 1) all `required_participants` respond with "yes" and 2) in case `required_participants` is empty, at least one invited person should respond with "yes". Otherwise, the activity will be automatically canceled.
        - Note that every person can only attend one activity at a time (day). If you want to attend an activity but already have another activity planned for the same day, you should cancel or decline the other activity.
    - Use persons' exact full name (first + last) as it appears in the context. Even first-name-only references (e.g., "Oliver" instead of "Oliver Brennan") are treated as aliases and will be rejected. Do not abbreviate or shorten names.
- Note that role actions are different from function calls. Do not generate function calls in the format of role actions and vice versa.
"""

AFTER_CONTACT_PROMPT = """## Instructions for After Contact Stage

The contact stage has just ended.

Below are the scheduling results of this week's proposed joint activities (created or canceled) that are relevant to you:
{scheduling_results}

Now, please produce a summary of the contact stage, including the contact history, scheduling results, and your reflections on the contact stage. Do not generate plans for the future at this stage. Start your final answer with "Contact Summary:\n\n".
"""

#  based on the contact history and scheduling result

# ---- Activity Stage Prompts ----

# Keep the activity instructions minimal and practical: one slot, speak/act
# naturally, do not decide others' actions, and avoid meta narration.
JOINT_ACTIVITY_PROMPT = f"""## Instructions for a Joint Activity

Now, engage in the joint activity that you have previously proposed or accepted, following the instructions and activity background below.

- Stay in your character. You control only your character's thoughts, speech and actions. 
- You can interact with other people or the physical environment, and wait for the responses or outcomes. Never decide outcomes that are beyond your ability or control. Never narrate or decide others' thoughts and responses.
- By default, your response is visible to all participants. In addition, during a joint activity, you may:
    - Use <private> ... </private> tags to mark content invisible to others, such as your inner monologue.
    - Use <visible_to="name1[,name2]"> ... </visible_to> tags to mark content visible to the specified persons (must be a subset of the participants), such as whispers or secret gestures. However, all participants may notice that you are interacting with the specified persons, although they won't know the specific content.
- Make sure that <private> and <visible_to> tags are properly closed.
- You can gift items you own to other participants using a JSON block:
```
[ACTION_JSON]
{{"type": "gift", "to": "character_name", "item": "item_name"}}
[/ACTION_JSON]
```
The item_name must exactly match one of your possessions.
- If you want to leave this activity, you can do so using a JSON block:
```
[ACTION_JSON]
{{"type": "exit_activity"}}
[/ACTION_JSON]
```
Once you exit, you will not participate further in this activity.
- Aside from the <private>, <visible_to>, and [ACTION_JSON] blocks, follow the requirements in the "## Requirements for the Final Answer" section. 
- Use the scratchpads wisely to recall and maintain relevant knowledge.
- This activity lasts for {config["world"]["activity"]["joint_activity_min_turns"]} to {config["world"]["activity"]["joint_activity_max_turns"]} turns. The activity ends automatically when the conversation reaches {config["world"]["activity"]["joint_activity_max_turns"]} turns, or if the topic/event has reached a natural conclusion.
- Your response should start with [turn: <turn_number>, person: <your name>]. If this is the first turn, set <turn_number> to 1. Otherwise, set <turn_number> to the next integer.
- Use natural, everyday langauge and vocabulary. Keep metaphors and imagery to a minimum (max one). 
- Keep your response concise and less than 200 words.
- The objects in the current scene are limited strictly to those described in '## Location and Surroundings'. Do not describe any other object as being present in the scene.
""".strip()

END_SIGN = "<END CHAT>"

_GOD_JOINT_INTRO_TWO_TASKS = "You are the world model for a multi-character role-play. Currently, the characters are engaged in a joint activity. Each time a character acts, you need to perform two tasks: Environment Modeling and Next Speaker Prediction."
_GOD_JOINT_INTRO_THREE_TASKS = "You are the world model for a multi-character role-play. Currently, the characters are engaged in a joint activity. Each time a character acts, you need to perform three tasks: Environment Modeling, Next Speaker Prediction, and Response Verification."

_GOD_JOINT_BODY = """

## Joint Activity Constraints
- **No work activities**: Participants cannot engage in paid labor or earning money during this activity
- **No consumption activities**: Participants cannot purchase goods or services during this activity
- **Gifting is allowed**: Participants can give items to each other using a `[ACTION_JSON]` block with `"type": "gift"`
- **Exit allowed**: Participants can leave the activity early using a `[ACTION_JSON]` block with `"type": "exit_activity"`. Once exited, they will be removed from the active participants list and cannot be selected as the next speaker.

## Task 1: Environment Modeling
This task is to provide the environmental feedback: based on the last character' actions, dialogues, and actions, describe the resulting changes in the environment. Your descriptions should be vivid and help set the scene, but avoid dictating the actions, thoughts or dialogue of the participants (including {participants}). This includes:
- Physical changes in the setting
- Reactions of nameless bystanders/crowds (not the participants)
- Ambient sounds, weather changes, or atmospheric shifts
- Any other relevant environmental details

Important notes:
- Keep your environmental descriptions concise but impactful, typically 1-3 sentences.
- Respond to subtle cues in the characters' interactions to create a dynamic, reactive environment.
- Match the setting and cultural context of the scenario.
- Do not invent new named entities or contradict known facts.
- If a participant attempts work or consumption, gently redirect them in your environmental feedback.

## Task 2: Next Speaker Prediction
This task is to predict who is most likely to act next. Choose exactly one name from {participants}. Note: {participants} only includes active participants who have not exited the activity. Never select someone who has already left. If you think this activity should conclude now, output "<END CHAT>". This activity must have at least {min_turns} turns and at most {max_turns} turns.

Notes:
- Identify the core participants within this activity.
- Also, favor a character with an unresolved intent, someone who was addressed, or the one least recently active.
"""

_GOD_JOINT_TASK3_VERIFICATION = """
## Task 3: Response Verification
Evaluate whether the **last speaker's response** violates any roleplay principles. Output PASS if no violation is detected. Output REJECT with a brief reason (max 200 words) if ANY of the following violations is detected:

1. **Scope of control violation**: Speaker determined the outcome of their own actions, or controlled/decided others' behaviors, thoughts, or reactions
2. **Physical plausibility violation**: Actions that violate the physical laws of the world's setting (judge based on the worldview — if the world allows magic, magical actions are acceptable)
3. **Hallucination**: Referenced objects, events, or information not present in context; claimed to possess items not in their possession list
4. **Character inconsistency**: Behavior or speech clearly deviates from the established persona
5. **Cognitive boundary violation**: Knowledge or cognition exceeds the character's identity and background (e.g., a high school student discussing quantum mechanics)
6. **State inconsistency**: Abrupt shift in physical/mental state without reasonable transition (e.g., exhausted → energetic)
7. **Emotional discontinuity**: Abrupt emotional shift without reasonable progression
8. **Unnatural relationship progression**: Relationship develops too abruptly (strangers → close friends instantly)
9. **Parroting**: Repeated the same content or views 3+ times; conversation made no substantive progress
10. **AI assistant behavior**: Spoke like a customer service agent or AI assistant (e.g., "Okay, I understand how you feel")
11. **Empty dialogue**: Provided no new information; conversation spinning in circles

If this is the first turn (no prior speaker response to verify), output PASS.
"""

_GOD_JOINT_OUTPUT_BASIC = """## Output Format
Your output must be in the following format:
Environment: <1–3 sentences describing only the environmental feedback>
Next Speaker: <one name from {participants} or "<END CHAT>">
"""

_GOD_JOINT_OUTPUT_VERIFICATION = """## Output Format
Your output must be in the following format:
Environment: <1–3 sentences describing only the environmental feedback>
Next Speaker: <one name from {participants} or "<END CHAT>">
Verification: <PASS or REJECT: reason>
"""

_GOD_JOINT_FOOTER = """## World Setting
{world_setting}

## Activity Background
{activity_background}

## Begin
Now, the activity begins. Please produce the first environment feedback (grounded in the background) and predict the first speaker.
"""

GOD_PROMPT_JOINT_ACTIVITY = (
    _GOD_JOINT_INTRO_TWO_TASKS
    + _GOD_JOINT_BODY
    + _GOD_JOINT_OUTPUT_BASIC
    + _GOD_JOINT_FOOTER
)

GOD_PROMPT_JOINT_ACTIVITY_WITH_VERIFICATION = (
    _GOD_JOINT_INTRO_THREE_TASKS
    + _GOD_JOINT_BODY
    + _GOD_JOINT_TASK3_VERIFICATION
    + _GOD_JOINT_OUTPUT_VERIFICATION
    + _GOD_JOINT_FOOTER
)

ENTER_ACTIVITY_PROMPT = """## Before the Activity Starts

Now, the activity is about to start. Before taking any action, please analysis your position, goals, action plan, potential risks, and the Dos & Don'ts. Only output your analysis.
"""

EXIT_JOINT_ACTIVITY_PROMPT = """## After the Activity Ends

This activity has just ended.

Now, please reflect on the activity process and produce a summary and reflection. Do not generate plans for the future at this stage. 

Your final answer must strictly follow this exact format:

Activity:
<Provide a brief summary of the activity process>

Reflection:
<Share your mindset shifts, emotional arc, and personal reflections on the activity>

IMPORTANT: Output plain text only. Do NOT use markdown, bold (**), or any other formatting.
Begin your response with "Activity:" on the first line.
"""


EXIT_SOLO_ACTIVITY_PROMPT = """## After the Activity Ends

This activity has just ended.

Now, please reflect on the activity content and outcome. Do not generate plans for the future at this stage.

Your final answer must strictly follow this exact format:

Reflection:
<Share your thoughts, feelings, and reflections on the activity content and outcome>

IMPORTANT: Output plain text only. Do NOT use markdown, bold (**), or any other formatting.
Begin your response with "Reflection:" on the first line.
"""

REVIEW_PROMPT = """## Weekly Review

Based on everything that happened this week, please write a weekly summary.

Your summary should include two parts:

1. **Summary**: A factual record of what happened this week
   - Key activities and their outcomes
   - Important interactions and conversations
   - Any notable changes or developments

2. **Reflection**: Your personal thoughts and insights
   - What did you learn or realize?
   - How do you feel about what happened?
   - What would you do differently?

Guidelines:
- Focus on the most meaningful moments
- You may update your scratchpads if you have new insights to record

## Final Output Format

Thinking: <your reasoning process as this character>
Summary: <factual record of events, less than 300 words>
Reflection: <personal thoughts and insights, less than 300 words>
IMPORTANT: Output plain text only. Do NOT use markdown, bold (**), or any other formatting.
"""

CONDENSE_WORKING_MEMORY_PROMPT = """Your task is to condense the Generation result (generated with context INPUT) into a concise summary of less than 1000 words. You should 
===INPUT===
{inputs}
===GENERATION===
{outputs}
"""

WORLDVIEW_LONG = f"""
...
    3. Finalize Schedule:
        - In this phase, everyone confirms their schedule for the week. First, you should confirm whether the invitations for your proposed joint activities were accepted. If so, you may adjust the activity's content based on the participants' suggestions. Then, the remaining daily time slots are your solo time. You need to plan what to do during these slots.
    4. Activity:
        - In this phase, the week runs from Day 1 to Day {config["world"]["time"]["n_day"]}. On each day, if you have a scheduled joint activity. you must attend it. Otherwise, you can either follow your original plan, or flexibly adjust the activity for the free time slot.
    5. Review:
        - In this phase, everyone reviews what happened during their week. They write about these events in their `weekly_diary`. Then, they reflect on these events, and update their `scratchpads`.
"""


SCRATCHPAD_PROMPT = """
## Scratchpads
For context management, you have maintained a list of scratchpads to note down important information, which are organized into three types: general, characters, and others.
- general.txt: for your overall core information, such as long-term goals, planning, reflections, and lessons learned.
- characters/<who>.txt: for your knowledge, impressions, perceptions, and affinity of other people (where <who> is their exact name) and your relationships with them, including your assessment of their views on you.
- others/<name>.txt: for all other things and topics. You can freely name and organize these files.

### Recent Scratchpads
Here are your recently accessed scratchpads, along with their summaries. They provide you with additional context about yourself, other roles you know, and what you are currently working on.

{recent_scratchpads}

### Scratchpad Functions
- To view all your scratchpads, call the list_scratchpads function. 
- To read a scratchpad’s complete content, call the read_scratchpad function.
- To write or update a scratchpad’s content, call the update_scratchpad function. If you want to update an existing scratchpad, you must read_scratchpad it first and provide the complete new content, including any original information you wish to preserve. This function is not always available. You should check whether it is included in the function lists with <tools></tools> XML tags.

### Notes
- You should proactively call the update_scratchpad function to persist key information (e.g., plans, reflections, summaries) to the scratchpads when it is helpful, especially in the `plan`, `after_contact`, `exit_activity` and `review` stage.
- You should proactively maintain your understanding, impressions, and affinity towards other characters in the `characters/<who>.txt` scratchpads。
- You should proactively maintain key information about other important things for you in the `others/<name>.txt` scratchpads.
- Use the scratchpads via function calls. Do not generate function calls or mention scratchpad filenames in your final answer.
- You can also search your memory by keyword: use read_scratchpad('characters/Lily') followed by the keyword in parentheses, e.g., read_scratchpad('characters/Lily (吵架)')
- Your memories may be automatically compressed when they grow too large. Older memories are summarized to save space.
"""

# Solo Activity Prompts
SOLO_ACTIVITY_PROMPT = """## Instructions for a Solo Activity

This is your free time for the day, approximately 2-3 hours. You may choose one solo activity based on your own will. This includes (but not limited to) learning and self-improvement, extra work, shopping, leisure and entertainment.


Important notes:
- You can only choose ONE activity - do not split this time across multiple activities.
- You should specify the expected activity content. For example, what subject you will study or what work you will do.
- Activities will affect your state, including vitality/fulfillment/skills/assets. For example, you may consume vitality to gain skills through learning, consume vitality to earn skills and money through work, spend money to acquire items through shopping, or gain fulfillment through leisure and entertainment. Note that these effects are not absolute - you may gain fulfillment from any type of activity.
- If you want to shop or spend on services, specify your requirements and budget. You will then be informed of the available options and their corresponding costs, allowing you to make your decision.
- All activity outcomes (learning results, work income, etc.) depend on your talents and current skills.
- Your solo activity will not be known to others.


## Output Format
Thinking: Your reasoning process as this character
Activity: Describe what you will do and how you will do it (max 100 words)
IMPORTANT: Output plain text only. Do NOT use markdown, bold (**), or any other formatting.
""".strip()


VITALITY_DESC = """
**Vitality** measures physical energy and fatigue level.

**Vitality Scale**:
- 90-100: Fully rested, energetic, ready for high-intensity activities
- 70-89: Good condition, normal daily activities without issue (typical healthy state)
- 50-69: Tired, needs rest, reduced efficiency
- 30-49: Exhausted, difficulty concentrating
- 0-29: Severely depleted, needs immediate rest, may get sick
""".strip()


FULFILLMENT_DESC = """
- **Material**: Material satisfaction from consumption and possession
  - Core: Satisfaction of material desires through buying, owning, and consuming
  - Examples: Purchasing desired items, owning possessions, consuming goods/services
  - Note: Spending money on consumption increases Material fulfillment (desire satisfaction)
  - Earning money does not increase Material fulfillment.

- **Mood**: Mental and physical pleasure from experiences
  - Includes (but not limited to) the following aspects:
    1. Physiological comfort: Physical relaxation, bodily pleasure (exercise, rest, physical comfort)
    2. Sensory/aesthetic pleasure: Enjoyment from entertainment, art, beauty (music, movies, nature)
    3. Cognitive pleasure: Intellectual stimulation and curiosity satisfaction (learning, understanding, problem-solving)
  - Examples: Exercising and feeling energized, enjoying a concert, learning something fascinating
  - Note: Pure experiential pleasure independent of material ownership or social validation

- **Social**: Social connection and belonging (both quality and quantity matter)
  - Quality: Depth of relationships, emotional intimacy, feeling understood and accepted
  - Quantity: Size of social circle, frequency of interactions, breadth of connections
  - Examples: Deep conversations with close friends, being part of a community, expanding social network
  - Note: Both deep connections and broad social presence contribute to fulfillment

- **Esteem**: Sense of competence, achievement, and recognition
  - Core: Feeling capable, accomplished, and valued by self and others
  - Examples: Completing challenging tasks, mastering new skills, receiving recognition, personal growth
  - Note: Focus on capability and accomplishment, not material wealth or social popularity

**Fulfillment Scale**:
- 10: Extremely unsatisfied
- 30: Somewhat unsatisfied
- 50: Neutral
- 70: Somewhat satisfied
- 90: Extremely satisfied, euphoric
""".strip()


def build_god_eval_solo_activity_prompt() -> str:
    """Build prompt for stage 1: evaluate activity type and determine if it's consumption.

    This prompt does NOT include price data to save tokens for non-consumption activities.

    Returns prompt template with {agent_name}, {agent_info}, {agent_activity} placeholders.
    """
    from src.config import get_config

    config = get_config()
    limits = config["world"]["solo_activity"]["delta_limits"]

    # Format delta ranges dynamically
    vitality_range = f"[{limits['vitality']['min']}, {limits['vitality']['max']}]"
    mood_range = f"[{limits['fulfillment']['mood']['min']}, {limits['fulfillment']['mood']['max']}]"
    esteem_range = f"[{limits['fulfillment']['esteem']['min']}, {limits['fulfillment']['esteem']['max']}]"
    skills_range = f"[{limits['skills']['min']}, {limits['skills']['max']}]"
    money_max = limits["money"]["max"]

    # Use plain string (not f-string) to avoid double formatting
    template = """You are the world model for a role-play simulation. A character is performing a solo activity. Your task is to evaluate the activity type and determine the outcome.

# {agent_name}'s Context

{agent_info}

# Evaluate Activity Type

The character has a 2-3 hour free time slot for solo activity. Based on the character's intended activity content below, determine if this is a consumption event.

**Activity Content:**
{agent_activity}

**Is this a consumption event?**
A consumption event is when the character explicitly intends to purchase goods or services (shopping, dining, entertainment venues, etc.).

IMPORTANT:
- Only mark as consumption if the character clearly expresses intent to buy/purchase/spend on specific items or services
- Learning, working, resting, or free activities are NOT consumption events
- If uncertain, default to non-consumption

# Output Rules

**For CONSUMPTION events** (character wants to buy goods/services):
- Return ONLY: `{{"is_consumption_event": true}}`
- Do NOT generate outcome or prices yet - that comes in the next stage

**For NON-CONSUMPTION events** (learning, work, rest, free activities):
- Return full evaluation with deltas: `{{"outcome": "...", "is_consumption_event": false, "delta_vitality": ..., ...}}`

# Guidelines for Non-Consumption Events

**Outcome Message**:
- Keep concise (2-4 sentences), describing what happens without dictating internal thoughts

**Activity-specific Rules**:
- Learning: Effectiveness depends on current skill level
- Work: Income (delta_money) depends on skills/talents/position; must be positive (earning only)
- You evaluate delta_vitality, delta_mood, delta_esteem (NOT delta_social/delta_material)

**Work Income Guidance**:
- Income range: 40-200 currency per 2-3 hour session
- Low-skilled (part-time, manual): 40-80
- Medium-skilled (tutoring, retail): 80-120
- High-skilled (professional services): 120-200
- Vitality cost: -5 to -1
- Mood: Usually negative or neutral (-5 to +5)
- Esteem: +1 for fulfilling work

__VITALITY_DESC__

**Fulfillment Dimensions**:
__FULFILLMENT_DESC__

**Delta Ranges** (system will clip to these limits):
- vitality: __VITALITY_RANGE__
- fulfillment:
  - mood: __MOOD_RANGE__
  - esteem: __ESTEEM_RANGE__ (changes slowly, small shifts only)
- skills: __SKILLS_RANGE__ (use existing skill names when applicable)
- money: [0, __MONEY_MAX__] for earning; must be 0 for non-work activities (no spending here)
- gain_items: must be empty [] for non-consumption (items are only gained through purchase)

# Output Format

**For consumption events**:
{{
  "is_consumption_event": true
}}

**For non-consumption events**:
{{
  "outcome": "You spent 3 hours studying algorithms. The material was challenging but you made steady progress.",
  "is_consumption_event": false,
  "delta_vitality": -4,
  "delta_fulfillment": {{"mood": 2, "esteem": 1}},
  "delta_skills": {{"computer_science": 3}},
  "delta_money": 0,
  "gain_items": []
}}
"""
    # Replace config values (use double underscore to avoid conflicts)
    template = template.replace("__VITALITY_DESC__", VITALITY_DESC)
    template = template.replace("__FULFILLMENT_DESC__", FULFILLMENT_DESC)
    template = template.replace("__VITALITY_RANGE__", vitality_range)
    template = template.replace("__MOOD_RANGE__", mood_range)
    template = template.replace("__ESTEEM_RANGE__", esteem_range)
    template = template.replace("__SKILLS_RANGE__", skills_range)
    template = template.replace("__MONEY_MAX__", str(money_max))

    return template.strip()


def build_god_generate_offers_prompt() -> str:
    """Build prompt for stage 2: generate consumption offers for the agent.

    This prompt includes price data from worldview.json as reference.

    Returns prompt template with {agent_name}, {agent_info}, {agent_activity}, {worldview} placeholders.
    """
    from src.config import get_config

    config = get_config()

    # Load and format price list
    world_name = config["world"]["name"]
    price_data = _load_price_data(world_name)
    price_list = _format_price_list(price_data)

    # If price list loading fails, raise error
    if not price_list:
        raise FileNotFoundError(
            f"Failed to load price list for world '{world_name}'. "
            f"Expected 'prices' key in: data/{world_name}/worldview.json"
        )

    # Use plain string (not f-string) to avoid double formatting
    template = """You are the world model. Act as the shop for this consumption activity.

# Character Context

{agent_info}

# Intended Items

{agent_activity}

# Shop Worldview

{worldview}

# Price Reference

__PRICE_LIST__

# Guidelines 
1. Outcome message: Describe what the character encounters/sees in the consumption scenario
2. Match options to the character's expressed needs and current financial situation.
3. Provide variety in price points (low/medium/high options when applicable)
4. Use the price list as a reference, but adapt to the specific scenario
5. For non-standard items not in the price list, infer reasonable prices consistent with the price list's pricing standards.
6. Each option should have: name (concise), price (integer), description (brief)
7. Prices must be uniform for all characters - do not adjust based on individual wealth or economic status.
8. If the intended items are not appropriate or avaialble, return empty consumption_options.

# Task

Generate consumption scenario and options. If items are clearly inappropriate (see worldview), return empty consumption_options.

Output JSON format:
{{
  "outcome": "brief description of what character encounters (2-3 sentences)",
  "consumption_options": [
    {{"name": "item name", "price": integer, "description": "brief description"}},
    ...
  ]
}}

Generate 2-4 options with varied price points. Use price reference but adapt to scenario.
"""
    # Replace price list (loaded from config/file)
    template = template.replace("__PRICE_LIST__", price_list)

    return template.strip()


SETTLE_DISCARD_PROMPT = """## Weekly Settlement

You currently have too many possessions, which exceeds the maximum limit of {max_items} items.

## Your Current Possessions
{possessions}

## Task
You must discard at least {discard_count} items to reduce your possessions to a manageable level.

Choose items that are:
- Least important to you
- Least useful for your current goals
- Redundant or replaceable

## Output Format
List the item names you want to discard (one per line, use exact names from the list above):
```
item_name1
item_name2
...
```

Now choose which items to discard:
"""


def build_god_eval_joint_activity_prompt() -> str:
    """Build prompt for evaluating joint activity outcomes for all participants.

    Returns prompt template with {activity_background}, {participants_info}, {dialog_history} placeholders.
    """
    from src.config import get_config

    config = get_config()
    limits = config["world"]["joint_activity"]["delta_limits"]

    # Format delta ranges dynamically
    vitality_range = f"[{limits['vitality']['min']}, {limits['vitality']['max']}]"
    mood_range = f"[{limits['fulfillment']['mood']['min']}, {limits['fulfillment']['mood']['max']}]"
    social_range = f"[{limits['fulfillment']['social']['min']}, {limits['fulfillment']['social']['max']}]"
    esteem_range = f"[{limits['fulfillment']['esteem']['min']}, {limits['fulfillment']['esteem']['max']}]"
    skills_range = f"[{limits['skills']['min']}, {limits['skills']['max']}]"

    template = """You are the world model for a role-play simulation. Multiple characters have just finished a joint activity. Your task is to evaluate the outcome and determine state changes for each participant.

# Activity Context

{activity_background}

# Participants

{participants_info}

# Dialog History

{dialog_history}

# Task: Evaluate Outcomes for Each Participant

Based on the activity content and each participant's involvement, evaluate their state changes (deltas), including delta_vitality, delta_fulfillment (mood, social and esteem) and delta_skills.

__VITALITY_DESC__

**Fulfillment Dimensions**:
__FULFILLMENT_DESC__

**Delta Ranges** (system will clip to these limits):
- vitality: __VITALITY_RANGE__
- fulfillment:
  - mood: __MOOD_RANGE__
  - social: __SOCIAL_RANGE__ (ONLY for joint activities; reflects social bonding)
  - esteem: __ESTEEM_RANGE__ (changes slowly, small shifts only)
- skills: __SKILLS_RANGE__ (use existing skill names when applicable; only for learning activities)

**Important Notes**:
- Joint activities do NOT allow work or consumption. Do not evaluate delta_money.
- Deltas should reflect each character's actual involvement and personality
- Social fulfillment reflects bonding and connection.
- Each participant must have an outcome, even if they were less active.

- Mood depends on activity enjoyment and compatibility with other participants
- Esteem can increase if the person feels respected or accomplished during the activity

# Output Format

Return a JSON object with each participant's name as key:

{{
  "Alice": {{
    "delta_vitality": -2,
    "delta_fulfillment": {{"mood": 4, "social": 5, "esteem": 2}},
    "delta_skills": {{"cooking": 2}}
  }},
  "Bob": {{
    "delta_vitality": -3,
    "delta_fulfillment": {{"mood": 3, "social": 4, "esteem": 1}},
    "delta_skills": {{"cooking": 1}}
  }}
}}

"""
    # Replace config values
    template = template.replace("__VITALITY_DESC__", VITALITY_DESC)
    template = template.replace("__FULFILLMENT_DESC__", FULFILLMENT_DESC)
    template = template.replace("__VITALITY_RANGE__", vitality_range)
    template = template.replace("__MOOD_RANGE__", mood_range)
    template = template.replace("__SOCIAL_RANGE__", social_range)
    template = template.replace("__ESTEEM_RANGE__", esteem_range)
    template = template.replace("__SKILLS_RANGE__", skills_range)

    return template.strip()


# =============================================================================
# Public Activity Prompts
# =============================================================================


def build_god_generate_public_events_prompt(
    agent_summaries: str,
    previous_events: str,
    current_time: str,
    n_days: int,
    previous_events_weeks: int,
    min_events: int,
    max_events: int,
    max_repeat_weeks: int,
    world_setting: str,
) -> str:
    """Build the prompt for the God Model to generate Public Activities.

    A Public Activity is an "open-themed gathering": participants sign up
    individually and gather around a shared theme/purpose, but have no direct
    interaction with each other during the activity (unlike a Joint Activity).

    Args:
        agent_summaries: Brief summaries of all characters.
        previous_events: List of activities created over the past N weeks.
        current_time: Current time (Y{year}-W{week}).
        n_days: Number of days in the current week.
        previous_events_weeks: How many weeks of past activities to look back over.
        min_events: Minimum number of activities to generate.
        max_events: Maximum number of activities to generate.
        max_repeat_weeks: Maximum number of weeks an activity may repeat.

    Returns:
        The prompt string.
    """
    return f"""# Task: Generate Public Events for This Week

You are the system administrator creating public events for this week.

## World Setting
{world_setting}

## What is a Public Event?

Public events are **open-entry, theme-based gatherings** where:
- Participants sign up individually (they don't come with friends)
- Participants share a common interest/purpose but **do NOT interact directly** with each other during the activity
- Each person focuses on the activity itself, not on socializing

Examples by category:
- Learning: interest classes (cooking, painting, calligraphy), skill workshops, lectures
- Sports: running clubs, fitness classes, yoga sessions, hiking groups
- Hobbies: photography outings, movie screenings, art exhibitions
- Community: volunteer activities (cleanup, tree planting), charity events

Key distinction from Joint Activities:
- Joint Activity: friends plan together, interact directly
- Public Event: strangers join individually, parallel participation without interaction

**NOT allowed as Public Events:**
- Shopping activities
- Any event primarily about purchasing goods

## Current Time
{current_time}

## Characters in the World
{agent_summaries}

## Previous Events (Last {previous_events_weeks} Weeks)
{previous_events if previous_events else "No previous events."}

## Requirements

Generate {min_events}-{max_events} public events. Each event should:
1. Be appropriate for the world setting and character demographics
2. Be an activity where participants do the same thing in parallel (not interacting)
3. Have a clear theme/purpose (skill development, hobby, community service, etc.)
4. Not duplicate recent events (check previous events list)
5. Decide which characters are eligible to participate based on the event's nature

## Output Format

Return a JSON array of events:

```json
[
  {{
    "event_name": "Cooking Class",
    "start_day": 2,
    "repeat_weeks": 3,
    "eligible_participants": "all",
    "description": "Learn traditional cooking techniques. Suitable for anyone interested in culinary arts."
  }},
  {{
    "event_name": "Yoga Session",
    "start_day": 3,
    "repeat_weeks": 4,
    "eligible_participants": ["Alice", "Carol", "Diana", "Emma"],
    "description": "Relaxing yoga practice for women."
  }},
  {{
    "event_name": "Freshman Orientation",
    "start_day": 5,
    "repeat_weeks": 2,
    "eligible_participants": ["Bob", "Charlie", "David"],
    "description": "Orientation session for first-year students."
  }}
]
```

Field specifications:
- `event_name`: concise and descriptive (in the world's language)
- `start_day`: 1 to {n_days} (which day of the week, where 1 is the first day)
- `repeat_weeks`: 1 to {max_repeat_weeks} (1 = one-time event, >1 = repeats weekly on the same day)
- `eligible_participants`: who can participate and will be notified about this event
  - `"all"`: everyone in the world is eligible
  - `["Name1", "Name2", ...]`: only these specific characters are eligible (use exact names from the character list above)
- `description`: what participants will do
"""


PUBLIC_SIGNUP_PROMPT = """## Public Events Available This Week

{events_list}

## What is a Public Event?

Public events are open-entry, theme-based gatherings where participants:
- Sign up individually based on shared interests
- Focus on the activity itself (not on socializing with others)
- May notice other participants but do NOT interact directly with them

## Available Phases
When signing up, choose ONE phase (time slot) to attend the event:
- dawn
- morning
- afternoon
- dusk
- night

## Instructions

Decide which events you want to sign up for based on your interests, goals, personality, current schedule, energy level and commitments.

- To sign up for an event, output a JSON block with "type": "signup", "event_name" matching exactly one of the events listed above, and "phase" with your chosen phase. Example:
```
[ACTION_JSON]
{{"type": "signup", "event_name": "Morning Yoga Flow", "phase": "morning"}}
[/ACTION_JSON]
```
- You can sign up for multiple events (if no time conflicts) or none.
- For recurring events (weekly repeat), each signup only applies to THIS week. You will need to sign up again next week if you want to continue participating.
"""


def build_god_eval_public_activity_prompt() -> str:
    """Build prompt for evaluating public activity outcomes for all participants.

    Returns prompt template with {activity_name}, {event_description}, {participants_info},
    {participation_descriptions} placeholders.
    """
    from src.config import get_config

    config = get_config()
    limits = config["world"]["public_activity"]["delta_limits"]

    # Format delta ranges dynamically (no material for public activity)
    vitality_range = f"[{limits['vitality']['min']}, {limits['vitality']['max']}]"
    mood_range = f"[{limits['fulfillment']['mood']['min']}, {limits['fulfillment']['mood']['max']}]"
    social_range = f"[{limits['fulfillment']['social']['min']}, {limits['fulfillment']['social']['max']}]"
    esteem_range = f"[{limits['fulfillment']['esteem']['min']}, {limits['fulfillment']['esteem']['max']}]"
    skills_range = f"[{limits['skills']['min']}, {limits['skills']['max']}]"

    template = """You are the world model for a role-play simulation. Multiple characters have just finished a public activity. Your task is to evaluate the outcome and determine state changes for each participant.

# Activity Context

Activity Name: {activity_name}
Description: {event_description}

# Participants

{participants_info}

# What Each Participant Did

{participation_descriptions}

# Task: Evaluate Outcomes for Each Participant

Based on the activity content and each participant's involvement, evaluate their state changes (deltas), including delta_vitality, delta_fulfillment (mood, social and esteem) and delta_skills.

__VITALITY_DESC__

**Fulfillment Dimensions**:
__FULFILLMENT_DESC__

**Delta Ranges** (system will clip to these limits):
- vitality: __VITALITY_RANGE__
- fulfillment:
  - mood: __MOOD_RANGE__
  - social: __SOCIAL_RANGE__ (agents are in a group setting, even without direct interaction)
  - esteem: __ESTEEM_RANGE__ (can increase if the person feels accomplished)
- skills: __SKILLS_RANGE__ (use existing skill names when applicable; only for learning activities)

**Important Notes for Public Activities**:
- Participants DO NOT interact directly with each other - they focus on the activity itself.
- Social fulfillment comes from being in a shared environment with others who have similar interests, NOT from direct interaction.
- Deltas should reflect each character's actual participation and personality.

# Output Format

Return a JSON object with each participant's name as key:

{{
  "Alice": {{
    "delta_vitality": -2,
    "delta_fulfillment": {{"mood": 3, "social": 2, "esteem": 0.5}},
    "delta_skills": {{"cooking": 2}}
  }},
  "Bob": {{
    "delta_vitality": -1,
    "delta_fulfillment": {{"mood": 2, "social": 1, "esteem": 1}},
    "delta_skills": {{"cooking": 1}}
  }}
}}

"""
    # Replace config values
    template = template.replace("__VITALITY_DESC__", VITALITY_DESC)
    template = template.replace("__FULFILLMENT_DESC__", FULFILLMENT_DESC)
    template = template.replace("__VITALITY_RANGE__", vitality_range)
    template = template.replace("__MOOD_RANGE__", mood_range)
    template = template.replace("__SOCIAL_RANGE__", social_range)
    template = template.replace("__ESTEEM_RANGE__", esteem_range)
    template = template.replace("__SKILLS_RANGE__", skills_range)

    return template.strip()


PUBLIC_ACTIVITY_PROMPT = """## Instructions for a Public Activity

You are participating in "{event_name}".

{event_description}

{other_participants_block}

Important notes:
- This is a group activity where multiple people participate at the same time.
- You can observe others present, but there is NO direct interaction or conversation with them.
- You should specify your activity content. For example, where you position yourself, what you focus on, what techniques you practice.


## Output Format
Thinking: Your reasoning process as this character
Activity: Describe what you will do and how you will do it (max 100 words)
IMPORTANT: Output plain text only. Do NOT use markdown, bold (**), or any other formatting.
"""


EXIT_PUBLIC_ACTIVITY_PROMPT = """## After the Public Activity

The public activity has ended.

## Other Participants and Their Activities

{other_participants_activities}

## Instructions

Now that the activity has ended, please:

1. **Reflect** on the activity - what did you experience and learn?

2. **Optionally update scratchpads** - If you noticed someone interesting among the other participants listed above:
   - For someone NEW (not in your character scratchpads): call `update_scratchpad(s_name="characters/<person_name>", content="<your impression>", create_new_scratchpad=True)`
   - For someone you ALREADY KNOW: call `update_scratchpad(s_name="characters/<person_name>", content="<updated impression>", create_new_scratchpad=False)`
   - You can ONLY create/update scratchpads for people who participated in this activity with you.

Your final answer must strictly follow this exact format:

Reflection:
<Share your thoughts, feelings, and reflections on the activity>

IMPORTANT: Output plain text only. Do NOT use markdown, bold (**), or any other formatting.
Begin your response with "Reflection:" on the first line.
"""


# =============================================================================
#                         ENCOUNTER ACTIVITY PROMPTS
# =============================================================================


# =============================================================================
#                      POSITION APPLICATION PROMPTS
# =============================================================================


def build_position_application_wishes_prompt(
    positions: List["Position"],
    current_position: Optional[Dict[str, Any]] = None,
    forced_out: bool = False,
) -> List[Dict[str, str]]:
    """Build prompt for agent to express position application wishes.

    Args:
        positions: Age-filtered Position objects (aged-out positions hidden)
        current_position: Agent's current position info dict with keys:
            - name: position_id (organization/role)
            - weekly_income: int
            - weekly_delta_skills: Dict[str, int]
        forced_out: If True, agent must leave current position (age limit)

    Returns:
        List of message dicts for LLM
    """
    # Format ALL positions for the prompt
    positions_text = []
    for pos in positions:
        pos_info = f"- **{pos.name}**\n"
        pos_info += f"  - Type: {pos.type}\n"
        pos_info += f"  - Description: {pos.description}\n"
        pos_info += f"  - Weekly Income: {pos.weekly_income}\n"
        if pos.weekly_delta_skills:
            skills_str = ", ".join(
                f"{k}: +{v}" for k, v in pos.weekly_delta_skills.items()
            )
            pos_info += f"  - Skills Growth: {skills_str}\n"
        if pos.min_age is not None or pos.max_age is not None:
            age_req = []
            if pos.min_age is not None:
                age_req.append(f"min age {pos.min_age}")
            if pos.max_age is not None:
                age_req.append(f"max age {pos.max_age}")
            pos_info += f"  - Age Requirements: {', '.join(age_req)}\n"
        if pos.min_skills:
            skills_req = ", ".join(f"{k} >= {v}" for k, v in pos.min_skills.items())
            pos_info += f"  - Skill Requirements: {skills_req}\n"
        positions_text.append(pos_info)

    positions_list = "\n".join(positions_text)

    assert current_position, "All agents must have current position (REQ-18)"
    current_position_name = current_position["name"]

    # Build current position info with income and skills growth
    current_pos_details = f"**{current_position_name}**\n"
    current_pos_details += f"  - Weekly Income: {current_position['weekly_income']}\n"
    if current_position["weekly_delta_skills"]:
        skills_str = ", ".join(
            f"{k}: +{v}" for k, v in current_position["weekly_delta_skills"].items()
        )
        current_pos_details += f"  - Skills Growth: {skills_str}\n"

    # Current position info
    if forced_out:
        current_pos_info = f"""
## Your Current Position (No Longer Available)

You previously held:
{current_pos_details}
**However, you have aged out of this position and can no longer hold it.** You MUST apply to a new position.
"""
    else:
        current_pos_info = f"""
## Your Current Position

You currently hold:
{current_pos_details}
You can keep your current position using:
- `<STAY_CURRENT>` tag
- Or include your current position name directly

The `<STAY_CURRENT>` tag or current position name means "keep current job" - this is always successful and does not compete with others.

**Important**: `<STAY_CURRENT>` can be placed at ANY priority level (1st, 2nd, or 3rd choice), not just first.
- Example 1: `<wishes><STAY_CURRENT></wishes>` - only want to keep current job
- Example 2: `<wishes>Dream Company/Manager, <STAY_CURRENT>, Another Org/Role</wishes>` - try for dream job first, fall back to current if rejected
- Example 3: `<wishes>New Role A, New Role B, <STAY_CURRENT></wishes>` - try new opportunities, keep current as safety net
"""

    prompt = f"""## Yearly Position Application Season

It's the end of the year and time to apply for positions for next year. Below are ALL available positions in the world.
{current_pos_info}
## Available Positions

{positions_list}

## Note on Requirements

- Positions you have aged out of are not shown
- Some positions have age or skill requirements — positions above your current age are shown as future goals
- Positions where you don't fully meet skill requirements may still accept you if your skills are semantically similar (e.g., "Teaching" is similar to "Education")
- Consider positions where you meet OR are close to meeting requirements
- Apply to positions that match your interests, even if you don't meet all requirements - use this as motivation to grow

## Instructions

Based on your personality, skills, career goals, interests, and life situation, choose up to 3 positions you want to apply for, in order of preference (first choice = most preferred).

Consider:
- Does this position match your skills and interests?
- Does the income meet your needs?
- Will this position help you grow in ways you want?
- Is this position aligned with your long-term goals?

## Output Format

Provide your choices using the EXACT position_id (format: "organization/role"):

<wishes>Organization1/Role1, Organization2/Role2, Organization3/Role3</wishes>

For example: <wishes>Fudan High School/English Teacher, City Library/Librarian, Community Center/Admin</wishes>

You may list fewer than 3 if you don't find 3 suitable positions.

Before giving your wishes, briefly explain your reasoning for each choice.
"""

    return [{"role": "user", "content": prompt}]


def build_god_evaluate_position_application_prompt(
    round_num: int,
    positions: Optional[List["Position"]],
    candidates: List[Dict],
    wishes: Optional[Dict[str, List[str]]],
    sub_round: int = 1,
) -> str:
    """Build prompt for God Model to evaluate position application (Round 1 only).

    Note: Round 2 has been removed. All agents have original positions
    and fallback to them if not matched in Round 1.

    Args:
        round_num: Must be 1 (wish round)
        positions: List of positions being evaluated (Round 1 batch)
        candidates: List of candidate info dicts
        wishes: Dict mapping agent_name to position_id wishes
        sub_round: Which wish is being evaluated (1, 2, or 3)

    Returns:
        Prompt string
    """
    if round_num != 1:
        raise ValueError(f"Only round_num=1 is supported, got {round_num}")

    # Round 1: Evaluate candidates for multiple positions (batch)
    # Build positions section
    positions_text = []
    for pos in positions:
        # Get applicants for this position
        applicants = [
            c["name"]
            for c in candidates
            if wishes and pos.name in wishes.get(c["name"], [])
        ]

        pos_info = f"### {pos.name}\n"
        pos_info += f"- Type: {pos.type}\n"
        pos_info += f"- Description: {pos.description}\n"
        pos_info += f"- Weekly Income: {pos.weekly_income}\n"
        pos_info += f"- Available Slots: {pos.available_slots()}\n"

        # Show requirements
        requirements = []
        if pos.min_age is not None:
            requirements.append(f"min_age={pos.min_age}")
        if pos.max_age is not None:
            requirements.append(f"max_age={pos.max_age}")
        if pos.min_skills:
            skills_str = ", ".join(f"{k}>={v}" for k, v in pos.min_skills.items())
            requirements.append(f"min_skills={{{skills_str}}}")
        if requirements:
            pos_info += f"- Requirements: {', '.join(requirements)}\n"

        if applicants:
            pos_info += f"- Applicants: {', '.join(applicants)}\n"
        positions_text.append(pos_info)

    positions_list = "\n".join(positions_text)

    # Build candidates section
    candidates_text = []
    for c in candidates:
        candidates_text.append(
            f"- **{c['name']}**\n"
            f"  - Age: {c['age']}\n"
            f"  - Skills: {c['skills']}\n"
            f"  - Brief: {c['brief']}"
        )

    candidates_list = "\n".join(candidates_text)

    return f"""# Task: Evaluate Candidates for Multiple Positions (Sub-round {sub_round})

You are evaluating candidates for multiple positions. This is sub-round {sub_round}, evaluating candidates' {sub_round}{"st" if sub_round == 1 else "nd" if sub_round == 2 else "rd"} choice wishes.

## Positions to Fill

{positions_list}

## Candidates

{candidates_list}

## Evaluation Criteria

1. **Age Eligibility (HARD CONSTRAINT)**: If a position has min_age or max_age, candidates outside that range MUST NOT be selected. This is non-negotiable.

2. **Skill Match**: Does the candidate have relevant skills for the position?
   - IMPORTANT: Semantic similarity counts! Skills with similar meanings should be considered equivalent.
   - Examples: "Teaching" ≈ "Education", "Communication" ≈ "Presentation", "Programming" ≈ "Coding"

3. **Requirements**: Check skill requirements flexibly
   - Be lenient on exact skill name matches - semantic similarity is acceptable
   - A candidate with "Education: 60" meets "Teaching >= 50" requirement

4. **Fit**: Is the candidate a good overall fit for this role?

## Important Notes

- This is sub-round {sub_round}, meaning this position is each applicant's {sub_round}{"st" if sub_round == 1 else "nd" if sub_round == 2 else "rd"} choice
- A candidate can only be selected for ONE position in this batch
- If a candidate is not qualified for any position, omit them from results
- Slots may remain unfilled if no qualified candidates

## Output Format

Return your decisions as a JSON object mapping position_id to selected candidate names:

```json
{{
  "Organization1/Role1": ["Alice", "Bob"],
  "Organization2/Role2": ["Charlie"],
  "Organization3/Role3": []
}}
```

Include ALL positions in the output, even if empty (no qualified candidates).
"""


def build_god_design_positions_prompt(
    agents_info: str,
    world_setting: str,
    min_capacity: int,
    max_capacity: int,
    income_min: int,
    income_max: int,
    max_work_capacity: int,
    age_min: int,
    age_max: int,
    existing_positions_info: str = "",
    min_position_count: int = 10,
    max_position_count: int = 10,
) -> str:
    """Build prompt for God Model to design positions for the world.

    This is called when positions.json doesn't exist, to generate initial positions
    based on the world setting and agent profiles.

    Args:
        agents_info: Summary of all agents (name, age, skills, brief)
        world_setting: Description of the world setting
        min_capacity: Minimum total capacity (sum of all positions' capacity)
        max_capacity: Maximum total capacity
        income_min: Minimum weekly income for work positions
        income_max: Maximum weekly income for work positions
        max_work_capacity: Maximum capacity per work position (ensures job diversity)
        age_min: Reference age minimum (most characters are this age or older)
        age_max: Reference age maximum (most characters are within this range)
        existing_positions_info: Info about existing positions from agents' profiles
        min_position_count: Minimum number of distinct position types
        max_position_count: Maximum number of distinct position types

    Returns:
        Prompt string
    """
    # Calculate income tiers for prompt guidance
    income_mid = income_min + (income_max - income_min) // 3
    income_high = income_min + (income_max - income_min) * 2 // 3
    non_work_income_max = income_max // 2  # Non-work income cap: half of work max

    # Build existing positions section if provided
    if existing_positions_info:
        existing_positions_section = f"""## Existing Positions (For Reference)

The following positions already exist and will be automatically included:

{existing_positions_info}

Your task is to design **NEW positions only**. The system will merge these existing positions automatically.

"""
    else:
        existing_positions_section = ""

    return f"""# Task: Design NEW Positions for a Simulated World

You are the world designer creating **new** positions (jobs/roles) for characters in a simulated world.

## World Setting

{world_setting}

## Characters in This World

{agents_info}

{existing_positions_section}## Age Reference

Most characters in this world are between {age_min} and {age_max} years old, but some positions (e.g., teachers, staff) may require older ages.

## Task

Design positions (job types/roles) for this world.

**Constraints**:
- **Number of position types**: You must create between **{min_position_count}** and **{max_position_count}** distinct positions
- **Total capacity** (sum of all positions' capacity): must be between {min_capacity} and {max_capacity}

Requirements:
1. **Position count**: Create at least {min_position_count} distinct position types (no more than {max_position_count})
2. **Fit the world setting**: Positions should make sense in this world
3. **Sufficient total capacity**: The sum of all positions' capacity must be at least {min_capacity}
4. **Vary in requirements**: Different positions should have different skill/age requirements
5. **Include both work and non-work**: Work positions have income; non-work (e.g., student) may also have income (e.g., allowance, stipend)
6. **Realistic capacity per position**: Some positions may have capacity=1 (e.g., principal), others may have capacity=20 (e.g., student)

## Position Structure

Each position should have:
- `organization`: Organization name (e.g., "Fudan High School")
- `role`: Role name within the organization (e.g., "English Teacher")
- **Unique ID (position_id)**: "organization/role" must be unique across all positions
- `type`: "work" or "non-work"
- `description`: Brief description of the role
- `weekly_income`: Income per week ({income_min}-{income_max} for work; 0-{non_work_income_max} for non-work)
- `weekly_delta_skills`: Skills gained per week (e.g., {{"teaching": 5, "communication": 2}})
- `min_age` (optional): Minimum age requirement
- `max_age` (optional): Maximum age requirement
- `min_skills` (optional): Minimum skill requirements (e.g., {{"teaching": 50}})
- `capacity`: How many agents can hold this position

## Guidelines

1. **Position count**: Create {min_position_count}-{max_position_count} distinct positions to ensure variety
2. **Total capacity** (sum of all capacity values) must be >= {min_capacity}
3. **Income distribution** (IMPORTANT):
   - Income range: {income_min}-{income_max} per week
   - **Most positions should pay {income_min}-{income_mid} per week** (entry-level, common jobs)
   - Only a **few high-skill/high-responsibility positions** should pay {income_high}-{income_max} (e.g., principal, senior manager)
   - This creates a realistic income hierarchy where reaching max income requires significant qualifications
4. **Non-work positions**: income range 0-{non_work_income_max} (e.g., student allowance, stipend)
5. **Skill growth should be modest**: +1 to +5 per skill per week
6. **Age restrictions**: Most characters are aged {age_min}-{age_max}, but some positions (e.g., teachers) may require older ages
7. **Capacity per work position**: Each work position's capacity should be <= {max_work_capacity} to ensure job diversity
8. **Organization-Role format**: Use separate `organization` and `role` fields; the combination "organization/role" must be unique

## Position Diversity (CRITICAL)

**Positions must reflect realistic diversity in life paths.** Design positions that represent the full spectrum of how characters might develop, based on the world setting. Consider:

1. **Career Trajectories**: Not everyone follows the same path. Include positions for:
   - Those pursuing further education/training
   - Those joining established organizations/institutions
   - Those in leadership/elite roles (rare, high-skill)
   - Those in common/everyday roles (majority)
   - Those in service/support roles

2. **Socioeconomic Diversity**: Create a realistic distribution:
   - Elite/prestigious positions (few slots, high requirements)
   - Skilled professional positions (moderate slots)
   - Common/entry-level positions (many slots, low requirements)
   - The income and skill requirements should form a realistic pyramid

3. **Organization Variety**: Characters should work in diverse types of organizations appropriate to this world setting. Avoid concentrating all positions in one or two organizations.

4. **Skill Path Diversity**: Different positions should develop different skill combinations, allowing characters to specialize in various directions.

5. **Encourage Competition**: For popular/elite positions, set **high entry barriers** (min_skills, min_age) and **limited slots** (low capacity). This motivates characters to continuously improve their abilities or accumulate career experience to qualify. Elite positions should feel earned, not given.

6. **Design Aspirational Positions**: Include some **challenging positions** that most characters cannot enter initially. These serve as long-term goals that characters can aspire to, gradually building skills over time to eventually qualify. Examples: leadership roles requiring years of experience, prestigious organizations with strict skill requirements.

7. **Support Different Life Philosophies**: Not everyone seeks elite positions. Design a range of positions that support different character motivations:
   - **Ambitious path**: High-competition, high-reward positions for those seeking achievement
   - **Balanced path**: Moderate positions offering stability and work-life balance
   - **Simple path**: Low-stress positions for those prioritizing life quality over career advancement

   The position system should allow characters to thrive whether they choose to compete fiercely or pursue a quieter, fulfilling life.

**Important**: Adapt these concepts to fit the specific world setting. A medieval fantasy world might have guilds, apprenticeships, and noble courts instead of modern institutions. A sci-fi world might have space stations, research colonies, and military academies. Design positions that are **thematically appropriate** to your world.

## Output Format

Return a JSON array of positions:

```json
[
  {{
    "organization": "Fudan High School",
    "role": "English Teacher",
    "type": "work",
    "description": "Responsible for high school English teaching",
    "weekly_income": 250,
    "weekly_delta_skills": {{"teaching": 5, "communication": 2}},
    "min_age": 22,
    "min_skills": {{"teaching": 50}},
    "capacity": 2
  }},
  {{
    "organization": "Fudan High School",
    "role": "Student",
    "type": "non-work",
    "description": "Student studying at high school",
    "weekly_income": 50,
    "weekly_delta_skills": {{"study": 10, "social": 3}},
    "min_age": {age_min},
    "max_age": 18,
    "capacity": 50
  }}
]
```
"""


def build_god_grow_positions_prompt(
    world_setting: str,
    existing_positions_info: str,
    agent_skills_info: str,
    count: int,
    age_min: int,
    age_max: int,
    income_max: int,
) -> str:
    """Build prompt for God Model to generate challenging new positions.

    Called at the start of each year (except the first) to add new positions
    that serve as growth targets for agents.

    Args:
        world_setting: Description of the world setting
        existing_positions_info: Summary of current positions
        agent_skills_info: Current skill distribution across all agents
        count: Number of new positions to generate
        age_min: Reference age minimum (most characters start around this age)
        age_max: Reference age maximum (most characters are within this range)
        income_max: Maximum weekly income (from config)

    Returns:
        Prompt string
    """
    return f"""# Task: Design NEW Challenging Positions

You are expanding the position system with **challenging positions** that serve as growth targets for characters.

## World Setting

{world_setting}

## Existing Positions

{existing_positions_info}

## Current Agent Skills Distribution

This shows the highest skill level for each skill across all agents:

{agent_skills_info}

## Age Reference

Most characters in this world are aged {age_min}-{age_max}, but some positions may require older ages.

## Task

Design exactly **{count}** NEW challenging **work** position(s).

## Requirements

### 1. Challenge Gap (CRITICAL)
Each new position's min_skills must be **at least 50 points higher** than the current highest agent skill for that skill type.

Example: If highest agent "teaching" = 80, then min_skills["teaching"] >= 130.

**Note**: You may introduce NEW skill types not listed above. For new skills, the threshold is 50 (since current max = 0).

### 2. Skill Cap
No skill requirement can exceed **500**.

### 3. Income Formula
Income is proportional to total skill requirements:
- `sum_skill = sum of all min_skills values`
- Income range: `[sum_skill * 0.2, min(sum_skill, {income_max})]`
- Maximum income is **{income_max}** per week

### 4. Scarcity
Capacity must be **1 or 2** (elite positions are rare).

### 5. Work Type
Must be `type: "work"`.

### 6. Unique Names
"organization/role" must NOT duplicate any existing position.

### 7. Skill Growth
Challenging positions should offer higher skill growth: **weekly_delta_skills values of 5-10**.

## Output Format

Return a JSON array of exactly {count} position(s):

```json
[
  {{
    "organization": "Elite Academy",
    "role": "Head Teacher",
    "type": "work",
    "description": "Senior teaching position requiring exceptional skills",
    "weekly_income": 300,
    "weekly_delta_skills": {{"teaching": 8, "leadership": 5}},
    "min_age": 25,
    "min_skills": {{"teaching": 200, "communication": 100}},
    "capacity": 1
  }}
]
```
"""


# =============================================================================
#                         YEARLY PROFILE UPDATE PROMPTS
# =============================================================================

# Fields that must NOT be updated (copied from previous year)
PROFILE_IMMUTABLE_FIELDS = [
    "name",
    "birthday",
    "birth_year",
    "gender",
    "position",  # position contains: organization, role, weekly_income, weekly_delta_skills
    "init_skills",
    "init_assets",
    "extra_income",  # non-work income (family support, investments, etc.)
]

# Fields that can be updated based on yearly experiences
PROFILE_UPDATABLE_FIELDS = [
    "brief_introduction",
    "appearance_and_impression",
    "personality_traits",
    "core_motivation",
    "conflicts",
    "values",
    "preferences",
    "details",
    "talents",
]


def build_god_yearly_profile_update_prompt(
    agent_name: str,
    current_profile: Dict,
    yearly_summaries: str,
    current_year: int,
    next_year: int,
) -> str:
    """Build prompt for GodModel to update character profile at year end.

    Args:
        agent_name: Name of the character
        current_profile: Current year's complete profile dict
        yearly_summaries: All weekly summaries from the past year
        current_year: Current year number
        next_year: Next year number

    Returns:
        Prompt string for profile update
    """
    # Format current profile as JSON for LLM
    profile_json = json.dumps(current_profile, ensure_ascii=False, indent=2)

    return f"""# Task: Update {agent_name}'s Profile for Year {next_year}

You are the world model. A year has passed in the simulation. Based on {agent_name}'s experiences throughout Year {current_year}, update their profile for the upcoming Year {next_year}.

## Current Profile (Year {current_year})

```json
{profile_json}
```

## {agent_name}'s Year in Review

The following are {agent_name}'s weekly summaries from Year {current_year}:

{yearly_summaries}

## Update Rules

### Fields that MUST NOT change (will be copied automatically):
- `name`, `birthday`, `birth_year`, `gender` - immutable personal facts
- `position` - job/role changes are handled by the position application system
- `init_skills`, `init_assets`, `extra_income` - tracked in state.jsonl

### Fields to update:

1. **appearance_and_impression**: Physical appearance changes very slowly over a year. Keep the core features (height, weight, facial features) intact. Only allow subtle aging-related changes or changes clearly indicated by the year's events (e.g., gained/lost weight if explicitly mentioned).

2. **personality_traits**:
   - `qualitative`: Update the narrative description based on personality evolution
   - `quantitative`: Each value can change by **at most ±5** per year. All values must stay in [0, 100].
     - IMPORTANT: Fields vary per character (e.g., some have `extraversion`, others `introversion`). Keep the same fields as the current profile unless the year's experiences clearly justify adding a new trait.

3. **talents**:
   - `qualitative`: Update based on developed or neglected talents
   - `quantitative`: Each value can change by **at most ±3** per year. All values must stay in [0, 100].
     - IMPORTANT: Fields vary per character. Keep the same fields as the current profile unless the year's experiences clearly justify adding a new talent.

4. **Other updatable fields**:
   - `brief_introduction`: Update to reflect growth/changes
   - `core_motivation`: May shift based on experiences
   - `conflicts`: Update internal/external conflicts
   - `values`: Beliefs may evolve gradually
   - `preferences`: Interests and lifestyle may change
   - `details`: Add significant events from this year to the character's history

## Output Format

Return a JSON object containing ONLY the updatable fields. The system will merge this with immutable fields.

```json
{{
  "brief_introduction": "...",
  "appearance_and_impression": "...",
  "personality_traits": {{
    "qualitative": "...",
    "quantitative": {{
      // Keep the SAME fields as current profile, e.g.:
      "extraversion": 75,  // or "introversion" if current has that
      "intuition": 75,
      "feeling": 35,       // or "thinking" if current has that
      "judging": 85,       // or "perceiving" if current has that
      "confidence": 75,
      "curiosity": 75,
      "empathy": 80,
      "responsibility": 95,
      "control": 70,
      "patience": 85
    }}
  }},
  "core_motivation": "...",
  "conflicts": "...",
  "values": "...",
  "preferences": "...",
  "details": "...",
  "talents": {{
    "qualitative": "...",
    "quantitative": {{
      // Keep the SAME fields as current profile
      "intelligence": 70,
      "creativity": 40,
      "leadership": 60,
      "communication": 50,
      "health": 60,
      "beauty": 70,
      "trustworthiness": 60,
      "honesty": 70,
      "integrity": 70
    }}
  }}
}}
```

**Important**:
- Keep the SAME quantitative fields as the current profile
- Only add new fields if the year's experiences strongly justify it
- Changes should be gradual and justified by the year's experiences
- Do NOT invent events not mentioned in the summaries
- Keep the character's core identity intact while allowing natural evolution
- Note: The system will automatically clip values that exceed delta limits (±5 for personality, ±3 for talents)
"""


# =============================================================================
# Social Ranking Prompt (for Social Reward calculation)
# =============================================================================

SOCIAL_RANKING_PROMPT = """## Private Social Evaluation

This is a **COMPLETELY PRIVATE** evaluation that **NO OTHER CHARACTER WILL EVER SEE**.
Be completely honest about your true feelings and judgments.

## People You Know

The following people are those you have interacted with or know about:

{known_names_list}

## Your Recent Interactions

{recent_interactions}

{delta_summary}

## Your Task

Score each person you know on TWO separate dimensions. Give each person a score from 0 to 100.

### Affection
How much you **personally like** them as a person.
Consider: emotional closeness, enjoyment of their company, personal affinity, comfort around them.

### Respect
How much you **admire** their abilities and character.
Consider: competence, accomplishments, reliability, wisdom and judgment.

## Scoring Scale

- **10** = extreme dislike / contempt
- **30** = dislike / low regard
- **50** = neutral baseline — neither good nor bad, the objective midpoint
- **70** = like / respect
- **90** = deep affection / great admiration

**Evaluate each person independently** against the 50 baseline. Do NOT compare people against each other.
Most acquaintances should fall in the 40-60 range. Only those you have genuinely strong feelings about should deviate significantly.

**Important Notes:**
- Affection and respect scores can be DIFFERENT for the same person
- This is YOUR honest personal view - there's no "right" answer
- Do NOT include yourself ({agent_name}) - you are scoring OTHERS only

## Output Format

Return your scores as JSON:

```json
{{
  "affection": {{"person_name": score, "another_person": score, ...}},
  "respect": {{"person_name": score, "another_person": score, ...}}
}}
```
"""


def build_social_ranking_prompt(
    agent_name: str,
    known_names: List[str],
    recent_interactions: str,
    delta_summary: str = "",
) -> List[Dict[str, str]]:
    """Build prompt for social ranking generation.

    Args:
        agent_name: Name of the agent doing the ranking (for self-exclusion reminder)
        known_names: List of people this agent knows
        recent_interactions: Summary of recent interactions with known people
        delta_summary: Scratchpad affection/respect delta totals for each person

    Returns:
        List of message dicts (to be appended to roleplay_prompt)
    """
    known_names_list = "\n".join(f"- {name}" for name in sorted(known_names))

    content = SOCIAL_RANKING_PROMPT.format(
        agent_name=agent_name,
        known_names_list=known_names_list,
        recent_interactions=recent_interactions
        if recent_interactions
        else "No recent interactions recorded.",
        delta_summary=(
            f"## Your Interaction History (recorded deltas)\n\n"
            f"The numbers below are affection/respect changes from your encounters "
            f"this year. Use them as reference, not as final scores.\n\n"
            f"{delta_summary}"
        ) if delta_summary
        else "No interaction history recorded yet.",
    )

    return [{"role": "user", "content": content}]


# ═══════════════════════════════════════════════════════════
# Post-Encounter Follow: 偶遇后跟随决策 prompt
# ═══════════════════════════════════════════════════════════

# 随同关系变化标记
POST_ENCOUNTER_FOLLOW_INSTRUCTION = """You are a social reasoning AI. Based on the conversation and upcoming schedules below, decide:

1. Did anyone express intent to follow or join another person's activity?
2. If so, what's the appropriate schedule change?
3. Who needs to be notified?

Output ONLY valid JSON with no markdown formatting:
{
    "follow_intent": {"agent": "who is following", "target": "who they're following", "type": "solo/joint/null"},
    "needs_change": true/false,
    "schedule_updates": [
        {"agent": "name", "type": "solo/joint", "location": "place", "participants": ["list", "of", "participants"]}
    ],
    "notifications": [{"from": "who", "to": "who", "message": "what to say"}],
    "description": "Short human-readable summary of what happens next"
}

Rules:
- If the conversation mentions "一起" or "join" or "加" → follow_type = "joint"
- If someone follows but does their own thing → follow_type = "solo"  
- If the followed person already has a joint plan, new participants can join (no limit)
- Notifications: only need to notify people who weren't in the original conversation
- If no follow intent was expressed, set needs_change = false"""


def build_post_encounter_follow_prompt(
    dialogue_history: str,
    schedule_lines: str,
) -> str:
    """Build the Post-Encounter Follow decision prompt.

    Args:
        dialogue_history: Full dialogue text from the encounter.
        schedule_lines: Each participant's upcoming schedule, formatted as lines.

    Returns:
        List of message dicts for the LLM call.
    """
    content = f"""## Recent Conversation

{dialogue_history}

## Upcoming Schedules

{schedule_lines}

## Decision
{POST_ENCOUNTER_FOLLOW_INSTRUCTION}"""
    return [{"role": "user", "content": content}]


# ── F2+F3: Post-Encounter Effects + Rumor Injection ─────────────────

POST_ENCOUNTER_EFFECTS_PROMPT = """You are analyzing a conversation that just happened. 
Extract ALL of:
1. Facts worth spreading as rumors (who did what, relationships, secrets, events)
2. What a bystander would casually overhear (filtered, vague summary of key points)
3. How each participant's feelings toward others changed during this conversation

Output a JSON object:
{
  "rumors": [
    {"content": "fact description", "topic": "activity|relationship|gossip|event|secret", "tags": ["keyword1", "keyword2"], "fidelity": 0.8}
  ],
  "perception": "1-3 sentence summary of what a bystander would overhear",
  "affection_deltas": [
    {"from": "speaker_name", "to": "listener_name", "delta": 3, "reason": "brief reason"}
  ],
  "respect_deltas": [
    {"from": "speaker_name", "to": "listener_name", "delta": 2, "reason": "brief reason"}
  ]
}

Rules:
- Rumors: factual, concise, one sentence each. 1-3 rumors total.
- Rumor fidelity (0.0-1.0): 1.0=firsthand witness, 0.7=heard from someone, 0.5=vague rumor. Default 1.0 if omitted.
- Perception: ignore pleasantries and small talk. Only what a passerby would notice and remember.
- Be vague about exact phrasing in perception — you only caught fragments.
- Affection deltas: integer between -10 and +10. +5 for warm/generous behavior, -5 for rude/cold, +2 for pleasant chat, -2 for disagreement.
- Respect deltas: integer between -10 and +10. +5 for impressive skill/intelligence shown, -5 for incompetence/dishonesty, +2 for competence, -2 for lack of judgment.
- Only include pairs that had meaningful interaction for each delta type.
- Do NOT speculate beyond what was actually said.
- Output ONLY valid JSON, no markdown or other text."""

RUMOR_INJECTION_BLOCK = """## Rumors You've Heard Recently

Here are rumors circulating in the world that you've recently heard or witnessed:

{rumor_lines}

Consider these rumors when planning your activities — they may influence your decisions."""
