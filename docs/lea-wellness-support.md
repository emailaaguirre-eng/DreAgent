# LEA Wellness / Health Journey Support (deferred)

**Status:** Roadmap only — **not implemented** in Smart LEA v1.  
**Suggested future branch:** `feat/lea-wellness-support`  
**Relationship to email work:** Separate capability area. Do **not** couple wellness to mail/calendar providers.

---

## Purpose

LEA should eventually help Dre with a healthy journey to lose weight and eat better—executive-style planning, tracking, reminders, and accountability—not medical care.

## Known user context (for future personalization design)

- Goal: significant healthy weight loss (~100 lbs total; ~180 lbs as a satisfying milestone).
- Prior stats shared for planning context: about 255 lbs, 5'6", age 50, female; mostly sedentary with treadmill walks about 4–5 times per week.
- Preferences: sustainable dietary change, health-conscious calorie tracking, progress trendlines, support as weight changes.

This context must **never** be hard-coded as clinical diagnosis data. Future implementation should use opt-in profile/preferences with clear consent.

## Capabilities (future)

LEA Wellness should help with:

- Meal planning and healthier swaps
- Grocery planning
- Calorie and protein tracking support
- Weekly progress summaries
- Walking/activity reminders
- Habit tracking
- Water / fiber / protein nudges
- Spotting patterns from logs
- Encouraging consistency **without shame**
- Helping prepare questions for a doctor or dietitian when appropriate

## Health guardrails (required)

- LEA is **not** a doctor and must not present as one.
- No dangerous crash-diet or extreme restriction advice.
- Prefer safe, sustainable weight-loss pacing; recommend medical guidance for extreme calorie restriction, medical conditions, medication questions, dizziness, fainting, chest pain, disordered eating signs, or rapid unexplained weight changes.
- Avoid shame-based language.
- Treat wellness as **executive support**: planning, tracking, reminders, accountability, organization.

## Architecture note

Implement as its own module/capability (e.g. mode, tools, or package under `src/lib/wellness/`), independent of:

- Multi-provider email (Outlook / Gmail)
- Calendar providers
- Incentives / Grant research

Share only generic LEA patterns (truthfulness, structure, mild persistence once ownership is fixed).

## Suggested implementation phases (later)

1. Docs + safety prompt addendum for wellness conversations only  
2. Optional structured log capture (meals, weight, activity) with privacy review  
3. Weekly summary + gentle nudges  
4. Doctor/dietitian question prep templates  

**Do not start coding wellness until a dedicated branch is approved.**
