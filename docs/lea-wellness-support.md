# LEA Wellness / Health Journey Support (earlier roadmap note)

**Status:** Lightweight roadmap history — **superseded for implementation** by  
[`docs/lea-health-weight-journey.md`](lea-health-weight-journey.md)  
(LEA Health / Weight Journey blueprint for **Next.js / TypeScript DreAgent**, Phase A foundation focus).

**Do not implement** from this file alone. Use the blueprint doc for architecture, privacy, Phase A scope, and branch plan.

**Relationship to email / Smart LEA v1:** Separate capability area. Do **not** couple wellness/health work to mail/calendar providers, Graph writes, Gmail/IMAP/SMTP, or conversation persistence.

---

## Purpose (historical)

LEA should help with a healthy weight and nutrition journey—planning, tracking support, accountability, and sustainable habits—not medical care.

## Planning context (non-clinical; design only)

High-level goals discussed for personalization design (store in **private, user-scoped** profile at implementation—not as the only source of truth in public code):

- Significant healthy weight loss over time (~100 lb class goal from a high start weight).
- **~180 lb** as an important satisfying **major milestone**.
- **~155 lb** as a later/stretch goal for some users.
- Mostly sedentary lifestyle with regular short treadmill walks.
- Sustainable eating change and calorie awareness without shame.

This context must **never** be treated as hard-coded clinical diagnosis data. Implementation uses opt-in private profile fields with consent.

## Capabilities (still future; see blueprint)

- Meal planning and realistic food fits  
- Grocery planning  
- Calorie and activity tracking support  
- Weekly progress summaries  
- Walking/activity habits  
- Pattern spotting from logs  
- Consistency without shame  
- Questions to prepare for a clinician when appropriate  

Full phased plan: **[`docs/lea-health-weight-journey.md`](lea-health-weight-journey.md)**.

## Health guardrails (still required)

- LEA is **not** a doctor and must not present as one.  
- No crash diets or extreme restriction recommendations.  
- Prefer sustainable pace; surface medical follow-up for concerning symptoms or aggressive restriction requests.  
- No shame language.  
- Treat health as executive-style support: plan, track, remind, organize.  

## Architecture note (updated)

Implement under DreAgent patterns (e.g. future `src/lib/health/*`), independent of:

- Multi-provider email (Outlook / Gmail)  
- Calendar write actions from chat  
- Incentives / Grant specialty research tools  

Share only generic LEA patterns (truthfulness, structured context, user-scoped storage once designed safely).

## Earlier suggested steps (replaced by phased blueprint)

1. Docs + safety (this note → now **blueprint** docs)  
2. Optional structured log capture with privacy review  
3. Weekly summary + gentle nudges  
4. Clinician question prep templates  

**Implementation order:** follow **`docs/lea-health-weight-journey.md`** Phase A → B → …, not this list.

---

*Planning branch for the blueprint: `feat/lea-health-weight-journey-plan`.*
