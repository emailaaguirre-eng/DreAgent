# LEA Health / Weight Journey

## Blueprint for Next.js / TypeScript DreAgent

**Status:** Planning / docs only — **not implemented**.  
**Planning branch:** `feat/lea-health-weight-journey-plan`  
**Implementation branch (future):** `feat/lea-health-phase-a-foundation`  
**Base main reference:** Smart LEA v1 merge `3752504168f2f986fcf0419c2cfe82624e3aa044`

This document is the **approved architectural blueprint** for Health / Weight Journey on the **Next.js / TypeScript DreAgent** product surface. It adapts the master Health, Weight & Nutrition specification to the current repo.  
For implementable Phase A detail, see **§ Phase A** below.

**Supersedes for implementation:** [`docs/lea-wellness-support.md`](lea-wellness-support.md) (kept as the earlier lightweight roadmap note).

---

## 1. Product principles (non-negotiable)

1. **Safety first** when requirements compete.
2. LEA is **not** a physician, endocrinologist, or registered dietitian.
3. **Celiac / gluten-free** is a **hard medical constraint**, not a lifestyle preference.
4. **Thyroid / no-thyroid** context requires careful handling; never medication or dose advice.
5. Thyroid-related metabolism / lab concerns should route to **clinician consult** using a **private profile clinician label** when set (not generic only).
6. **No shame-based** language; no compensatory punishment (skip meals, “burn it off”).
7. Weight loss guidance must prefer **sustainable** pacing.
8. **1,200 calories/day** is a **software safety floor** for this product’s guardrail (not presented as a universal medical rule for every person).
9. **1,750 calories/day** is the working target seed; it must **not** be silently lowered when weight drops.
10. **Any** calorie target change requires **explicit user approval** and an **audit history** row.
11. **Deterministic code** is the source of truth for REE, TDEE estimates, scenarios, progress math, and floors—not the LLM inventing numbers.
12. The LLM **explains** structured engine results and helps plan; it is **not** the calculator.
13. Do **not** dump full health history into the model every turn; pass **minimized structured snippets** only.
14. Health data is **sensitive**: private, **user-scoped**, minimize logs, no third-party health analytics without explicit approval.
15. **Do not commit** private lab results, medication dosages, or full personal food diaries into public repo docs or fixtures.
16. **Phase A and this branch do not change Smart LEA v1 runtime** (chat/mail/calendar behavior).

---

## 2. Architecture correction

| Concept | Reality for this product |
|--------|---------------------------|
| Primary LEA app | **Next.js / TypeScript DreAgent** (`src/app`, `src/lib`, App Router APIs) |
| Not primary | Older private Python / FastAPI / SQLite concepts as the main surface |
| Standalone tracker | PyQt desktop `health_tracker` is **reference and optional import source only** |
| Runtime dependency | **No PyQt**, no second health microservice |

Do **not** couple health to:

- Gmail / IMAP / SMTP  
- Graph send / calendar create from chat  
- Multi-provider mail abstraction  
- Conversation persistence UI (known ownership risks)

---

## 3. How this fits current DreAgent

### Reuse

- Domain modules under `src/lib/*` with pure functions + thin API routes later  
- Deterministic helpers outside the model (same spirit as `src/lib/ai/executive.ts`)  
- Offline scripts (`scripts/*.ts` + `tsx`) for golden checks  
- Supabase client patterns in `src/lib/db/supabase.ts` and SQL in `src/lib/db/schema.sql`  
- Client identity patterns (`userId` / `x-user-id`)—with a **hard privacy note**: weak client ID is **not** sufficient for multi-user PHI without real ownership/auth  

### Future module home (not created on this planning branch)

```text
src/lib/health/           # calculations, types, safety, privacy snippets
src/lib/db/               # health tables SQL + repository
src/app/api/health/       # optional Phase A+/B APIs
scripts/lea-health-checks.ts
docs/lea-health-weight-journey.md   # this file
```

### Where health logic must not live

- `src/lib/outlook/*`  
- Graph write paths  
- Public system prompts as the **only** store of personal profile numbers  
- Free-text LLM as TDEE / safety calculator  

**Product identity:** Health is a **capability of Lea** (primary assistant), not a new top-level agent in the Smart LEA chrome.

---

## 4. Storage and privacy

### Recommended production store

**Supabase / Postgres** (aligned with existing schema style), with:

- Every row scoped by `user_id`  
- Server-side access only for health CRUD where possible  
- No full-table dump into logs or into the model context  

### Profile seeding

- Real personal values live in **private, user-scoped storage** after consent/activate.  
- Repo docs may describe **field names and rules**, not live labs or diary contents.  
- Prefer bootstrap / private config / first-run activation over hard-coding a full medical narrative into git.

### LLM context policy (all phases)

**May send (examples of minimized structure):**

- Active calorie target, floor, latest weight summary, milestone progress band  
- One scenario comparison result (“2 lb/week below floor—do not recommend”)  
- Hard flags: gluten-free required; thyroid clinician consult suggested y/n  

**Must not send every turn:** full log history, full target audit trail, lab tables.

---

## 5. Phase A — foundation only

Phase A is **library + schema + tests (+ docs)** when implemented. It is **not** chat NL logging, meals, restaurant safety, dashboard charts, or import UIs.

### 5.1 Schemas (logical)

**`health_profiles`** (user-scoped)

Illustrative fields (implementation names may match project conventions):

- Weights: starting, latest (nullable until first log), milestone (**180**), ultimate (**155**), optional active goal  
- Anthropometrics for estimates: height, age (or birth year), calculation sex (for equations only)  
- Activity setting (e.g. sedentary + exercise habit notes—not double-counted exercise math without design)  
- Calories: `current_calorie_target` (seed **1750**), `calorie_safety_floor` (**1200**), planning loss rate default **~1 lb/week**  
- REE method id (default Mifflin–St Jeor)  
- Celiac: gluten-free, reason, strict avoidance  
- Kitchen: dedicated gluten-free air fryer flags  
- Thyroid: status (e.g. no thyroid / under care), **clinician label** field (private), no medication-change APIs  
- Consent / seed version timestamps  

**`health_daily_logs`**

- Unique `(user_id, entry_date)`  
- Optional: weight, calories, walking miles/minutes, other activity, hunger/energy/sleep/stress, notes  
- Partial updates must not wipe omitted fields (**merge rules**—tested in pure code even before chat)

**`health_calorie_target_history`**

- previous/new target, effective date, approved_at, approval_status, reason  
- optional maintenance / trend context at change time  
- never silent overwrite  

### 5.2 Deterministic engine responsibilities

| Function | Notes |
|----------|--------|
| REE | Mifflin–St Jeor (or documented alternative); values are **estimates** |
| TDEE / maintenance | Documented activity factor; avoid double-counting treadmill without explicit design |
| Scenarios | 0.5 / 1.0 / 1.5 / 2.0 lb/week; rough **3500 kcal ≈ 1 lb** heuristic labeled as planning only |
| Floor | Never recommend or **approve** target &lt; **1200** |
| Milestones | Progress to **180** and **155** independently from start (~**255** seed concept) |
| Target apply | Propose → require explicit approval → write profile + history |

Weight change alone **must not** reduce the active target.

### 5.3 Pure safety helpers (no medical diagnosis)

- Escalate-to-clinician **message kinds** for thyroid/metabolism concerns when data warrants—not from single-day noise.  
- No functions that set or change medication dose.  
- Celiac remaining a hard filter flag for later meal phases.

### 5.4 Proposed future file layout (not created yet)

```text
src/lib/health/types.ts
src/lib/health/constants.ts      # floor, methods, heuristic labels (no private labs)
src/lib/health/calculations.ts
src/lib/health/targets.ts
src/lib/health/logs.ts           # pure merge helpers
src/lib/health/summary.ts
src/lib/health/safety.ts
src/lib/health/privacy.ts        # LLM snippet builder
src/lib/health/thyroid.ts        # pure escalation helpers
src/lib/health/celiac.ts
src/lib/db/health-schema.sql     # or schema.sql section
src/lib/db/health-repository.ts
scripts/lea-health-checks.ts
package.json → test:lea-health
```

Optional later: `src/app/api/health/*` for profile/target endpoints.

### 5.5 Phase A tests (when coding starts)

Cover at least: REE; maintenance; no silent target cut; approve/reject paths; floor 1200; all four loss-rate scenarios; 2 lb/week below-floor flagged; log merge purity; milestone % 180/155; profile flags exist for celiac/thyroid clinician; regression `test:lea-executive` remains green.

### 5.6 Phase A acceptance checklist

- [ ] Calculations deterministic in TypeScript services  
- [ ] 1200 floor enforced on apply/recommend  
- [ ] No silent target cuts on weight change  
- [ ] 180 + 155 milestones represented  
- [ ] Target history/audit works with explicit approval  
- [ ] No medication/dose advice APIs  
- [ ] Celiac hard constraint on profile model  
- [ ] Thyroid clinician escalation rule represented as pure helper  
- [ ] Privacy snippet helper defined (even if chat not wired)  
- [ ] Offline health checks + existing Smart LEA checks pass  
- [ ] No Smart LEA v1 runtime behavior change  

---

## 6. Later phases (B–G summary)

| Phase | Focus |
|-------|--------|
| **B** | Natural-language daily logging via chat tools; same-day merge |
| **C** | Trends, weekly coaching summary, dashboard/chart |
| **D** | Food preferences, kitchen safety learning |
| **E** | Meal ideas, remaining-cal awareness, pantry/grocery; restaurant **uncertainty**, no hallucinated safety |
| **F** | Thyroid timeline fields UX and contextual escalations |
| **G** | Idempotent CSV / SQLite import from desktop tracker; hardening |

One master product vision; ship in coherent PRs.

---

## 7. PyQt / desktop tracker

**Port concepts:** daily log shape, date upsert idea, progress framing, CSV columns, non-clinical disclaimers.  
**Do not port:** Qt UI, AppData as production cloud store, PyQt dependency.  
**Later:** optional import from CSV (and optionally desktop SQLite export), idempotent by `(user_id, date)`.

Until Phase B ships, users may keep logging in the desktop app and use LEA only for coaching from **minimized** pasted summaries.

---

## 8. Risks and blockers

| Risk | Mitigation |
|------|------------|
| Health privacy / PHI in git or logs | User-scoped DB; snippet policy; no private labs in fixtures |
| Weak client `user_id` | Single-tenant until real auth; no list endpoints across users |
| Medical overreach | Floor, approval, deterministic engine; LLM secondary |
| Couling to mail/deploy | Separate branches; no Graph writes or Gmail |
| LLM inventing celiac-safe restaurants | Out of Phase A; uncertainty language later |
| Double-counting activity | Simple Phase A activity model; logs separate |

---

## 9. Branch plan

| Branch | Purpose |
|--------|---------|
| `feat/lea-health-weight-journey-plan` | **This docs-only plan** |
| `feat/lea-health-phase-a-foundation` | Phase A code (future) |
| `feat/lea-health-phase-b-nl-logging` | NL log + tools |
| `feat/lea-health-phase-c-dashboard` | UI chart / cards |
| Later D–G | As table above |

**Do not** implement Phase A on the planning branch.

---

## 10. Explicit non-goals (planning + Phase A)

- Meal generation, restaurant safety claims, full macro system  
- Medication or thyroid **dose** advice  
- Gmail / IMAP / SMTP / Graph send/create  
- Conversation persistence UI  
- cPanel deploy or `.env` / secrets changes for this plan  
- Smart LEA v1 runtime changes  
- Creating `src/lib/health` or migrations **on the planning branch**  

---

## 11. Milestone framing (product)

| Point | Role |
|-------|------|
| Starting weight (seed concept ~255 lb) | Baseline for progress math |
| **180 lb** | **Major milestone**—genuine success if that is where the user is happy |
| **155 lb** | Stretch / ultimate goal (~100 lb total from start concept) |

Never frame stopping around 180 as failure.

---

## 12. Coaching posture (all phases)

Supportive, factual, practical, nonjudgmental.  
High-calorie days and missed walks are **information**, not moral failure.  
When the plan is working: LEA should be willing to say **do not lower calories** merely to go faster.

---

## 13. Related docs

- [`docs/lea-wellness-support.md`](lea-wellness-support.md) — superseded for **implementation**; historical lightweight roadmap  
- [`docs/lea-executive-eval.md`](lea-executive-eval.md) — Smart LEA v1 offline eval (separate track)  
- Master Health / Weight / Nutrition specification (external product authority)—follow when implementing; this file is the **DreAgent-shaped** Phase A blueprint  

---

## 14. Next step after docs merge

1. Human approval of this blueprint (granted for planning).  
2. Open `feat/lea-health-phase-a-foundation` from updated `main`.  
3. Implement Phase A **only** as defined here.  
4. Keep LEA live deploy / Smart LEA ops on a separate track.  
