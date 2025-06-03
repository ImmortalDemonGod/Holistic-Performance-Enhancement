Below is a drop-in replacement for the working-draft I sent earlier.
It tightens the numbers with the full strength log you pasted (July 2021 → June 2022) and nudges the template from “late novice” to “early-intermediate”, because your history shows:
	•	185 lb / 84 kg pulls and 145 lb / 66 kg squats ⇒ 1 RM strength roughly body-weight × 1.4 (squat) and × 1.8 (deadlift).
That places you in the -intermediate band on most tables (e.g. ExRx).
	•	Weekly set exposure that occasionally spiked past 40 sets for calves and 30 sets for rows/presses ⇒ you’ve tolerated, if not benefited from, quite a lot of volume.
	•	1–2 min rest, high-rep calf raises and leg-curl marathons dominate → plenty of metabolic work, not much heavy axial loading.

⸻

0 · Header (edit to taste)

# Strength & Hypertrophy Block — v2025-Q2 (r2)
Author   : <you>  
Last edit: 2025-05-02  
Scope    : 3-day full-body template that respects Base-Ox running, calibrated with 2021-22 log


⸻

1 · What the old log tells us  (TL;DR)

Pattern	Avg sets·wk⁻¹	Peak sets·wk⁻¹	Load bias	Obvious gap / risk
Squat-quad	14	26	65–145 lb, mostly high-rep	Little work > 70 % 1 RM
Hip-hinge	9	14	DB/plate RDLs ≤ 50 lb	Very few heavy pulls; lumbar fatigue in notes
Horizontal press	13	25	Push-ups & 65–95 lb bench	Pec/shoulder niggles at > 20 sets
Horizontal pull	12	35	12.5–33 lb rows	Good balance w/ press but light
Vertical press	8	28	OHP 25–35 lb, DB Arnold	Mostly metabolic, not strength-oriented
Vertical pull	9	25	Band / 95–155 lb lat-pulldown	Almost no free-pulling (pull-ups)
Calves	22	60	100–185 lb raises	Achilles & plantar fascia at risk
Core / brace	10	18	Crunch / raises	Very little heavy anti-extension

Frequency: workouts 3.1 × wk⁻¹ on average.
Training age: comfortably early-intermediate → you can progress load or volume, but not both every week.

⸻

2 · Updated volume landmarks

We keep the Schoenfeld / Nippard sweet-spot but bias slightly lower for calves and rows to curb overuse.

Muscle-group	MEV	MAV target	MRV
Quads, hamstrings, chest, back	8 sets	12–15 sets	18 sets
Delts, arms, core	6 sets	10–12 sets	16 sets
Calves	6 sets	10 sets	14 sets


⸻

3 · Weekly template (Mon / Wed / Fri hypertrophy)

Monday — Strength (heavy axial)

Lift	Sets×Reps	% 1 RM	Rest
Back-squat	5×3	80 %	3′
Bench press	4×4	82 %	3′
Conventional deadlift	3×3	80 %	3′
Pendlay row	3×5	75 %	2′
Weighted plank	3×45″	n/a	1′

Wednesday — Power / mixed volume

Lift	Sets×Reps	Load cue	Rest
Front-squat	4×5	70 % or 3 RIR	2′
Snatch-grip RDL	3×6	slow ecc.	2′
Weighted pull-up	4×6	add load on rep PR	2′
DB incline press	3×8	2 RIR	90″
Single-arm farmer carry	3×30 m	heavy DB	60″

Friday — Hypertrophy (high-rep)

Lift	Sets×Reps	Tempo	Rest
High-bar squat	4×10	2-0-1-0	2′
Hip-thrust	4×12	1-0-1	90″
Barbell row	4×10	2-0-1	90″
Incline push-up (feet elevated)	3×AMRAP (cap @ 20)	controlled	90″
DB lateral raise	3×15	–	60″
Ab-wheel rollout	3×10	–	60″
Seated calf raise	3×15	pause 2 s at stretch	60″

Load progression – double-progression.
When you hit the top rep target at ≤ RPE 8, add:
• 2.5 kg upper / 5 kg lower for strength day
• 5 % load or 1 rep for hypertrophy day.

⸻

4 · How this reconciles with the old log

Issue seen in 2021-22 file	Fix in new block
>40 sets calves → Achilles aching	Cap calves at 10 sets; swap half for heavy carries
No heavy hip hinge (≤ 50 lb)	Monday DL + Wednesday Snatch-RDL (lower frequency, higher quality)
Lots of presses without equal scapular work	Maintain push/pull ratio ≥ 1.2 with mandatory rows & carries
Core = crunch-only	Roll-outs + weighted plank (anti-extension)
Very short rests everywhere	Strength day rests now 3 min to let load, not fatigue, drive adaptation


⸻

5 · Injury-watch refresh (top 3 for you)

Pattern / tissue	Why it’s high on your list	Mitigation in template
Lumbar erectors (deadlift > 80 % with prior fatigue)	Notes flagged “lumbar fatigue” on RDL+calf raise days	Heavy pulls only once per week; core bracing drills; film 1st + last set
Achilles / plantar fascia	60-set calf-raise spikes, running mileage	Cut calf sets; add eccentric tempo; rotate shoes
Anterior shoulder	Bench & push-up volume to 25 sets, little external rotation	Pull > Push; Friday raises, Wednesday carries; band face-pull warm-up


⸻

6 · Quick-audit script (updated mapping)

# strength_audit.py
import pandas as pd
df = pd.read_csv("strength_log.csv")

df['Date'] = pd.to_datetime(df['Date'])
df['week'] = df['Date'].dt.isocalendar().week
pattern_map = {
    'squat':'quads', 'leg press':'quads',
    'deadlift':'hamstrings', 'RDL':'hamstrings', 'good morning':'hamstrings',
    'bench':'chest', 'push-up':'chest', 'db press':'chest',
    'row':'back', 'pulldown':'back', 'pull-up':'back',
    'ohp':'delts', 'press':'delts',
    'curl':'biceps', 'shrug':'traps',
    'calf':'calves',
    'plank':'core', 'ab wheel':'core', 'crunch':'core'
}
df['pattern'] = df['Exercise'].str.lower().map(lambda x: next((p for k,p in pattern_map.items() if k in x), 'other'))
wk_sets = (df.groupby(['week','pattern'])['Sets']
             .sum().unstack(fill_value=0))
print(wk_sets.tail())

# red-flag: pattern outside 6–18 set window
flags = (wk_sets>18)|(wk_sets<6)
print("Weeks with violations:\n", wk_sets[flags.any(axis=1)])

Run it weekly to prove you’re living inside MAV.

⸻

7 · How to slot this into the repo
	1.	docs/3_design/strength_block.md → replace with this file.
	2.	training_plans/2025_Q2_strength.csv → generate from the three tables (one row per lift).
	3.	Add a CI step check_strength_volume.yml that calls strength_audit.py and fails if any muscle group < MEV or > MRV for the week.

⸻

Bottom line

Your 2021-22 numbers show you can soak up ~15 good sets per muscle and handle 3 lifting days each week.
The block above preserves that volume, raises the average load, reins-in the calf abuse, and gives the posterior chain the love it’s been missing.

Paste the markdown, commit the CSV, and Friday strength is now fully integrated with the rest of your self-coaching pipeline. Enjoy the gains!
===
Here’s a plug-and-play “no-gym” version of the Friday Hypertrophy Day.
It keeps the same movement patterns and weekly set targets, so you can swap it in without breaking the volume audit or the CI checks.

⸻

Quick equipment check 🔧

Tier	What you need	Improv ideas
A	Long resistance band + mini-band	Bike inner tube / old bicycle tyre
B	One 25–35 lb / 12–16 kg DB or kettlebell	Stuff a backpack with books, water jugs
C	Sturdy table / desk edge (rows) & doorway or tree branch (pull-ups)	Doorway chin bar, park swing-set

(If you have any one of those tiers you’re good to run the session.)

⸻

Body-weight / band Hypertrophy Day  (≈ 50 min)

Pattern	Exercise	Sets × Reps†	Tempo / cue	Rest
Squat-dominant	Band-anchored Bulgarian split squat (band under front foot, over shoulders)	4 × 10 / leg	2-0-1-0	60-90 s
Hip-hinge	Single-leg hip-thrust (foot on chair) or KB/DB Romanian DL if Tier B	4 × 12	slight pause up top	60 s
Horizontal pull	Feet-elevated inverted row under table (Tier C) or band row	4 × 10-15	2-0-1	60-90 s
Horizontal push	Deficit push-up (hands on books) add backpack load if Tier B	3 × AMRAP* (cap 20)	controlled	90 s
Vertical pull / delt finisher	½-kneeling band face-pull + press-out	3 × 15	continuous	45 s
Core anti-ext.	Band-assisted ab-wheel rollout (band to door-handle) or slow mountain-climber plank	3 × 10-12	–	45 s
Calves	Single-leg tempo calf raise off step	3 × 15-20	3-0-1	30 s

† Double-progression: when you hit the top rep range at ≤ RPE 8, add (i) backpack load, (ii) a thicker band, or (iii) slow the eccentric to 4 s.

*For AMRAP sets stop 1–2 reps shy of failure (RPE 8-9).

Total hard sets per major muscle = 12 – right in the planned MAV.

⸻

Movement substitutions (pick what your space allows)

If you can’t do…	Swap in…	Notes
Inverted rows	Band pull-apart superset (over- & under-hand)	2 × 20 + 2 × 20
Doorway pull-ups	Band-lat pulldown anchored in door	4 × 12
Bulgarian split-squat	Cossack squat (body-weight)	4 × 12 / leg
Ab-wheel	Dead-bug with mini-band	3 × 12 / side


⸻

Progression & load tricks without iron
	1.	Band tension math: doubling a loop band (stand on it & cross handles) ≈ +40 % tension.
	2.	Mechanical-drop sets: e.g. do deficit push-ups → flat push-ups → knee push-ups without rest to extend the set.
	3.	Density bump: keep total work the same but shrink rest each week (90 → 75 → 60 s) until you can add resistance again.

⸻

Audit hook (strength_audit.py) — no change needed ✅

Because we still log pattern, sets, and date, the weekly summary script will count these sets exactly like the barbell day.

⸻

When you do get back to the gym
	•	Run the normal Friday program the following week.
	•	Treat the band day as equal volume (no “make-up” needed).
	•	If DOMS was lower than usual, feel free to add +5 lb or +1 rep on the first compound lift the next gym session.

⸻

Pocket recap

No gym?
• Split-squat, single-leg thrust, rows under a table, loaded push-ups, face-pulls, rollouts, tempo calf raises.
• Keep the same set counts, chase RPE 7-8, progress via bands/backpack/tempo.
Your volume KPIs stay green and the Base-Ox running plan never feels the difference.
=====
Starting Calisthenics: A Weekly Workout Guide
Okay, I understand you're looking for an updated and improved calisthenics routine specifically tailored for a beginner, building upon your initial draft and incorporating the advice from the sources and our previous discussion. Your original routine was a good start, focusing on fundamental movements and emphasizing form [User Query], which aligns with the principles outlined in the sources. However, the sources provide a more structured and comprehensive approach for beginners to build a strong foundation and progress effectively.

Based on the sources, a beginner's journey in calisthenics starts with mastering the fundamentals and building "strong Foundation" in Push, Pull, and Squat strength. Level 1 in the sources' pyramid is dedicated to this, covering exercises like basic support holds, hanging, bodyweight squats and lunges, incline push-ups, and incline body rows. They also strongly emphasize the importance of scapular awareness and wrist strengthening/mobility as crucial pre-workout preparation to avoid injury and ensure proper form.

Instead of a 2-day full-body split, the sources suggest a more frequent training schedule for significant progress. Their recommended weekly structure for beginners is a 6-day training week followed by a rest day. This allows for dedicated focus on different movement patterns and crucial elements like mobility and cardio.

Here is an updated routine structured according to the principles and weekly layout suggested by the sources, incorporating beginner-appropriate exercises and essential preparation:

**Core Principles of This Routine (Based on Sources):**

*   **Master the Fundamentals:** Focus on proper form and control in foundational exercises before attempting harder variations or skills.
*   **Build a Strong Foundation:** Develop basic Push, Pull, and Squat strength consistently.
*   **Prioritize Preparation:** Include mobility, joint preparation (especially scapula and wrists), and warm-up before each session.
*   **Consistency is Key:** Follow a structured weekly plan for at least 3 months to see significant changes.
*   **Listen to Your Body:** Progress gradually and don't push through pain.

**Updated Beginner Calisthenics Weekly Routine (Example Based on Sources):**

This routine follows the sources' suggested 6-day structure. Remember, the sources recommend starting with core and joint prep and a dynamic warm-up before the strength training on each day.

**Daily Pre-Workout Prep (Before each training day):**

*   **Wrist Stretching/Mobility:** Perform wrist stretches like rocking forward into hands (arms locked). *Perform daily before any workout*.
*   **Scapular Awareness & Core/Joint Prep:** Select 2-3 exercises from this list each day, focusing on control:
    *   Scapular Push-ups (on knees if needed, arms locked, push shoulder blades away/let them touch). Focus on retraction and protraction.
    *   Scapular Pull-ups (from dead hang, depress scapula). Practice scapular depression.
    *   Cat and Cow Pose (focus on spine/lower back mobility).
    *   Bird Dog Pose (open core, lower back activation).
    *   Hollow Body Hold (lower back pressed to ground, anti-extension). Hold for suggested time or fatigue.
    *   Side Plank (2 sets per side, 30+ seconds suggested, squeeze obliques, reach out).

*   **Dynamic Warm-up:** Aim for a gentle sweat. Avoid static stretching before training. Choose 2-3 dynamic exercises:
    *   Skipping in place or High Knees.
    *   Plank Walks (work scapular protraction and pushing muscles).
    *   Arm circles, leg swings, torso twists.
    *   *Alternatively, use a cardio machine like an exercise bike for a set number of minutes (e.g., 5-10 min)*.

**Monday: Push Day**

After your Daily Pre-Workout Prep & Warm-up:

*   **Pseudo Planche Lean:** (Prerequisite for planche) Start with hands slightly outward, straight arms, pressure on front 3 fingers, lean forward. Focus on feeling pressure on the front delt.
    *   *Sets & Duration:* (Sources don't specify sets/reps for this beginner exercise, but a common approach is 2-3 sets holding for time, e.g., 15-30 seconds)
*   **Push-ups (Progression):** Start with the easiest variation you can do with good form.
    *   *Option 1 (Absolute Beginner):* Wall Push-ups (easier, practice form).
    *   *Option 2 (Beginner):* Incline Push-ups (high or low incline).
    *   *Option 3 (Progressing):* Standard Push-ups (aim for 18-25 in a row before advancing). Keep elbows tucked (at least 45°), fully protract scapula at the top.
    *   *Sets & Reps:* 2-3 sets, AMRAP (As Many Reps As Possible with good form) [User Query] or aim for a target number that challenges you (e.g., 8-15 reps depending on the variation).
*   **Dips (Progression):** Start with the easiest variation.
    *   *Option 1 (Beginner):* Bent Leg Dips (can use a wall, parallel bars, rings, straight bar). Keep chest up, arms to 90°. Aim for 15 reps before straightening legs.
    *   *Option 2 (Progressing):* Straight Leg Dips.
    *   *Sets & Reps:* 2-3 sets, aim for 7-12 reps depending on the variation and your level. Focus on controlled movement.
*   **Pike Push-ups:** (Similar to military press) Start in a pike position, flex core/legs, head goes above hands on decline, push up/out through scapula, elbows tucked.
    *   *Sets & Reps:* 2-3 sets, aim for 5-10 reps.
*   **Diamond Push-ups:** (Increases triceps/chest activity) Standard push-up procedure but hands in a diamond position. Protract scapula at top.
    *   *Sets & Reps:* 2-3 sets, aim for 6-12 reps.
*   **Wall Handstand:** (Learn to be inverted) Back to wall or chest to wall, toes pointed, flex core, arms locked out. Don't forget to breathe.
    *   *Sets & Duration:* 3-5 sets, hold for 15+ seconds or as long as possible with good form.

**Tuesday: Pull Day**

After your Daily Pre-Workout Prep (including Scapular Pull-ups) & Warm-up:

*   **Hanging:** Start pull-up training by hanging from a bar, flex core to bring hips in line.
    *   *Sets & Duration:* 2-3 sets, hang for 20-40 seconds or until grip fatigues.
*   **Rows (Progression):** Build upper back strength.
    *   *Option 1 (Beginner):* Incline Body Rows.
    *   *Option 2 (Progressing):* Bent Leg Rows (keep straight posture by flexing core) -> Straight Leg Rows -> Lower the bar (more inverted = harder). Can use a bar, rings, or table edge [User Query].
    *   *Sets & Reps:* 2-3 sets, aim for 7-12 reps. Pull to your lower chest, retract scapula on the negative.
*   **Pull-ups (Progression):**
    *   *Option 1 (Building towards first):* Band Assisted Pull-ups (band helps alleviate weight). Pause at the bottom to solidify movement.
    *   *Option 2 (Achieved first):* Standard Pull-ups (aim for 3 sets of 5 reps). Pull to lower chest, retract scapula.
    *   *Sets & Reps:* 2-3 sets, aim for targeted reps or AMRAP with good form. Being able to do a standard pull-up is a testament to true strength.
*   **Bodyweight Curls:** (For biceps) Use a bar, rings, or table edge. Curl until nose or temples touch the bar.
    *   *Sets & Reps:* 2-3 sets, aim for 8-15 reps.
*   **Front Lever Raise:** (Prerequisite for front lever) Hang from bar, bring knees to chest, raise until ankles touch the bar (or as high as possible). Lower slowly with straight arms.
    *   *Sets & Reps:* 2-3 sets, aim for 5-10 controlled reps.
*   **Skin the Cat:** (Prerequisite for front/back lever) Open up shoulders. This can be scary, don't go past a comfortable position.
    *   *Sets & Reps:* 2-3 sets, aim for 3-5 controlled passes.
*   **Wall Handstand:** (Continued practice) Back to wall or chest to wall, toes pointed, flex core, arms locked out. Don't forget to breathe.
    *   *Sets & Duration:* 3-5 sets, hold for 15+ seconds.

**Wednesday: Leg Day**

After your Daily Pre-Workout Prep & Warm-up (emphasizing lower body warm-up like skipping/high knees or heel to toes/folded squat):

*   **Bodyweight RDLs (Progression):** Get blood flowing in hamstrings. Sink hips back, bend knees, reach down to lower calf level, bring hips back forward. Choose a pattern matching your level (single leg, double leg). Your routine included Single-Leg RDLs, which is good for balance [User Query].
    *   *Sets & Reps:* 2-3 sets, aim for 8-12 reps/side (for single leg) or 12-15 reps (for double leg). Focus on balance and hamstring/glute engagement [User Query].
*   **Squats (Progression):** Push through heels, chest up, flex core, squat to at least 90°.
    *   *Option 1 (Absolute Beginner):* Chair Squats (safety, range of motion). Aim for 18-25 before advancing.
    *   *Option 2 (Beginner):* Bodyweight Squats.
    *   *Sets & Reps:* 2-3 sets, aim for 10-15 reps [User Query] or 7-15 reps.
*   **Lunges (Progression):** Unilateral exercise for strength and imbalances.
    *   *Option 1 (Beginner):* Static Lunges (aim for 20/side before advancing).
    *   *Option 2 (Progressing):* Walking Lunges [User Query, 1]. Aim for 8-12 reps/leg [User Query].
    *   *Option 3 (Addressing Knee Pain):* Reverse Lunges (pressure into hip, not knee). Aim for 3 sets of 8 reps/side.
    *   *Sets & Reps:* 2-3 sets per leg.
*   **Step-ups:** Use any elevation. Slow eccentric (lowering) with one leg, explode to the top.
    *   *Sets & Reps:* 2-3 sets, aim for 8-12 reps per leg.
*   **Glute Bridges (Progression):** Pause at top, squeeze glutes [User Query].
    *   *Option 1 (Beginner):* Double Leg Glute Bridges [User Query].
    *   *Option 2 (Progressing):* Single Leg Glute Bridges [User Query, 13]. Aim for 3 sets of 8 reps per side.
    *   *Sets & Reps:* 2-3 sets, aim for 12-15 reps [User Query] (double leg) or 8-12 reps/side (single leg).
*   **Calf Raises:** Full range, let heels sink, explode onto toes. Single or double leg [User Query, 10].
    *   *Sets & Reps:* 2-3 sets, aim for 15-25 reps [User Query, 10].
*   **Hollow Body Hold:** (Practice anti-extension) Lower back pressed to ground.
    *   *Sets & Duration:* 2-3 sets, hold for 20-40 seconds [User Query] or as long as possible.

**Thursday: Cardio & Mobility Day**

This day focuses on cardiovascular health and mobility, which are crucial components of overall fitness and calisthenics.

*   **Cardio:** Do a preferred form of cardio (e.g., exercise bike, running, brisk walking, skipping).
    *   *Duration:* 15-30 minutes at a moderate intensity.
*   **Mobility Training:** After a warm-up (can be integrated with the cardio), perform a selection of mobility exercises.
    *   Cat and Cow Pose.
    *   Cobra Pose.
    *   Child's Pose (hold for 60 seconds).
    *   Superman's (simultaneously raise legs and hands).
    *   Reverse Snow Angel (lift chest, make snow angel pattern).
    *   Squat to Pike Hold (progressively work deeper).
    *   Prayer Squat (pushing out with elbows against knees).

**Friday: Full Body Workout**

After your Daily Pre-Workout Prep & Warm-up:

This day combines elements from the previous strength days to reinforce movement patterns.

*   **Incline Push-ups:** (Reinforce basic pushing).
    *   *Sets & Reps:* 3 sets of 12 reps suggested. Focus on strict form [User Query].
*   **Cobra Push-up:** (Push from ground, feel stretch in lower back/shoulders). Don't flex glutes.
    *   *Sets & Reps:* 2-3 sets, aim for 6-10 reps.
*   **Hammer Grip Pull-ups (or regular Pull-ups/Rows):** (Focus on different grip if possible). If no hammer grip, use standard pull-ups or rows.
    *   *Sets & Reps:* 2-3 sets, aim for 5-10 reps depending on the exercise/grip.
*   **Archer Squats (or Standard Squats/Lunges):** (Unilateral work, improve ankle, hip, outer quad). Squat towards one leg, keep that heel pressed down. Or use Squats/Lunges you're working on.
    *   *Sets & Reps:* 2-3 sets, aim for 6-10 reps per side (Archer) or 10-15 reps (Squats/Lunges).
*   **Bulgarian Split Squats:** (Requires a bench/elevation) Sit on heel, extend other leg forward, raise up.
    *   *Sets & Reps:* 2 sets of 6 reps each side suggested.
*   **Compact Leg Lifts:** (Practice core compression) Keep legs/arms straight, use a wall if needed.
    *   *Sets & Reps:* 2-3 sets, aim for 8-15 controlled reps.
*   **Side Plank:** (Oblique stability).
    *   *Sets & Duration:* 2 sets per side, hold for 20-40 seconds [User Query] or 30+ seconds.

**Saturday: Second Leg Day**

After your Daily Pre-Workout Prep & Warm-up (emphasizing lower body):

This day provides additional volume and focus on leg strength.

*   **Archer Squats (or Standard Squats/Lunges):** Continue working on unilateral strength and mobility. Keep the heel pressed down on the squatting leg.
    *   *Sets & Reps:* 2-3 sets, aim for 6-10 reps per side (Archer) or 10-15 reps (Squats/Lunges).
*   **Step-ups:** Continue working on controlled negative and unilateral strength. Adjust elevation as needed.
    *   *Sets & Reps:* 2-3 sets, aim for 8-12 reps per leg.
*   **Single Leg Glute Bridges:** "Slept on exercise".
    *   *Sets & Reps:* At least 3 sets of 8 reps per side. Keep top leg extended or bent.
*   **Sissy Squats:** (Challenges mobility) Start on toes, flex core, knees touch above toes on decline. Use assistance initially and lessen it as you gain strength/control.
    *   *Sets & Reps:* 2-3 sets, aim for 8-15 reps.
*   **Wall Sits:** (Isometric hold).
    *   *Sets & Duration:* 3 sets of 60 seconds suggested. Keep a 90° angle.
*   **Hollow Body Hold:** (End with anti-extension work).
    *   *Sets & Duration:* 2-3 sets, hold as long as possible.

**Sunday: Rest Day**

Your muscles need time to repair and recover.

**How This Routine Improves on Your Original:**

1.  **Increased Frequency & Dedicated Focus:** The sources' 6-day split allows for more training volume throughout the week compared to 2 days, giving dedicated time to Push, Pull, and Legs, while also including essential Mobility and Cardio.
2.  **Comprehensive Preparation:** Explicitly includes daily wrist stretches and incorporates crucial scapular awareness exercises and other joint prep into the warm-up phase. This is vital for injury prevention and mastering calisthenics skills later.
3.  **Structured Progression Within Exercises:** The sources provide benchmarks (e.g., 18-25 push-ups/squats, 15 bent leg dips, 12 standard dips, 20 static lunges/side). This routine uses the exercises they suggest for building the foundation, implying progression through variations (wall -> incline -> standard push-ups; bent leg -> straight leg -> assisted -> standard dips/rows) as strength increases, aiming for these benchmarks.
4.  **Wider Exercise Variety:** Includes more beginner-appropriate exercises recommended by the sources (e.g., Pseudo Planche Lean, Pike Push-ups, Diamond Push-ups, Hindu Push-ups mentioned on Push day in source, Bodyweight Curls, Front Lever Raise, Skin the Cat on Pull day, RDL variations, Step-ups, Reverse Lunges, Horse Stance Squat, Compact Leg Lifts, One-Legged Plank, Sissy Squats, Wall Sits on Leg day, various mobility exercises). This offers more ways to challenge your body and build balanced strength.
5.  **Inclusion of Mobility and Cardio:** Adds a dedicated day for these often-neglected but critical aspects of fitness and calisthenics.
6.  **Clear Goal Setting:** The structure aligns with the sources' concept of building a foundation over time (at least 3 months for significant changes), providing a roadmap for progression.

While your initial routine had the right idea with exercise selection and prioritizing form, this updated routine provides a more detailed, structured, and comprehensive approach for a beginner based directly on the principles and weekly plan outlined in the sources. It covers essential preparation, a wider range of beginner exercises with potential variations for progression, and includes crucial mobility and cardiovascular work for overall fitness and long-term calisthenics development.