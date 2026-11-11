# openarm2026
This is a roadmap of Waste handling robot development.


## First Step: “OpenArm should pick up the cup wherever the cup is on the table.”

What I would do for your OpenArm ACT dataset
Since you're starting with one cup on a table, I'd use several levels of difficulty.
Stage 1 — Fixed position
Start with maybe 50–100 demonstrations.

Keep the cup approximately in the same location.

Purpose:
- verify teleoperation
- verify gripper control
- verify camera configuration
- verify ACT training pipeline
- verify that the robot can actually learn the basic task

Table
+--------------------------------+
|                                |
|             🥤                 |
|                                |
+--------------------------------+

This is your pipeline validation dataset, not your final dataset.


Stage 2 — Small position variation
Then collect perhaps 200–300 demonstrations.

Move the cup around a small region:

+--------------------------------+
|        🥤     🥤               |
|                                |
|   🥤              🥤           |
|                                |
|          🥤                    |
+--------------------------------+


For example:
- left/right: ±10 cm
- forward/back: ±10 cm
- small orientation changes
The robot now has to learn:
"Where is the cup?"

rather than:
"Move my arm to this one fixed coordinate."


Stage 3 — Larger workspace variation
Then expand the region:

+--------------------------------+
| 🥤                 🥤          |
|                                |
|          🥤                    |
|                                |
|                         🥤     |
|                                |
|       🥤                       |
+--------------------------------+

Now the cup could appear at many locations within the reachable and visually reliable workspace.

X: ±25–30 cm
Y: ±20–25 cm
Z: fixed by table

You don't need exact numbers yet. They should ultimately be determined from your OpenArm workspace and camera visibility.

But there's an even more important point
Don't only vary position.

If your goal is robust real-world manipulation, vary the initial conditions.

| Variable | Training variation |
|---|---|
| Cup X position | Yes |
| Cup Y position | Yes |
| Cup rotation | Yes |
| Cup distance from robot | Yes |
| Lighting | Moderate |
| Camera viewpoint | Mostly fixed initially |
| Cup color | Several |
| Cup size | Small variation |
| Cup orientation | Yes |
| Exact starting arm position | Some variation |

However: don't over-randomize
This is where people can accidentally make an ACT dataset worse.

Suppose you have only 1,000 demonstrations and do this:
100 cup positions
×
10 cup colors
×
10 lighting conditions
×
10 camera configurations
You effectively have very few examples for each condition.

ACT may struggle.

Instead, control the variation.

I'd recommend:
Phase A
Same robot + same table + same cameras + same cup

Only vary:

cup position

This isolates the most important capability:
spatial generalization.

Then later add:

cup orientation → cup appearance → object variation → environmental variation.

Your teleoperation strategy should also change
This is particularly important.

Don't think:
"I need to record 1,000 successful trajectories."

Think:
"I need to record demonstrations that cover the task's state space."

For example:
       🥤

       approach
↓
align
↓
grasp
↓
lift

One more thing: don't make "anywhere" literally anywhere
Your final claim should be something like:
"The robot can pick up the cup from arbitrary positions within the defined reachable workspace."

Not:
"The robot can pick up a cup anywhere."

There are physical limitations:
- arm workspace
- self-collision
- table collision
- camera visibility
- gripper reachability
- occlusion
- singularities
- joint limits
So define a valid workspace region.

                OpenArm

                   ↓
        ┌─────────────────────┐
        │                     │
        │    VALID CUP        │
        │    WORKSPACE        │
        │                     │
        │   🥤       🥤       │
        │                     │
        │       🥤            │
        │                     │
        └─────────────────────┘
                 TABLE


                                  ACT CUP PICKING
                       │
                       ▼
              ┌─────────────────┐
              │ MuJoCo scene    │
              │ OpenArm + table │
              │ + cup           │
              └────────┬────────┘
                       │
                       ▼
              Teleoperation
                       │
                       ▼
          ┌─────────────────────────┐
          │ Dataset collection      │
          │                         │
          │ 1. Fixed position       │
          │ 2. Small variation      │
          │ 3. Large variation      │
          │ 4. Workspace coverage   │
          └────────────┬────────────┘
                       │
                       ▼
                 Train ACT
                       │
                       ▼
              Evaluate on NEW
              cup positions
                       │
              ┌────────┴────────┐
              ▼                 ▼
          Seen position     Unseen position
              │                 │
              ▼                 ▼
            Easy          Generalization


            And the most important test is not the training position.

If you train with:
cup positions:
A B C D E
then test with:
F G H I J

If it successfully picks up F–J, you have actual evidence that your ACT policy learned spatial generalization rather than memorizing the demonstrations.

For your OpenArm waste-sorting direction, this is also the much better foundation: first prove position generalization with one cup, then add object/appearance variation, then move to multiple objects and sorting.