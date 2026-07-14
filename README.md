# ai-mario

Watch a tiny artificial "brain" teach itself to play a Mario-style platformer, right in your browser.

**▶️ [Open the live demo](https://ebrithilnogare.github.io/AI-Mario/)**

## What is this?

Little characters keep trying to run through a level full of pipes, pits, spikes, enemies and coins. At first they fail almost immediately. Each round (a "generation"), the best ones are kept and slightly tweaked, and over time they learn to jump, dodge and reach the flag — no one programs how, they figure it out by trial and error.

## How to use it

1. Open the [demo](https://ebrithilnogare.github.io/AI-Mario/).
2. Press **TRAIN** and watch the graph climb as the brains improve.
3. Press **WATCH BEST** any time to see the current best player run a level.
4. Press **NEW LEVEL** to throw a fresh, unseen level at them.

## Things you can tweak

- **Inputs** — what the character is allowed to "see" (nearby enemies, coins, gaps, walls, and more).
- **Network** — how big the brain is.
- **Rewards** — what counts as good or bad: reaching the flag, grabbing coins, stomping enemies, dying, jumping, wasting time.
- **Maps & threads** — how many levels each player is tested on, and how much of your computer to use for faster learning.

## Reading the graph

- **Black line** — the best score so far.
- **Grey line** — the average of everyone.
- **Blue line** — how the current best does on a secret level it never practices on, showing whether it's truly learning or just memorizing.

No installation, no accounts, nothing to download — it all runs in your browser.
