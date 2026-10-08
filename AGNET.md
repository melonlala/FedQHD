# Secrets
- Read credentials from ‘.env‘ in the project root; never print or commit it.
- Before creating a ‘.env‘, always add ‘.env‘ to ‘.gitignore‘ first.
- Pipe secrets via stdin/env vars into commands, never as literal CLI args.

# Env
- coding environment is conda activate hdc_fedrl
- everything stored in /home/yuchen/projects/fedqhd

# Rules
- Overlap main results in /results  and report in /docs by updated experiment design
- save history in /history
- add agent tag "claude_made" or "codex_made"
- use wandb for training log, return wandb id and explaination to check on progress on website
- keep the report simple and honest

# Writing instruction
Default to using clear, concise paragraphs, each developing one main idea. Use lists only when the information is genuinely parallel, sequential, or easier to compare, and avoid nested lists unless the hierarchy cannot be expressed clearly in prose. Use plain, simple language: familiar words, concrete examples, and precise verbs. Prefer active voice and direct statements.

Make sure to state the main point clearly and early, then develop it with the explanation and detail the reader needs. Let each sentence build on what came before. Develop the points that matter and provide enough support to be useful.

Use plain language over jargon, and reference technical details only to the degree that it helps illustrate an idea or your work to the user. Communicate complex concepts in a clear and cohesive manner, and calibrate your writing to the level of background knowledge assumed from the user's prompt and context.

Avoid using slop words or phrases like "Bottom Line:" in conclusions, "delve," "foster," "leverage," "it's worth noting," "importantly," "Question? Answer." or "This isn't about X. It's about Y.", "genuinely" or hyphenated compound descriptions and adjectives. Do not use concluding summary statements such as "In short:..", "The simplest mental model is:...".

State the intended action directly. Avoid adding what you won't do, what will remain unchanged, or how you'll separate or categorize results. Do not use contrastive framing such as "X, not Y" or "X—not Y" that introduces an unprompted alternative that the user didn't ask about. Avoid invented compound labels like "exact-head checks" and "editorial-row layouts", vague qualifiers, and canned transitions; use plain verbs and prepositions to state the actual relationship directly.