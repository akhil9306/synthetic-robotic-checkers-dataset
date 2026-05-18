# Synthetic Robotic Checkers Dataset Generation with Gemma 4
### Teaching Robots to See and Act

### The Problem: Robotics is Starving for Multimodal Training Data
Robotics will be won or lost on data. Robots that operate in the physical world need to reason from raw visual input and translate that reasoning into precise, sequenced physical actions. This is the Vision-Language-Action (VLA) problem and it is genuinely hard to solve at scale.
The bottleneck is not only hardware or model architecture. It is labeled training data that pairs visual observations with structured reasoning chains and executable action sequences. Collecting this data in the real world is slow, expensive, and dangerous. Human teleoperation is rate-limited by the number of operators. Scripted demos produce brittle data that does not generalize.
This project asks a different question: What if a frontier multimodal model could play the role of the robot brain, generating its own training data from simulation?

### The Approach: Gemma 4 as the Robot's Mind
This pipeline uses Gemma 4 (E4B-it) as the reasoning and action-planning agent inside a fully simulated robotic checkers environment. The model receives only a top-down camera image of the board, produces structured chain-of-thought reasoning, and emits a sequence of low-level robot primitives that the left arm of a [Trossen WidowX dual-arm robot](https://github.com/TrossenRobotics/trossen_arm_mujoco) executes inside MuJoCo physics simulation.
Every turn the model takes becomes a training sample. After enough games, the result is a richly annotated dataset of (image, system_prompt, thinking, actions) tuples which over here is the format needed to fine-tune a VLA.
The key insight is that Gemma 4's native multimodal capability and structured output compliance make it uniquely suited for this role. Earlier models either could not reliably parse a board image under a strict JSON schema, or hallucinated illegal moves too frequently to produce useful signal. Gemma 4's image encoder (2,496 tokens per frame) combined with its instruction-following fidelity made reliable structured generation achievable on commodity cloud GPUs.

### Architecture
The pipeline has five interlocking components:

1. The MuJoCo Simulation Layer
A procedurally generated MuJoCo scene renders the full 8×8 checkers board, all 24 pieces as cylindrical meshes, and the Trossen dual-arm robot. Two cameras are used: a top-down orthographic view for LLM input and an oblique view for recording. The scene is deterministically initialized from checkers_core.initializeBoard() at the start of every game.

2. Gemma 4 on vLLM
The model is served via vLLM 0.20.0 with tensor parallelism. A custom system prompt defines the robot's coordinate frame, the eight allowed primitives, and a strict output format: a <thinking> block with STATE:, PLAN:, and ACTIONS: labels, followed by an <actions> block of valid JSON. The model must ground every action to a board square in algebraic notation (e.g., b6).

3. The RL Opponent Stack
The opponent is a curriculum of three pretrained RL checkers models (Linear, Medium, Deep) blended with a random policy using an annealing schedule: random moves dominate early games, deep RL dominates late. This produces training data across a meaningful range of board complexity which include shallow endgames and contested midgames both appear in the dataset, which is important for training a generalizable downstream policy.

4. Best-of-N Validation and Scoring
This is the most important engineering contribution of the project. A naive pipeline that accepts the first output from the LLM produces a dataset contaminated with illegal moves, malformed JSON, and incorrect capture counts. Instead, each turn triggers candidates_per_turn=4 independent samples from Gemma 4 at temperature 0.9. Each candidate is passed through an 8-check validator:

Structural tags present (<thinking>, <actions>)
Thinking labels present (STATE:, PLAN:, ACTIONS:)
Actions are parseable JSON
Every op is in the allowed primitive set
The primitive skeleton follows the required ordering
(from, to) squares are recoverable and well-formed
The move is legal according to checkers_core
The number of remove_captured calls matches actual captures

Candidates are then scored with a weighted rubric (legality worth 3×, captures and structure worth 1× each) plus a heuristic board evaluation from the strongest RL model. The best-scoring candidate is executed; if no candidate passes the hard gates, up to two retry rounds are issued before a legal fallback is substituted. Turns where the fallback was used are flagged was_legal=False so downstream training can filter them out.

5. Windowed SFT Sample Construction
After each game, turns are sliced into windows of 1–6 consecutive turns. Each window becomes one supervised fine-tuning sample, containing the full message history (system prompt, alternating user-image / assistant-response turns), per-turn metadata, and a terminal flag with the true game outcome for windows that reach the final ply. This multi-turn structure allows a student model trained on this data to learn both single-step action prediction and longer-horizon board reasoning.

### Why Gemma 4 Specifically
Three properties of Gemma 4 were non-negotiable for this pipeline:
Native multimodal input. The model needed to consume a raw rendered PNG of the board and infer piece positions without any hand-crafted board serialization. Gemma 4's image encoder handles this reliably. The board image encodes 2,496 tokens. The model attends to the full visual context of all 24 pieces simultaneously.
Structured output compliance. The action schema is strict: a JSON list where every entry has an op from an allowed set and an arg that is either a valid algebraic square or null. The best-of-N scoring system amplifies the value of compliance where a model with higher baseline compliance produces more candidates that survive the hard gates per round, which directly reduces GPU time and retry overhead.
Scale efficiency. The E4B model (~8 GB at fp16) fits on two T4s or a single A100 with headroom for the fp8 KV cache and full context (8,192 tokens). This made the pipeline reproducible on free cloud tiers, not just expensive infrastructure.

### Technical Challenges and Solutions
CUDA version mismatch. vLLM 0.20.0 was built against CUDA 13.0 while Kaggle ships 12.8. The notebook includes an automated upgrade that installs the CUDA 13.1 toolkit and matching PyTorch wheels before restarting the session.
Gemma 4 attention backend constraints. Gemma 4's heterogeneous head dimensions force the TRITON_ATTN backend, which is incompatible with TurboQuant and requires --enforce-eager to disable CUDA graph capture in TP mode. These constraints are documented in the notebook with explanations, not just workarounds.
NCCL stability on Kaggle. T4 pairs have no NVLink, so NCCL P2P and SHM must be disabled to prevent hangs during tensor-parallel all-reduce. The pipeline detects the platform and sets these environment variables automatically.
Board diffing without move annotations. checkers_core.possibleMoves returns full next-board states, not (from, to, captured) tuples. A custom diff_boards function recovers the move by comparing two board states square by square, identifying which piece moved and which squares were vacated by captures.

### The Dataset
The generated dataset is published at [akhil9306/robot-checkers-dataset](https://huggingface.co/datasets/akhil9306/robot-checkers-dataset) on HuggingFace Hub. Each record contains the full multi-turn message history in HF Datasets-compatible format, the rendered PNG board images, per-turn validation metadata, and candidate trace logs. The JSONL, Parquet, and CSV index formats are all included, making it directly loadable with datasets.load_dataset(...) and compatible with standard fine-tuning pipelines.
The dataset is not the end goal it is the scaffold. The intended downstream use is fine-tuning a smaller VLA policy (e.g., Gemma 4 E2B or a purpose-built action model) on this data, creating a model that can operate a robot arm from raw camera input without the full inference cost of the 4B generator.

### What This Enables
Synthetic data generation pipelines like this one are how the robotics field will escape the teleoperation bottleneck. A single run of this pipeline on a single or multi GPU clster produces a dataset that would take dozens of hours of human operator time to collect in the real world. The best-of-N validation layer means the data is clean enough to train on directly, without a separate human review pass and as minimal as possible structural errors.
The checkers domain is intentionally legible the rules are well-defined, the legality oracle is deterministic, and the visual input is structured enough to verify model behavior. But the architecture generalizes: replace the checkers simulator with any MuJoCo environment, swap the RL opponent for a scripted policy or a second LLM, and the same pipeline produces VLA training data for manipulation, navigation, or assembly tasks.
Gemma 4 made this tractable. The combination of reliable multimodal grounding, strict instruction following, and efficient inference at the E4B scale meant that the pipeline could run end-to-end on free cloud hardware and produce a dataset worth publishing. That is the bar this project was designed to clear.

### References and Acknowledgements

1. [MuJoCo](https://github.com/google-deepmind/mujoco)
2. [Trossen Arm MuJoCo](https://github.com/TrossenRobotics/trossen_arm_mujoco) 
3. [Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
4. [vLLM](https://github.com/vllm-project/vllm)