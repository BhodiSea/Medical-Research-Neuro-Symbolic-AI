premedpro-ai/  # Root: Consolidated monorepo for Dharma Engine in Premedpro
├── .git/  # Git repo (with submodules for OSS)
├── .gitignore  # Ignore build artifacts, venvs, etc.
├── README.md  # Overview, setup, phasing (middleman to independent), new features explained
├── LICENSE  # Your license + OSS attributions
├── CREDITS.md  # List of OSS sources and modifications
├── docs/  # Architecture diagrams (e.g., via Draw.io), API specs, ethical guidelines
│   ├── architecture.md  # Detailed system map, including new features like 10th man and simulations
│   └── ethical_memory_design.md  # Docs on simulation-based ethical training (inspired by Keleth PoC)
├── config/  # Shared configs (e.g., ethical rules YAML, simulation scenarios, API keys securely referenced)
│   ├── ethical_constraints.yaml  # Seed for ethical memory (e.g., core principles for simulations)
│   ├── simulation_scenarios.yaml  # Predefined human life templates for ethical training
│   └── agent_domains.yaml  # Definitions for 10 agents' expertise and 10th man rules
├── core/  # Hybrid Neuro-Symbolic Engine (Python primary; ~40% OSS, extended for memory/sim integrations)
│   ├── symbolic/  # Symbolic reasoning layer (extended for ethical memory graphs)
│   │   ├── nstk/  # Submodule: Forked IBM NSTK (for LNNs; extend with ethical rule verification)
│   │   ├── nucleoid/  # Submodule: Forked Nucleoid (for knowledge graphs; **extend with ethical_memory_graph.py for long-term ethics storage**)
│   │   ├── peirce/  # Submodule: Forked PEIRCE (for inference loops; integrate with 10th man dissent)
│   │   └── custom_logic.py  # Extensions: Symbolic rules for query formulation and simulation analysis
│   ├── neural/  # Neural learning layer (extended for learning from queries/simulations)
│   │   ├── torchlogic/  # Submodule: Forked TorchLogic (for weighted logic; fine-tune on ethical sim data)
│   │   ├── symbolicai/  # Submodule: Forked SymbolicAI (for LLM fusion; adapt for internal narrative gen in sims)
│   │   └── custom_neural.py  # Extensions: QM-inspired uncertainty models, trained on query responses
│   └── hybrid_bridge.py  # Custom: Fuse symbolic/neural; now includes hooks for audit logs and ethical memory updates
├── math_foundation/  # Mathematical Core (Julia primary; extended for simulation metrics)
│   ├── qft_qm.jl  # Custom: QFT/QM analogs (branching sim paths for life alternatives)
│   ├── thermo_entropy.jl  # Custom: Entropy-based truth/ethics (now measures ethical alignment in sims)
│   └── python_wrapper.py  # Custom: PyJulia integration; add functions for sim entropy scoring
├── ethical_audit/  # Safety Layer (Rust primary; **expanded for further auditability and thinking traces**)
│   ├── Cargo.toml  # Rust manifest
│   ├── src/  # Rust source
│   │   ├── main.rs  # Entry for testing
│   │   ├── consciousness_detector.rs  # Custom: Extended for sim-based ethical checks
│   │   ├── privacy_enforcer.rs  # Custom: Differential privacy for sim data
│   │   ├── audit_trail.rs  # Custom: Traceable logs; **now with english_reasoning_logger.rs for plain English traces**
│   │   └── **thinking_auditor.rs**  # **New: Module for auditing agent reasoning paths, integrating 10th man logs**
│   └── py_bindings/  # Custom: PyO3 crates to expose Rust to Python (e.g., for real-time audit in sims/queries)
├── middleman/  # Data Pipeline (Mixed Rust/Python; extended for query/learning feeds)
│   ├── interceptor.rs  # Custom Rust: Secure data capture from users/APIs
│   └── learning_loop.py  # Custom Python: Pipe data to sims/queries; **now feeds into ethical memory**
├── orchestration/  # Agentic/Phased Control (Python; **expanded for 10th man, querying, and simulations**)
│   ├── openssa/  # Submodule: Forked OpenSSA (for agents; adapt for multi-agent hierarchies)
│   ├── agents/  # Custom extensions: Domain-specific agents
│   │   ├── base_agent.py  # Shared class with access to ethical memory
│   │   ├── domain_expert.py  # Polymath agents (1-9) with specialized domains
│   │   └── **tenth_man_agent.py**  # **New: Devil's Advocate logic—disagrees with consensus, draws from ethical memory**
│   │   └── multi_agent_deliberation.py  # Consensus mechanism with 10th man rule
│   ├── **external_ai_integration/**  # **New submodule: For continuous autonomous querying**
│   │   ├── api_wrappers.py  # Clients for Grok4, GPT-4, Claude4, Gemini 2.5
│   │   └── query_orchestrator.py  # Logic for generating/processing queries (e.g., ethics topics first)
│   ├── phase_manager.py  # Custom: Switch from middleman (query-heavy) to independent (sim/memory-only)
│   ├── api_endpoints.py  # Custom: FastAPI for premedpro integration
│   └── **simulation/**  # **New: For long-term ethical memory via human life simulations**
│       ├── life_simulator.py  # Core sim generator (text-based narratives, inspired by Keleth PoC)
│       ├── ethical_distiller.py  # Extracts lessons from sims into memory (entropy-based)
│       └── sim_debater.py  # Integrates 10th man—agents debate sim outcomes for ethical refinement
├── utils/  # Shared Utilities (Cross-language; expanded for new features)
│   ├── data_streams.py  # Custom: Processing user/AI inputs; **now includes sim data feeders**
│   ├── testing/  # Custom tests: Unit/integration (e.g., ethical sim scenarios, query mocks)
│   └── build_scripts/  # Scripts: e.g., build.sh to compile Rust/bind to Python
├── docker/  # Deployment (for enterprise/local runs; updated for sim/query scaling)
│   ├── Dockerfile  # Multi-stage: Python/Rust/Julia, with GPU support for neural sims
│   └── docker-compose.yml  # Services for layers (e.g., simulation as scalable microservice)
└── scripts/  # Automation (expanded for new workflows)
    ├── setup.sh  # Clone submodules, install deps
    ├── train_ethical_memory.sh  # Script to run batch simulations for ethics bootstrap
    └── **autonomous_learning.sh**  # **New: Triggers querying and sim loops for continuous learning**