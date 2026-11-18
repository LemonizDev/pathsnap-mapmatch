# PathSnap Agent Instructions

## Project

PathSnap is an ML-based GPS trajectory map-matching system for the hackathon. It takes noisy GPS trajectories, identifies likely road candidates, and reconstructs the most likely route using a model plus HMM/Viterbi-style sequence logic.

Repository: `https://github.com/LemonizDev/pathsnap-mapmatch.git`

The canonical branch is `main`.

## Two-laptop setup

- Backend laptop: runs the FastAPI/API and ML pipeline.
- Frontend laptop: runs the user interface and calls the backend over the local network.

The frontend must use the backend laptop's LAN IP, not `localhost`, for example `http://192.168.1.5:8000`. The backend must listen on `0.0.0.0) when it needs to be reached from the frontend laptop. Both laptops must be on the same network, and the backend port must be allowed through the firewall if necessary.

## Git workflow

Before starting work:

    git checkout main
    git pull --ff-only origin main

After a focused change:

    git status
    git add <specific-files>
    git commit -m "Describe the change"
    git push origin main

Do not force-push, reset shared history, or discard another person's work. Do not push unless explicitly authorized. The frontend agent normally pulls changes but does not push.

Before pulling, preserve or commit local work; never use destructive commands to discard it.

## Data and generated files

- Keep the supplied dataset in `Student folder/`.
- Do not delete, rename, or modify supplied raw data unless explicitly required.
- Do not commit secrets, `.env` files, virtual environments, caches, logs, model artifacts, or dependency folders.
- Respect `.gitignore` and avoid committing large generated outputs.

## Implementation rules

- Read the existing code and documentation before changing architecture.
- Make the smallest change that satisfies the task.
- Preserve evaluator-compatible interfaces and behavior.
- Add or update tests for meaningful behavior changes.
- Keep backend and frontend changes separated when possible.
- Do not invent test results, metrics, routes, or API contracts.
- Document assumptions in the final report.

## Handoff requirements

The backend agent must report the exact startup command, host and port, health endpoint, LAN reachability, and required environment variables.

The frontend agent must report the exact install and startup commands, frontend URL and port, API base URL location, and whether it was tested against the backend laptop's LAN IP.

## Verification before the 11:30 demonstration

1. Start the backend on the backend laptop.
2. Confirm its health endpoint locally.
3. Confirm it listens on the LAN interface.
4. Start the frontend on the frontend laptop.
5. Configure the frontend with the backend laptop's LAN IP.
6. Submit one representative trajectory through the UI.
7. Confirm the backend receives it and returns a valid result.
8. Keep a command-line or direct API fallback ready if the UI or network fails.

When reporting completion, include commands actually run and their observed results.

