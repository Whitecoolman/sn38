# Validating on SN38

## Overview

Validators evaluate miner models and set weights on-chain. The validator runs inside a TEE (Trusted Execution Environment) on Phala Cloud to keep the evaluation dataset private.

## Prerequisites

- Registered validator on SN38 with sufficient stake
- [Phala Cloud](https://cloud.phala.com/) account with a GPU TEE instance provisioned
- OpenAI API key (for the LLM judge in Stage 2)

## Deploy

### 1. Install and log in to Phala CLI

```bash
npm install -g phala
phala login
```

### 2. Deploy the validator

```bash
phala deploy \
  -c docker-compose.validator.yml \
  --pre-launch-script scripts/prelaunch.sh \
  -e HOTKEY_FILE_CONTENT="$(cat ~/.bittensor/wallets/validator/hotkeys/default)" \
  -e OPENAI_API_KEY=sk-xxx
```

> **Note**: Replace the hotkey path with your actual wallet path (e.g. `~/.bittensor/wallets/<your-wallet>/hotkeys/<your-hotkey>`).

### 3. Verify attestation

```bash
phala cvms attestation
```

The `compose-hash` in the event log proves the validator is running the correct, unmodified code.

## Frequency

The validator runs **once per round** (typically weekly). Execution time depends on the number of miner submissions and model inference speed. The validator exits automatically after setting weights — no need to run it 24/7.
