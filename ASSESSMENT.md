# Original self-study assessment

Not an official NUS rubric.

## Is the moving average a diffusion model?

No. It lacks a trained denoiser/score model, conditioning, a reverse diffusion schedule and a generative sampling process.

## Is low temperature a factuality control?

No. It concentrates a distribution but cannot make unsupported claims true.

## Build task

Run a licensed pretrained text or image model locally if hardware allows; record model/version/license, seed, settings, outputs and a structured failure analysis. Do not buy compute.

## Suggested scoring, 100 points
- Reproducible clean run and tests: 25
- Correct mechanism and explanation: 25
- Evaluation with a held-out set or explicit mechanism checks: 20
- Failure cases, provenance and limits: 20
- Readable demo and documentation: 10

Pass suggestion: 70, with no fabricated results, private-data exposure or unsupported production claims. This is a self-study convention.
