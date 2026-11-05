# Obtainable: source-label agreement and recorded behavior

Fixed 209 families: 153 condition and 56 identity. Accuracy, requests and completion use every expected family; missing or abstaining final answers remain in the denominator.
Each arm requires all 209 episode keys before any score is shown. -- means unavailable, never zero. New pixels do not certify sufficiency.
Grounded accuracy, evidence-found rate, correct-evidence-use rate and unsupported-decision rate are null until independent evidence sets exist.

| Model | Initial acc. | Full-access acc. | Agent condition | Agent identity | Agent overall | Request | New image | Repeated image | Finish |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 28.2 | 60.8 | 29.4 | 53.6 | 35.9 | 12.9 | 12.4 | 0.5 | 64.1 |
| Qwen3-VL-8B | 34.9 | 72.7 | 45.8 | 71.4 | 52.6 | 5.7 | 5.7 | 0.0 | 85.2 |
| Qwen2.5-VL-7B | 9.6 | 10.5 | 0.0 | 0.0 | 0.0 | 0.5 | 0.5 | 0.5 | 76.6 |
| InternVL3.5-8B | 49.3 | 53.1 | 0.0 | 17.9 | 4.8 | 8.6 | 8.6 | 2.4 | 4.8 |
| Qwen3-VL-2B | 51.2 | 61.7 | 19.6 | 46.4 | 26.8 | 1.9 | 1.0 | 1.0 | 45.9 |
| Qwen3-VL-4B | 15.3 | 40.7 | 10.5 | 46.4 | 20.1 | 5.7 | 5.7 | 0.0 | 34.9 |
| LLaVA-OneVision-7B | 30.6 | 40.7 | 0.7 | 0.0 | 0.5 | 10.5 | 2.4 | 3.3 | 0.5 |
| SmolVLM2-2.2B | 50.7 | 50.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| InternVL3.5-14B | 30.6 | 52.6 | 1.3 | 58.9 | 16.7 | 8.1 | 8.1 | 0.0 | 29.2 |
| Pixtral-12B | 30.1 | 42.6 | 32.0 | 7.1 | 25.4 | 8.6 | 8.6 | 0.0 | 62.2 |
| GPT-6.1 Sol | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| GPT-6.1 Terra | -- | -- | 75.8 | 67.9 | 73.7 | 100.0 | 100.0 | 0.5 | 100.0 |
| GPT-5.4 mini | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| GPT-4.1 mini | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| GPT-6 Luna | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| GPT-6 Luna (Decisions) | -- | -- | 49.7 | 23.2 | 42.6 | 79.9 | 78.0 | 23.9 | 99.5 |

Coverage by arm (initial / full / interactive):

- Qwen3.5-9B: 209/209/209 of 209.
- Qwen3-VL-8B: 209/209/209 of 209.
- Qwen2.5-VL-7B: 209/209/209 of 209.
- InternVL3.5-8B: 209/209/209 of 209.
- Qwen3-VL-2B: 209/209/209 of 209.
- Qwen3-VL-4B: 209/209/209 of 209.
- LLaVA-OneVision-7B: 209/209/209 of 209.
- SmolVLM2-2.2B: 209/209/209 of 209.
- InternVL3.5-14B: 209/209/209 of 209.
- Pixtral-12B: 209/209/209 of 209.
- GPT-6.1 Sol: 0/0/75 of 209.
- GPT-6.1 Terra: 144/144/209 of 209.
- GPT-5.4 mini: 0/0/0 of 209.
- GPT-4.1 mini: 0/0/0 of 209.
- GPT-6 Luna: 0/0/18 of 209.
- GPT-6 Luna (Decisions): 0/0/209 of 209.

The Decisions row is an additional interface for GPT-6 Luna, not a sixteenth distinct model.
