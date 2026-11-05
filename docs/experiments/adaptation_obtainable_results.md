# Post-training on Obtainable

All rows use the same 209 families (153 condition, 56 identity) and the interactive interface. Accuracy is source-label agreement, not grounded accuracy. Missing/invalid final answers and abstentions score zero. Finish includes explicit abstentions; efficiency includes every attempt. New image counts requests returning previously unseen RGB pixels, not sufficient evidence. Sufficient scores are untouched.

| Model | Policy | Condition acc. % | Identity acc. % | Overall acc. % | New image: condition % | New image: identity % | Request % | Finish % | Time s | Output tokens | Model calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | Base | 0.0 | 0.0 | 0.0 | 0.0 | 1.8 | 0.5 | 76.6 | 15.4 | 280 | 6.63 |
| Qwen2.5-VL-7B | SFT | 51.6 | 75.0 | 57.9 | 0.0 | 0.0 | 0.0 | 100.0 | 5.1 | 112 | 1.00 |
| Qwen2.5-VL-7B | RL | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 | 1.4 | 100.0 | 4.4 | 69 | 2.67 |
| Qwen2.5-VL-7B | SFT+RL | 52.9 | 62.5 | 55.5 | 0.0 | 0.0 | 0.0 | 100.0 | 4.7 | 112 | 1.00 |
| Qwen3-VL-2B | Base | 19.6 | 46.4 | 26.8 | 0.0 | 3.6 | 1.9 | 45.9 | 13.4 | 409 | 7.68 |
| Qwen3-VL-2B | SFT | 54.2 | 73.2 | 59.3 | 0.0 | 0.0 | 0.0 | 100.0 | 6.2 | 139 | 1.00 |
| Qwen3-VL-2B | RL | 13.1 | 64.3 | 26.8 | 0.0 | 0.0 | 0.0 | 44.5 | 17.4 | 408 | 7.02 |
| Qwen3-VL-2B | SFT+RL | 50.3 | 73.2 | 56.5 | 0.0 | 1.8 | 30.6 | 96.2 | 8.2 | 205 | 1.55 |
| Qwen3-VL-8B | Base | 45.8 | 71.4 | 52.6 | 5.2 | 7.1 | 5.7 | 85.2 | 20.5 | 397 | 5.19 |
| Qwen3-VL-8B | SFT | 63.4 | 76.8 | 67.0 | 53.6 | 0.0 | 39.2 | 100.0 | 8.0 | 135 | 1.40 |
| Qwen3-VL-8B | RL | 51.6 | 66.1 | 55.5 | 5.9 | 8.9 | 7.2 | 92.8 | 23.1 | 395 | 4.73 |
| Qwen3-VL-8B | SFT+RL | 66.0 | 83.9 | 70.8 | 60.1 | 0.0 | 44.0 | 100.0 | 9.2 | 153 | 1.47 |
