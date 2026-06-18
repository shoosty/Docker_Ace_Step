# ACE-Step v59 deploy notes

What v59 ships (handler-only — no image-side surgery):

1. **Sweet-spot recipe baked into handler defaults.** After the 9-test
   E-series listening matrix on 2026-06-18, Stephen confirmed two
   defaults that the v58 handler had wrong:
   - `dcw_enabled` flipped from `True` → `False`
   - `guidance_scale` flipped from `7.0` → `10.0`
   - Everything else (28 steps, ODE sampler, shift 3.0, CFG window
     0.0→0.95) is unchanged from v58.
2. **Documented the failure modes in the docstring** so future-Stephen
   doesn't repeat the recipe search. Three landmines worth knowing:
   - DCW `mode="double"` (the v58 default) was scored "terrible" /
     scrambled across multiple genres.
   - Combining `inference_steps=50` with `guidance_scale=10` (the two
     best individual knobs) was scored "annoying drums / awful" — knobs
     don't compound on this model.
   - Raising guidance past 10 (e.g. 12) was scored "crystal voice then
     muffled."
3. **Studio is already in sync.** `feature/ace-v59-sweetspot` in
   `shoosty-studio` sends `dcw_enabled: false` + `guidance_scale: 10.0`
   on every ACE submit, so quality is good on the existing v58 endpoint
   right now. v59's value-add is: once deployed, the studio can stop
   sending those overrides (or any caller can omit them safely).

What v59 does NOT ship (deferred to v60):
- `flash_attn` import failure on RTX 5090 + CUDA 12.8 (symbol mismatch).
  Currently falls back to PyTorch sdpa kernels — works but slower + more
  CPU-bound. Fix is a CUDA-12.8-ABI flash_attn wheel rebuild.
- `nano-vllm` triton ABI break (`triton_key` import error). Falls back
  to PyTorch LM backend — works but slower. Fix is pinning a compatible
  triton wheel version.

Wall-clock impact when v60 lands: expected ~19s → ~8-10s on a 60s song
at the current 28-step / guidance-10 recipe. Quality impact of v60: zero
— same audio out, just produced faster.

## Build + push

```bash
cd ~/Code/Docker_Ace_Step
git checkout feature/v59-sweetspot-defaults
docker build --platform linux/amd64 -t shoosty1/ace-step:v59 .
docker push shoosty1/ace-step:v59
```

## RunPod endpoint update

RunPod console → endpoint `2y4tcs922vur4c` → New Release →
`shoosty1/ace-step:v59`. Smoke test from `/preview/generate` (or curl
the `/run` URL with a minimal `{caption, lyrics, duration}` payload —
no quality params needed now). Audio should sound the same as the
verified-good Path B output from today's E7 / D-series tests.

## Rollback

Two paths:

- **Tag swap** — flip the endpoint back to `shoosty1/ace-step:v58`. v58
  image is unchanged from before today's work; the bad quality returns
  but the studio's per-request overrides still keep output good.
- **Snapshot rebuild** — see `PINNED.md` for the v55 rollback recipe.
  v55 is the deepest-known-good (first end-to-end song success
  2026-06-13). Use this if v58 and v59 both behave unexpectedly.
