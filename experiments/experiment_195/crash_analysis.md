CLASSIFICATION: FIX

The crash is a `torch.compile` / Inductor bug: `InductorError: ValueError: The argument '-(I)' is not comparable.`

The new state-augmented attention code concatenates memory tokens with real chunk tokens (`torch.cat([mem_k, kc], dim=2)`) and then applies `masked_fill` with a concatenated boolean mask. This creates tensor stride patterns that cause sympy (used internally by Inductor's loop simplifier) to produce imaginary number expressions it can't compare, crashing in `sizevars.py:stride_vars`.

Fix: extracted the entire augmented chunk attention loop into a standalone function `_augmented_chunk_attn` decorated with `@torch.compiler.disable`. This makes Inductor skip compilation for just the problematic local attention path while the rest of the model (SDPA layers, MLP, embeddings, optimizer) remains fully compiled. The experimental idea (memory tokens competing in softmax) is preserved exactly.
