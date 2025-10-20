from typing import Any, Iterable, Optional, Tuple

class EnvCompat:
    """
    Wraps any Gym/Gymnasium env and exposes the Gymnasium-style API:
      reset() -> (obs, info)
      step(a) -> (obs, reward, terminated, truncated, info)

    Use GymCompat.make(...) to construct and wrap from either backend.
    """
    def __init__(self, env, backend: str):
        self.env = env
        self._backend = backend  # 'gymnasium' or 'gym'

        # passthrough common attributes if they exist
        for attr in ("observation_space", "action_space", "spec", "metadata", "reward_range"):
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

    # ---------- public factory ----------
    @classmethod
    def make(
        cls,
        id: str,
        *,
        backend: str = "auto",                  # 'gymnasium' | 'gym' | 'auto'
        register_modules: Optional[Iterable[str]] = None,  # e.g. ('nasim',) or ('nasimemu',)
        prefer: Optional[str] = None,           # optional hard preference if both exist
        **kwargs: Any,
    ) -> "EnvCompat":
        """
        Build env `id` from whichever backend works, importing modules that
        register envs at import time (e.g. nasim / nasimemu).

        Heuristics:
          - If register_modules is not given, we guess from the id string.
          - If backend='auto', we try Gymnasium first, then Gym (or reversed if prefer='gym').
        """
        errors = {}

        if 'observation_format' in kwargs.keys():
            graph = 'graph' in kwargs['observation_format']
        else:
            graph = False

        # 1) Import modules that perform registration (no-op if not installed).
        if register_modules is None and id:
            low = id.lower()
            if "nasimemu" in low:
                register_modules = ("nasimemu",)
            elif "nasim" in low:
                register_modules = ("nasim",)
            else:
                register_modules = ()

        for mod in register_modules:
            try:
                __import__(mod)
            except Exception as e:
                # don't fail here; we might still succeed if already registered
                errors[f"import:{mod}"] = f"{type(e).__name__}: {e}"

        # 2) Decide candidate backends and import them.
        candidates = []
        if backend in ("gymnasium", "auto"):
            try:
                import gymnasium as gymn  # type: ignore
                candidates.append(("gymnasium", gymn))
            except Exception as e:
                errors["gymnasium_import"] = f"{type(e).__name__}: {e}"
        if backend in ("gym", "auto"):
            try:
                import gym as oldgym  # type: ignore
                candidates.append(("gym", oldgym))
            except Exception as e:
                errors["gym_import"] = f"{type(e).__name__}: {e}"

        # Optional preference ordering.
        if prefer is not None:
            candidates.sort(key=lambda x: 0 if x[0] == prefer else 1)
        else:
            # Sensible default: if the id looks like nasim, try Gymnasium first;
            # if it looks like nasimemu, try Gym first.
            low = id.lower()
            if "nasimemu" in low:
                candidates.sort(key=lambda x: 0 if x[0] == "gym" else 1)
            elif "nasim" in low:
                candidates.sort(key=lambda x: 0 if x[0] == "gymnasium" else 1)

        # 3) Try to create the env with each backend.
        last_exc = None
        for name, gm in candidates:
            try:
                env = gm.make(id, **kwargs)
                return cls(env, backend=name)
            except Exception as e:
                last_exc = e

        # 4) If we get here, we failed; raise a helpful error.
        detail = "; ".join(f"{k}={v}" for k, v in errors.items())
        raise RuntimeError(
            f"GymCompat.make failed for id='{id}'. Tried backends: {[n for n,_ in candidates]}. "
            + (f"Notes: {detail}. " if detail else "")
            + (f"Last error: {type(last_exc).__name__}: {last_exc}" if last_exc else "")
        )

    # ---------- normalized API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Gymnasium path
        try:
            return self.env.reset(seed=seed, options=options)
        except TypeError:
            # Old Gym variants
            obs = None
            if seed is not None:
                # best-effort seeding across history
                try:
                    obs = self.env.reset(seed=seed)
                except TypeError:
                    if hasattr(self.env, "seed"):
                        self.env.seed(seed)
                    obs = self.env.reset()
            else:
                obs = self.env.reset()
            return obs, {}

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:  # Gymnasium
            return out
        # Old Gym: (obs, reward, done, info)
        obs, reward, done, info = out
        terminated = bool(done)
        truncated = bool(info.get("TimeLimit.truncated", False))
        return obs, reward, terminated, truncated, info

    # ---------- passthrough helpers ----------
    def render(self, *a, **k):  # noqa: D401
        return self.env.render(*a, **k)

    def close(self):
        return self.env.close()