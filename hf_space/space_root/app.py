# HF Space root entrypoint (gradio SDK reads `app_file: app.py`).
#
# This is a thin shim — the real app lives in the `hf_space` package. The repo
# root is on sys.path when HF executes this file, so `from hf_space.app import
# main` resolves (hf_space has __init__.py).
#
# `import spaces` MUST precede torch on ZeroGPU; it is absent in local/CI envs,
# so it is guarded. The heavy ZeroGPU engine is wired in a later stage (S46+);
# today this just launches the existing backend-streaming Gradio shell.
try:
    import spaces  # noqa: F401  (ZeroGPU runtime hook; absent locally)
except ImportError:
    pass

from hf_space.app import main

main()
