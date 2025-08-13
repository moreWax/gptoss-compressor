#!/usr/bin/env python3
# CLI wrapper for de-quantizing GPT-OSS MXFP4 weights and compressing with LLM Compressor.
#
# Features:
# - Dequantize MXFP4 -> BF16/FP16
# - Quantize with GPTQ and/or AWQ
# - Schemes: W4A16, W8A8, W4A8 (experimental, GPTQ-only)
# - Optional 2:4 sparsity (SparseGPT)
# - Per-group controls: experts vs. non-expert weights
# - W4A8 runtime detection/installer (QQQ CUTLASS kernels)
# - verify-runtime subcommand with optional FP16 vs W4A8 microbench

import argparse
import os
import sys
import subprocess
from typing import List, Optional

# ---------- Helpers for W4A8 backend ----------
def _detect_w4a8_backend_name() -> str:
    try:
        import importlib
        try:
            q = importlib.import_module("QQQ")
            ver = getattr(q, "__version__", "unknown")
            return f"QQQ (cutlass), pkg=QQQ, ver={ver}"
        except Exception:
            q = importlib.import_module("qqq")
            ver = getattr(q, "__version__", "unknown")
            return f"QQQ (cutlass), pkg=qqq, ver={ver}"
    except Exception:
        return "unavailable"

def has_w4a8_runtime() -> bool:
    """Best-effort check: QQQ installs a Python package 'QQQ' (capitalized) or 'qqq'."""
    try:
        import importlib
        try:
            importlib.import_module("QQQ")
            return True
        except Exception:
            importlib.import_module("qqq")
            return True
    except Exception:
        return False

def install_qqq(non_interactive: bool = False) -> bool:
    """Install QQQ CUTLASS kernels from GitHub via pip. Returns True on success."""
    print("[compress_gptoss] Installing QQQ CUTLASS kernels from GitHub (builds CUDA extensions)...", flush=True)
    cmd = [sys.executable, "-m", "pip", "install", "-U", "git+https://github.com/HandH1998/QQQ.git"]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[compress_gptoss] QQQ installation failed: {e}", flush=True)
        return False
    return has_w4a8_runtime()

def log(msg: str):
    print(f"[compress_gptoss] {msg}", flush=True)


# ---------- Transformers (dequant) ----------
try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import Mxfp4Config
        _HAS_MXFP4CONFIG = True
    except Exception:
        Mxfp4Config = None
        _HAS_MXFP4CONFIG = False
except Exception as e:
    print("Transformers is required for the dequantization step:", e, file=sys.stderr)
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    Mxfp4Config = None
    _HAS_MXFP4CONFIG = False

# ---------- LLM Compressor ----------
try:
    from llmcompressor import oneshot  # type: ignore
    from llmcompressor.modifiers.quantization import GPTQModifier  # type: ignore
    from llmcompressor.modifiers.awq import AWQModifier  # type: ignore
    from llmcompressor.modifiers.obcq import SparseGPTModifier  # type: ignore
except Exception as e:
    oneshot = None  # type: ignore
    GPTQModifier = None  # type: ignore
    AWQModifier = None  # type: ignore
    SparseGPTModifier = None  # type: ignore

# ---------- Commands ----------
def cmd_dequantize(args: argparse.Namespace) -> None:
    if AutoModelForCausalLM is None:
        raise RuntimeError("Transformers not available. Install it to use dequantize.")

    import torch
    src = args.src
    dst = args.dst
    os.makedirs(dst, exist_ok=True)

    # Determine project root to check for local model directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Prefer a local model directory matching 'src' under project root
    local_src_path = os.path.join(project_root, src)
    if os.path.isdir(local_src_path):
        log(f"Found local model directory at: {local_src_path}. Using local model.")
        src = local_src_path
        skip_cache = True
    else:
        skip_cache = False
        # Determine Hugging Face cache directory: env vars or project root
        env_cache = os.environ.get("HF_HOME") or os.environ.get("HF_CACHE") or os.environ.get("TRANSFORMERS_CACHE")
        if env_cache:
            hf_cache_dir = os.path.expanduser(env_cache)
        else:
            hf_cache_dir = project_root
        log(f"Checking Hugging Face cache at: {hf_cache_dir}")

    if not skip_cache:
        # Check for a local cache directory matching the model name under hub
        try:
            owner, repo = src.split('/', 1)
            cache_folder = f"models--{owner}--{repo}"
            model_cache_dir = os.path.join(hf_cache_dir, 'hub', cache_folder)
        except ValueError:
            model_cache_dir = os.path.join(hf_cache_dir, 'hub', src.replace('/', '--'))
        if os.path.isdir(model_cache_dir) and os.path.isfile(os.path.join(model_cache_dir, 'config.json')):
            found_in_cache = True
            log(f"Model {src} found in Hugging Face hub cache directory {model_cache_dir}. Using cache.")
        else:
            log(f"Model {src} not found in Hugging Face hub cache directory {model_cache_dir}. Will download from remote.")
    log(f"Checking Hugging Face cache at: {hf_cache_dir}")

    # Try to find the model in cache first
    from transformers.utils import cached_file, EntryNotFoundError
    found_in_cache = False
    try:
        # Try to resolve config.json in cache
        _ = cached_file(src, "config.json", cache_dir=hf_cache_dir)
        found_in_cache = True
        log(f"Model {src} found in Hugging Face cache. Will load from cache.")
    except EntryNotFoundError:
        log(f"Model {src} not found in Hugging Face cache. Will download from remote.")
    except Exception as e:
        log(f"Cache check error: {e}. Will attempt to download.")

    qconf = None
    if _HAS_MXFP4CONFIG:
        qconf = Mxfp4Config(dequantize=True)  # upcast MXFP4->BF16 on load

    # Use sequential device mapping to minimize memory usage
    device_map = getattr(args, 'device_map', 'auto')
    if device_map == 'cpu-only':
        device_map = "cpu"

    log(f"Using device_map: {device_map} for memory-efficient loading")

    # Build kwargs for model loading
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if args.dtype.lower() == "bf16" else torch.float16,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,  # Allow custom model code
        "cache_dir": hf_cache_dir,
    }
    # Apply GPU memory limit if requested
    if hasattr(args, 'max_gpu_memory') and args.max_gpu_memory:
        # max_memory can accept dict mapping device id to memory string
        try:
            model_kwargs['max_memory'] = {0: args.max_gpu_memory}
            log(f"Applying GPU memory limit: device 0 -> {args.max_gpu_memory}")
        except Exception as e:
            log(f"Failed to apply max_memory: {e}")
    # Only add quantization_config if it's not None and valid
    if qconf is not None:
        model_kwargs["quantization_config"] = qconf
    # Only add offload_folder if specified
    offload_folder = getattr(args, 'offload_folder', None)
    if offload_folder:
        model_kwargs["offload_folder"] = offload_folder

    # NEW: get the config first with trust_remote_code, then pass it in
    try:
        config = AutoConfig.from_pretrained(src, trust_remote_code=True, cache_dir=hf_cache_dir)
    except Exception as e:
        log(f"Failed to load AutoConfig for {src}: {e}")
        raise

    # Load model with the config; handle rope_scaling quirk and retry on CAS errors
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(src, config=config, **model_kwargs)
        except ValueError as e:
            if "rope_scaling" in str(e):
                log("Detected rope_scaling config issue, trying with config override...")
                # Convert YARN rope scaling to standard format if needed
                if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
                    rope_config = config.rope_scaling
                    if isinstance(rope_config, dict) and 'rope_type' in rope_config:
                        config.rope_scaling = {
                            'type': rope_config.get('rope_type', 'linear'),
                            'factor': rope_config.get('factor', 1.0)
                        }
                        log(f"Converted rope_scaling: {config.rope_scaling}")
                # Retry with adjusted config
                model = AutoModelForCausalLM.from_pretrained(src, config=config, **model_kwargs)
            else:
                raise
    except Exception as e:
        err = str(e)
        # Retry download on CAS service errors
        if any(x in err for x in ["CAS service error", "Reqwest Error", "error decoding response body"]):
            log("Encountered CAS service error, retrying download with force_download=True...")
            try:
                model = AutoModelForCausalLM.from_pretrained(src, config=config, force_download=True, **model_kwargs)
            except Exception as e2:
                log(f"Retry with force_download failed: {e2}")
                raise
        else:
            log(f"All loading attempts failed. Last error: {e}")
            raise

    # Basic diagnostics and sanity checks
    try:
        total_params = sum(p.numel() for p in model.parameters())
        bytes_per_param = 2 if getattr(args, 'dtype', 'fp16').lower() == 'fp16' else 2  # bf16/fp16 both 2 bytes
        est_gb = (total_params * bytes_per_param) / (1024 ** 3)
        model_cls = model.__class__.__name__
        try:
            from transformers import AutoConfig as _AutoConfig
            cfg = _AutoConfig.from_pretrained(src, trust_remote_code=True)
            model_type = getattr(cfg, 'model_type', 'unknown')
            archs = getattr(cfg, 'architectures', [])
        except Exception:
            model_type = 'unknown'
            archs = []
        log(f"Loaded model class: {model_cls} (model_type={model_type})")
        log(f"Parameter count: {total_params:,} (~{est_gb:.1f} GB at {args.dtype.upper()})")
        # Strict sanity: if user targets GPT-OSS and model doesn't look like GPT-OSS, abort
        expect_gptoss = any(x in src.lower() for x in ["gpt-oss", "gpt_oss"]) or any(
            isinstance(a, str) and "gptoss" in a.lower() for a in archs
        )
        if expect_gptoss and not (
            (isinstance(model_type, str) and model_type.lower() == "gpt_oss") or
            (isinstance(model_cls, str) and "gptoss" in model_cls.lower())
        ):
            log("ERROR: The loaded config/model doesn't identify as GPT-OSS (model_type != 'gpt_oss').")
            log("       This typically happens if Transformers fell back to Llama config.")
            log("       Ensure trust_remote_code=True and avoid overriding config with Llama.")
            raise RuntimeError("Loaded wrong architecture (not GPT-OSS)")
        # Heuristic warning for obviously wrong size when user expects 120B
        if any(x in src.lower() for x in ["120b", "gpt-oss-120b"]) and est_gb < 100:
            log("WARNING: Estimated FP16 size < 100GB for a '120B' model. This suggests a mis-loaded architecture.")
            log("         Ensure trust_remote_code is honored and no config override forces Llama.")
    except Exception:
        pass

    # Tokenizer: trust remote code as well
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True, use_fast=False)

    if args.dtype.lower() == "fp16":
        log("Model loaded as FP16. No further casting needed.")
    else:
        log("Model loaded as BF16. No further casting needed.")

    log(f"Saving to: {dst}")
    model.save_pretrained(dst, safe_serialization=True)
    # Overwrite saved config with the source config to preserve custom fields (e.g., GPT-OSS model_type)
    try:
        from transformers import AutoConfig as _AutoConfig
        src_cfg = _AutoConfig.from_pretrained(src, trust_remote_code=True)
        # Avoid including unserializable attrs; rely on Transformers' to_dict
        cfg_path = os.path.join(dst, "config.json")
        import json
        with open(cfg_path, "w") as f:
            json.dump(src_cfg.to_dict(), f, indent=2)
        log("Wrote source config to output config.json to preserve GPT-OSS settings.")
    except Exception as e:
        log(f"Note: could not resave source config.json ({e}); proceeding.")
    tok.save_pretrained(dst)
    log("Done.")

# ---------- Spec helpers ----------
def _parse_legacy_spec(spec: Optional[str]) -> Optional[str]:
    if spec is None:
        return None
    s = spec.strip().lower()
    if s in ("int4", "w4a16"):
        return "W4A16"
    if s in ("w8a8", "int8"):
        return "W8A8"
    if s in ("w4a8",):
        return "W4A8"
    raise ValueError(f"Unknown spec '{spec}'. Use int4/w4a16, int8/w8a8, or w4a8.")

def _resolve_pair(dtype: Optional[str], scheme: Optional[str]) -> Optional[str]:
    """Resolve (dtype, scheme) to 'W4A16', 'W8A8', or 'W4A8'."""
    if not dtype and not scheme:
        return None
    d = (dtype or "").lower()
    sc = (scheme or "").lower()
    if d == "int4":
        if not sc:
            sc = "w4a16"
        if sc not in ("w4a16", "w4a8"):
            raise ValueError("For int4, scheme must be 'w4a16' or 'w4a8'.")
        return "W4A8" if sc == "w4a8" else "W4A16"
    if d == "int8":
        if not sc:
            sc = "w8a8"
        if sc != "w8a8":
            raise ValueError("For int8, scheme must be 'w8a8'.")
        return "W8A8"
    raise ValueError("dtype must be one of: int4, int8.")

def _build_gptq_modifier(scheme: str, targets, ignore, group_size, symmetric, block_size, dampening_frac, sequential_update, offload_hessians):
    return GPTQModifier(
        scheme=scheme,
        targets=targets,
        ignore=ignore,
        group_size=group_size,
        symmetric=symmetric,
        block_size=block_size,
        dampening_frac=dampening_frac,
        sequential_update=sequential_update,
        offload_hessians=offload_hessians,
    )

def _build_awq_modifier(spec: str, targets, ignore, group_size, symmetric):
    if spec != "W4A16":
        raise ValueError("AWQ supports weight-only INT4 here; use int4/w4a16.")
    return AWQModifier(
        targets=targets,
        ignore=ignore,
        num_bits=4,
        group_size=group_size,
        symmetric=symmetric,
    )

def _make_quant_stage(algo: str, spec: str, targets, ignore, args: argparse.Namespace):
    if algo == "gptq":
        return _build_gptq_modifier(
            scheme=spec,
            targets=targets,
            ignore=ignore,
            group_size=args.group_size,
            symmetric=args.symmetric,
            block_size=args.block_size,
            dampening_frac=args.dampening_frac,
            sequential_update=not args.no_sequential_update,
            offload_hessians=args.offload_hessians,
        )
    elif algo == "awq":
        return _build_awq_modifier(
            spec=spec,
            targets=targets,
            ignore=ignore,
            group_size=args.group_size,
            symmetric=args.symmetric,
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

def _detect_experts_regex(model_in: str) -> str:
    """Try to load model on 'meta' to infer an expert regex. Fallback: 're:.*experts.*'."""
    default = r"re:.*experts.*"
    try:
        if AutoModelForCausalLM is None:
            return default
        # Use meta device mapping to load only structure, not weights
        model = AutoModelForCausalLM.from_pretrained(
            model_in,
            device_map="meta",
            torch_dtype=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,  # May be needed for some models
        )
        names = [n.lower() for n, _ in model.named_modules()]
        keys = ["experts", "expert", "moe", "mixtureofexperts", "mixture_of_experts"]
        hits = [n for n in names if any(k in n for k in keys)]
        if not hits:
            return default
        if any("experts" in h for h in hits):
            return r"re:.*experts.*"
        return r"re:.*(expert|moe).*"
    except Exception:
        return default

def _ensure_w4a8_support_or_handle(args, who: str, spec: str) -> str:
    """If W4A8 requested, ensure runtime exists. Optionally try-install QQQ or fall back to W4A16."""
    if spec != "W4A8":
        return spec
    if has_w4a8_runtime():
        log(f"W4A8 runtime detected for {who}.")
        return spec

    log(f"W4A8 requested for {who}, but runtime kernel not detected.")
    if args.try_install_qqq:
        do_install = True
        if not args.yes:
            resp = input("Install QQQ CUTLASS kernels now? [y/N]: ").strip().lower()
            do_install = resp in ("y", "yes")
        if do_install:
            ok = install_qqq(non_interactive=args.yes)
            if ok:
                return "W4A8"
            else:
                log("QQQ installation failed.")
        else:
            log("User declined QQQ installation.")

    if args.allow_fallback:
        log(f"Falling back {who} scheme to W4A16.")
        return "W4A16"
    raise RuntimeError("W4A8 requested but runtime not available. Use --try-install-qqq or --allow-fallback.")

def build_recipe(args: argparse.Namespace):
    if oneshot is None:
        raise RuntimeError("llmcompressor not available. Install llmcompressor to quantize.")

    base_ignore: List[str] = [
        "lm_head",
        r"re:.*embed_tokens.*",
        r"re:.*mlp\.router.*",
    ] + (args.extra_ignore or [])

    recipe = []

    if args.with_sparse:
        if SparseGPTModifier is None:
            raise RuntimeError("SparseGPTModifier unavailable. Check llmcompressor install.")
        recipe.append(
            SparseGPTModifier(
                sparsity=args.sparsity,
                mask_structure=args.mask,
                targets="Linear",
                ignore=base_ignore,
                sequential_update=True,
            )
        )

    experts_regex = None
    if args.auto_detect_experts:
        experts_regex = _detect_experts_regex(args.model_in)
    if not experts_regex:
        experts_regex = args.experts_regex or r"re:.*experts.*"

    exp_algo = (args.experts_algo or args.algo).lower()
    mw_algo  = (args.mw_algo or args.algo).lower()

    exp_spec = _resolve_pair(args.experts_type, args.experts_scheme) if (args.experts_type or args.experts_scheme) else _parse_legacy_spec(args.experts) or "W4A16"
    mw_spec  = _resolve_pair(args.mw_type, args.mw_scheme) if (args.mw_type or args.mw_scheme) else _parse_legacy_spec(args.mw) or "W4A16"

    if exp_algo == "awq" and exp_spec != "W4A16":
        raise ValueError("Experts: AWQ requires INT4/W4A16 (weight-only).")
    if mw_algo == "awq" and mw_spec != "W4A16":
        raise ValueError("Model-weights: AWQ requires INT4/W4A16 (weight-only).")
    if exp_algo == "gptq" and exp_spec == "W4A8":
        exp_spec = _ensure_w4a8_support_or_handle(args, "experts", exp_spec)
    if mw_algo == "gptq" and mw_spec == "W4A8":
        mw_spec = _ensure_w4a8_support_or_handle(args, "model-weights", mw_spec)

    exp_ignore = list(base_ignore)
    exp_targets = ["Linear", experts_regex]

    mw_ignore = list(base_ignore) + [experts_regex]
    mw_targets = "Linear"

    recipe.append(_make_quant_stage(exp_algo, exp_spec, exp_targets, exp_ignore, args))
    recipe.append(_make_quant_stage(mw_algo, mw_spec, mw_targets, mw_ignore, args))

    return recipe

def cmd_quantize(args: argparse.Namespace) -> None:

    recipe = build_recipe(args)
    # Determine project root to check for local model directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(project_root, args.model_in)
    if os.path.isdir(local_model_path):
        log(f"Found local model directory at: {local_model_path}. Using local model.")
        args.model_in = local_model_path
        skip_cache = True
    else:
        skip_cache = False
    if not skip_cache:
        # Determine Hugging Face cache directory: env vars or project root
        env_cache = os.environ.get("HF_HOME") or os.environ.get("HF_CACHE") or os.environ.get("TRANSFORMERS_CACHE")
        if env_cache:
            hf_cache_dir = os.path.expanduser(env_cache)
        else:
            hf_cache_dir = project_root
        log(f"Checking Hugging Face cache at: {hf_cache_dir}")
        from transformers.utils import cached_file, EntryNotFoundError
        found_in_cache = False
        try:
            _ = cached_file(args.model_in, "config.json", cache_dir=hf_cache_dir)
            found_in_cache = True
            log(f"Model {args.model_in} found in Hugging Face cache. Will load from cache.")
        except EntryNotFoundError:
            log(f"Model {args.model_in} not found in Hugging Face cache. Will load from local or remote.")
        except Exception as e:
            log(f"Cache check error: {e}. Will attempt to load.")

    if args.dry_run:
        log("Dry run: constructed recipe (no execution):")
        for i, m in enumerate(recipe):
            log(f"  Stage {i+1}: {m.__class__.__name__} -> {getattr(m, '__dict__', {})}")
        return

        dataset = args.dataset if args.dataset else (args.calib_jsonl if args.calib_jsonl else "open_platypus")

    out = args.output_dir
    if not out:
        parts = ["gpt-oss-120b"]
        if args.with_sparse:
            parts.append(args.mask.replace(":", "of"))
        parts.append(f"{(args.experts_algo or args.algo).upper()}-{(args.mw_algo or args.algo).upper()}")
        parts.append(f"experts-{(args.experts_type or args.experts or 'w4a16').lower()}-{(args.experts_scheme or '').lower()}".strip("-"))
        parts.append(f"mw-{(args.mw_type or args.mw or 'w4a16').lower()}-{(args.mw_scheme or '').lower()}".strip("-"))
        out = "-".join([p for p in parts if p])

    log("Running oneshot compression:")
    log(f"  model={args.model_in}")
    log(f"  experts: algo={ (args.experts_algo or args.algo).upper() }, spec={ (args.experts_type or args.experts or 'w4a16') } { (args.experts_scheme or '') }")
    log(f"  mw:      algo={ (args.mw_algo or args.algo).upper() }, spec={ (args.mw_type or args.mw or 'w4a16') } { (args.mw_scheme or '') }")
    log(f"  dataset={dataset}")
    log(f"  output_dir={out}")
    log(f"  max_seq_length={args.max_seq_length}  num_calibration_samples={args.num_calibration_samples}")

    oneshot(
        model=args.model_in,
        dataset=dataset,
        recipe=recipe,
        output_dir=out,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
        # Pass memory optimization options if supported by llmcompressor
        device_map=getattr(args, 'device_map', 'auto'),
        offload_dir=getattr(args, 'offload_folder', None),
    )
    log("Done.")

def cmd_install_kernel(args: argparse.Namespace) -> None:
    if has_w4a8_runtime():
        log("W4A8 runtime (QQQ) already present.")
        return
    if args.yes:
        ok = install_qqq(non_interactive=True)
    else:
        resp = input("Install QQQ CUTLASS kernels now? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            log("Aborted by user.")
            return
        ok = install_qqq(non_interactive=False)
    if not ok:
        raise SystemExit(1)

def cmd_verify_runtime(args: argparse.Namespace) -> None:
    print("=== Runtime Verification ===")
    # Torch / CUDA
    try:
        import torch
        import os
        print(f"PyTorch: {torch.__version__}")
        print(f"Torch CUDA version: {getattr(torch.version, 'cuda', 'unknown')}")
        try:
            import torch.backends.cudnn as cudnn
            print(f"cuDNN version: {getattr(cudnn, 'version', lambda: 'unknown')()}")
        except Exception:
            pass
        # Show common Docker GPU envs
        print(f"ENV CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        print(f"ENV NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', '')}")
        print(f"CUDA available (torch): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            print(f"CUDA device count: {count}")
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                print(f"  [{i}] {name} (CC {cap[0]}.{cap[1]})")
            dev = torch.cuda.current_device()
            name = torch.cuda.get_device_name(dev)
            cap = torch.cuda.get_device_capability(dev)
            print(f"Current device: {dev} -> {name}, CC {cap[0]}.{cap[1]}")
            if (cap[0], cap[1]) >= (8, 0):
                print("✓ Compute capability >= 8.0 (Ampere+), suitable for INT4/INT8 tensor cores.")
            else:
                print("✗ Compute capability < 8.0; W4A8 kernels may not be supported.")
        else:
            print("✗ CUDA not available; cannot run GPU kernels.")
    except Exception as e:
        print("! Error while checking PyTorch/CUDA:", e)

    # nvidia-smi availability (container GPU exposure)
    try:
        import subprocess
        print("nvidia-smi (host toolkit):")
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if out.returncode == 0:
            print(out.stdout.strip() or "(no output)")
        else:
            # Try without -L for broader compatibility
            out2 = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if out2.returncode == 0:
                print(out2.stdout.splitlines()[0] if out2.stdout else "nvidia-smi OK")
            else:
                print("✗ nvidia-smi not available in container or no GPU runtime.")
    except Exception as e:
        print("! nvidia-smi check failed:", e)

    # QQQ presence
    try:
        backend = _detect_w4a8_backend_name()
        print(f"W4A8 backend: {backend}")
    except Exception as e:
        print("! Error while checking W4A8 backend:", e)

    # Optional: try a trivial tensor op to ensure CUDA is healthy
    try:
        import torch
        if torch.cuda.is_available():
            a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
            b = torch.randn(128, 128, device="cuda", dtype=torch.float16)
            c = a @ b
            torch.cuda.synchronize()
            print("✓ CUDA matmul sanity check passed.")
    except Exception as e:
        print("! CUDA sanity check failed:", e)

    # Optional micro-benchmark
    if getattr(args, "bench", False):
        try:
            import torch
            M, K, N, iters = args.M, args.K, args.N, args.iters
            print(f"\n--- Microbench: M={M}, K={K}, N={N}, iters={iters} ---")

            # Baseline FP16 Linear
            x = torch.randn(M, K, device="cuda", dtype=torch.float16)
            lin = torch.nn.Linear(K, N, bias=False, device="cuda", dtype=torch.float16)
            for _ in range(10):
                _ = lin(x); torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(iters):
                _ = lin(x)
            t1.record(); torch.cuda.synchronize()
            fp16_ms = t0.elapsed_time(t1) / iters
            print(f"PyTorch FP16 Linear: {fp16_ms:.3f} ms/iter")

            # QQQ QuantLinear (if available)
            qqq_ms = None
            backend = _detect_w4a8_backend_name()
            try:
                from QQQ.gptq.qlinear import QuantLinear as _QQQQuantLinear  # type: ignore
                QQQQuantLinear = _QQQQuantLinear
            except Exception:
                try:
                    from qqq.gptq.qlinear import QuantLinear as _QQQQuantLinear  # type: ignore
                    QQQQuantLinear = _QQQQuantLinear
                except Exception as e:
                    QQQQuantLinear = None
                    print("QuantLinear not importable from QQQ; skipping W4A8 path.", e)

            if QQQQuantLinear is not None:
                qlin = QQQQuantLinear(4, 128, K, N, bias=False).to("cuda")
                x8 = torch.randn(M, K, device="cuda", dtype=torch.float16)
                for _ in range(10):
                    _ = qlin(x8); torch.cuda.synchronize()
                t2 = torch.cuda.Event(enable_timing=True); t3 = torch.cuda.Event(enable_timing=True)
                t2.record()
                for _ in range(iters):
                    _ = qlin(x8)
                t3.record(); torch.cuda.synchronize()
                qqq_ms = t2.elapsed_time(t3) / iters
                print(f"{backend} QuantLinear (W4A8): {qqq_ms:.3f} ms/iter")

            if qqq_ms is not None:
                speedup = fp16_ms / qqq_ms if qqq_ms > 0 else float('nan')
                print(f"Speedup vs FP16: {speedup:.2f}x")
            else:
                print("W4A8 path unavailable; only FP16 baseline measured.")
        except Exception as e:
            print("! Microbench failed:", e)

    print("=== End Verification ===")

# ---------- Argparse ----------
def main() -> None:
    #_print_transformers_version_and_upgrade()

    # Main parser
    parser = argparse.ArgumentParser(
        prog="compress_gptoss",
        description="Dequantize MXFP4 to BF16/FP16 and compress GPT-OSS-120B with GPTQ/AWQ (optional 2:4 sparsity).",
        epilog="""
Examples:
  %(prog)s dequantize --src openai/gpt-oss-120b --dst ./local-model
  %(prog)s quantize --model-in ./local-model --experts-type int4 --experts-scheme w4a8
  %(prog)s install-kernel --yes
  %(prog)s verify-runtime --bench

For more information, see the USAGE.md file.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True
    )
    
    # Add version information
    parser.add_argument(
        '--version', 
        action='version', 
        version='%(prog)s 1.0.0'
    )
    
    sub = parser.add_subparsers(
        dest="cmd", 
        required=True,
        title="Commands",
        description="Available commands (use 'COMMAND --help' for command-specific help)",
        metavar="COMMAND"
    )

    # dequantize
    d = sub.add_parser(
        "dequantize", 
        help="De-MXFP4 to BF16/FP16 and save a local checkpoint",
    )
    d.add_argument("--src", type=str, default="openai/gpt-oss-120b", metavar="PATH", help="source model (HuggingFace repo or local path)")
    d.add_argument("--dst", type=str, default="gpt-oss-120b-fp16", metavar="DIR", help="output directory for dequantized model")
    d.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="fp16", metavar="TYPE", help="final weight data type (default: %(default)s)")
    d.add_argument("--device-map", type=str, choices=["auto", "sequential", "cpu-only"], default="sequential", help="device mapping strategy: auto, sequential, or cpu-only (default: %(default)s)")
    d.add_argument("--offload-folder", type=str, metavar="DIR", help="directory for offloading weights to disk (enables larger models)")
    d.add_argument("--max-gpu-memory", type=str, metavar="MEMORY", help="maximum GPU memory per device (e.g., '10GB') to constrain model loading")
    d.set_defaults(func=cmd_dequantize)

    # quantize
    q = sub.add_parser(
        "quantize", 
        help="Quantize model using GPTQ/AWQ with optional 2:4 sparsity",
        description="Apply quantization (GPTQ/AWQ) to model weights with optional 2:4 sparsity via SparseGPT.",
        epilog="""
Examples:
  %(prog)s --model-in ./my-model --experts-type int4 --experts-scheme w4a16
  %(prog)s --model-in ./my-model --algo awq --with-sparse
  %(prog)s --model-in ./my-model --experts-type int4 --experts-scheme w4a8 --mw-type int8
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    q.add_argument("--model-in", type=str, default="gpt-oss-120b-fp16", 
                   metavar="PATH", help="path to local FP16/BF16 model from dequantize step")

    # Global default algo
    q.add_argument("--algo", type=str, choices=["gptq", "awq"], default="gptq", 
                   metavar="ALG", help="default quantization algorithm (default: %(default)s)")

    # Per-group algos
    q.add_argument("--experts-algo", type=str, choices=["gptq", "awq"], 
                   metavar="ALG", help="algorithm for expert layers")
    q.add_argument("--mw-algo", type=str, choices=["gptq", "awq"], 
                   metavar="ALG", help="algorithm for non-expert model weights")

    # Per-group type/scheme pairs (W4A8 via int4 + scheme w4a8)
    q.add_argument("--experts-type", type=str, choices=["int4", "int8"], 
                   metavar="TYPE", help="expert layers data type")
    q.add_argument("--experts-scheme", type=str, 
                   metavar="SCHEME", help="expert layers scheme (w4a16, w4a8 for int4; w8a8 for int8)")
    q.add_argument("--mw-type", type=str, choices=["int4", "int8"], 
                   metavar="TYPE", help="model weights data type")
    q.add_argument("--mw-scheme", type=str, 
                   metavar="SCHEME", help="model weights scheme (w4a16, w4a8 for int4; w8a8 for int8)")

    # Back-compat combined spec flags
    q.add_argument("--experts", type=str, 
                   metavar="SPEC", help="(deprecated) combined spec for experts: int4|w4a16|w4a8|int8|w8a8")
    q.add_argument("--mw", type=str, 
                   metavar="SPEC", help="(deprecated) combined spec for model weights: int4|w4a16|w4a8|int8|w8a8")

    # Expert selection
    q.add_argument("--auto-detect-experts", action="store_true", 
                   help="auto-detect expert layer regex from module names")
    q.add_argument("--experts-regex", type=str, 
                   metavar="REGEX", help="override expert layer regex pattern")

    # Sparsity (global)
    q.add_argument("--with-sparse", action="store_true", 
                   help="enable 2:4 sparsity (SparseGPT) before quantization")
    q.add_argument("--sparsity", type=float, default=0.5, 
                   metavar="RATIO", help="global sparsity ratio for SparseGPT (default: %(default)s)")
    q.add_argument("--mask", type=str, default="2:4", 
                   metavar="N:M", help="N:M mask structure (default: %(default)s)")

    # Quant knobs
    q.add_argument("--group-size", type=int, default=128, 
                   metavar="SIZE", help="group size for INT4 quantization (default: %(default)s)")
    q.add_argument("--symmetric", action="store_true", 
                   help="use symmetric quantization")
    q.add_argument("--block-size", type=int, default=128, 
                   metavar="SIZE", help="block size for GPTQ Hessian solves (default: %(default)s)")
    q.add_argument("--dampening-frac", type=float, default=0.01, 
                   metavar="FRAC", help="GPTQ dampening fraction (default: %(default)s)")
    q.add_argument("--no-sequential-update", action="store_true", 
                   help="disable sequential layer-by-layer updates")
    q.add_argument("--offload-hessians", action="store_true", 
                   help="offload Hessians to reduce peak RAM usage (GPTQ only)")

    # Calibration
    q.add_argument("--dataset", type=str, default="open_platypus", 
                   metavar="NAME", help="calibration dataset name from HuggingFace (default: %(default)s)")
    q.add_argument("--calib-jsonl", type=str, 
                   metavar="FILE", help="local JSONL file with calibration texts (overrides --dataset)")
    q.add_argument("--num-calibration-samples", type=int, default=256, 
                   metavar="N", help="number of calibration samples (default: %(default)s)")
    q.add_argument("--max-seq-length", type=int, default=2048, 
                   metavar="LEN", help="maximum sequence length for calibration (default: %(default)s)")

    # IO / misc
    q.add_argument("--output-dir", type=str, 
                   metavar="DIR", help="output directory for compressed checkpoint")
    q.add_argument("--extra-ignore", type=str, action="append", default=[], 
                   metavar="PATTERN", help="additional ignore patterns (can be repeated)")
    q.add_argument("--dry-run", action="store_true", 
                   help="show recipe without executing")

    # W4A8 kernel handling
    q.add_argument("--try-install-qqq", action="store_true", 
                   help="offer to install QQQ CUTLASS kernels if W4A8 requested but missing")
    q.add_argument("--yes", action="store_true", 
                   help="assume 'yes' to all prompts (non-interactive mode)")
    q.add_argument("--allow-fallback", action="store_true", 
                   help="fallback W4A8 to W4A16 if runtime unavailable")

    # Memory optimization
    q.add_argument("--device-map", type=str, choices=["auto", "sequential", "cpu-only"], default="sequential",
                   help="device mapping strategy for memory optimization (default: %(default)s)")
    q.add_argument("--offload-folder", type=str, metavar="DIR",
                   help="directory for offloading weights to disk during quantization")

    q.set_defaults(func=cmd_quantize)

    # install-kernel
    k = sub.add_parser(
        "install-kernel", 
        help="Install QQQ CUTLASS kernels for W4A8 support",
        description="Install QQQ CUTLASS kernels to enable W4A8 quantization runtime support.",
        epilog="Example: %(prog)s --yes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    k.add_argument("--yes", action="store_true", 
                   help="assume 'yes' to all prompts (non-interactive mode)")
    k.set_defaults(func=cmd_install_kernel)

    # verify-runtime
    v = sub.add_parser(
        "verify-runtime", 
        help="Verify GPU, CUDA, and W4A8 runtime environment",
        description="Check system capabilities including GPU, CUDA availability, and W4A8 (QQQ) runtime presence.",
        epilog="""
Examples:
  %(prog)s                    # basic verification
  %(prog)s --bench            # include performance microbenchmark
  %(prog)s --bench --M 32     # custom microbenchmark parameters
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    v.add_argument("--bench", action="store_true", 
                   help="run FP16 vs W4A8 performance microbenchmark")
    v.add_argument("--M", type=int, default=16, 
                   metavar="ROWS", help="token batch size for microbenchmark (default: %(default)s)")
    v.add_argument("--K", type=int, default=4096, 
                   metavar="COLS", help="input features for microbenchmark (default: %(default)s)")
    v.add_argument("--N", type=int, default=4096, 
                   metavar="COLS", help="output features for microbenchmark (default: %(default)s)")
    v.add_argument("--iters", type=int, default=50, 
                   metavar="N", help="timing iterations for microbenchmark (default: %(default)s)")
    v.set_defaults(func=cmd_verify_runtime)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
