"""Add RMS matching to the installed DeepSpeed Muon and its ZeRO-2 path."""
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
import re


def patch(root):
    names = ["runtime/zero/muon/original_muon.py",
             "runtime/zero/stage_1_and_2.py", "runtime/engine.py"]
    sources = {name: (root / name).read_text() for name in names}

    def replace(source, old, new, count=1):
        if source.count(new) == count:
            return source
        assert source.count(old) == count, f"Unexpected DeepSpeed source: {old}"
        return source.replace(old, new)

    source = replace(sources[names[0]],
        'ns_method="gram", is_expert_group=False):',
        'ns_method="gram", is_expert_group=False, adjust_lr_fn="original"):')
    source = replace(source, "    orig_dtype = grad.dtype",
        '    assert adjust_lr_fn in ("original", "match_rms_adamw")\n'
        "    orig_dtype = grad.dtype")
    source = replace(source,
        "        scale = max(1, update.size(-2) / update.size(-1))**0.5",
        '        scale = (0.2 * max(update.shape[-2:])**0.5 if adjust_lr_fn == "match_rms_adamw"\n'
        "                 else max(1, update.size(-2) / update.size(-1))**0.5)")
    source = replace(source,
        "        update *= max(1, grad.size(-2) / grad.size(-1))**0.5",
        "        # Match AdamW's typical update RMS using the complete matrix shape.\n"
        '        scale = (0.2 * max(update.shape[-2:])**0.5 if adjust_lr_fn == "match_rms_adamw"\n'
        "                 else max(1, update.size(-2) / update.size(-1))**0.5)\n"
        "        update *= scale")
    source = replace(source,
        'momentum=0.95, ns_method="gram"):',
        'momentum=0.95, ns_method="gram", adjust_lr_fn="original"):', count=2)
    source = replace(source,
        "defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_method=ns_method)",
        "defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,\n"
        "                        ns_method=ns_method, adjust_lr_fn=adjust_lr_fn)", count=2)
    source = replace(source,
        '                group["ns_method"] = group.get("ns_method", "gram")',
        '                group["ns_method"] = group.get("ns_method", "gram")\n'
        '                group["adjust_lr_fn"] = group.get("adjust_lr_fn", "original")\n'
        '                assert group["adjust_lr_fn"] in ("original", "match_rms_adamw")', count=2)
    call = 'adjust_lr_fn=group.get("adjust_lr_fn", "original"),'
    if source.count(call) != 4:
        source, count = re.subn(
            r'(?m)^(\s*)ns_method=group.get\("ns_method", "gram"\),$',
            lambda m: m[0] + "\n" + m[1] + call, source)
        assert count == 4
    sources[names[0]] = source
    sources[names[1]] = replace(sources[names[1]],
        "                                         ns_method=ns_method,",
        "                                         ns_method=ns_method,\n"
        '                                         adjust_lr_fn=self.optimizer.param_groups[param_group_idx].get("adjust_lr_fn", "original"),')
    sources[names[2]] = replace(sources[names[2]],
        'for key in ["lr", "momentum", "weight_decay", "muon_lr", "ns_method"]:',
        'for key in ["lr", "momentum", "weight_decay", "muon_lr", "ns_method", "adjust_lr_fn"]:')

    for name, source in sources.items():
        compile(source, str(root / name), "exec")
    for name, source in sources.items():
        path = root / name
        backup = path.with_suffix(".py.before_moss_rms")
        if not backup.exists():
            backup.write_text(path.read_text())
        path.write_text(source)
        print(f"Patched {path}")


if __name__ == "__main__":
    assert version("deepspeed") == "0.19.6"
    patch(Path(find_spec("deepspeed").origin).parent)
