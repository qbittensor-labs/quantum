import gc
import sys

import bittensor as bt


def clear_all_quimb_caches():
    bt.logging.info("Starting quimb cache sweep...")
    cleared_count = 0
    quimb_modules = 0

    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("quimb"):
            continue

        quimb_modules += 1

        if mod is None:
            continue

        try:
            for attr_name in dir(mod):
                try:
                    attr = getattr(mod, attr_name)

                    if hasattr(attr, "cache_clear"):
                        attr.cache_clear()
                        cleared_count += 1

                except Exception:
                    continue

        except Exception:
            continue

    gc.collect()
    bt.logging.info(f"Quimb cache sweep complete: {quimb_modules} modules checked, {cleared_count} caches cleared")
