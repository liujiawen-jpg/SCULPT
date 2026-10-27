"""No cross-project cache discovery in the portable release.

Current-run, byte-bound empirical caches in ConnectionRuntime are retained.
Historical experiment directories are never searched or imported.
"""
def configure_observation_reuse(runtime):
    runtime._observation_reuse = []
    runtime._observation_reuse_counts = {'precision': 0, 'source': 0, 'supervision': 0}

def try_reuse_condition(runtime, path, purpose, env, target):
    return False
