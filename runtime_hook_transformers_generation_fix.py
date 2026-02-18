
try:
    import transformers.generation as g
    from transformers.generation.utils import GenerationMixin
    from transformers.generation.configuration_utils import GenerationConfig
    g.GenerationMixin = GenerationMixin
    g.GenerationConfig = GenerationConfig
except Exception:
    pass
