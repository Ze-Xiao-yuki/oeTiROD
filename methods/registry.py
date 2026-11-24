from importlib import import_module

METHODS = {
    "FT": ("ft", "FineTuning", "TiROD.py", "连续微调"),
    "CUMUL": ("cumul", "Cumulative", "TiROD_CUMUL.py", "累积训练"),
    "REP": ("rep", "Replay", "TiROD_REP.py", "普通回放"),
    "TREP": ("trep", "TemporalReplay", "TiROD_TREP.py", "时序回放"),
    "KMREP": ("kmrep", "KMeansReplay", "TiROD_KMREP_ResNet.py", "KMeans代表样本回放"),
    "LREP": ("lrep", "LatentReplay", "TiROD_LREP.py", "冻结骨干的回放"),
    "LWF": ("lwf", "LearningWithoutForgetting", "TiROD_LWF.py", "输出蒸馏"),
    "EWC": ("ewc", "ElasticWeightConsolidation", "TiROD_EWC.py", "Fisher参数约束"),
    "SID": ("sid", "SelectiveDistillation", "TiROD_SID.py", "分类与中间层蒸馏"),
    "LD": ("ld", "LatentDistillation", "TiROD_LD.py", "冻结骨干的潜在蒸馏"),
}


def create_method(name, config):
    module, cls, _, _ = METHODS[name.upper()]
    return getattr(import_module(f"methods.{module}"), cls)(config)
