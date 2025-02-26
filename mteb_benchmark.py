import mteb
import os

from eval.evaluation import evaluation_mteb

MODEL_LIST = [
    "x2bee/KoModernBERT-base-mlm-ecs_V01",
    # "BM-K/KoSimCSE-roberta-multitask",
    # "dragonkue/BGE-m3-ko",
    # "nlpai-lab/KURE-v1",
    # "nlpai-lab/KoE5",
    # "jinaai/jina-embeddings-v3",
    # "klue/roberta-base",
    # "klue/bert-base",
]

OUTPUT_DIR = "./result"

TASK_LIST = [
    "KLUE-TC",
    "MIRACLReranking",
    "SIB200ClusteringS2S",
    "KLUE-STS",
    "KorSTS",
    "KLUE-NLI",
    "PawsXPairClassification",
    "PubChemWikiPairClassification",
    "AutoRAGRetrieval",
    "Ko-StrategyQA",
    "BelebeleRetrieval",
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MrTidyRetrieval",
    "MultiLongDocRetrieval",
    "XPQARetrieval",
]

if __name__ == '__main__':
    evaluation_mteb(MODEL_LIST, OUTPUT_DIR, 8, TASK_LIST)
    # model = mteb.get_model("x2bee/KoModernBERT-base-mlm-ecs_V01")
    # # tasks = mteb.get_benchmark("MTEB(kor, v1)")
    # result = evaluation.run(model, output_folder="./result", encode_kwargs={"batch_size": 32})
    
