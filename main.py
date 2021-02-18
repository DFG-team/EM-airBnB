from pipeline_summaryDeepER import start_pipeline_with_DeepER
from pipeline_summary import start_pipeline_summary

if __name__ == "__main__":
    start_pipeline_summary("DatasetRome\\")
    start_pipeline_with_DeepER("DatasetRomeDeepER\\")