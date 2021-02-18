from pipeline import start_pipeline
from pipeline_summary import start_pipeline_summary
from pipeline_ditto import start_pipeline_ditto

if __name__ == "__main__":
    start_pipeline_summary("DatasetAmsterdam\\")
    start_pipeline_ditto("DatasetAmsterdamDitto\\")
