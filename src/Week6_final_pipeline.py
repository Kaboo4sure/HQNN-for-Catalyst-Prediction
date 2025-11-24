from week1_exploration import run_week1_exploration
from week2_preprocessing import run_week2_preprocessing
from week2b_consistency import run_week2b_consistency
from week3_models import run_week3_models
from week4_hqnn import run_week4_hqnn
from week5_explainability import run_week5_explainability

def run_week6_pipeline():
    run_week1_exploration()
    run_week2_preprocessing()
    run_week2b_consistency()
    run_week3_models()
    run_week4_hqnn()
    run_week5_explainability()

if __name__ == "__main__":
    run_week6_pipeline()
