import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.week6_final_pipeline import run_week6_pipeline

if __name__ == "__main__":
    run_week6_pipeline()
