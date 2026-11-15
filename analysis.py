from market_study.market_evidence import main as run_source_audit
from market_study.model_readiness import main as run_readiness
from market_study.prepare_modeling_data import main as run_preparation
from market_study.statistical_evidence import main as run_statistics
from market_study.train_price_models import main as run_training
from market_study.evaluate_rent_models import main as run_evaluation
from market_study.publish_portfolio import main as run_publication
from market_study.verify_portfolio import main as verify_publication


def main():
    run_source_audit()
    run_readiness()
    run_preparation()
    run_statistics()
    run_training()
    run_evaluation()
    run_publication()
    verify_publication()
    print("PASS - Reproducible rental-market study completed")


if __name__ == "__main__":
    main()
