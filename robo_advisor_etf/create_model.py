import pandas as pd
import argparse
import optimization


if __name__ == "__main__":
    # Collect args
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--filepath",
        default="etf_data/combined_etf_amsterdam_sustainable.csv",
        help="Input filepath of csv",
        type=str,
    )
    parser.add_argument(
        "--output_filepath",
        default="models/portfolio.model",
        help="Output filepath of model",
        type=str,
    )

    # Parse the args
    args = parser.parse_args()

    # Read data
    etf_data = pd.read_csv(args.filepath)

    # Create new model object
    model = optimization.ETFPortfolioOptimizer(etf_data)

    # Save model
    model.save(args.output_filepath)
