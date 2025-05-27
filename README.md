# market-prediction-workbench

Market Prediction Workbench for stock forecasting.

## Setup

1.  Clone the repository.
2.  Ensure you have Poetry installed.
3.  Install dependencies:
    ```bash
    poetry install
    ```
4.  Activate the virtual environment:
    ```bash
    poetry shell
    ```
5.  Set up pre-commit hooks (if not already done by someone else who committed `.git/hooks`):
    ```bash
    poetry run pre-commit install
    ```

## Project Structure

-   `data/`: Raw and processed data.
-   `notebooks/`: EDA and scratch work.
-   `src/market_prediction_workbench/`: Core Python package.
-   `conf/`: Configuration files.
-   `experiments/`: Experiment logs and outputs.
-   `tests/`: Unit tests.
