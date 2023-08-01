from urbantrips import initialize_environment
from urbantrips import process_transactions
from urbantrips import run_postprocessing
from urbantrips import create_viz


def main():

    initialize_environment.main()
    process_transactions.main()
    run_postprocessing.main()
    create_viz.main()
    !streamlit run urbantrips/dashboard/dashboard.py
if __name__ == "__main__":
    main()
