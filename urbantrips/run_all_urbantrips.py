from urbantrips import initialize_environment
from urbantrips import process_transactions
from urbantrips import run_postprocessing
from urbantrips import create_viz
from urbantrips import run_dashboard

import warnings

# Filter and suppress specific warnings
warnings.filterwarnings("ignore")

def main():

    initialize_environment.main()
    process_transactions.main()
    run_postprocessing.main()
    create_viz.main()
    run_dashboard.main()
    
if __name__ == "__main__":
    main()
