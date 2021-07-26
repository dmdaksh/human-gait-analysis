# create a log directory to store logs if logs directory doesn't exist
mkdir -p logs

# env vars
export LOG_LEVEL=debug

# running main file
python3 -m src.main --exclude_surface CALIB
