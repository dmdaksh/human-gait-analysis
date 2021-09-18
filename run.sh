# create a log directory to store logs if logs directory doesn't exist
mkdir -p logs
mkdir -p results
# env vars
export LOG_LEVEL=info

# set bFloatTensor
# export XLA_USE_BF16=1

# running main file
python3 -m gait.main --exclude_surface CALIB
