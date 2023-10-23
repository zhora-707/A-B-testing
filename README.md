# A-B-testing

# Makefile

# Define Python interpreter
PYTHON = python

# Define the name of the main Python script
MAIN_PY = bandit.py

# Define input and output files
LOGS_PY = logs.py
REWARDS_CSV = bandit_rewards.csv

# Target to run the main Python script
run: $(MAIN_PY)
	$(PYTHON) $(MAIN_PY)

# Target to check logs
check-logs: $(LOGS_PY)
	$(PYTHON) $(LOGS_PY)

# Clean up generated files
clean:
	# Remove any generated log files or other temporary files
	rm -f *.log

