from models.baselines import run_pipeline
    # run_model, hyperparam_search

# running config for multitask debugging
test_loss = run_pipeline(sweep=False)

# running hyperparam search for multitask debugging
test_loss = run_pipeline(sweep=True)

# run_model()

# hyperparam_search()
