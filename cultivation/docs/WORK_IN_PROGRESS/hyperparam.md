### A Beginner's Guide to Automated Hyperparameter Tuning

Imagine your machine learning model is a complex machine with dozens of tuning knobs. These knobs—like `learning_rate`, `batch_size`, or the number of layers in a neural network—are its **hyperparameters**. Finding the right combination is crucial for performance, but turning them by hand is slow, tedious, and often relies on guesswork.

Automated hyperparameter tuning is a systematic way to find the best settings for these knobs. This guide will teach you the core principles so you can implement it in your own projects.

#### The Core Idea: An Experiment-Driven Loop

At its heart, hyperparameter tuning is a simple loop:

1.  **Define a Search Space:** Tell the computer which knobs to turn and the range of values for each knob (e.g., "try learning rates between 0.01 and 0.0001").
2.  **Run an Experiment:** Pick one combination of settings from the search space.
3.  **Train & Evaluate:** Train a model with these settings for a short period.
4.  **Measure Performance:** Score the model using a single, clear metric (e.g., validation loss or accuracy).
5.  **Repeat:** Use the score from the last experiment to intelligently choose the *next* combination of settings to try. Repeat this for a set number of experiments.

After many experiments, you will have a winner: the set of hyperparameters that produced the best score.

#### Choosing Your Tool: Don't Reinvent the Wheel

You don't need to build the "intelligent chooser" yourself. Libraries like **Optuna** (used in `JARC-Reactor`), Hyperopt, or Ray Tune are excellent at this. We'll focus on the principles using Optuna because it's intuitive and powerful.

---

### The Four Pillars of Implementation

To add hyperparameter tuning to your project, you need to build four key components.

#### Pillar 1: The Objective Function (Your Experiment)

This is the most important piece. It's a single Python function that runs one experiment. It must do the following:

1.  **Accept a `trial` object:** Optuna passes this object to your function. You'll use it to get the hyperparameter values for this specific experiment.
2.  **Define the Hyperparameters:** Use `trial.suggest_*()` methods to sample values from your search space.
3.  **Build Your Model:** Initialize your model using the hyperparameter values from the `trial`.
4.  **Train and Validate:** Train the model for a *short* time (e.g., a few epochs). You don't need a full training run; you just need enough to see if it's promising.
5.  **Return a Performance Score:** The function must return a single number (a float) that Optuna should either `minimize` (e.g., `validation_loss`) or `maximize` (e.g., `accuracy`).

**Example from `JARC-Reactor`, simplified for clarity:**

```python
# This is a simplified version of jarc_reactor/optimization/objective.py
import optuna

def objective(trial):
    # 1. & 2. Get hyperparameter values for this trial
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    encoder_layers = trial.suggest_int("encoder_layers", 1, 6)
    
    # 3. Build your model with these specific values
    # (Imagine you have a function to create your model from a config)
    trial_config = create_config_for_trial(learning_rate, encoder_layers)
    model = MyModel(config=trial_config) 
    
    # 4. Train the model for a few epochs
    trainer = MyTrainer(max_epochs=5, ...) # Train for only 5 epochs
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # 5. Get the final validation loss and return it
    validation_loss = trainer.get_final_val_loss()
    
    # Handle the case where training failed (e.g., loss is NaN)
    if validation_loss is None or torch.isnan(validation_loss):
        raise optuna.TrialPruned() # Tell Optuna this trial failed

    return validation_loss 
```

#### Pillar 2: The Search Space (Your Blueprint)

You need to define the "knobs" and their "ranges". This is your search space. It's good practice to centralize this in a configuration file, just like `JARC-Reactor` does in `jarc_reactor/config.py`.

**Key Principles for Defining a Search Space:**

*   **Start Small:** Don't try to tune 20 parameters at once. Start with the most important ones, like the learning rate, number of layers, and model dimensions.
*   **Use the Right Scale:** For learning rates, use a `log` scale (`suggest_float(..., log=True)`), as they often vary by orders of magnitude.
*   **Handle Dependencies:** Sometimes one parameter depends on another. For example, in `JARC-Reactor`, `d_model` must be divisible by `heads`. You should enforce this logic inside your objective function.

**A simple search space definition:**

```python
# This is where you define the ranges for your hyperparameters
search_space = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    "num_layers": {"type": "int", "low": 2, "high": 8},
    "dropout_rate": {"type": "float", "low": 0.1, "high": 0.5},
}
```

#### Pillar 3: The Study Conductor (The Orchestrator)

This is the main script that manages the entire optimization process. It creates an Optuna `study` and tells it to run the `objective` function many times.

**Key features:**

*   **Create a `study`:** This object tracks all trials, their parameters, and their results.
*   **Specify `direction`:** Tell Optuna whether to `minimize` or `maximize` the score from your objective function.
*   **Use `storage`:** This is crucial. By telling Optuna to save the study to a database file (e.g., `storage="sqlite:///my_study.db"`), you can stop and resume the tuning process at any time.
*   **Call `study.optimize()`:** This starts the loop.

**Example from `jarc_reactor/optimization/optimize.py`, simplified:**

```python
import optuna

# 1. Create the study and link it to a database file for persistence
study = optuna.create_study(
    study_name="my_model_tuning",
    storage="sqlite:///my_study.db",  # Allows you to pause and resume
    direction="minimize",             # We want to minimize validation_loss
    load_if_exists=True               # Resume the study if it already exists
)

# 2. Tell the study to run your objective function for 100 trials
# The 'objective' function is the one we defined in Pillar 1.
study.optimize(objective, n_trials=100)

# 3. After it's done, you can get the best results
print("Best trial found:")
print(f"  Value (min val_loss): {study.best_trial.value}")
print(f"  Params: {study.best_trial.params}")
```

#### Pillar 4: Pruning (The Efficiency Hack)

Training models is expensive. **Pruning** is the technique of automatically stopping unpromising trials early. If a trial's performance is poor after the first epoch, why waste time training it for four more?

You implement this with a **callback**. A callback is a small piece of code that the training framework (like PyTorch Lightning) runs at specific points, such as the end of an epoch.

**How to implement pruning:**

1.  Inside your `objective` function, create a callback that is linked to the Optuna `trial`.
2.  At the end of each validation step or epoch, the callback reports the current performance (`val_loss`) to Optuna using `trial.report()`.
3.  It then asks Optuna `trial.should_prune()`. If Optuna, based on the performance of other trials, thinks this one is a lost cause, this will return `True`.
4.  If it should be pruned, the callback raises a `optuna.TrialPruned` exception, which gracefully stops the training for that trial.

**Simplified Pruning Callback:**

```python
# Inside your objective function...
# This callback would be passed to your trainer
class OptunaPruningCallback(YourFrameworkCallback):
    def __init__(self, trial):
        self.trial = trial

    def on_validation_end(self, current_val_loss):
        # Report the current performance to Optuna
        self.trial.report(current_val_loss, step=current_epoch)
        
        # Check if Optuna thinks this trial should be stopped
        if self.trial.should_prune():
            raise optuna.TrialPruned()

# In your objective function:
# trainer = MyTrainer(..., callbacks=[OptunaPruningCallback(trial)])
```

---

### Your Workflow for a New Project

1.  **Isolate Hyperparameters:** Go through your project and move all the hard-coded numbers you want to tune (learning rate, layer sizes, etc.) into a central configuration object.
2.  **Write the Objective Function:** Create a new script, `tune.py`. In it, define your `objective(trial)` function. This will be the heart of your work. It needs to be able to build and train your model from the parameters given by the `trial`.
3.  **Create the Conductor Script:** In the same `tune.py` file, write the logic to create the Optuna `study` and call `study.optimize()`.
4.  **Run and Wait:** Start the tuning process. Let it run for a significant number of trials (50-200 is a good start).
5.  **Analyze and Use:** Once finished, examine the `study.best_trial`. Update your main configuration file with these new, optimized values and run a full training session to get your final, high-performing model.