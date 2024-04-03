import subprocess
import os


def run_abides(agent_name, seed):
    # Set the command to run the ABIDES simulation with the given agent and seed
    command = f"python abides.py -c {agent_name} --random_seed {seed}"

    # Run the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Return the output
    return result.stdout


def compare_strategies():
    # Define the agents and seeds to compare
    agents = ["yshah72_dopamine", "SimpleAgent"]
    seeds = [123, 456, 789]

    # Run the simulations and collect the results
    results = {}
    for agent in agents:
        for seed in seeds:
            key = f"{agent}_seed_{seed}"
            results[key] = run_abides(agent, seed)

    # Compare the results (this is just a placeholder, replace with your own comparison logic)
    for key, result in results.items():
        print(f"Results for {key}:")
        print(result)


# Run the comparison
compare_strategies()