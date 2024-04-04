import subprocess
import os
import random
import sys
import pickle
def run_abides(agent_name, seed):
    # Set the command to run the ABIDES simulation with the given agent and seed
    command = f"python abides.py -c rmsc01 -l rmsc01_{seed}_{agent_name} -s {seed} -a {agent_name}"

    # Run the command and capture the output
    # result = subprocess.run(command, shell=True, check=True, stdout=)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    result, error = process.communicate()
    if error:
        print(f"Error running command: {command}")
        print(error)
        sys.exit(1)


    # Return the output
    return result


def compare_strategies():
    # Define the agents and seeds to compare
    agents = ["contributed_traders.yshah72_dopamine.yshah72_dopamine", "contributed_traders.SimpleAgent"]
    seeds = seeds = [random.randint(1000000000, 2 ** 32) for _ in range(1000)]
    print(seeds)

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
    with open('/home/dopamine/results.pkl', 'wb') as f:
        pickle.dump(results, f)

# Run the comparison
compare_strategies()