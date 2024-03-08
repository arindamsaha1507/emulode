import argparse
import numpy as np
import matplotlib.pyplot as plt


def initialize_parameters(args):
    params={}
    paramnames=['mu_h',  # human death rate
        'd_h',      # disease-induced death rate
        'beta_h',        # infection rate for humans
        'sigma_h',  # incubation rate for humans
        'gamma',    # recovery rate
        'mu_v',     # vector death rate
        'beta_v',        # infection rate for vectors
        'sigma_v',  # incubation rate for vectors
        'total_population_h',  # total human population
        'total_population_v',  # total vector population
        'initial_S_h',  # initial susceptible fraction of humans
        'initial_E_h',  # initial exposed fraction of humans
        'initial_I_h',  # initial infected fraction of humans
        'initial_R_h',   # initial recovered fraction of humans
        'initial_S_v',  # initial susceptible fraction of vectors
        'initial_E_v',  # initial exposed fraction of vectors
        'initial_I_v',  # initial infected fraction of vectors
        'initial_R_v',   # initial recovered fraction of vectors
        'lambda_h',  # human birth rate
        'lambda_v'   # vector birth rate
        ]
    for paramname in paramnames:
        if not hasattr(args, paramname):
            raise ValueError('Parameter {} is missing'.format(paramname))
        else:
            params[paramname]=getattr(args, paramname)
    # params = {
    #     'mu_h': 0.000039139,  # human death rate
    #     'd_h': 1 / 20.0,      # disease-induced death rate
    #     'beta_h': 0.3,        # infection rate for humans
    #     'sigma_h': 1 / 10.0,  # incubation rate for humans
    #     'gamma': 1 / 10.0,    # recovery rate
    #     'mu_v': 1 / 14.0,     # vector death rate
    #     'beta_v': 0.3,        # infection rate for vectors
    #     'sigma_v': 1 / 10.0,  # incubation rate for vectors
    #     'total_population_h': 100.0,  # total human population
    #     'total_population_v': 1000.0,  # total vector population
    #     'initial_S_h': 0.90,  # initial susceptible fraction of humans
    #     'initial_E_h': 0.05,  # initial exposed fraction of humans
    #     'initial_I_h': 0.05,  # initial infected fraction of humans
    #     'initial_R_h': 0.0,   # initial recovered fraction of humans
    #     'initial_S_v': 0.90,  # initial susceptible fraction of vectors
    #     'initial_E_v': 0.05,  # initial exposed fraction of vectors
    #     'initial_I_v': 0.05,  # initial infected fraction of vectors
    #     'initial_R_v': 0.0,   # initial recovered fraction of vectors
    #     'lambda_h': 0.000039139 * 100.0,  # human birth rate
    #     'lambda_v': (1 / 14.0) * 1000.0   # vector birth rate
    # }
    return params

def initialize_state(params):
    state = {
        'S_h': int(params['initial_S_h'] * params['total_population_h']),
        'E_h': int(params['initial_E_h'] * params['total_population_h']),
        'I_h': int(params['initial_I_h'] * params['total_population_h']),
        'R_h': int(params['initial_R_h'] * params['total_population_h']),
        'S_v': int(params['initial_S_v'] * params['total_population_v']),
        'E_v': int(params['initial_E_v'] * params['total_population_v']),
        'I_v': int(params['initial_I_v'] * params['total_population_v']),
        'R_v': int(params['initial_R_v'] * params['total_population_v']),
        'I_d_h': 0
    }
    return state

def calculate_rates(state, params):
    N_h = state['S_h'] + state['E_h'] + state['I_h'] + state['R_h']
    N_v = state['S_v'] + state['E_v'] + state['I_v'] + state['R_v']
    rates = {
        'rate_birth_h': params['lambda_h'],
        'rate_death_susceptible_h': params['mu_h'] * state['S_h'],
        'rate_exposed_h': params['beta_h'] * state['S_h'] * state['I_v'] / N_h,
        'rate_death_exposed_h': params['mu_h'] * state['E_h'],
        'rate_infection_h': params['sigma_h'] * state['E_h'],
        'rate_death_infectious_h': params['mu_h'] * state['I_h'],
        'rate_death_dueto_infection_h': params['d_h'] * state['I_h'],
        'rate_recovery_h': params['gamma'] * state['I_h'],
        'rate_death_recovered_h': params['mu_h'] * state['R_h'],
        'rate_birth_v': params['lambda_v'],
        'rate_death_susceptible_v': params['mu_v'] * state['S_v'],
        'rate_exposed_v': params['beta_v'] * state['S_v'] * state['I_h'] / N_h,
        'rate_death_exposed_v': params['mu_v'] * state['E_v'],
        'rate_infection_v': params['sigma_v'] * state['E_v'],
        'rate_death_infectious_v': params['mu_v'] * state['I_v']
    }
    return rates

def update_state(state, event_type):
    if event_type == "Birth":
        state['S_h'] += 1
    elif event_type == "Death Susceptible":
        state['S_h'] -= 1
    elif event_type == "Exposed":
        state['S_h'] -= 1
        state['E_h'] += 1
    elif event_type == "Death Exposed":
        state['E_h'] -= 1
    elif event_type == "Infection":
        state['E_h'] -= 1
        state['I_h'] += 1
    elif event_type == "Death Infection":
        state['I_h'] -= 1
    elif event_type == "Disease Induced Death":
        state['I_h'] -= 1
        state['I_d_h'] += 1
    elif event_type == "Recovery":
        state['I_h'] -= 1
        state['R_h'] += 1
    elif event_type == "Death Recovery":
        state['R_h'] -= 1
    elif event_type == "Birth_v":
        state['S_v'] += 1
    elif event_type == "Death Susceptible_v":
        state['S_v'] -= 1
    elif event_type == "Exposed_v":
        state['S_v'] -= 1
        state['E_v'] += 1
    elif event_type == "Death Exposed_v":
        state['E_v'] -= 1
    elif event_type == "Infection_v":
        state['E_v'] -= 1
        state['I_v'] += 1
    elif event_type == "Death Infection_v":
        state['I_v'] -= 1
    return state

def run_simulation(args):
    params = initialize_parameters(args)
    state = initialize_state(params)
    event_types = list(calculate_rates(state, params).keys())
    event_type_mapping = {
    'rate_birth_h': 'Birth',
    'rate_death_susceptible_h': 'Death Susceptible',
    'rate_exposed_h': 'Exposed',
    'rate_death_exposed_h': 'Death Exposed',
    'rate_infection_h': 'Infection',
    'rate_death_infectious_h': 'Death Infection',
    'rate_death_dueto_infection_h': 'Disease Induced Death',
    'rate_recovery_h': 'Recovery',
    'rate_death_recovered_h': 'Death Recovery',
    'rate_birth_v': 'Birth_v',
    'rate_death_susceptible_v': 'Death Susceptible_v',
    'rate_exposed_v': 'Exposed_v',
    'rate_death_exposed_v': 'Death Exposed_v',
    'rate_infection_v': 'Infection_v',
    'rate_death_infectious_v': 'Death Infection_v'
}

    t = 0
    events = []
    t_max=args.tmax
    while t < t_max:
        rates = calculate_rates(state, params)
        total_rate = sum(rates.values())
        rate_sums = np.cumsum(list(rates.values()))
        if state['I_h'] == 0 or state['E_h'] == 0:
            break
        dt = np.random.exponential(1 / total_rate)
        t += dt
        event_prob = np.random.uniform(0, 1)
        for i, event_type in enumerate(event_types):
            if i == 0 and event_prob < rate_sums[i] / total_rate:
                event_type = "Birth"
            if i > 1 and i < len(event_types) - 1 and event_prob > rate_sums[i - 1]/total_rate and event_prob < rate_sums[i] / total_rate:
                event_type = event_type_mapping[event_types[i]]
            if i == len(event_types) - 1 and event_prob > rate_sums[i - 1]/total_rate and event_prob < rate_sums[i] / total_rate:
                event_type = "Death Infection_v"

            state = update_state(state, event_type)
            events.append((t, state['S_h'], state['E_h'], state['I_h'], state['R_h'], state['S_v'], state['E_v'], state['I_v'], state['R_v'], state['I_d_h'], event_type))
            # print(state['S_v'], state['E_v'], state['I_v'], state['R_v'])
    return events

def plot_results(events):
    # Extract data for plotting
    times = [event[0] for event in events]
    S_values = [event[1] for event in events]
    E_values = [event[2] for event in events]
    I_values = [event[3] for event in events]
    R_values = [event[4] for event in events]

    S_v_values = [event[5] for event in events]
    E_v_values = [event[6] for event in events]
    I_v_values = [event[7] for event in events]

    I_h_d_values= [event[8] for event in events]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot human variables on the left y-axis
    ax1.plot(times, S_values, label='Susceptible (Human)', linestyle='--', color='tab:blue')
    ax1.plot(times, E_values, label='Exposed (Human)', linestyle='-', color='tab:orange')
    ax1.plot(times, I_values, label='Infectious (Human)', linestyle='-', color='tab:green')
    ax1.plot(times, I_h_d_values, label='Disease deaths (Human)', linestyle='-', color='tab:red')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of individuals (Human)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis for vector variables
    ax2 = ax1.twinx()
    ax2.plot(times, S_v_values, label='Susceptible (Vector)', linestyle='--', color='tab:cyan')
    ax2.plot(times, E_v_values, label='Exposed (Vector)', linestyle='-', color='tab:purple')
    ax2.plot(times, I_v_values, label='Infectious (Vector)', linestyle='-', color='tab:pink')
    ax2.set_ylabel('Number of individuals (Vector)', color='tab:cyan')
    ax2.tick_params(axis='y', labelcolor='tab:cyan')

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Stochastic SEIR-SEI Model Simulation')
    plt.grid(True)
    plt.savefig('../Data/StochasticSEIR_SEI.png')
    plt.show()


if __name__ == "__main__":
    tmax=10
    parser=argparse.ArgumentParser()
    defaults_vals = {'--tmax': 50,  # Maximum time for simulation
                     '--path': '../Data/',  # Path to save the events
                     '--output_file': 'SEIR_SEI_events.csv',  # Output file name
        '--mu_h': 0.000039139,  # human death rate
        '--d_h': 1 / 20.0,      # disease-induced death rate
        '--beta_h': 0.3,        # infection rate for humans
        '--sigma_h': 1 / 10.0,  # incubation rate for humans
        '--gamma': 1 / 10.0,    # recovery rate
        '--mu_v': 1 / 14.0,     # vector death rate
        '--beta_v': 0.3,        # infection rate for vectors
        '--sigma_v': 1 / 10.0,  # incubation rate for vectors
        '--total_population_h': 100.0,  # total human population
        '--total_population_v': 1000.0,  # total vector population
        '--initial_S_h': 0.90,  # initial susceptible fraction of humans
        '--initial_E_h': 0.05,  # initial exposed fraction of humans
        '--initial_I_h': 0.05,  # initial infected fraction of humans
        '--initial_R_h': 0.0,   # initial recovered fraction of humans
        '--initial_S_v': 0.90,  # initial susceptible fraction of vectors
        '--initial_E_v': 0.05,  # initial exposed fraction of vectors
        '--initial_I_v': 0.05,  # initial infected fraction of vectors
        '--initial_R_v': 0.0,   # initial recovered fraction of vectors
        '--lambda_h': 0.000039139 * 100.0,  # human birth rate
        '--lambda_v': (1 / 14.0) * 1000.0   # vector birth rate
    }
    defaults_help={'--tmax': 'Maximum time for simulation','--path': 'Path to save the events','--output_file': 'Output file name',
                   '--mu_h': 'human death rate',
                   '--d_h': 'disease-induced death rate',
                   '--beta_h': 'infection rate for humans',
                   '--sigma_h': 'incubation rate for humans',
                   '--gamma': 'recovery rate',
                   '--mu_v': 'vector death rate',
                   '--beta_v': 'infection rate for vectors',
                   '--sigma_v': 'incubation rate for vectors',
                   '--total_population_h': 'total human population',
                   '--total_population_v': 'total vector population',
                   '--initial_S_h': 'initial susceptible fraction of humans',
                   '--initial_E_h': 'initial exposed fraction of humans',
                   '--initial_I_h': 'initial infected fraction of humans',
                   '--initial_R_h': 'initial recovered fraction of humans',
                   '--initial_S_v': 'initial susceptible fraction of vectors',
                   '--initial_E_v': 'initial exposed fraction of vectors',
                   '--initial_I_v': 'initial infected fraction of vectors',
                   '--initial_R_v': 'initial recovered fraction of vectors',
                   '--lambda_h': 'human birth rate',
                   '--lambda_v': 'vector birth rate'}
    for key, value in defaults_vals.items():
        parser.add_argument(key, type=type(value), default=value, help=defaults_help[key])

    args=parser.parse_args()
    events=run_simulation(args)
    save_path = '../Data/'
    with open(save_path+'SEIR_SEI_events.csv', 'w') as f:
        for event in events:
            f.write(str(event)+'\n')
    # #load events from file
    # events = []
    # with open(save_path+'SEIR_SEI_events.txt', 'r') as f:
    #     for line in f:
    #         events.append(eval(line))
    plot_results(events)