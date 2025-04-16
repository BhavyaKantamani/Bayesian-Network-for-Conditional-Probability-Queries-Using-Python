# Name: Bhavya Sri Kantamani
# UTA ID: 1002118109


import sys
import pandas as pd

# Function to calculate the probability of a conjunction of conditions from the dataset
def bxk_calculate_probability(conditions):
    """
    Calculate the probability of a conjunction of conditions from the dataset.
    Args:
        conditions (list of tuples): Each tuple contains a variable and its value ('T' or 'F').
    Returns:
        float: The probability of the conjunction of conditions.
    """
    try:
        match_count = 0  # Counter for matching records
        for record in bxk_dataset:  # Iterate over each record in the dataset
            match = True  # Assume the record matches initially
            for condition in conditions:  # Check each condition
                if condition[1] == 'T' and record[bxk_column_map[condition[0]]] != 1:
                    match = False  # Condition is not satisfied
                    break
                elif condition[1] == 'F' and record[bxk_column_map[condition[0]]] != 0:
                    match = False  # Condition is not satisfied
                    break
            if match:  # If all conditions are satisfied
                match_count += 1
        return match_count / bxk_total_records  # Return the fraction of matching records
    except KeyError as e:
        raise ValueError(f"Invalid variable in conditions: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error in bxk_calculate_probability: {e}")

def bxk_validate_query(query):
    """
    Validate the format and variables in the query.
    Args:
        query (str): The input query string.
    Returns:
        None: If the query is valid.
    Raises:
        ValueError: If the query is invalid.
    """
    valid_variables = {'B', 'G', 'C', 'F'}  # Acceptable variables
    parts = query.split("given")
    
    # Extract query and evidence variables
    query_vars = parts[0].strip().split() if parts[0].strip() else []
    evidence_vars = parts[1].strip().split() if len(parts) > 1 and parts[1].strip() else []

    all_vars = query_vars + evidence_vars

    # Check if at least one variable is present
    if not all_vars:
        raise ValueError("Query must contain at least one variable.")

    # Check for invalid variable format or values
    for var in all_vars:
        if len(var) != 2 or var[0].upper() not in valid_variables or var[1].upper() not in {'T', 'F'}:
            raise ValueError(f"Invalid query format: {var}. Use variables like Bt, Gf, etc.")
    
    # Check for missing variables
    if len(query_vars) == 0:
        raise ValueError("Query variables are missing. Specify at least one query variable.")
    if len(evidence_vars) == 0 and "given" in query:
        raise ValueError("Evidence variables are missing after 'given'. Provide evidence or remove 'given'.")
    
    # Check for conflicts (query variables also in evidence, except exact matches)
    conflicts = [var for var in query_vars if var in evidence_vars]
    if conflicts and set(query_vars) != set(evidence_vars):
        raise ValueError(f"Conflicting variables found: {conflicts}. A variable cannot be in both query and evidence unless they match exactly.")

    # Exact matches are valid and will result in probability 1.0
    if set(query_vars) == set(evidence_vars):
        print("Query matches evidence exactly. Result: Probability = 1.0")
        return


def bxk_get_baseball_game_probability(value):
    """
    Retrieve the probability of a baseball game being on TV.
    Args:
        value (str): 'T' or 'F' indicating True or False.
    Returns:
        float: The probability of the baseball game status.
    """
    try:
        if value == 'T':
            return bxk_baseball_probabilities.iat[0, 0]  # Probability of B=True
        elif value == 'F':
            return bxk_baseball_probabilities.iat[0, 1]  # Probability of B=False
        else:
            raise ValueError(f"Invalid value for Baseball Game: {value}")
    except IndexError as e:
        raise ValueError(f"Baseball Game probability not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error in bxk_get_baseball_game_probability: {e}")

def bxk_get_george_tv_probability(baseball_value, george_value):
    """
    Retrieve the probability of George watching TV given a baseball game.
    Args:
        baseball_value (str): 'T' or 'F' for Baseball Game.
        george_value (str): 'T' or 'F' for George Watching TV.
    Returns:
        float: The conditional probability of George Watching TV.
    """
    try:
        if baseball_value == 'T':
            if george_value == 'T':
                return bxk_george_tv_probs.iat[0, 0]  # P(G=True | B=True)
            elif george_value == 'F':
                return bxk_george_tv_probs.iat[0, 1]  # P(G=False | B=True)
        elif baseball_value == 'F':
            if george_value == 'T':
                return bxk_george_tv_probs.iat[1, 0]  # P(G=True | B=False)
            elif george_value == 'F':
                return bxk_george_tv_probs.iat[1, 1]  # P(G=False | B=False)
        raise ValueError(f"Invalid value combination for Baseball Game: {baseball_value}, George Watching TV: {george_value}")
    except IndexError as e:
        raise ValueError(f"George TV probability not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error in bxk_get_george_tv_probability: {e}")

def bxk_get_cat_food_probability(cat_food_value):
    """
    Retrieve the probability of running out of cat food.
    Args:
        cat_food_value (str): 'T' or 'F' indicating True or False.
    Returns:
        float: The probability of Cat Food status.
    """
    try:
        if cat_food_value == 'T':
            return bxk_cat_food_probs.iat[0, 0]  # P(C=True)
        elif cat_food_value == 'F':
            return bxk_cat_food_probs.iat[0, 1]  # P(C=False)
        else:
            raise ValueError(f"Invalid value for Cat Food: {cat_food_value}")
    except IndexError as e:
        raise ValueError(f"Cat Food probability not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error in bxk_get_cat_food_probability: {e}")

def bxk_get_feeding_probability(george_value, cat_food_value, feeding_value):
    """
    Retrieve the probability of George feeding the cat based on other variables.
    Args:
        george_value (str): 'T' or 'F' for George Watching TV.
        cat_food_value (str): 'T' or 'F' for Out of Cat Food.
        feeding_value (str): 'T' or 'F' for George Feeding.
    Returns:
        float: The conditional probability of feeding.
    """
    try:
        if george_value == 'T' and cat_food_value == 'T':
            if feeding_value == 'T':
                return bxk_feeding_probs.iat[0, 2]  # P(F=True | G=True, C=True)
            elif feeding_value == 'F':
                return bxk_feeding_probs.iat[0, 3]  # P(F=False | G=True, C=True)
        elif george_value == 'T' and cat_food_value == 'F':
            if feeding_value == 'T':
                return bxk_feeding_probs.iat[1, 2]  # P(F=True | G=True, C=False)
            elif feeding_value == 'F':
                return bxk_feeding_probs.iat[1, 3]  # P(F=False | G=True, C=False)
        elif george_value == 'F' and cat_food_value == 'T':
            if feeding_value == 'T':
                return bxk_feeding_probs.iat[2, 2]  # P(F=True | G=False, C=True)
            elif feeding_value == 'F':
                return bxk_feeding_probs.iat[2, 3]  # P(F=False | G=False, C=True)
        elif george_value == 'F' and cat_food_value == 'F':
            if feeding_value == 'T':
                return bxk_feeding_probs.iat[3, 2]  # P(F=True | G=False, C=False)
            elif feeding_value == 'F':
                return bxk_feeding_probs.iat[3, 3]  # P(F=False | G=False, C=False)
        raise ValueError(f"Invalid value combination for George Watching TV: {george_value}, Cat Food: {cat_food_value}, Feeding: {feeding_value}")
    except IndexError as e:
        raise ValueError(f"Feeding probability not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error in bxk_get_feeding_probability: {e}")

# Function to compute conditional probabilities
def bxk_compute_conditional_probability(*variables):
    """
    Compute conditional probabilities from given variables.
    Args:
        variables: Variable-value pairs (e.g., 'Bt', 'Gf').
    Returns:
        float: The computed conditional probability.
    """
    try:
        # Extract variable-value pairs
        baseball = [var for var in variables if var.startswith('B')][0]
        george = [var for var in variables if var.startswith('G')][0]
        cat_food = [var for var in variables if var.startswith('C')][0]
        feeding = [var for var in variables if var.startswith('F')][0]

        # Retrieve probabilities
        baseball_prob = bxk_get_baseball_game_probability(baseball[1])
        george_prob = bxk_get_george_tv_probability(baseball[1], george[1])
        cat_food_prob = bxk_get_cat_food_probability(cat_food[1])
        feeding_prob = bxk_get_feeding_probability(george[1], cat_food[1], feeding[1])

        return baseball_prob * george_prob * cat_food_prob * feeding_prob
    except ValueError as e:
        print(f"Error in bxk_compute_conditional_probability: {e}")
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected error in bxk_compute_conditional_probability: {e}")

# Function to generate all combinations of true/false values for unspecified variables
def bxk_generate_combinations(variables):
    """
    Generate all combinations of true/false values for unspecified variables.
    Args:
        variables (list of tuples): Specified variable-value pairs.
    Returns:
        list of lists: All combinations of true/false values for remaining variables.
    """
    try:
        included_vars = [v[0] for v in variables]  # Extract variables already included
        all_vars = ['B', 'G', 'C', 'F']  # Define all possible variables
        missing_vars = list(set(all_vars) - set(included_vars))  # Identify missing variables
        queue = [variables]  # Initialize queue with current variables
        for var in missing_vars:
            new_queue = []
            for current_set in queue:
                new_queue.append(current_set + [(var, 'T')])  # Add 'T' value for missing variable
                new_queue.append(current_set + [(var, 'F')])  # Add 'F' value for missing variable
            queue = new_queue
        # Convert tuples to strings for compatibility
        return [[f"{v[0]}{v[1]}" for v in combination] for combination in queue]
    except Exception as e:
        raise RuntimeError(f"Unexpected error in bxk_generate_combinations: {e}")

# --- Main Logic ---

try:
    # Load training data
    file_name = sys.argv[1]  # Get input file name from command line
    bxk_column_map = {'B': 0, 'G': 1, 'C': 2, 'F': 3}  # Map column names to indices

    # Read and process the dataset
    with open(file_name, 'r') as file:
        raw_data = file.readlines()  # Read all lines from the file

    bxk_total_records = len(raw_data)  # Count the total number of records
    bxk_dataset = [[int(value) for value in line.strip().split()] for line in raw_data]  # Parse each line into integers
except FileNotFoundError as e:
    raise RuntimeError(f"Training data file not found: {e}")
except Exception as e:
    raise RuntimeError(f"Unexpected error during dataset loading: {e}")

# Calculate Joint Probability Distributions (JPDs)
try:
    b_true = round(bxk_calculate_probability([('B', 'T')]), 9)
    b_false = round(1 - b_true, 9)
    g_b_true = round(bxk_calculate_probability([('G', 'T'), ('B', 'T')]) / b_true, 9)
    g_b_false = round(1 - g_b_true, 9)
    g_b_false_true = round(bxk_calculate_probability([('G', 'T'), ('B', 'F')]) / b_false, 9)
    g_b_false_false = round(1 - g_b_false_true, 9)
    c_true = round(bxk_calculate_probability([('C', 'T')]), 9)
    c_false = round(1 - c_true, 9)
    f_gc_true = round(bxk_calculate_probability([('F', 'T'), ('G', 'T'), ('C', 'T')]) / bxk_calculate_probability([('G', 'T'), ('C', 'T')]), 9)
    f_not_gc_true = round(1 - f_gc_true, 9)
    f_gc_false = round(bxk_calculate_probability([('F', 'T'), ('G', 'T'), ('C', 'F')]) / bxk_calculate_probability([('G', 'T'), ('C', 'F')]), 9)
    f_not_gc_false = round(1 - f_gc_false, 9)
    f_g_false_c_true = round(bxk_calculate_probability([('F', 'T'), ('G', 'F'), ('C', 'T')]) / bxk_calculate_probability([('G', 'F'), ('C', 'T')]), 9)
    f_not_g_false_c_true = round(1 - f_g_false_c_true, 9)
    f_g_false_c_false = round(bxk_calculate_probability([('F', 'T'), ('G', 'F'), ('C', 'F')]) / bxk_calculate_probability([('G', 'F'), ('C', 'F')]), 9)
    f_not_g_false_c_false = round(1 - f_g_false_c_false, 9)
except ZeroDivisionError as e:
    raise ValueError(f"Probability calculation error (division by zero): {e}")
except Exception as e:
    raise RuntimeError(f"Unexpected error in JPD calculation: {e}")

# Create pandas DataFrames for JPDs
bxk_baseball_probabilities = pd.DataFrame([[b_true, b_false]], columns=['Baseball', 'No Baseball'])
bxk_george_tv_probs = pd.DataFrame([[g_b_true, g_b_false], [g_b_false_true, g_b_false_false]],
                               columns=['George Watches TV', 'George Doesn\'t Watch TV'], index=['B=True', 'B=False'])
bxk_feeding_probs = pd.DataFrame(
    [
        ['T', 'T', f_gc_true, f_not_gc_true],
        ['T', 'F', f_gc_false, f_not_gc_false],
        ['F', 'T', f_g_false_c_true, f_not_g_false_c_true],
        ['F', 'F', f_g_false_c_false, f_not_g_false_c_false],
    ],
    columns=['George Watches TV', 'Out of Cat Food', 'Feeds', 'Doesn\'t Feed']
)
bxk_cat_food_probs = pd.DataFrame([[c_true, c_false]], columns=['Out of Cat Food', 'Not Out of Cat Food'])

# Interactive query system
if len(sys.argv) == 2:
    print("\nBaseball Probabilities:\n", bxk_baseball_probabilities)
    print("\nGeorge Watching TV Probabilities:\n", bxk_george_tv_probs)
    print("\nFeeding Probabilities:\n", bxk_feeding_probs)
    print("\nCat Food Probabilities:\n", bxk_cat_food_probs)

while True:
    query = input("Enter query (or type 'none' to exit): ").strip()
    if query.lower() == 'none':
        break
    try:
        # Validate the query
        bxk_validate_query(query)

        # Parse query into variables and evidence
        query_parts = query.split("given")
        query_vars = [q.upper() for q in query_parts[0].strip().split()] if query_parts[0].strip() else []
        evidence_vars = [e.upper() for e in query_parts[1].strip().split()] if len(query_parts) > 1 and query_parts[1].strip() else []

        # Generate combinations for numerator and denominator
        query_combination = query_vars + evidence_vars
        numerator_combinations = bxk_generate_combinations(query_combination)
        numerator = sum(bxk_compute_conditional_probability(*combo) for combo in numerator_combinations)
        denominator = 1 if not evidence_vars else sum(
            bxk_compute_conditional_probability(*combo) for combo in bxk_generate_combinations(evidence_vars)
        )
        if denominator == 0:
            print("Probability cannot be computed (zero denominator).")
        else:
            print(f"Probability: {round(numerator / denominator, 9)}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
