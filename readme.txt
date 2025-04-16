Name : Bhavya Sri Kantamani
UTA ID: 1002118109


Programming Language Used
  • Language: Python
  • Version: Python 3.9.7

Code Structure:
The code is structured as follows:
1. Main Script (bnet.py):
  • Contains all the functions and logic for processing input, validating queries,
      calculating probabilities, and displaying results.
  • Implements Bayesian reasoning to compute probabilities based on a dataset.
  • Includes helper functions for conditional probabilities and joint 
      probability distributions (JPDs).
2.Input File (data.txt):
  • The dataset file provided as input.
  • Each line in the file contains values for the variables:
  • B (Baseball Game on TV)
  • G (George Watching TV)
  • C (Cat Food Status)
  • F (George Feeding the Cat)
  • The values in each line are binary (1 for True, 0 for False), separated by spaces.
3.Generated Output:
  • Probabilities are printed to the console for interactive queries.

How to Run the Code:
1.Prepare the Input File:
  • Place the dataset in the same directory as the script.
  • The input file should be named training_data.txt and contain one record per line with space-separated binary values (e.g., 1 0 1 0).
2.Running the Code:
  • Ensure Python 3.9.7 (or compatible) is installed on your machine.
  • Run the script from the command line as follows: 
     python bnet.py training_data.txt
3.Interactive Query System:
  • After the script starts, the pre-computed probabilities are displayed.
  • Enter a query in the format: <Query Variables> given 
    <Evidence  Variables>
  • Example: Bt Gf given Ct
  • To exit, type none.
4.Specific Notes:
  • Ensure training_data.txt is correctly formatted before running the script.
  • Follow query guidelines for valid variable names and formats.

Running on ACS Omega:
  This code does not require ACS Omega but can run on it if Python 3.9.7 is installed. Ensure the dataset and script are uploaded to ACS Omega and follow the instructions to execute.

Important Notes:
  • Queries must use valid variable names (B, G, C, F) and values (T or F).
  • Ensure the dataset file is in the correct format and accessible.
  • If compilation is required, include specific instructions for the environment setup.
