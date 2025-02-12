import numpy as np
from thesis_code.analysis.algo4 import fit_state_sequence  # Import function

def test_fit_state_sequence(debug_flags=None):
    """
    Tests fit_state_sequence by computing and optionally printing C, Lambda, V, and S.
    
    Parameters:
    - debug_flags (dict): Dictionary controlling what to print.
      Example:
        {
            "print_C": True,
            "print_Lambda": False,
            "print_V": True,
            "print_S": True
        }
    """
    if debug_flags is None:
        debug_flags = {"print_C": True, "print_Lambda": True, "print_V": True, "print_S": True}

    # Step 1: Define a simple observation matrix Y (T x D)
    Y = np.array([[1, 0.2],  
                  [0.5, 0.2],  
                  [1.2, 2.1],
                  [9, 8],
                  [5.5, 10]])

    # Step 2: Define known state parameters theta (num_states x D)
    theta = np.array([[1, 1],  
                      [2, 2]])  

    # Step 3: Create a small probability simplex grid C (num_states x N)
    C = np.array([[0.8, 0.2],   
                  [0.2, 0.8]])  

    if debug_flags.get("print_C", False):
        print("\n📌 **C (Probability Simplex Grid)**:\n", C)

    # Step 4: Define lambda penalty
    lambda_penalty = 1  

    # Step 5: Compute Lambda matrix
    Lambda = (lambda_penalty / 4) * np.sum(np.abs(C[:, :, None] - C[:, None, :]), axis=0)

    if debug_flags.get("print_Lambda", False):
        print("\n📌 **Lambda (Jump Penalty Matrix)**:\n", Lambda)

    # Step 6: Run fit_state_sequence
    S_test = fit_state_sequence(Y, theta, C, lambda_penalty, num_states=2)

    if debug_flags.get("print_V", False):
        print("\n📌 **V (Dynamic Programming Table) - Needs to be printed inside fit_state_sequence**")

    if debug_flags.get("print_S", False):
        print("\n📌 **S (Estimated State Sequence)**:\n", S_test)
        print("Row sums of S_test:", S_test.sum(axis=1))

    # Step 7: Validate S
    for row in S_test:
        assert any(np.allclose(row, C[:, i]) for i in range(C.shape[1])), "❌ S contains invalid values!"
    
    #print("\n✅ **Test Passed: S contains only valid probability simplex vectors from C!**")

# Run the test with debugging options
test_fit_state_sequence(debug_flags={
    "print_C": False,      # Print C matrix?
    "print_Lambda": False, # Print Lambda matrix?
    "print_V": False,     # Print V matrix? (Should be printed inside fit_state_sequence)
    "print_S": True       # Print S matrix?
})