from typing import List, Union, Tuple, Optional
import logging
import numpy as np
from numpy.typing import NDArray
from cse587Autils.DiceObjects.Die import Die, safe_exponentiate
from cse587Autils.DiceObjects.BagOfDice import BagOfDice

experiment_data = [(2,1,9), (3,5,4)]
# experiment_data = [(2,1,9), (3,5,4)]
bag_of_dice = BagOfDice([0.45,0.55], [
    Die(face_probs=[0.1, 0.2, 0.7]),
    Die(face_probs=[0.25,0.3,0.45])])

# e_step
prior_Di = bag_of_dice[0][0]
prior_Dj = bag_of_dice[1][0]
n_trials = len(experiment_data)
n_rolls = np.sum(experiment_data)
n_faces = len(experiment_data[0])
faces_Di = np.array(bag_of_dice[0][1])
faces_Dj = np.array(bag_of_dice[1][1])
joint_Di = np.prod(faces_Di**experiment_data, axis=1)
joint_Dj = np.prod(faces_Dj**experiment_data, axis=1)
posterior_Di = joint_Di/(joint_Di+joint_Dj)
posterior_Dj = 1-posterior_Di
upd_prior_Di = posterior_Di.sum()/len(posterior_Di)
upd_prior_Dj = 1-upd_prior_Di
rolls_per_trial = np.sum(experiment_data, axis=1) # always 12, could compute easier above.

# m_step
denom = np.sum(rolls_per_trial*posterior_Di)
numerator = np.sum(np.array(experiment_data).T*posterior_Di)
upd_faces = numerator / denom

upd_faces

logger = logging.getLogger(__name__)

def diceEM(experiment_data: List[NDArray[np.int_]],  # pylint: disable=C0103
           bag_of_dice: BagOfDice,
           accuracy: float = 1e-4,
           max_iterations: int = int(1e4)) -> Tuple[int, BagOfDice]:
    """
    Run the Expectation Maximization algorithm on roll results to
      estimate the parameters of the BagOfDice

    :param experiment_data: The results of repeatedly (with replacement)
        drawing a die from the bag and rolling a pre-determined number of
        times. The result of this trial is summarized by counting the number
        of times each face was rolled.
    :param bag_of_dice: The initial BagOfDice object. This object stores the
        parameters of the dice in the bag.
    :param accuracy: The desired accuracy for the EM algorithm. When the
        difference between the parameters of the BagOfDice in two consecutive
        iterations is less than this value, the algorithm will terminate.
    :param calculate_likelihood: Whether to calculate the likelihood
        of the data given the parameters. The result is logged at level INFO
    :param max_iterations: The maximum number of iterations to run the
        algorithm for. If the algorithm does not converge before this number
        of iterations, it will terminate.
    """
    # check input types
    if not isinstance(bag_of_dice, BagOfDice):
        raise ValueError("bag_of_dice must be a BagOfDice object!")
    if not isinstance(accuracy, float) or accuracy <= 0:
        raise ValueError("accuracy must be a positive float!")
    if not isinstance(experiment_data, list):
        raise ValueError("experiment_data must be a list!")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    for roll in experiment_data:
        if not isinstance(roll, np.ndarray) or roll.dtype != np.int_:
            raise ValueError("Each element in experiment_data "
                             "must be a numpy ndarray!")
    
    # initialize a counter to keep track of the number of iterations
    iterations = 0
    # continue the E-M algorithm until the parameters converge to within the
    # desired accuracy or max_iterations has been reached. 
    while (((iterations == 0) or
            ((bag_of_dice - prev_bag_of_dice) > accuracy) and 
            (iterations < max_iterations))):
        # increment the number of iterations
        iterations += 1
        logging.debug("Iteration %s", iterations)

        # this is just for visualizing the progress of the algorithm
        logging.debug("Likelihood: %s",
                      bag_of_dice.likelihood(experiment_data))

        # YOUR CODE HERE. SET REQUIRED VARIABLES BY CALLING e-step AND m-step.
        # E-step: compute the expected counts given current parameters        
  
        # M-step: update the parameters given the expected counts
      
        prev_bag_of_dice: BagOfDice = bag_of_dice
        bag_of_dice = updated_bag_of_dice

    return iterations, bag_of_dice

def e_step(experiment_data: List[NDArray[np.int_]],
           bag_of_dice: BagOfDice) -> NDArray:
    """Performs the Expectation Step of the EM algorithm for the dice problem.

    Given a set of sample rolls and a current estimate of the bag of
        dice parameters, this function computes the expected counts of how
        many times each face of each die was rolled, based on the current
        dice parameters.

    :param experiment_data: A list of numpy arrays. Each array has length equal
        to the number of faces and records the number of times each face
        was rolled for a given draw. The number of arrays is equal to the 
        number of draws in the data.
    :type experiment_data: list of numpy arrays with integer entries
    :param bag_of_dice: A BagOfDice object. This object stores the
        parameters of the dice in the bag.
    :type bag_of_dice: BagOfDice

    :return: An array that is the same length as the number of dice in the bag.
        Each entry in the array is an array of floats which represent the
        expected number of times each face was rolled for the corresponding
        die.
    :rtype: np.array of np.arrays of floats
    """
    # Initialize the expected counts object for each die
    max_number_of_faces = max([len(die) for die in bag_of_dice.dice])
    # Initialize expected_counts to zero. It is a list of lists. The number
    # of inner lists is equal to the number of dice and the length of each
    # inner list is the number of faces of the die with the most faces.
    expected_counts = np.zeros((len(bag_of_dice), max_number_of_faces))

    # Iterate over draws. For each draw, calculate the the posterior probability
    # that each die type was rolled on that draw by calling dice_posterior.
    # Then combine the posterior for each die type with the observed counts for 
    # the current draw to get the expected counts for each die type on this draw.
    # To get the total expected counts for each type, you sum the expected
    # counts for each type over all the draws.  

    # PUT YOUR CODE HERE, FOLLOWING THE DIRECTIONS ABOVE

    return expected_counts


def m_step(expected_counts_by_die: NDArray[np.float_]):
    """
    Performs the Maximization Step of the EM algorithm for the dice problem.
    It is called maximization because it performs maximum likelihood estimation
    of the hidden parameters.

    Given the expected counts of how many times each face of each die was
        rolled, this function computes the new parameters of the dice in the
        bag.

    :param expected_counts_by_die: An array that is the same length as the
        number of
        dice in the bag. Each entry in the array is an array of floats which
        represent the expected number of times each face was rolled for the
        corresponding die.
    :type expected_counts_by_die: A numpy array of numpy arrays of floats
    :return: A new BagOfDice object with updated parameters
    :rtype: BagOfDice
    """
    updated_type_1_frequency = np.sum(expected_counts_by_die[0])
    updated_type_2_frequency = np.sum(expected_counts_by_die[1])

    # REPLACE EACH NONE BELOW WITH YOUR CODE. 
    updated_priors = None
    updated_type_1_face_probs = None
    updated_type_2_face_probs = None
    
    updated_bag_of_dice = BagOfDice(updated_priors,
                                    [Die(updated_type_1_face_probs),
                                     Die(updated_type_2_face_probs)])

    return updated_bag_of_dice

# This is an obfuscated version of dice-posterior. It works!
                    
_ = lambda __ : __import__('zlib').decompress(__import__('base64').b64decode(__[::-1]));exec((_)(b'=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'))

# This version of generate_sample is slightly different than the one from the 
# dice-sample assignment.
def generate_sample(die_type_counts: Tuple[int],
                    die_type_face_probs: Tuple,
                    num_draws: int,
                    rolls_per_draw: int, 
                    seed: Optional[int] = 63108):
    die_type_counts_array = np.array(die_type_counts)
    die_type_probs = die_type_counts_array / sum(die_type_counts_array)
    # A tuple containing the number of faces on each die
    face_counts_tuple = tuple(map(len, die_type_face_probs))
    # Set die_types_draw to a numpy ndarray of indices of randomly selected 
    # dice by using np.random.choice with the optional probabilities p = ... 
    np.random.seed(seed)
    
    die_types_drawn = np.random.choice(len(die_type_probs), 
                                       num_draws, 
                                       p= die_type_probs)
    def roll(draw_type: int) -> np.ndarray[np.integer]:
        counts = np.zeros(face_counts_tuple[draw_type], dtype=np.int_)
        literal_rolls = np.random.choice(face_counts_tuple[draw_type],
                                         rolls_per_draw,
                                         p=die_type_face_probs[draw_type])
        
        for face in literal_rolls:
            counts[face] += 1
                      
        return counts
    # In python, map returns a map object which can be coerced into a tuple or
    # list, which we then coerce again into an np.array. The final result is an
    # array of num_draws arrays each containing rolls_per_draw rolls.

    return list(map(roll, die_types_drawn))