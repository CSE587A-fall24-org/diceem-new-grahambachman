from typing import List, Union, Tuple, Optional
import logging
import numpy as np
from numpy.typing import NDArray
from cse587Autils.DiceObjects.Die import Die, safe_exponentiate
from cse587Autils.DiceObjects.BagOfDice import BagOfDice

logger = logging.getLogger(__name__)

def generate_sample(die_type_counts: Tuple[int],
                    die_type_face_probs: Tuple,
                    num_draws: int,
                    rolls_per_draw: int, 
                    seed: Optional[int] = 63108):
    die_type_counts_array = np.array(die_type_counts)
    die_type_probs = die_type_counts_array / sum(die_type_counts_array)
    face_counts_tuple = tuple(map(len, die_type_face_probs))
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
    return list(map(roll, die_types_drawn))
def dice_posterior(sample_draw: List[int], 
                   die_type_probs: Tuple[float],
                   dice: Tuple[Die]) -> float:
    if len(dice) != 2:
        raise ValueError('This code requires exactly 2 dice')
    if len(dice[0]) != len(dice[1]):
        raise ValueError('This code requires two dice with the same number of faces')
    if len(sample_draw) != len(dice[0]):
        raise ValueError('The sample draw is a list of observed counts for the \
                         faces. Its length must be equal to the number of faces \
                         on the dice.')
    prior_Di = die_type_probs[0]
    prior_Dj = die_type_probs[1]
    likelihood_Di = np.prod([safe_exponentiate(dice[0].face_probs[d], sample_draw[d]) for d in range(len(sample_draw))])
    likelihood_Dj = np.prod([safe_exponentiate(dice[1].face_probs[d], sample_draw[d]) for d in range(len(sample_draw))])
    posterior_Di = (prior_Di*likelihood_Di)/(prior_Di*likelihood_Di+prior_Dj*likelihood_Dj)
    return posterior_Di
def e_step(experiment_data: List[NDArray[np.int_]],
           bag_of_dice: BagOfDice) -> NDArray:
    max_number_of_faces = max([len(die) for die in bag_of_dice.dice])
    expected_counts = np.zeros((len(bag_of_dice), max_number_of_faces))
    for trial in experiment_data:
        posterior_D_i = dice_posterior(trial,
                       (bag_of_dice[0][0], bag_of_dice[1][0]),
                       (bag_of_dice[0][1], bag_of_dice[1][1]))
        posterior_D_j = 1-posterior_D_i
        expected_counts[0] += posterior_D_i * trial
        expected_counts[1] += posterior_D_j * trial
    return expected_counts
def m_step(expected_counts_by_die: NDArray[np.float_]):
    updated_type_1_frequency = np.sum(expected_counts_by_die[0])
    updated_type_2_frequency = np.sum(expected_counts_by_die[1])
    updated_priors = np.sum(expected_counts_by_die, axis=1) / np.sum(expected_counts_by_die)
    updated_type_1_face_probs = expected_counts_by_die[0] / np.sum(expected_counts_by_die[0])
    updated_type_2_face_probs = expected_counts_by_die[1] / np.sum(expected_counts_by_die[1])
    updated_bag_of_dice = BagOfDice(updated_priors,
                                    [Die(updated_type_1_face_probs),
                                     Die(updated_type_2_face_probs)])
    return updated_bag_of_dice
def diceEM(experiment_data: List[NDArray[np.int_]],  # pylint: disable=C0103
           bag_of_dice: BagOfDice,
           accuracy: float = 1e-4,
           max_iterations: int = int(1e4)) -> Tuple[int, BagOfDice]:
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
    iterations = 0
    while (((iterations == 0) or
            ((bag_of_dice - prev_bag_of_dice) > accuracy) and 
            (iterations < max_iterations))):
        iterations += 1
        logging.debug("Iteration %s", iterations)
        logging.debug("Likelihood: %s",
                      bag_of_dice.likelihood(experiment_data))     
        expected_counts_by_die = e_step(experiment_data, bag_of_dice)
        updated_bag_of_dice = m_step(expected_counts_by_die)
        prev_bag_of_dice: BagOfDice = bag_of_dice
        bag_of_dice = updated_bag_of_dice
    return iterations, bag_of_dice