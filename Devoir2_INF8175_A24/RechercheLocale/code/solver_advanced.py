# Huy Viet Nguyen, 2136374
# Alaa Eddine Chenak, 

import time
import random
from typing import Generator, Iterator
from math import exp, floor
from copy import deepcopy

from schedule import Schedule


type Transition = tuple[int, int]  # Transition d'un cours d'un créneau vers un autre


def get_timer(duration: int = 300, time_margin: int = 5):
    assert duration > time_margin

    start = time.time()
    return lambda: time.time() - start > duration - time_margin

def get_random_solution(schedule: Schedule) -> dict:
    """Génère une solution en attribuant aléatoirement un créneau pour
    chaque cours.
    """
    solution = dict()
    
    courses = list(schedule.course_list)
    solution[courses.pop()] = 1

    for course in courses:
        slot = random.randint(1, len(solution))
        solution[course] = slot
        for node in schedule.get_node_conflicts(course):
            other_slot = solution.get(node)
            if other_slot is not None and other_slot == slot:
                solution[course] = len(solution)

    return solution

def get_neighborhood(
        slots: list, 
        transitions_per_branches: int,
        number_of_branches: int,
        timeout: lambda: bool
    ) -> Generator[list[Transition], None, None]:
    """Génère un voisinage pour la recherche. Pour éviter de faire une
    nouvelle copie de la solution à chaque voisin, seul des listes de 
    transitions d'une certaine taille pour aller d'un voisin vers un 
    autre (qu'on va appeler branches) sont données.
    """

    transitions = []
    for _ in range(number_of_branches):
        for _ in range(transitions_per_branches):
            if timeout(): yield []

            i = random.randint(0, len(slots) - 1)
            j = (i + random.randint(0, len(slots) - 2)) % len(slots)  # Moyen pour éviter des transitions (i, i) qui ne fait rien

            transitions.append((slots[i], slots[j]))

        yield transitions

def apply_transitions(course_slots, conflicts, transitions, timeout) -> list[Transition] | None:
    actual_transitions = []
    for transition in transitions:
        if timeout(): return []

        i, j = transition

        if course_slots.get(i) is None or course_slots.get(j) is None:
            continue

        course = course_slots[i].pop()
        course_conflicts = conflicts[course]

        if any(course in course_conflicts for course in course_slots[j]):
            course_slots[i].append(course)
        else:
            course_slots[j].append(course)
            actual_transitions.append(transition)

            if len(course_slots[i]) == 0:
                del course_slots[i]

    return actual_transitions

def revert_transitions(course_slots, transitions: list, timeout):
    transitions.reverse()
    for transition in transitions:
        if timeout(): return

        j, i = transition

        course = course_slots[i].pop()
        course_slots.setdefault(j, list()).append(course)

        if len(course_slots[i]) == 0:
            del course_slots[i]

def select(
        course_slots: dict[int, list],
        conflicts: dict[str, set],
        neighborhood: Iterator[list[Transition]],
        temperature: int,
        timeout: lambda: bool
    ) -> bool:

    for transitions in neighborhood:
        initial_score = len(course_slots)

        transitions = apply_transitions(course_slots, conflicts, transitions, timeout)

        score_variation = initial_score - len(course_slots)
        if score_variation > 0:
            return True  # Amélioration de la solution; on garde la solution actuelle
        
        else:
            return False
        #prob = exp(-1 * score_variation / temperature)
        #if random.random() > prob:
            #return False  # On garde la solution malgré une dégration
        
        # On a fini le parcours d'une branche de transitions; on rétourne à la solution initiale
        #revert_transitions(course_slots, transitions, timeout)
            
def local_search(
        schedule: Schedule,
        initial_solution: dict,
        timeout: lambda: bool,
        stagnation_counter: int=20
    ) -> tuple[dict, int]:

    conflicts: dict[str, set] = { course:schedule.get_node_conflicts(course) for course in schedule.course_list }
    
    course_slots: dict[int, list] = dict()
    for course, slot in initial_solution.items():
        course_slots.setdefault(slot, list()).append(course)

    initial_score = len(course_slots)
    best_solution = deepcopy(course_slots)
    best_score = initial_score

    i = 0
    while not timeout():
        # L'équation pour déterminer le nombre de transitions par branche a été trouvé expérimentalement.
        # Plus le score initial est grand, plus le nombre de transitions sera grand.
        # Si la solution s'améliore (soit le score diminue), moins il aura de transitions par branche.
        # Si la solution s'empire (soit le score augmente), plus il aura de transitions par branche.
        transitions_per_branch = max(1, floor(2 * (initial_score + best_score) / (10 + initial_score + (0.5 * best_score))))
        
        neighborhood = get_neighborhood(list(course_slots.keys()), transitions_per_branch, 10, timeout)
        is_improvement = select(course_slots, conflicts, neighborhood, 3, timeout)

        if is_improvement:
            i = 0
            best_solution = deepcopy(course_slots)
        else:
            i += 1

        if i == stagnation_counter:
            break

    solution = dict()
    for slot, courses in enumerate(best_solution.values()):
        for course in courses:
            solution[course] = slot + 1

    return solution, len(best_solution)

def solve(schedule: Schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """
    # Add here your agent
    timeout = get_timer(100, 0)

    random_solution = get_random_solution(schedule)
    best_solution = deepcopy(random_solution)
    best_eval = schedule.get_n_creneaux(random_solution)

    i = 0
    while not timeout():
        try:
            solution, evaluation = local_search(schedule, random_solution, timeout, len(best_solution))
        except Exception as e:
            i += 1
            continue  # TODO: Corriger l'erreur. Je ne sais pas c'est quoi l'erreur.

        if timeout(): break

        if best_eval > evaluation:
            best_solution = solution
            best_eval = evaluation
            print("Meilleure solution:", schedule.get_n_creneaux(solution), 'Erreurs:', i)

    return best_solution
