# Huy Viet Nguyen, 2136374
# Alaa Eddine Chenak, 1976567

import time
import random
from copy import deepcopy

from schedule import Schedule

type Transition = tuple[int, int]  # Transition d'un cours d'un créneau vers un autre


def get_timer(duration: int = 300, time_margin: int = 5):
    """
    Génère une fonction qui indique si le temps est écoulé ou non.
    La marge de temps est pour donner assez de temps pour que le code 
    sorte de sa recherche et retourne une solution. 
    """
    assert duration > time_margin

    start = time.time()
    return lambda: time.time() - start > duration - time_margin

def get_random_solution(schedule: Schedule) -> dict:
    """
    Génère une solution en attribuant aléatoirement un créneau pour
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

def local_search(
        schedule: Schedule,
        initial_solution: dict,
        timeout: lambda: bool,
        stagnation_counter: int
    ) -> tuple[dict, int]:
    # Conflits pour chaque cours. Pour simplifier l'accès au conflits d'un cours.
    conflicts: dict[str, set] = { course:schedule.get_node_conflicts(course) for course in schedule.course_list }
    
    # Conversion de créneaux de réponse dict(cours, crénau) à des créneaux pour la 
    # recherche dict(crénau, liste des cours). Ceci va faciliter l'évaluation des solutions 
    # (qui est le nombre de clés) et la suppression de créneaux.
    course_slots: dict[int, list] = dict()
    for course, slot in initial_solution.items():
        course_slots.setdefault(slot, list()).append(course)

    best_solution = deepcopy(course_slots)  # La meilleure solution avant la recherche.

    i = 0
    while not timeout():
        # On trouve un voisin préférablement loin de la solution initiale,
        # puis on le garde ou non comme une solution. Ensuite, on continue 
        # la recherche à partir du voisin.
        # 
        # Expérimentalement, si le nombre de transitions pour trouver un voisin
        # est suffisament grand, la recherche va converger rapidement vers une
        # très bonne solution. Cependant, plus le nombre de transitions est
        # grand, plus le coût en temps de calcul sera grand.
        number_transitions = 20000
        transitions = get_neighbor(list(course_slots.keys()), number_transitions, timeout)
        solution = select(course_slots, conflicts, transitions, timeout)

        if solution is not None:
            i = 0
            best_solution = solution
        else:
            i += 1

        if i == stagnation_counter:
            break

    # Fin de la recherche, conversion des créneaux-recherche à des créneaux-réponse.
    solution = dict()
    for slot, courses in enumerate(best_solution.values()):
        for course in courses:
            solution[course] = slot + 1

    return solution, len(best_solution)

def get_neighbor(
        slots: list, 
        number_transitions: int,
        timeout: lambda: bool
    ) -> list[Transition]:
    """
    Génère un voisin pour la recherche. Pour éviter de faire une
    nouvelle copie de la solution à chaque voisin, seul une liste de 
    transitions pour l'atteindre est retournée.
    """
    transitions = []
    for _ in range(number_transitions):
        if timeout(): []

        i = random.randint(0, len(slots) - 1)
        j = (i + random.randint(0, len(slots) - 2)) % len(slots)  # Moyen pour éviter des transitions (i, i) qui ne fait rien

        transitions.append((slots[i], slots[j]))

    return transitions

def apply_transitions(course_slots, conflicts, transitions, timeout) -> list[Transition] | None:
    """
    Applique les transitions sur les créneaux horaires donnés. Si une 
    transition est impossible, on ignore la transition et on continue 
    pour les autres.
    """
    for transition in transitions:
        if timeout(): return []

        i, j = transition

        if course_slots.get(i) is None or course_slots.get(j) is None:
            continue

        course = course_slots[i].pop()
        course_conflicts = conflicts[course]

        if any(course in course_conflicts for course in course_slots[j]):
            course_slots[i].append(course)  # On annule la transition
        else:
            course_slots[j].append(course)  # On applique la transition

            if len(course_slots[i]) == 0:
                del course_slots[i]

def select(
        course_slots: dict[int, list],
        conflicts: dict[str, set],
        transitions: list[Transition],
        timeout: lambda: bool
    ) -> dict | None:
    """
    Applique les transitions pour obtenir une nouvelle solution, si
    la nouvelle solution est meilleure, on retourne une nouvelle 
    copie de la solution, sinon on retourne rien.
    """
    initial_score = len(course_slots)

    apply_transitions(course_slots, conflicts, transitions, timeout)

    score_variation = initial_score - len(course_slots)
    if score_variation > 0:
        return deepcopy(course_slots)
    else:
        return None

def solve(schedule: Schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """
    # Add here your agent
    timeout = get_timer()

    old_solution = get_random_solution(schedule)
    best_solution = deepcopy(old_solution)
    best_eval = schedule.get_n_creneaux(old_solution)

    stagnation_counter = best_eval

    while not timeout():
        try:
            # Techniquement une seule recherche locale est faite, car on fait toujours la recherche sur
            # old_solution qui représente l'endroit où la recherche s'est arrêtée. Le stagnation_counter
            # sert plus à obtenir la meilleure solution actuelle que la recherche a trouvée.
            solution, evaluation = local_search(schedule, old_solution, timeout, stagnation_counter)
        except Exception:
            continue  # Pour éviter que la recherche se termine à cause d'une erreur et aucune solution n'est retournée.

        if timeout(): break

        if best_eval > evaluation:
            best_solution = solution
            best_eval = evaluation
            print("Meilleure solution:", best_eval)

    return best_solution
