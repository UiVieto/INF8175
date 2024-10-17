# Huy Viet Nguyen, 2136374
# Alaa Eddine Chenak, 

import time
import random

from schedule import Schedule


def get_neighborhood(solution: dict, schedule: Schedule, timeout: lambda: bool, max_neighborhood_size: int=500):
    """Retourne un générateur du voisinage d'une solution et d'une certaine taille maximale. 
    Chaque élément du voisinage va différer d'au maximum un crénau par rapport à la solution donnée. 
    Si la fonction timeout retroune vrai, aucun voisinage est retourné et l'exécution s'arrête
    lorsque possible.

    :solution: dict
    """
    if timeout(): return []

    yield solution.copy()

    counter = 0
    number_slots = schedule.get_n_creneaux(solution)
    courses = list(solution.keys())
    random.shuffle(courses)

    for course in courses:
        if timeout(): return []

        if counter < max_neighborhood_size:
            neighbor = solution.copy()
            neighbor[course] = random.randint(1, number_slots)
            counter += 1

            yield neighbor
        else:
            return

def get_valid_neighbors(neighboors: list[dict], schedule: Schedule, timeout: lambda: bool) -> list[dict]:
    valid_neighbors = []

    for possible_solution in neighboors:
        if timeout(): return []

        if sum(possible_solution[a[0]] == possible_solution[a[1]] for a in schedule.conflict_list) == 0:
            valid_neighbors.append(possible_solution)

    return valid_neighbors

def select(neighbors: list[dict], schedule: Schedule, timeout: lambda: bool) -> dict | None:
    if timeout(): return dict()

    min_slots = schedule.get_n_creneaux(min(neighbors, key=lambda neighbor: schedule.get_n_creneaux(neighbor)))
    
    possible_solutions = []
    for neighbor in neighbors:
        if timeout(): return dict()

        if schedule.get_n_creneaux(neighbor) == min_slots:
            possible_solutions.append(neighbor)

    return possible_solutions[random.randint(0, len(possible_solutions) - 1)]

def eval(solution: dict, schedule: Schedule) -> int:
    return schedule.get_n_creneaux(solution)

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

def local_search(schedule: Schedule, initial_solution: dict, timeout: lambda: bool, stagnation_counter: int=5):
    best_solution = initial_solution
    counter = 0
    
    while not timeout():
        keys = list(best_solution.keys())
        random.shuffle(keys)
        neighborhood = get_neighborhood(best_solution, schedule, timeout)
        valid_neighbors = get_valid_neighbors(neighborhood, schedule, timeout)
        current_solution = select(valid_neighbors, schedule, timeout)

        if not timeout():
            if eval(best_solution, schedule) > eval(current_solution, schedule):
                best_solution = current_solution
                counter = 0
            else:
                counter += 1

            if counter >= stagnation_counter:
                return best_solution
                    
    return best_solution

def solve(schedule: Schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """
    # Add here your agent
    timeout = get_timer()

    random_solution = get_random_solution(schedule)
    best_solution = random_solution

    i = 0
    with open('stats.txt', 'a') as f:
        while not timeout():
            i += 1
            start = time.time()
            solution = local_search(schedule, random_solution, timeout, (schedule.get_n_creneaux(random_solution) // 5) + 1)

            if eval(best_solution, schedule) > eval(solution, schedule):
                f.write(f"Search count: {i}, Time spent in search: {time.time() - start}, Solution: {schedule.get_n_creneaux(solution)}\n")
                best_solution = solution

        f.write(f"Number of searches: {i}\n")

    return best_solution
