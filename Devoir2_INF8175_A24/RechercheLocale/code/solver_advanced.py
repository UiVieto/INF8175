import time
import random

from schedule import Schedule
import solver_naive

# TODO: implémenter un itérateur pour le voisinage
class Neighborhood():
    def __init__(self, initial_solution, max_neighborhood_size: int=3000):
        self.solution = initial_solution
        self.max_size = max_neighborhood_size

    def __iter__(self): 
        return self
    
    def __next__(self):
        return


def get_neighborhood(solution: dict, timeout: lambda: bool, max_neighborhood_size: int=3000) -> list[dict]:
    """Retourne un générateur du voisinage d'une solution et d'une certaine taille maximale. 
    Chaque élément du voisinage va différer d'au maximum un crénau par rapport à la solution donnée. 
    Si la fonction timeout retroune vrai, aucun voisinage est retourné et l'exécution s'arrête
    lorsque possible.

    :solution: dict
    """
    if timeout(): return []

    neighborhood = []
    shuffled_courses = list(solution.keys())
    random.shuffle(shuffled_courses)

    for course in shuffled_courses:
        for other_course in shuffled_courses:
            if timeout(): return []

            if len(neighborhood) > max_neighborhood_size:
                return neighborhood

            neighbor = solution.copy()
            neighbor[other_course] = solution[course]
            neighborhood.append(neighbor)
    
    return neighborhood

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

def get_timer(duration: int = 600, time_margin: int = 5):
    assert duration > time_margin

    start = time.time()
    return lambda: time.time() - start > duration - time_margin

def local_search(schedule, initial_solution: dict, timeout: lambda: bool, stagnation_counter: int=10):
    best_solution = initial_solution
    counter = 0
    
    while not timeout():
        neighborhood = get_neighborhood(best_solution, timeout)
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
    timeout = get_timer(120, 20)

    naive_solution = solver_naive.solve(schedule)
    best_solution = naive_solution
    
    while not timeout():
        solution = local_search(schedule, naive_solution, timeout)

        if eval(best_solution, schedule) > eval(solution, schedule):
            best_solution = solution
    
    return best_solution