
def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01):
    graded = [ (fitness(x, target), x) for x in pop]
    graded = [ x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]

    # randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)

    # mutate some individuals
    for individual in parents:
        if mutate > random.random():
            pos_to_mutate = randint(0, len(individual)-1)
            # this mutation is not ideal, because it
            # restricts the range of possible values,
            # but the function is unaware of the min/max
            # values used to create the individuals,
            individual[pos_to_mutate] = randint(
                min(individual), max(individual))

    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) // 2
            child = male[:half] + female[half:]
            children.append(child)

    parents.extend(children)
    return parents



def grade(pop, target):
    'Find average fitness for a population.'
    summed = reduce(add, (fitness(x, target) for x in pop), 0)
    return summed / (len(pop) * 1.0)


from operator import add
def fitness(individual, target):
    sum = reduce(add, individual, 0)
    return abs(target-sum)

from random import randint
def individual(length, min, max):
    'Create a member of the population.'
    return [ randint(min,max) for x in range(1,length) ]


def population(count, length, min, max):
    return [ individual(length, min, max) for x in range(1,count) ]

target = 371
p_count = 100
i_length = 5
i_min = 0
i_max = 100
tolerate=1
scores=[]
individuals=[]
p = population(p_count, i_length, i_min, i_max)
for iter in range(1,5):
    p = evolve(p, target)
    for individual in p:
        score=fitness(individual, 371)
        individuals.append(individual)
        scores.append(score)
        if score < tolerate:
            break
