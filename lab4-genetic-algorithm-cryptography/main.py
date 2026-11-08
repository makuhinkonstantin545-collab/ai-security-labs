from collections import Counter
import string
ALPHABET = string.ascii_lowercase
ALPHABET_SIZE = len(ALPHABET)
ENGLISH_FREQS = {
    'e': 12.70, 't': 9.06, 'a': 8.17, 'o': 7.51,
    'i': 6.97, 'n': 6.75, 's': 6.33, 'h': 6.09,
    'r': 5.99, 'd': 4.25, 'l': 4.03, 'c': 2.78,
    'u': 2.76, 'm': 2.41, 'w': 2.36, 'f': 2.23,
    'g': 2.02, 'y': 1.97, 'p': 1.93, 'b': 1.29,
    'v': 0.98, 'k': 0.77, 'j': 0.15, 'x': 0.15,
    'q': 0.10, 'z': 0.07
}

import random

POPULATION_SIZE = 100
MAX_GENERATIONS = 100
MUTATION_RATE = 0.1

def initialize_population():
    return [
        random.randint(1, ALPHABET_SIZE - 1)
        for _ in range(POPULATION_SIZE)
    ]

def crossover(parent1, parent2):
    return parent2, parent1

def mutate(key):
    if random.random() < MUTATION_RATE:
        key += random.choice([-1, 1])
        key = key % ALPHABET_SIZE

    return key

def genetic_algorithm(ciphertext):
    population = initialize_population()

    best_key = None
    best_fitness = 0

    for generation in range(MAX_GENERATIONS):

        fitnesses = evaluate_population(
            ciphertext,
            population
        )

        best_index = max(
            range(len(fitnesses)),
            key=lambda i: fitnesses[i]
        )

        if fitnesses[best_index] > best_fitness:
            best_fitness = fitnesses[best_index]
            best_key = population[best_index]

        parents = select_parents(
            population,
            fitnesses
        )

        new_population = [parents[0]]

        while len(new_population) < POPULATION_SIZE:

            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            child1, child2 = crossover(
                parent1,
                parent2
            )

            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.append(child1)

            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    return best_key, best_fitness

def select_parents(population, fitnesses):
    sorted_indices = sorted(
        range(len(fitnesses)),
        key=lambda i: fitnesses[i],
        reverse=True
    )

    elite_count = POPULATION_SIZE // 2

    parents = [
        population[i]
        for i in sorted_indices[:elite_count]
    ]

    return parents

def evaluate_population(ciphertext, population):
    fitnesses = []

    for key in population:
        fitness = calculate_fitness(
            ciphertext,
            key,
            ENGLISH_FREQS
        )

        fitnesses.append(fitness)

    return fitnesses

def calculate_fitness(ciphertext, key, expected_freqs):
    decoded_text = decrypt(ciphertext, key)

    text_letters = [
        char for char in decoded_text.lower()
        if char in ALPHABET
    ]

    if not text_letters:
        return 0

    total_letters = len(text_letters)
    actual_counts = Counter(text_letters)

    fitness_score = 0

    for char, expected_freq in expected_freqs.items():
        actual_freq = (
            actual_counts.get(char, 0) / total_letters
        ) * 100

        fitness_score += abs(actual_freq - expected_freq)

    return 10000.0 / (fitness_score + 1e-6)

def caesar_cipher(text, key):
    result = ""

    for char in text:
        if char.isalpha():
            if char.isupper():
                base = ord('A')
            else:
                base = ord('a')

            shifted = (ord(char) - base + key) % 26
            result += chr(base + shifted)
        else:
            result += char

    return result


def encrypt(text, key):
    return caesar_cipher(text, key)


def decrypt(text, key):
    return caesar_cipher(text, -key)


message = "HELLO WORLD"
key = 3

encrypted = encrypt(message, key)
decrypted = decrypt(encrypted, key)

print("Исходный текст:", message)
print("Ключ:", key)
print("Зашифрованный текст:", encrypted)
print("Расшифрованный текст:", decrypted)

test_text = "the quick brown fox jumps over the lazy dog"

test_encrypted = encrypt(test_text, 5)

fitness = calculate_fitness(
    test_encrypted,
    5,
    ENGLISH_FREQS
)

print("\nПроверка Fitness Function")
print("Исходный текст:", test_text)
print("Зашифрованный текст:", test_encrypted)
print("Оценка Fitness:", fitness)

population = initialize_population()

print("\nПроверка популяции:")
print("Количество особей:", len(population))
print("Популяция:", population)
test_population = initialize_population()

test_fitnesses = evaluate_population(
    test_encrypted,
    test_population
)

parents = select_parents(
    test_population,
    test_fitnesses
)

print("\nПроверка селекции:")
print("Размер исходной популяции:", len(test_population))
print("Количество родителей:", len(parents))
print("Лучший Fitness:", max(test_fitnesses))
print("Выбранные родители:", parents)

print("\nПроверка Crossover:")

child1, child2 = crossover(5, 10)

print("Родитель 1:", 5)
print("Родитель 2:", 10)
print("Потомок 1:", child1)
print("Потомок 2:", child2)

print("\nПроверка мутации:")

test_key = 5

for i in range(10):
    mutated_key = mutate(test_key)

    print(
        f"Попытка {i + 1}: "
        f"исходный ключ = {test_key}, "
        f"после мутации = {mutated_key}"
    )

print("\nЗапуск генетического алгоритма:")

best_key, best_fitness = genetic_algorithm(test_encrypted)

decrypted_text = decrypt(test_encrypted, best_key)

print("Найденный ключ:", best_key)
print("Лучший Fitness:", best_fitness)
print("Исходный зашифрованный текст:", test_encrypted)
print("Результат расшифровки:", decrypted_text)