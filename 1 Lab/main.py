import csv
import random

from cards_data import banks, payment_systems_codes
from generation_functions import (generate_name, generate_passport, generate_snils, generate_unique_card, generate_doctors_data,
                                  generate_analyses_date, generate_price, generate_visit_date, generate_next_visit_date)


while True:
    n = int(input("Введите кол-во ячеек. Число должно быть больше 50.000. \nКол-во: "))
    if n < 50_000:
        print("Ошибка! Введите число больше чем 49.999")
    else:
        break

print("Введите веса для банков. В сумме должно быть больше 0!")
bank_weights = []
for bank in banks:
    weight = float(input(f"Введите вес для {bank}: "))
    bank_weights.append(weight)

print("Введите веса для платёжных систем. В сумме должно быть больше 0!")
payment_systems = list(payment_systems_codes.keys())
payment_weights = []
for system in payment_systems:
    weight = float(input(f"Введите вес для {system}: "))
    payment_weights.append(weight)

unique_patients = []
for _ in range(int(n*0.7)):
    name = generate_name()
    passport = generate_passport()
    snils = generate_snils()
    visit_date = generate_visit_date()
    analyses_date = generate_analyses_date(visit_date)
    card = generate_unique_card(bank_weights, payment_weights)
    
    unique_patients.append({
        "ФИО": name,
        "Паспортные данные": passport,
        "СНИЛС": snils,
        "Дата посещения врача": visit_date,
        "Дата получения анализов": analyses_date,
        "Карта оплаты": card
    })

slots = []
for _ in range(n):
    patient = random.choice(unique_patients)

    doctor = generate_doctors_data(False)
    symptoms = generate_doctors_data(doctor)[0]
    analyses = generate_doctors_data(doctor)[1]
    price = generate_price()

    record = {
        "ФИО": patient["ФИО"],
        "Паспортные данные": patient["Паспортные данные"],
        "СНИЛС": patient["СНИЛС"],
        "Симптомы": ', '.join(symptoms),
        "Выбор врача": doctor,
        "Дата посещения врача": patient["Дата посещения врача"],
        "Анализы": ', '.join(analyses),
        "Дата получения анализов": patient["Дата получения анализов"],
        "Стоимость анализов": price,
        "Карта оплаты": ', '.join(patient["Карта оплаты"])
    }

    patient["Дата посещения врача"] = generate_next_visit_date(patient["Дата получения анализов"])
    patient["Дата получения анализов"] = generate_analyses_date(patient["Дата посещения врача"])

    slots.append(record)

with open('1 Lab/dataset.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["ФИО", "Паспортные данные", "СНИЛС", "Симптомы", "Выбор врача", "Дата посещения врача", "Анализы", "Дата получения анализов", "Стоимость анализов", "Карта оплаты"])
    writer.writeheader()
    writer.writerows(slots)

print("Данные успешно записаны в файл dataset.csv")
