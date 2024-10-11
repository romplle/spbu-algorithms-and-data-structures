from datetime import datetime, timedelta
import random

from names_data import male_names, female_names, last_names, male_patronymics, female_patronymics
from doctors_data import doctors_data
from cards_data import banks, bin_codes, payment_systems_codes


def generate_name():
    gender = random.choice(["male", "female"])
    if gender == "male":
        name = random.choice(last_names) + ' ' + random.choice(male_names) + ' ' + random.choice(male_patronymics)
    else:
        name = random.choice(last_names) + 'а ' + random.choice(female_names) + ' ' + random.choice(female_patronymics)
    return name

def generate_passport():
    countries = ["Россия", "Беларусь или Казахстан"]
    weights = [99, 1]
    country = random.choices(countries, weights=weights, k=1)[0]
    if country == "Россия":
        series = f"{random.randint(1000, 9999)}"
        number = f"{random.randint(100000, 999999)}"
        return f"{series} {number}"
    
    else:
        series = ''.join(random.choices('ABCEHIKMOPTXY', k=2))
        number = f"{random.randint(1000000, 9999999)}"
        return f"{series} {number}"

def generate_snils():
    return f"{random.randint(0, 999):03}" + '-' f"{random.randint(0, 999):03}" + '-' f"{random.randint(0, 999):03}" + ' ' f"{random.randint(0, 99):02}" 

def generate_doctors_data(doctor):
    if (not doctor):
        random_doctor = random.choice(list(doctors_data.keys()))
        return random_doctor
    else:
        symptoms, tests = doctors_data[doctor]

        random_symptoms = random.sample(symptoms, random.choices(range(1, 11), weights=[6, 7, 8, 7, 6, 5, 4, 3, 2, 1])[0])
        random_tests = random.sample(tests, random.choices(range(1, 6), weights=[2, 3, 2, 1, 1])[0])

        return random_symptoms, random_tests
  
def generate_visit_date():
    today = datetime.now()
    visit_date = today + timedelta(days=random.randint(-183 * 3, 30))
    
    while visit_date.weekday() >= 5:
        visit_date += timedelta(days=1)
    
    visit_time = timedelta(hours=random.randint(9, 18), minutes=random.randint(0, 59))
    visit_date = visit_date.replace(hour=0, minute=0, second=0, microsecond=0) + visit_time
    
    return visit_date

def generate_analyses_date(visit_date):
    analyses_date = visit_date + timedelta(hours=random.randint(24, 72))

    while analyses_date.weekday() >= 5:
        analyses_date += timedelta(days=1)

    if analyses_date.hour < 9:
        analyses_date = analyses_date.replace(hour=random.randint(9, 10), minute=random.randint(0, 59))
    elif analyses_date.hour >= 18:
        analyses_date += timedelta(days=1)
        analyses_date = analyses_date.replace(hour=random.randint(9, 10), minute=random.randint(0, 59))
        
    return analyses_date

def generate_next_visit_date(analyses_date):
    next_visit_date = analyses_date + timedelta(days=random.randint(0, 30), hours=random.randint(24, 48))
    
    while next_visit_date.weekday() >= 5:
        next_visit_date += timedelta(days=1)
    
    if next_visit_date.hour < 9:
        next_visit_date = next_visit_date.replace(hour=random.randint(9, 10), minute=random.randint(0, 59))
    elif next_visit_date.hour >= 18:
        next_visit_date += timedelta(days=1)
        next_visit_date = next_visit_date.replace(hour=random.randint(9, 10), minute=random.randint(0, 59))
    
    return next_visit_date

used_cards = {}
def generate_card(bank_weights, payment_weights):
    bank = random.choices(banks, weights=bank_weights, k=1)[0]
    bin_code = random.choice(bin_codes[bank])
    payment_system = random.choices(list(payment_systems_codes.keys()), weights=payment_weights, k=1)[0] 

    card_number = f"{payment_systems_codes[payment_system]}{bin_code}{random.randint(0, 9999999999):010}"
    
    if card_number in used_cards:
        used_cards[card_number] += 1
    else:
        used_cards[card_number] = 1

    return card_number, bank, payment_system

def generate_unique_card(bank_weights, payment_weights):
    while True:
        card_number, bank, payment_system = generate_card(bank_weights, payment_weights)
        if used_cards[card_number] <= 5:
            return card_number, bank, payment_system

def generate_price():
    return str(random.randint(500, 10000)) + ' руб.'
