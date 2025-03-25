import pandas as pd

from doctors_data import doctors_departments, symptoms_categories, analyses_categories

def anonymize_fullname(fullname):
    name = fullname.split()[2] 
    if name.endswith('а'):
        return 'Ж'
    else:
        return 'М'
    
def anonymize_passport(passport):
    return '*' * 4 + ' ' + '*' * 6

def anonymize_snils(snils):
    return '*' * 3 + '-' + '*' * 3 + '-' + '*' * 3 + ' ' + '*' * 2

def anonymize_symptoms(symptoms):
    symptom_list = symptoms.split(", ")
    
    has_internal = False
    has_external = False
    
    for symptom in symptom_list:
        if symptom in symptoms_categories['Внутренние']:
            has_internal = True
        elif symptom in symptoms_categories['Внешние']:
            has_external = True
    
    if has_internal and has_external:
        return "Смешанные"
    elif has_internal:
        return "Внутренние"
    elif has_external:
        return "Внешние"
    else:
        return "Неизвестные"

def anonymize_doctor(doctor):
    for department, doctors in doctors_departments.items():
        if doctor in doctors:
            return department
        
    return 'X'

def anonymize_analyses(analyses):
    analyses = [analysis.strip() for analysis in analyses.split(',')]
    first_analysis = analyses[0] 
    for analyses_category, analyses_list in analyses_categories.items():
        if first_analysis in analyses_list:
            return analyses_category
    
    return 'Неизвестный'

def anonymize_cost(cost):
    cost_value = int(cost.split()[0])
    if cost_value <= 3500:
        return '0-3500 руб.'
    elif 3501 < cost_value <= 7000:
        return '3501-7000 руб.'
    else:
        return '7001+ руб.'

def anonymize_card(card):
    return card.split(', ')[1]
