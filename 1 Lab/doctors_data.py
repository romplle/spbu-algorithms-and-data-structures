doctors_data = {
    "Терапевт": ([
        "Головная боль", "Повышенная температура", "Слабость", 
        "Кашель", "Заложенность носа", "Боль в горле", 
        "Тошнота", "Одышка", "Боль в животе", "Головокружение"
    ], [
        "Общий анализ крови", "Биохимический анализ крови", 
        "Анализ мочи", "Рентген грудной клетки", "ЭКГ"
    ]),
    "Педиатр": ([
        "Сыпь", "Плач без причины", "Снижение аппетита", 
        "Повышенная температура", "Кашель", "Нарушение сна", 
        "Диарея", "Боль в ухе", "Рвота", "Конъюнктивит"
    ], [
        "Общий анализ крови", "Анализ кала", 
        "УЗИ брюшной полости", "Общий анализ мочи", "Анализ на паразитов"
    ]),
    "Хирург": ([
        "Острая боль в животе", "Травмы", "Раны", 
        "Кровотечение", "Абсцесс", "Опухоли", 
        "Боль в суставах", "Грыжа", "Переломы", "Ожоги"
    ], [
        "УЗИ", "МРТ", "Рентген", "Компьютерная томография", "Анализ крови на свертываемость"
    ]),
    "Невролог": ([
        "Головокружение", "Онемение конечностей", "Тремор", 
        "Головная боль", "Шум в ушах", "Проблемы с координацией", 
        "Судороги", "Потеря сознания", "Нарушение речи", "Слабость в конечностях"
    ], [
        "МРТ головного мозга", "ЭЭГ", "ЭНМГ", "Общий анализ крови", "УЗИ сосудов шеи"
    ]),
    "Кардиолог": ([
        "Боль в груди", "Одышка", "Сердцебиение", 
        "Головокружение", "Слабость", "Отек ног", 
        "Повышенное давление", "Тахикардия", "Потливость", "Потеря сознания"
    ], [
        "ЭКГ", "ЭхоКГ", "Мониторирование Холтера", "Стресс-тест", "Анализ на липиды"
    ]),
    "Гастроэнтеролог": ([
        "Тошнота", "Боль в животе", "Изжога", 
        "Диарея", "Запор", "Вздутие", 
        "Горечь во рту", "Отрыжка", "Рвота", "Снижение аппетита"
    ], [
        "Гастроскопия", "Анализ кала на скрытую кровь", 
        "УЗИ брюшной полости", "Анализ на хеликобактер", "Биохимический анализ крови"
    ]),
    "Офтальмолог": ([
        "Зуд в глазах", "Покраснение глаз", "Сухость глаз", 
        "Боль в глазах", "Снижение зрения", "Светобоязнь", 
        "Слезотечение", "Появление пятен перед глазами", "Двоение в глазах", "Чувство инородного тела"
    ], [
        "Офтальмоскопия", "Тонометрия", "Периметрия", "УЗИ глазного яблока", "КТ орбиты"
    ]),
    "Отоларинголог": ([
        "Боль в ухе", "Потеря слуха", "Заложенность носа", 
        "Боль в горле", "Шум в ушах", "Снижение обоняния", 
        "Храп", "Частые ангины", "Гнойные выделения из уха", "Нарушение равновесия"
    ], [
        "Аудиометрия", "Риноскопия", "Ларингоскопия", "МРТ пазух носа", "Тимпанометрия"
    ]),
    "Дерматолог": ([
        "Сыпь", "Зуд", "Покраснение кожи", 
        "Высыпания", "Трещины на коже", "Прыщи", 
        "Псориаз", "Перхоть", "Пигментные пятна", "Грибковые поражения"
    ], [
        "Дерматоскопия", "Анализ на аллергены", "Биопсия кожи", "Микроскопическое исследование кожи", "Анализ крови на антитела"
    ]),
    "Гинеколог": ([
        "Боль внизу живота", "Нарушение менструального цикла", "Выделения из влагалища", 
        "Боль при мочеиспускании", "Зуд", "Болезненные менструации", 
        "Кровотечения", "Боль при половом акте", "Бесплодие", "Опухоли"
    ], [
        "УЗИ органов малого таза", "Цитология", "Кольпоскопия", "Анализ на ВПЧ", "Гормональный профиль"
    ]),
    "Уролог": ([
        "Боль при мочеиспускании", "Частые мочеиспускания", "Кровь в моче", 
        "Боль внизу живота", "Задержка мочи", "Ночное мочеиспускание", 
        "Боль в паху", "Снижение либидо", "Эректильная дисфункция", "Нарушения мочеиспускания"
    ], [
        "Анализ мочи", "УЗИ почек", "Урофлоуметрия", "ПСА", "Цистоскопия"
    ]),
    "Эндокринолог": ([
        "Повышенная жажда", "Сухость во рту", "Изменение веса", 
        "Усталость", "Потоотделение", "Частое мочеиспускание", 
        "Тремор", "Головокружение", "Проблемы с памятью", "Снижение либидо"
    ], [
        "Анализ на гормоны щитовидной железы", "Глюкозотолерантный тест", "Анализ крови на сахар", "УЗИ щитовидной железы", "Анализ мочи на сахар"
    ]),
    "Психиатр": ([
        "Депрессия", "Тревожность", "Бессонница", 
        "Галлюцинации", "Апатия", "Нарушение концентрации", 
        "Маниакальное поведение", "Панические атаки", "Социальная изоляция", "Фобии"
    ], [
        "Психологическое тестирование", "ЭЭГ", "МРТ головного мозга", "Нейропсихологическое обследование", "Анализ на уровень гормонов"
    ]),
    "Инфекционист": ([
        "Лихорадка", "Сыпь", "Боль в горле", 
        "Тошнота", "Рвота", "Диарея", 
        "Повышенная температура", "Одышка", "Кашель", "Покраснение глаз"
    ], [
        "Анализ на антитела", "ПЦР-анализ", "Бакпосев", "Анализ мочи", "Анализ крови на инфекции"
    ]),
    "Стоматолог": ([
        "Боль в зубе", "Кровоточивость десен", "Зубной налет", 
        "Чувствительность зубов", "Опухание десен", "Зубной камень", 
        "Плохой запах изо рта", "Кариес", "Пародонтоз", "Трещины зубов"
    ], [
        "Рентген зубов", "Ортопантомограмма", "Мазок на микрофлору", "КТ челюсти", "Анализ слюны"
    ]),
    "Онколог": ([
        "Необъяснимая потеря веса", "Опухоли", "Боль в костях", 
        "Увеличенные лимфоузлы", "Кровотечения", "Хроническая усталость", 
        "Лихорадка", "Кожные изменения", "Длительный кашель", "Постоянные боли"
    ], [
        "Биопсия", "КТ", "МРТ", "ПЭТ", "Анализ на онкомаркеры"
    ]),
    "Фтизиатр": ([
        "Кашель с кровью", "Потеря веса", "Лихорадка", 
        "Потливость ночью", "Одышка", "Боль в груди", 
        "Хронический кашель", "Слабость", "Увеличенные лимфоузлы", "Усталость"
    ], [
        "Анализ мокроты", "Рентген грудной клетки", "Туберкулиновая проба", "МРТ легких", "КТ легких"
    ]),
    "Ревматолог": ([
        "Боль в суставах", "Отеки суставов", "Утренняя скованность", 
        "Ограничение движений", "Слабость в конечностях", 
        "Повышенная температура", "Усталость", "Остеопороз", 
        "Деформация суставов", "Высыпания"
    ], [
        "Общий анализ крови", "Ревматоидный фактор", 
        "Анализ мочевой кислоты", "МРТ суставов", "Денситометрия"
    ]),
    "Аллерголог": ([
        "Сыпь", "Зуд", "Насморк", 
        "Чихание", "Кашель", "Одышка", 
        "Слезотечение", "Заложенность носа", "Покраснение глаз", "Отеки"
    ], [
        "Кожные пробы", "Анализ на иммуноглобулин E", 
        "Провокационные тесты", "Общий анализ крови", "Тест на аллерген-специфические антитела"
    ]),
    "Пульмонолог": ([
        "Кашель", "Одышка", "Боль в груди", 
        "Хронический кашель", "Кровохарканье", 
        "Частые простуды", "Затрудненное дыхание", "Хрипы", "Одышка при физической нагрузке", "Усталость"
    ], [
        "Спирометрия", "КТ грудной клетки", "МРТ легких", "Анализ мокроты", "Рентген грудной клетки"
    ]),
    "Гематолог": ([
        "Анемия", "Кровотечения", "Бледность", 
        "Увеличенные лимфоузлы", "Слабость", 
        "Повышенная температура", "Частые инфекции", "Головокружение", "Боль в костях", "Гематомы"
    ], [
        "Общий анализ крови", "Биохимический анализ крови", 
        "Анализ на коагулограмму", "Биопсия костного мозга", "Ферритин"
    ]),
    "Иммунолог": ([
        "Частые инфекции", "Увеличенные лимфоузлы", "Слабость", 
        "Повышенная температура", "Сыпь", "Аллергические реакции", 
        "Боль в суставах", "Хроническая усталость", "Отек слизистых", "Повышенная чувствительность к инфекциям"
    ], [
        "Иммунограмма", "Анализ на иммуноглобулины", 
        "Анализ на антитела", "Общий анализ крови", "ПЦР-анализ"
    ]),
    "Флеболог": ([
        "Отеки ног", "Боль в ногах", "Усталость ног", 
        "Варикозное расширение вен", "Тяжесть в ногах", 
        "Судороги", "Зуд на ногах", "Сосудистые звездочки", "Изменение цвета кожи на ногах", "Тромбы"
    ], [
        "УЗИ вен нижних конечностей", "Допплеровское исследование", 
        "Анализ на свертываемость крови", "Анализ на D-димер", "Флебография"
    ]),
    "Травматолог": ([
        "Травмы", "Боль в костях", "Переломы", 
        "Вывихи", "Гематомы", "Отек", 
        "Невозможность двигаться", "Боль при движении", "Деформации конечностей", "Открытые раны"
    ], [
        "Рентген", "МРТ", "КТ", 
        "УЗИ суставов", "Анализ крови на инфекцию"
    ]),
    "Нефролог": ([
        "Боль в пояснице", "Частое мочеиспускание", "Боль при мочеиспускании", 
        "Отек ног", "Кровь в моче", "Изменение цвета мочи", 
        "Высокое давление", "Слабость", "Повышенная температура", "Ночное мочеиспускание"
    ], [
        "Анализ мочи", "УЗИ почек", 
        "Биохимический анализ крови", "КТ почек", "Анализ на креатинин"
    ]),
    "Ангиохирург": ([
        "Тромбы", "Отеки ног", "Варикозное расширение вен", 
        "Боль в ногах при ходьбе", "Холодные конечности", 
        "Судороги", "Изменение цвета кожи", "Трофические язвы", "Потеря чувствительности", "Отсутствие пульса на ногах"
    ], [
        "Ангиография", "Допплеровское исследование сосудов", 
        "МРТ сосудов", "КТ сосудов", "Анализ на коагулограмму"
    ]),
    "Гепатолог": ([
        "Боль в правом подреберье", "Желтуха", "Тошнота", 
        "Потеря аппетита", "Темная моча", "Светлый стул", 
        "Повышенная температура", "Слабость", "Зуд кожи", "Вздутие живота"
    ], [
        "УЗИ печени", "Биохимический анализ крови", 
        "Анализ на вирусы гепатита", "Фиброскан", "Анализ на билирубин"
    ]),
    "Диетолог": ([
        "Избыточный вес", "Недостаток веса", "Слабость", 
        "Нарушение пищеварения", "Ожирение", "Диарея", 
        "Запор", "Повышенный аппетит", "Отсутствие аппетита", "Тошнота"
    ], [
        "Анализ на уровень сахара в крови", "Анализ на холестерин", 
        "Тест на пищевую непереносимость", "Гормональный анализ", "Биохимический анализ крови"
    ]),
    "Токсиколог": ([
        "Тошнота", "Рвота", "Головная боль", 
        "Головокружение", "Спутанность сознания", "Одышка", 
        "Судороги", "Кожные высыпания", "Боль в животе", "Обморок"
    ], [
        "Анализ крови на токсины", "Анализ мочи на токсины", 
        "ЭКГ", "УЗИ органов брюшной полости", "Биохимический анализ крови"
    ]),
    "Спортивный врач": ([
        "Травмы", "Боль в мышцах", "Судороги", 
        "Боль в суставах", "Утомляемость", "Растяжения", 
        "Травмы связок", "Гематомы", "Боль при физической нагрузке", "Вывихи"
    ], [
        "МРТ", "УЗИ суставов", "КТ", 
        "ЭКГ при физической нагрузке", "Анализ на электролиты"
    ]),
    "Мануальный терапевт": ([
        "Боль в спине", "Боль в шее", "Ограничение движений", 
        "Невралгия", "Сколиоз", "Грыжа диска", 
        "Остеохондроз", "Головная боль", "Онемение конечностей", "Проблемы с осанкой"
    ], [
        "МРТ позвоночника", "Рентген позвоночника", 
        "УЗИ мягких тканей", "Общий анализ крови", "Электромиография"
    ]),
    "Сексолог": ([
        "Эректильная дисфункция", "Проблемы с либидо", "Проблемы с оргазмом", 
        "Преждевременная эякуляция", "Боль во время полового акта", 
        "Психологические проблемы в сексуальной жизни", "Асексуальность", 
        "Отсутствие сексуального удовлетворения", "Проблемы в паре", "Фригидность"
    ], [
        "Гормональный профиль", "Анализ на тестостерон", 
        "Психологическое тестирование", "Анализ на инфекции", "Консультация психолога"
    ]),
    "Проктолог": ([
        "Боль в области ануса", "Кровотечение из ануса", "Запор", 
        "Геморрой", "Трещины", "Зуд в анальной области", 
        "Пролапс прямой кишки", "Ощущение инородного тела", "Гнойные выделения", "Нарушение стула"
    ], [
        "Колоноскопия", "Ректороманоскопия", 
        "Анализ кала", "Анализ крови на инфекции", "Биопсия"
    ]),
    "Физиотерапевт": ([
        "Боль в мышцах", "Боль в суставах", "Восстановление после травм", 
        "Слабость в конечностях", "Ограничение подвижности", 
        "Растяжения", "Остеоартрит", "Реабилитация после операций", "Паралич", "Спазмы мышц"
    ], [
        "Электротерапия", "Магнитотерапия", 
        "Ультразвуковая терапия", "Лазеротерапия", "Лечебная гимнастика"
    ]),
    "Педиатр-невролог": ([
        "Головная боль", "Онемение", "Судороги", 
        "Задержка психомоторного развития", "Тремор", 
        "Повышенная возбудимость", "Головокружение", "Тики", "Потеря сознания", "Проблемы с координацией"
    ], [
        "МРТ головного мозга", "ЭЭГ", 
        "ЭхоЭГ", "Общий анализ крови", "УЗИ сосудов головного мозга"
    ]),
    "Гериатр": ([
        "Проблемы с памятью", "Утомляемость", "Потеря ориентации", 
        "Головокружение", "Снижение слуха", "Трудности с передвижением", 
        "Нарушение сна", "Потеря веса", "Одышка", "Слабость"
    ], [
        "Общий анализ крови", "Кардиограмма", 
        "УЗИ сердца", "Тест на деменцию", "Биохимический анализ крови"
    ]),
    "Трихолог": ([
        "Выпадение волос", "Сухость кожи головы", "Зуд кожи головы", 
        "Ломкость волос", "Облысение", "Себорея", 
        "Перхоть", "Покраснение кожи головы", "Жирность волос", "Шелушение кожи головы"
    ], [
        "Трихограмма", "Анализ крови на витамины", 
        "Гормональный анализ", "Анализ на минеральный баланс", "Микроскопия волос"
    ]),
    "Анестезиолог": ([
        "Подготовка к операции", "Анализ переносимости анестезии", 
        "Контроль за состоянием пациента во время операции", 
        "Головокружение после операции", "Слабость", "Боль после операции", 
        "Нарушение дыхания", "Проблемы с сердечным ритмом", "Аллергическая реакция на анестезию", "Тошнота"
    ], [
        "ЭКГ", "Биохимический анализ крови", 
        "Общий анализ крови", "Анализ на свертываемость", "Спирометрия"
    ]),
    "Диабетолог": ([
        "Частое мочеиспускание", "Жажда", "Потеря веса", 
        "Повышенная утомляемость", "Головокружение", 
        "Зуд кожи", "Проблемы с заживлением ран", "Одышка", "Судороги", "Зрение"
    ], [
        "Анализ на глюкозу", "Глюкозотолерантный тест", 
        "Анализ на гликированный гемоглобин", "УЗИ почек", "ЭКГ"
    ]),
    "Реабилитолог": ([
        "Ограниченная подвижность", "Мышечная слабость", "Боль в суставах", 
        "Последствия травм", "Парез", "Проблемы с координацией", 
        "Паралич", "Снижение чувствительности", "Онемение", "Восстановление после операций"
    ], [
        "Физическая терапия", "Массаж", 
        "Электростимуляция", "Миостимуляция", "Магнитотерапия"
    ]),
    "Нарколог": ([
        "Зависимость от алкоголя", "Зависимость от наркотиков", "Абстинентный синдром", 
        "Нарушение сна", "Тремор", "Депрессия", 
        "Головная боль", "Тошнота", "Агрессивное поведение", "Потеря аппетита"
    ], [
        "Тест на наркотики", "Анализ крови на токсины", 
        "Психологическое тестирование", "Анализ на алкоголь", "Анализ мочи"
    ]),
    "Миколог": ([
        "Грибковые поражения кожи", "Зуд", "Трещины на коже", 
        "Грибковые инфекции ногтей", "Шелушение кожи", "Покраснение", 
        "Высыпания", "Боль в ногтях", "Изменение цвета ногтей", "Прыщи"
    ], [
        "Микроскопия кожи", "Анализ на грибки", 
        "Посев на микрофлору", "Биопсия кожи", "Анализ крови на инфекции"
    ]),
    "Ангиолог": ([
        "Отеки ног", "Боль в ногах", "Варикозное расширение вен", 
        "Судороги", "Тромбоз", "Боль при ходьбе", 
        "Онемение конечностей", "Изменение цвета кожи на ногах", "Язвы на ногах", "Слабость в ногах"
    ], [
        "УЗИ сосудов", "Ангиография", 
        "Допплерография", "Анализ крови на свертываемость", "МРТ сосудов"
    ]),
    "Гомеопат": ([
        "Хроническая усталость", "Аллергические реакции", "Проблемы с пищеварением", 
        "Частые простуды", "Нарушение сна", "Головная боль", 
        "Нервозность", "Боли в суставах", "Гормональный дисбаланс", "Повышенная раздражительность"
    ], [
        "Гомеопатическое обследование", "Консультация по выбору препаратов", 
        "Иммунограмма", "Анализ крови", "Анализ на аллергены"
    ]),
    "Венеролог": ([
        "Высыпания в интимной зоне", "Боль при мочеиспускании", "Выделения", 
        "Зуд в половых органах", "Язвы на коже", "Жжение", 
        "Покраснение", "Боль при половом акте", "Прыщи", "Покраснение кожи"
    ], [
        "Анализ на ИППП", "Анализ крови на сифилис", 
        "Анализ мочи", "Мазок на микрофлору", "ПЦР-анализ"
    ]),
    "Сомнолог": ([
        "Хроническая усталость", "Бессонница", "Апноэ", 
        "Храп", "Частые пробуждения ночью", "Сонливость днем", 
        "Нарушение дыхания во сне", "Тревожные сны", "Ночной пот", "Головные боли по утрам"
    ], [
        "Полисомнография", "Оксигенометрия", 
        "Анализ на гормоны", "ЭЭГ", "Мониторинг сна"
    ]),
    "Логопед": ([
        "Нарушение речи", "Картавость", "Заикание", 
        "Задержка речевого развития", "Дислексия", "Трудности с произношением звуков", 
        "Трудности с письмом", "Нарушение слуха", "Афазия", "Дисграфия"
    ], [
        "Логопедическое обследование", "Консультация с психологом", 
        "Аудиометрия", "Речевые тесты", "Психологическое тестирование"
    ]),
    "Кинезиолог": ([
        "Нарушение осанки", "Сколиоз", "Боль в позвоночнике", 
        "Мышечная слабость", "Нарушение координации", "Головная боль", 
        "Боль в суставах", "Слабость конечностей", "Хроническая усталость", "Скованность движений"
    ], [
        "Тест на мышечную силу", "Оценка биомеханики тела", 
        "Кинезиологическая диагностика", "Физическая терапия", "Массаж"
    ]),
    "Остеопат": ([
        "Боль в спине", "Мышечные спазмы", "Головная боль", 
        "Боль в шее", "Боль в суставах", "Нарушение осанки", 
        "Ограниченная подвижность", "Головокружение", "Усталость", "Скованность движений"
    ], [
        "Остеопатическое обследование", "Мануальная терапия", 
        "Массаж", "Тест на гибкость", "Функциональная диагностика"
    ]),
    "Вертебролог": ([
        "Боль в позвоночнике", "Скованность в спине", "Головные боли", 
        "Ограниченная подвижность шеи", "Боль в пояснице", "Онемение конечностей", 
        "Боль в суставах", "Нарушение осанки", "Слабость в ногах", "Мышечные спазмы"
    ], [
        "МРТ позвоночника", "Рентген", 
        "ЭМГ", "УЗИ суставов", "Компьютерная томография"
    ])
}