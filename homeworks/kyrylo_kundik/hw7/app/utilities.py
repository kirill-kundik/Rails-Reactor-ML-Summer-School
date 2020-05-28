import logging

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from app.config import PROJECT_ROOT


def load_env():
    env_path = PROJECT_ROOT / '.env'
    load_dotenv(dotenv_path=env_path)


def set_up_logging(log_file: str, verbose: bool):
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        filename=log_file,
                        filemode='a')
    if verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)

    logging.info('ARGS PARSED, LOGGING CONFIGURED.')


all_features = ['street_name', 'city_name', 'total_square_meters', 'living_square_meters', 'kitchen_square_meters',
                'rooms_count', 'floor', 'wall_type', 'inspected', 'construction_year', 'heating', 'seller', 'water',
                'building_condition', 'floors_count']

num_features = ['total_square_meters', 'living_square_meters', 'kitchen_square_meters', 'rooms_count', 'floor',
                'construction_year', 'floors_count', 'inspected']

cat_features = ['street_name', 'city_name', 'wall_type', 'heating', 'seller', 'water', 'building_condition']

all_columns = ['Unnamed: 0', 'Unnamed: 0.1', 'total_square_meters', 'living_square_meters', 'kitchen_square_meters',
               'rooms_count',
               'floor', 'inspected', 'construction_year', 'floors_count', 'street_name_other',
               'street_name_бульвар', 'street_name_дорога', 'street_name_майдан', 'street_name_переулок',
               'street_name_плато', 'street_name_проезд', 'street_name_проспект', 'street_name_улица',
               'street_name_шоссе', 'city_name_Berezan', 'city_name_Burshtyn', 'city_name_Gnivan',
               'city_name_Kelmentsi', 'city_name_Noviy Rozdil', 'city_name_rzishev', 'city_name_Івано-Франківськ',
               'city_name_Авдєєвка', 'city_name_Акимовка', 'city_name_Александрия', 'city_name_Александровка',
               'city_name_Алупка', 'city_name_Алушта', 'city_name_Алчевск', 'city_name_Ананьев', 'city_name_Андрушевка',
               'city_name_Апостоловo', 'city_name_Артемовск', 'city_name_Арциз', 'city_name_Ахтырка',
               'city_name_Балаклея', 'city_name_Балта', 'city_name_Бар', 'city_name_Барановка', 'city_name_Барвенково',
               'city_name_Барышевка', 'city_name_Бахмач', 'city_name_Баштанка', 'city_name_Белая Церковь',
               'city_name_Белгород-Днестровский', 'city_name_Белогорье', 'city_name_Белозерка', 'city_name_Белополье',
               'city_name_Беляевка', 'city_name_Бердичев', 'city_name_Бердянск', 'city_name_Берегово',
               'city_name_Бережаны', 'city_name_Березанка', 'city_name_Березань', 'city_name_Березно',
               'city_name_Березовка', 'city_name_Берислав', 'city_name_Бершадь', 'city_name_Близнюки',
               'city_name_Бобровица', 'city_name_Богодухов', 'city_name_Богородчаны', 'city_name_Богуслав',
               'city_name_Болград', 'city_name_Борислав', 'city_name_Борисполь', 'city_name_Боровая',
               'city_name_Бородянка', 'city_name_Борщев', 'city_name_Бровары', 'city_name_Броды', 'city_name_Брянка',
               'city_name_Буковель', 'city_name_Бурштын', 'city_name_Бурынь', 'city_name_Буск', 'city_name_Буча',
               'city_name_Бучач', 'city_name_Біла Церква', 'city_name_Валки', 'city_name_Васильевка',
               'city_name_Васильков', 'city_name_Ватутино', 'city_name_Великая Александровка',
               'city_name_Великая Михайловка', 'city_name_Великая Писаревка', 'city_name_Великий Березный',
               'city_name_Верхнеднепровск', 'city_name_Веселиново', 'city_name_Вижница', 'city_name_Винница',
               'city_name_Виноградов', 'city_name_Виньковцы', 'city_name_Владимир-Волынский', 'city_name_Владимирец',
               'city_name_Вознесенск', 'city_name_Волноваха', 'city_name_Воловец', 'city_name_Володарск-Волынский',
               'city_name_Волочиск', 'city_name_Волчанск', 'city_name_Вольногорск', 'city_name_Вольнянск',
               'city_name_Вышгород', 'city_name_Вінниця', 'city_name_Гадяч', 'city_name_Гайворон', 'city_name_Гайсин',
               'city_name_Галич', 'city_name_Геническ', 'city_name_Герца', 'city_name_Глобино', 'city_name_Глубокая',
               'city_name_Глухов', 'city_name_Гнивань', 'city_name_Голая Пристань', 'city_name_Горловка',
               'city_name_Горностаевка', 'city_name_Городенка', 'city_name_Городище', 'city_name_Городня',
               'city_name_Городок', 'city_name_Горохов', 'city_name_Гоща', 'city_name_Гребенка', 'city_name_Гуляйполe',
               'city_name_Гурзуф', 'city_name_Гусятин', 'city_name_Двуреченское', 'city_name_Дебальцево',
               'city_name_Деражня', 'city_name_Дергачи', 'city_name_Джанкой', 'city_name_Дзержинск',
               'city_name_Диканька', 'city_name_Димитров', 'city_name_Днепродзержинск', 'city_name_Днепропетровск',
               'city_name_Добровеличковка', 'city_name_Доброполье', 'city_name_Докучаевск', 'city_name_Долина',
               'city_name_Долинская', 'city_name_Доманевка', 'city_name_Донецк', 'city_name_Драбов',
               'city_name_Дрогобич', 'city_name_Дрогобыч', 'city_name_Дружковка', 'city_name_Дубляны',
               'city_name_Дубно', 'city_name_Дубровица', 'city_name_Дунаевцы', 'city_name_Евпатория',
               'city_name_Еланец', 'city_name_Емильчино', 'city_name_Жашков', 'city_name_Железный Порт',
               'city_name_Желтые Воды', 'city_name_Жидачов', 'city_name_Жидачов\t', 'city_name_Житомир',
               'city_name_Жмеринка', 'city_name_Жовтневый район', 'city_name_Жолква', 'city_name_Залещики',
               'city_name_Залізний Порт', 'city_name_Запорожье', 'city_name_Заставна', 'city_name_Затока',
               'city_name_Зачепиловка', 'city_name_Збараж', 'city_name_Зборов', 'city_name_Звенигородка',
               'city_name_Здолбунов', 'city_name_Зеньков', 'city_name_Змиев', 'city_name_Знаменка',
               'city_name_Золотоноша', 'city_name_Золочев', 'city_name_Иванков', 'city_name_Ивано-Франковск',
               'city_name_Ивановка', 'city_name_Измаил', 'city_name_Изюм', 'city_name_Изяслав', 'city_name_Ильинцы',
               'city_name_Ильичевск', 'city_name_Ирпень', 'city_name_Иршавa', 'city_name_Ичня', 'city_name_Кагарлык',
               'city_name_Казатин', 'city_name_Каланчак', 'city_name_Калиновка', 'city_name_Калуш',
               'city_name_Каменец-Подольский', 'city_name_Каменка', 'city_name_Каменка-Бугская',
               'city_name_Каменка-Днепровская', 'city_name_Камень-Каширский', 'city_name_Канев', 'city_name_Карловка',
               'city_name_Каховка', 'city_name_Кегичевка', 'city_name_Кельменцы', 'city_name_Киверцы', 'city_name_Киев',
               'city_name_Киево-Святошинский', 'city_name_Килия', 'city_name_Кировоград', 'city_name_Кицмань',
               'city_name_Київ', 'city_name_Кобеляки', 'city_name_Ковель', 'city_name_Козелец', 'city_name_Козова',
               'city_name_Коломия', 'city_name_Коломыя', 'city_name_Коминтерновское', 'city_name_Комсомольск',
               'city_name_Конотоп', 'city_name_Константиновка', 'city_name_Коростень', 'city_name_Коростышев',
               'city_name_Корсунь-Шевченковский', 'city_name_Корюковка', 'city_name_Косов', 'city_name_Костополь',
               'city_name_Котовск', 'city_name_Краматорск', 'city_name_Красилов', 'city_name_Красноармейск',
               'city_name_Красноград', 'city_name_Краснодон', 'city_name_Красноперекопск', 'city_name_Красный Лиман',
               'city_name_Красный Луч', 'city_name_Кременец', 'city_name_Кременная', 'city_name_Кременчуг',
               'city_name_Кривой Рог', 'city_name_Кролевец', 'city_name_Крыжополь', 'city_name_Кузнецовск',
               'city_name_Куйбышево', 'city_name_Купянск', 'city_name_Кіровоград', 'city_name_Ладыжин',
               'city_name_Лановцы', 'city_name_Лебедин', 'city_name_Ленино', 'city_name_Летичев', 'city_name_Липовец',
               'city_name_Лисичанск', 'city_name_Литин', 'city_name_Лозовая', 'city_name_Лохвица', 'city_name_Лубны',
               'city_name_Луганск', 'city_name_Лугины', 'city_name_Луцк', 'city_name_Лысянка', 'city_name_Львов',
               'city_name_Львів', 'city_name_Любомль', 'city_name_Макаров', 'city_name_Макеевка', 'city_name_Малин',
               'city_name_Маневичи', 'city_name_Маньковка', 'city_name_Марганец', 'city_name_Мариуполь',
               'city_name_Марковка', 'city_name_Марьинка', 'city_name_Межгорье', 'city_name_Мелитополь',
               'city_name_Меловое', 'city_name_Мена', 'city_name_Миколаїв', 'city_name_Миргород', 'city_name_Мироновка',
               'city_name_Мисхор', 'city_name_Михайловка', 'city_name_Млынов', 'city_name_Могилев-Подольский',
               'city_name_Монастыриска', 'city_name_Монастырище', 'city_name_Моршин', 'city_name_Мостиска',
               'city_name_Мукачево', 'city_name_Надворная', 'city_name_Нежин', 'city_name_Немиров', 'city_name_Нетешин',
               'city_name_Николаев', 'city_name_Никополь', 'city_name_Новая Водолага', 'city_name_Новая Каховка',
               'city_name_Новая Одесса', 'city_name_Новгород-Северский', 'city_name_Новгородкa',
               'city_name_Новий Роздол', 'city_name_Нововолынск', 'city_name_Нововоронцовка',
               'city_name_Новоград-Волынский', 'city_name_Новогродовка', 'city_name_Новоднестровск',
               'city_name_Новомосковск', 'city_name_Новониколаевка', 'city_name_Новоселица', 'city_name_Новотроицкое',
               'city_name_Новоукраинка', 'city_name_Новые Санжары', 'city_name_Носовка', 'city_name_Обухов',
               'city_name_Овидиополь', 'city_name_Овруч', 'city_name_Одесса', 'city_name_Оратов',
               'city_name_Орджоникидзе', 'city_name_Оржица', 'city_name_Острог', 'city_name_Очаков',
               'city_name_Павлоград', 'city_name_Первомайск', 'city_name_Первомайский', 'city_name_Перемышляны',
               'city_name_Перечин', 'city_name_Переяслав-Хмельницкий', 'city_name_Першотравенск', 'city_name_Печенеги',
               'city_name_Пирятин', 'city_name_Погребище', 'city_name_Подволочиск', 'city_name_Пологи',
               'city_name_Полонное', 'city_name_Полтава', 'city_name_Попасная', 'city_name_Попельня',
               'city_name_Прилуки', 'city_name_Приморск', 'city_name_Пустомыты', 'city_name_Путивль',
               'city_name_Радехов', 'city_name_Радивилов', 'city_name_Радомышль', 'city_name_Раздельная',
               'city_name_Ракитное', 'city_name_Рахов', 'city_name_Рени', 'city_name_Решетиловка', 'city_name_Ржищев',
               'city_name_Ровеньки', 'city_name_Ровно', 'city_name_Рожище', 'city_name_Рожнятов', 'city_name_Романов',
               'city_name_Ромны', 'city_name_Рубежное', 'city_name_Саки', 'city_name_Самбор', 'city_name_Сарата',
               'city_name_Сарны', 'city_name_Сахновщина', 'city_name_Свалява', 'city_name_Сватовo',
               'city_name_Свердловск', 'city_name_Светловодск', 'city_name_Севастополь', 'city_name_Северодонецк',
               'city_name_Селидово', 'city_name_Семеновка', 'city_name_Середина-Буда', 'city_name_Симферополь',
               'city_name_Синельниково', 'city_name_Скадовск', 'city_name_Сквирa', 'city_name_Сколе',
               'city_name_Славута', 'city_name_Славутич', 'city_name_Славянск', 'city_name_Смела',
               'city_name_Снигиревка', 'city_name_Снятин', 'city_name_Сокаль', 'city_name_Соленое', 'city_name_Сосница',
               'city_name_Сосновка', 'city_name_Ставище', 'city_name_Старая Выжевка', 'city_name_Старобельск',
               'city_name_Староконстантинов', 'city_name_Старый Самбор', 'city_name_Сторожинец', 'city_name_Стрый',
               'city_name_Судак', 'city_name_Сумы', 'city_name_Тальное', 'city_name_Тараща', 'city_name_Татарбунари',
               'city_name_Теофиполь', 'city_name_Теплик', 'city_name_Теплодар', 'city_name_Теребовля',
               'city_name_Терновка', 'city_name_Тернополь', 'city_name_Тетиев', 'city_name_Токмак',
               'city_name_Томаковка', 'city_name_Томашполь', 'city_name_Торез', 'city_name_Тростянец',
               'city_name_Трускавец', 'city_name_Тульчин', 'city_name_Турийск', 'city_name_Турка', 'city_name_Тывров',
               'city_name_Тысменица', 'city_name_Тячев', 'city_name_Угледар', 'city_name_Ужгород',
               'city_name_Ульяновка', 'city_name_Умань', 'city_name_Фастов', 'city_name_Феодосия', 'city_name_Феодосія',
               'city_name_Харцызск', 'city_name_Харьков', 'city_name_Херсон', 'city_name_Хмельник',
               'city_name_Хмельницкий', 'city_name_Хмельницький', 'city_name_Хорол', 'city_name_Хотин',
               'city_name_Христиновка', 'city_name_Хуст', 'city_name_Царичанка', 'city_name_Цюрупинск',
               'city_name_Чаплинка', 'city_name_Чемеровцы', 'city_name_Червоноармейск', 'city_name_Червоноград',
               'city_name_Черкассы', 'city_name_Чернигов', 'city_name_Чернобай', 'city_name_Черновцы',
               'city_name_Чернухи', 'city_name_Черняхов', 'city_name_Чертков', 'city_name_Чечельник',
               'city_name_Чигирин', 'city_name_Чоп', 'city_name_Чугуев', 'city_name_Чуднов', 'city_name_Чутово',
               'city_name_Шаргород', 'city_name_Шахтарск', 'city_name_Шацк', 'city_name_Шевченково',
               'city_name_Шепетовка', 'city_name_Широкое', 'city_name_Шишаки', 'city_name_Шостка', 'city_name_Шумское',
               'city_name_Щорс', 'city_name_Энергодар', 'city_name_Южноукраинск', 'city_name_Южный',
               'city_name_Юрьевка', 'city_name_Яворов', 'city_name_Яготин', 'city_name_Ялта', 'city_name_Ямполь',
               'city_name_Яремча', 'city_name_Ярмолинцы', 'city_name_Ясиноватая', 'wall_type_other', 'wall_type_СИП',
               'wall_type_армированная 3D Панель', 'wall_type_армированный железобетон', 'wall_type_бескаркасная',
               'wall_type_блочно-кирпичный', 'wall_type_бутовый камень', 'wall_type_газобетон', 'wall_type_газоблок',
               'wall_type_дерево и кирпич', 'wall_type_железобетон', 'wall_type_инкерманский камень',
               'wall_type_каркасно-каменная', 'wall_type_керамзитобетон', 'wall_type_керамический блок',
               'wall_type_керамический кирпич', 'wall_type_кирпич', 'wall_type_монолит', 'wall_type_монолитно-блочный',
               'wall_type_монолитно-каркасный', 'wall_type_монолитно-кирпичный', 'wall_type_монолитный железобетон',
               'wall_type_облицовочный кирпич', 'wall_type_панель', 'wall_type_пеноблок',
               'wall_type_ракушечник (ракушняк)', 'wall_type_сборно-монолитная', 'wall_type_сборный железобетон',
               'wall_type_силикатный кирпич', 'heating_other', 'heating_без отопления', 'heating_индивидуальное',
               'heating_централизованное', 'seller_other', 'seller_от застройщика', 'seller_от посредника',
               'seller_от представителя застройщика', 'seller_от представителя хозяина (без комиссионных)',
               'seller_от собственника', 'water_other', 'water_автоматическая с колодца', 'water_колодец',
               'water_колодец • автоматическая с колодца', 'water_скважина', 'water_скважина • колодец',
               'water_централизованное (водопровод)', 'water_централизованное (водопровод) • колодец',
               'water_централизованное (водопровод) • скважина', 'building_condition_other',
               'building_condition_нормальное', 'building_condition_отличное', 'building_condition_требует',
               'building_condition_удовлетворительное', 'building_condition_хорошее']


def preprocess_row(params):
    data = pd.DataFrame(np.zeros((1, len(all_columns))), columns=all_columns)

    for k in params:
        if k in num_features:
            data.at[0, k] = params[k]
        elif k in cat_features:
            if k + '_' + params[k] in all_columns:
                data.at[0, k + '_' + params[k]] = 1

    return data
