NumPy(Numerical Python extensions)
import numpy as np
np.array()
    expenses = [0, 312, 4232, 0, 1958, 3062, 1454]
    purchases = [0, 1, 5, 0, 3, 5, 4]
    cashback = [0, 3, 138, 0, 4, 14, 12]
    transactions = np.array([expenses, purchases, cashback])
    print(transactions)
    print(type(transactions))

    [[   0  312 4232    0 1958 3062 1454]
    [   0    1    5    0    3    5    4]
    [   0    3  138    0    4   14   12]]
    <class 'numpy.ndarray'>
    
x.ravel()
    transactions.ravel()
    array([   0,  312, 4232,    0, 1958, 3062, 1454,    0,    1,    5,    0,    3,    5,    4,    0,    3,  138,    0,    4,   14,   12]) # в результате работы кода мы получили одномерный массив со списком всех значений
    
np.zeros(y, x) — создает массив указанной формы, заполненный нулями;
np.ones(y, x) — создает массив указанной формы, заполненный единицами.
    C = np.ones((2, 3))
    print(C)
    [[1. 1. 1.]
    [1. 1. 1.]]
np.eye()
    D = np.eye(4)
    print(D)

    [[1. 0. 0. 0.]
    [0. 1. 0. 0.]
    [0. 0. 1. 0.]
    [0. 0. 0. 1.]]
    
    D = np.eye(4, 5)
    print(D)
    [[1. 0. 0. 0. 0.]
    [0. 1. 0. 0. 0.]
    [0. 0. 1. 0. 0.]
    [0. 0. 0. 1. 0.]]
np.nan  #NaN
    x = np.nan
x[0][:5]    #к матрице можно обратиться по слайсу — диапазону индексов. 
    transactions = np.array([[0, 312, 4232, 0, 1958, 3062, 1454], [0, 1, 5, 0, 3, 5, 4], [0, 3, 138, 0, 4, 14, 12]])
    print(transactions[0][:5]) # конечное значение не включается в диапазон
    print(sum(transactions[0][:5]))
    [   0  312 4232    0 1958]
    6502
x.transpose()    
x.T
    print(transactions.T) # метод T предоставляет более удобный и краткий способ для транспонирования матрицы
np.arange(начало, конец, шаг, тип данных) #позволяет работать с данными типа float.

np.random.rand(a,b) #матрица a на b со значениями [0,1)
np.random.randint()
    A = np.random.randint(low = 1, high = 10, size = (4, 4))#в интервале [1, 10)
    print(A)
    [[9 3 4 8]
    [8 5 4 7]
    [4 1 9 8]
    [2 6 6 9]]
    
np.min(A) = 1  — минимальное значение
np.max(A) = 9 — максимальное значение
np.mean(A) = 5.8125 — среднее значение
np.median(A) = 6.0 — медиана
np.sum(A) = 93 — сумма
#Если мы передадим в качестве второго параметра 0, то функция будет агрегировать данные по по столбцам, а если 1 — по строкам.
np.min(A, 0) = array([2, 1, 4, 7]) — минимальные значения по столбцам
np.max(A, 0) = array([9, 6, 9, 9]) — максимальные значения по столбцам
np.mean(A, 0) = array([5.75, 3.75, 5.75, 8.  ]) — средние значения по столбцам
np.median(A, 0) = array([6., 4., 5., 8.]) — медианные значения по столбцам
np.sum(A, 0) = array([23, 15, 23, 32]) — сумма значений по столбцам

np.min(A, 1) = array([3, 4, 1, 2]) — минимальные значения по строкам
np.max(A, 1) = array([9, 8, 9, 9]) — максимальные значения по строкам
np.mean(A, 1) = array([6.  , 6.  , 5.5 , 5.75]) — средние значения по строкам
np.median(A, 1) = array([6., 6., 6., 6.]) — медианные значения по строкам
np.sum(A, 1) = array([24, 24, 22, 23]) — сумма значений по строкам

Pandas
import pandas as pd

example_list = [1, 'Hello', True] # создаем список                                              
example_series = pd.Series(example_list) # преобразуем список в series с помощью метода pd.Series()

# Чтобы посмотреть, что у нас получилось, используем print(example_series) или display(example_series)                  
print(example_series)         
0        1
1        Hello
2        True 

.iloc[2]
# Извлекаем элемент на позиции 2 (нумерация с нуля)
element = example_series.iloc[2] print(element)
True

pd.DataFrame
sales_list=[43,56,78,45] price_list=[340,500,299,100]
data_dict = {'sales':sales_list,'price':price_list}
df=pd.DataFrame(data_dict)
Или
df=pd.DataFrame(data={'sales':sales_list,'price':price_list})
Но в таком случае название словаря может быть только data, другие наименования не допускаются.

pd.read_csv
    df=pd.read_csv('2019.csv')
    # если бы у вас был Excel-файл, команда была бы очень похожей: df=pd.read_excel('2019.xlsx')

    df=pd.read_csv(‘C/Users/IvanPupkin/Downloads/archive/2019.csv ’)

# Сброс ограничений на количество выводимых рядов
pd.set_option('display.max_rows', None)

display(df.head()) # покажет первые 5 строк
display(df.tail(3)) # покажет последние 3 строки

display(df.shape)
display(df.shape[0]) display(df.shape[1])
(156, 9) # наша матрица имеет 156 строк и 9 столбцов 156 # мы можем по индексу обращаться к отдельному элементу размера: сейчас мы вывели число строк… 9 # … а сейчас число столбцов

display(df.info())
какие типы данные в ней лежат и есть ли пропущенные значения.

Чтобы вывести один столбец, укажем его название в квадратных скобках. Выведем только названия стран:
display(df[['Country or region']])

display(df[['Country or region']].head(3))
display(df[['Country or region']].tail(7))

display(df[['Country or region', 'Score']])

display(df['Country or region'][5])
display(df['Score'][100])

filtered_df = df[df['Country or region'] == 'Russia'] # после двойного равно указываем нужное значение

логическое ИЛИ |
    filtered3_df = df[(df['Country or region'] == 'Russia') | (df['Country or region'] == 'China')] 
    # обратите внимание, что каждое условие взято в круглые скобки  

isin()
    countries_to_select = ['Russia', 'China'] # создаем список из нужных нам критериев
    filtered3_df = df[df['Country or region'].isin(countries_to_select) 
    display(filtered3_df)# выводим только те строки, где есть элементы из списка

логическое И &
    filtered4_df = df[(df['Score'] >= 7) & (df['Social support'] <= 1.5)]
    display(filtered4_df)

df['Overall_rank_sum10']=df['Overall rank']+10 # мы создали новый столбец 'Overall_rank_sum10' и прибавили к каждой строке 10

df['GDP_Support']=df['GDP per capita']+df['Social support']

df['GDP_Support_round']=df['GDP_Support'].round(2)
display(df)

df.drop
df=df.drop(['Overall_rank_sum10','GDP_Support','GDP_Support_round'], axis=1) # перечислим через запятую все названия столбцов, которые мы хотим удалить и укажем, что мы хотим удалить элементы по вертикальной оси (axis = 1)
display(df)# проверим, что столбцы правда удалились

#по части стран сделать доп.столбцы
df=pd.read_csv('2018.csv')
co = ['Portugal','Spain', 'Italy','Greece']
df['Sc']=(7.632-df['Score']).round(1)
df['%GDP per capita']=(df['GDP per capita']/df['Score']*100).round(1)
df['%Social support']=(df['Social support']/df['Score']*100).round(1)
df4=df[df['Country or region'].isin(co)][['Overall rank','Country or region', 'Score','Sc','%GDP per capita','%Social support']]
display(df4)
OR
df2=df[['Overall rank','Country or region', 'Score','Sc','%GDP per capita','%Social support']]
df3=df2[df2['Country or region'].isin(co)]
display(df3)


df['Score'].min()
                        
df['Score'].max()
                        
df['Score'].median()
              
df['Score'].mean()

#Найдем среднее значение индекса счастья для стран из топ-10 рейтинга:
df[df['Overall rank'] <= 10]['Score'].mean()       
#Найдем среднее значение индекса счастья для Финляндии и Норвегии:
df[(df['Country or region'] == 'Finland') | (df['Country or region'] == 'Norway')]['Score'].mean()            

sort_values()
#По умолчанию используется сортировка по возрастанию True
display(df.sort_values(by=['Score'], ascending=True))      

display(df.sort_values('GDP per capita',ascending=False)['Country or region'][10])      

display(df.sort_values(by=['Social support','GDP per capita'],ascending=False))

display(df.sort_values(by=['Social support','GDP per capita'],ascending=[False, True]))      

# перезаписывать датафрейм    
df = df.sort_values(by=['Score'])
  
round(df[df['Country or region'].isin(countries_to_select)]['Score'].median(),2)
>>6.3    


query()
    filtered_df = df[df['Country or region'] == 'Russia']
    display(filtered_df) 
    -> 
    filtered_df2 = df.query('`Country or region`=="Russia"')
    display(filtered_df2)    
#название столбца выделено другим типом одинарных кавычек — ``; но они используются не всегда, а только когда в название столбца есть пробелы.

' `a a` == "a" '
`` если есть пробелы, чтобы query понял, что это один столбец
#Если название столбца состоит только из одного слова, а сам столбец содержит только числовые данные, то все выглядит гораздо проще. Выведем только те строки, в которых значение индекса счастья больше 7.5:
    filtered_df3 = df.query('Score==7.5') # нам потребовались только одинарные кавычки для обозначения всего условия
    display(filtered_df3)       

    filtered_df4=df.query('Score>7.5 or (`Social support`>1.4 and `GDP per capita`>1.4)') # следим за кавычками и порядком выполнения действий
    display(filtered_df4)

#Чтобы «объяснить» query, что мы используем именно переменную, нам надо поставить специальный символ @ .    
@
    score_lvl=7.5                                                       
    filtered_df5=df.query('Score>score_lvl') #такой код приведет к ошибке

    score_lvl=7.5                                                   
    filtered_df5=df.query('Score>@score_lvl') #такой код успешно выполнится
    display(filtered_df4)

Группировка, уменьшение размерности
groupby()

product=['морковь','картофель','яблоки','груши']
product_cat=['овощи','овощи','фрукты','фрукты']price=[35, 50, 65, 70]
data_dict = {'продукт':product,'категория':product_cat,'цена':price} # собираем из списков словарь
df_ex=pd.DataFrame(data_dict) # преобразуем словарь в датафрейм
display(df_ex)  

grouped_ex = df_ex.groupby('категория') ['цена'] # в круглых скобках укажем параметр, по которому агрегируем (в данному случае хотим агрегировать данные по категориям); в квадратных - какие значения используем для расчетов (мы хотим поработать с ценами)

grouped_ex = df_ex.groupby('категория') ['цена'] 
display(grouped_ex.mean()) 
>>  категория
    овощи     42.5
    фрукты    67.5
    Name: цена, dtype: float64
    
OR

.agg('max')
(df_ex.groupby('категория'))['цена'].agg('mean')# обратите внимание, что в данном случае mean нужно взять в кавычки 
#(['mean']) даст датафрейм

df_new = df.groupby('col_n', as_index=False).agg({'col_1':'sum', 'col_2':'sum'})#Параметр as_index позволяет превратить результат в датафрейм, который можно использовать дальше, как обычный датафрейм
df_new = df.groupby(['col_n', 'col_m'], as_index=False).agg({'col_3':'sum', 'col_4':'sum'})

reset_index()
df_new.reset_index() # индексы будут снова числеными

pivot_table().
values — какие данные мы будем агрегировать;
index — строки нашей будущей таблицы;
columns —  столбцы нашей будущей таблицы.
aggfunc — какую агрегирующую функцию используем (mean, max, min, sum, median).

data = {'Город': ['Москва', 'Москва', 'СПб', 'СПб', 'СПб'],
'Пол': ['М', 'Ж', 'М', 'М', 'Ж'],'Зарплата': [1000, 1200, 1100, 1050, 1300]} # если вы уже уверенно чувствуете себя в создании списков и преобразовании их в словарь, то можно делать это в один шаг
df_ex2 = pd.DataFrame(data)  
#Создадим сводную таблицу, которая покажет нам средние значения по городам:
pivot_table_1 = df_ex2.pivot_table(values='Зарплата', index='Город', aggfunc='mean')
display(pivot_table_1) 
>>	Город	Зарплата
    Москва	1100.0
    СПб	1150.0

pivot_table_3 = df_ex2.pivot_table(values='Зарплата', index='Город', columns='Пол', aggfunc='mean')
display(pivot_table_3)

>>  Пол	Ж	М
    Город		
    Москва	1200.0	1000.0
    СПб	1300.0	1075.0

df.pivot_table(['Score'], ['rank_group'], aggfunc='mean')).round(2) 
df.pivot_table(['Score'], ['rank_group'], aggfunc='mean')).round(2) 

#Выведем максимальное и минимальное зарплат по городам для каждого пола
pivot_table_3 = df_ex2.pivot_table(values='Зарплата', index='Город', columns='Пол', aggfunc=['max', 'min'])
>>      max	            min
Пол	    Ж	    М	    Ж	    М
Город				
Москва	1200	1000	1200	1000
СПб	    1300	1100	1300	1050

Преобразование строк

#Чтобы изменить тип данных
astype(type)
df['Overall rank']=df['Overall rank'].astype('float')

#Вспомним, какие методы и функции мы уже успели изучить за курс. Многие из них можно применять к строкам при работе с датафреймами Pandas.
int / str / float Перевод типов данных Не работает, но есть аналог astype()
math. floor / ceil / fabs / sqrt / log Математические операции Работают для строк
len Определение длины элементы Работает для строк
Обращение по индексам (слайсы) [] Создание слайсов, выделение фрагмента текста Работает для строк
isdigit Определение числа Работает для строк
find Поиск значения Работает для строк
count Подсчет вхождения доступен Работает для строк
lower / upper Изменение регистра Работает для строк
replace Замена символа Работает для строк регулярные выражения Регулярные выражения Могут применять к строкам
split / join Разделение и объединение фрагментов Работает для строк

df.columns  #вывод названия столбцов
    print(df.columns)
    Index(['Overall rank', 'Country or region', 'Score', 'GDP per capita',  
                       'Social support', 'Healthy life expectancy',
                       'Freedom to make life choices', 'Generosity'
                       'Perceptions of corruption', 'Country or region_lower',
                       'Country or region_lower_2'],
                     dtype='object')     
                     
# Приведем все названия столбцов к нижнему регистру с помощью str.lower()
df.columns=df.columns.str.lower()
df.columns
Index(['overall rank', 'healthy life expectancy', 'country or region', 'score','gdp per capita', 'social support', 'freedom to make life choices','generosity', 'perceptions of corruption'],dtype='object') #теперь все названия столбцов в нижнем регистр  

lambda
df['Country or region_lower']=df['Country or region'].apply(lambda x x.lower())

df['Int_Score']=df['Score'].astype('int')
df['Int_Score']=df['Score'].apply(lambda x:int(x))
df['Int_Score']=df['Score'].apply(lambda x:x).astype('int')

Практическое задание 3 к уроку 15 Часть 3 - Библиотека Pandas
import pandas as pd
df=pd.read_csv('2019.csv')
#df['Mix']=df['Country or region'].apply(lambda x: len(x.split(' ')))
#df['Mix2']=df['Mix'].apply(lambda x: 1 if x==1 else 2)
df['Mix']=df['Country or region'].apply(lambda x: 1 if len(x.split(' '))==1 else 2)
df.groupby('Mix', as_index=False).agg({'Score':'mean'})['Score'][0].round(2)

Визуализация в Pandas

гистограмма
plot()

plot() — функция, которая позволяет строить линейные графики, гистограммы, круговые диаграммы и другие типы графиков;
kind='hist' — параметр, передаваемый в plot() для построения гистограммы;

df['No of student_int'].plot(kind='hist')
#По горизонтальной оси Х находятся так называемые корзины (bins). Это диапазоны значений, на которые данные разбиваются при построении гистограммы.
#По вертикальной оси Y отмечается частота (frequency), то есть количество появлений значений в каждой корзине гистограммы.

вручную укажем количество корзин (bins);
нарисуем сетку (grid) для удобства чтения и визуализации данных;
поменяем соотношение сторон (figsize);
добавим название (title);
дадим названия осям (xlabel и ylabel);
ограничим отображение по оси Х (xlim), чтобы убрать пустоту справа.
df['No of student_int'].plot(kind='hist', bins=25, grid=True, figsize=(10,5), title='Количество студентов', xlabel='Кол-во студентов', ylabel='Частота', xlim=(0,150000) )

df_t=pd.concat([df1,df2,df3]) # unionall таблиц

аналог join - merge

%%time #время выполения ячейки , всегда в начале


Matplotlib
!pip install matplotlib

import matplotlib.pyplot as plt
%matplotlib inline#команда, которая позволит отображать графики и визуализации в ячейке вывода прямо под кодом
#Без этой команды после каждого графического вызова вам придется использовать отдельную функцию plt.show().

plt.plot(x, y, fmt, **kwargs)# график — ломаная линия по нескольким точкам


х и y — последовательности значений для осей х и y соответсвенно, это может быть как рукописная строка, так и значения из таблицы;
fmt — строка форматирования, определяющая тип графика и внешний вид линии;
**kwargs — дополнительные параметры для настройки внешнего вида графика: цвета, стиля линии и так далее.

plt.plot([1, 2, 7, 3, 1, -3])

def y_func(x):
    return ((x**2)+(5*x) + 7)
x = np.arange(-5, 5, 0.05)
y = y_func(x)
plt.plot(x, y)

plt.plot(x, y)
plt.title('График функции')  # зададим название
plt.xlabel('x', fontsize=12) # зададим название и размер шрифта для оси Х
plt.ylabel('y', fontsize=12) # зададим название и размер шрифта для оси Y
plt.grid(True) # добавим сетку

plt.plot(x, y, x, y1) #График двух функций

**kwargs  «keyword arguments»
- непрерывная линия
-- штриховая линия
-. штрихпунктирная линия
: пунктирная линия

'b' синий
'g' зеленый
'r' красный
'c' голубой cyan [saɪˈæn]
'm' пурпурный   magenta
'y' желтый
'k' черный  #The "K" in CMYK stands for key, a traditional word for a black printing plate.
'w' белый
-----------------------
def y_func(x):
    return ((x**2)+(5*x) + 7)
def y_func_1(x):
    return (-(x**2)+(4*x) + 8)

x = np.arange(-5, 5, 0.05)
y = y_func(x)
y1 = y_func_1(x)

plt.plot(x, y,'--b', x, y1,'r') # добавляем для первого графика синий цвет и пунктирную линию, а для второго — красный цвет
plt.title('График двух функций')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.grid(True)
plt.text(-5, 42, 'При значении х=0 вторая функция больше первой')
-----------------------

plt.bar #столбчатая диаграмма


SNS (Seaborn)
pip install seaborn

import seaborn as sns

histplot
название датафрейма (data);
название столбца (х), на основе которого мы будем строить визуализацию;
sns.histplot(data=gapminder.query('year==2007'), x='lifeExp',  bins=5)
sns.histplot(data=gapminder.query('year'==2007'), x='lifeExp', binwidth=5) 

data=gapminder.query('year==2007')
x='lifeExp'
bins=5
binwidth=5 #отвечает за ширину корзины
kde True или False #True, мы указываем на необходимость наложить на гистограмму плавную линию распределения. 
hue [hjuː] хюю оттенок ='continent' выделить в корзине распределение континентов
multiple='dodge'    отдельно каждый континент из continent
stat='percent'  значение в процентах, 100% это сумма всех столбиках
estimator='sum' # если хотим, чтобы были не среднии показатели, а сумма
errorbar='None'# убираем отображение погрешности

barplot()#столбчатая диаграмма
х — это категории, по которым мы будем выводить данные: пол, возраст, страны и так далее;
y — это показатель, который мы хотим проанализировать в разрезе выбранных категорий.
sns.barplot(data=gapminder.query('year==2007'), x='continent', y='lifeExp')

#Передать в функцию barplot параметр для дополнительной категоризации hue и присвоить ему значение year. Благодаря этому параметру SNS раскрасит данные за начало и конец периода в разные цвета.
sns.barplot(data=gapminder.query('year==2007 or year==1957'), x='continent', y='lifeExp', hue='year')
sns.barplot(data=gapminder.query('(continent=="Africa" or continent=="Oceania")'), x='year', y='pop',hue='continent')

Матрица рассеяния
pairplot()
sns.pairplot(gapminder)
sns.pairplot(gapminder, hue = 'continent')

bins — позволяет указать количество корзин для гистограммы;
binwidth — позволяет указать ширину корзины для гистограммы;
kde=True — позволяет отрисовать на гистограмме линию распределения;
hue — позволяет разделить данные по категориальной переменной;
multiple=dodge — позволяет отобразить несколько гистограмм, смещенных относительно друг друга;
stat=percent — позволяет представить гистограмму в процентном соотношении;
estimator — позволяет настроить статистическую функцию для подсчета значений по каждому столбцу (mean, sum, count и др.)


API и Python    -интерфейс прикладного программирования

