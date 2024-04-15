											-----взять часть столбца по разделителю
select 'Dublin,Ireland' as adress
,split_part('Dublin,Ireland',',',1) as city
,split_part('Dublin,Ireland',',',2) as country 
											-----взять часть столбца по количеству знаков
select left ('North Chicago, Illinois', 13)
select right ('North Chicago, Illinois', 8)
											-----изменить регистр столбца
select  s.sub_industry
,upper(s.sub_industry)
,lower(s.sub_industry)
											-----заменить, выдаст:Netflix
select replace('Netflix Inc.','Inc.','')

select
,case when f.symbol = 'AAPL' then 'Apple'
	when f.symbol = 'FB'  then 'Facebook'
	else 'Other'
end as category
,case 
	when f.capital_expenditures <=-10000000000 then 'extra-high'
	when f.capital_expenditures >-10000000000 and f.capital_expenditures <=-5000000000 then 'high'
	when f.capital_expenditures >-5000000000  and f.capital_expenditures <=-1000000000 then 'medium'
	when f.capital_expenditures >-1000000000  and f.capital_expenditures <=0 then 'low'
end as capital_expenditures_category
										-----OR--------
,case 
	when f.capital_expenditures <=-10000000000 then 'extra-high'
	when f.capital_expenditures <=-5000000000 then 'high'
	when f.capital_expenditures <=-1000000000 then 'medium'
	when f.capital_expenditures <=0 then 'low'
end capital_expenditures_category
											-----уникальные значения
count(distinct client_id)
/* aboba */
from fundamentals f  

--
SELECT * FROM таблица_1
LEFT JOIN таблица_2
ON таблица_1.столбец_1 = таблица_2.столбец_1
LEFT JOIN таблица_3
ON таблица_1.столбец_1 = таблица_3.столбец_1
--
from securities s
left join fundamentals f
inner join
on s.symbol = f.symbol		
/*------OR------*/		
using(symbol,period_ending) --убирает дублирование столбцов при попытке вывести результат объединения
--

--
SELECT *  FROM таблица_1
UNION
SELECT *  FROM таблица_2
UNION
SELECT *  FROM таблица_3
--
select s.security
from securities s
where s.sector = 'Financials'
union all
select s.security
from securities s
where s.sub_industry = 'Banks'
/*
Если дублирующихся строк нет, то  UNION и UNION ALL дадут одинаковый результат. Однако UNION ALL будет выполняться быстрее и создаст меньшую нагрузку на хранилище данных, так как он не предполагает ненужной в данном случае фильтрации.
*/

with companies as (
select s.symbol, s."security", f.capital_expenditures, s.sector
from securities s
left join fundamentals f
on s.symbol = f.symbol
where f.capital_expenditures > -200000000
and f.period_ending = '31.12.2016'
),
CTE_name_2 AS (*некий запрос*),
CTE_name_3 AS (*некий запрос*),
CTE_name_4 AS (*некий запрос*),
CTE_name_5 AS (*некий запрос*)
select *
from companies
---
select *
from (
select s.symbol, s."security", f.capital_expenditures, s.sector
from securities s
left join fundamentals f
on s.symbol = f.symbol
where f.capital_expenditures > -200000000
and f.period_ending = '31.12.2016'
) as companies

select *
from prices p
where p.symbol in (select s.symbol from securities s where s.sector = 'Financials')

/*Не рекомендуется использовать оператор IN с подзапросом, который возвращает большое количество значений. Для ускорения запроса лучше воспользоваться другим оператором, например, JOIN*/

select *
from fundamentals f
where f.capital_expenditures > (select avg(f.capital_expenditures) from fundamentals f)
---
--три(четыре) типа оконных функций: агрегирующие, ранжирующие и функции смещения, (аналитические)
/*Аналитические функции предоставляют информацию о распределении данных и используются в основном для статистического анализа — например, поиска медианы, интегрального распределения, перцентилей.*/
row_number(), min(marginality) 
over(
partition by client_id
order by current_month desc
)
from store_data

ROW_NUMBER() --номер строки
/* ROW_NUMBER() активно применяют для борьбы с дублирующимися строками в запросах. Например, если в результате SELECT появляется несколько строк, то используют WHERE ROW_NUMBER() = 1, чтобы вывести только одну такую строку.*/
LAG() -- обращается к данным из предыдущей строки, а функция 
LEAD(marginality, 2)  -- из на 2 следующей 
/*Функции используют, чтобы сравнивать значение каждой строки с предыдущей или со следующей. Например, при анализе биржевой информации для вычисления изменения курса акций необходимо сравнить данные за текущий день с данными за предыдущий.*/


where (symbol = 'AAPL' or symbol= 'FB') and "close">100 and date between '2015-10-23' and '2015-10-30'
where symbol ='AAPL'

where s.symbol in ('AAPL','FB')

where s.sub_industry like '%software%'
where lower(s.sub_industry) like '%software%'
									-----
where s."security" is not null

group by client_id, current_month

having min(marginality) < 5

order by sector asc, sub_industry desc --по убыванию

limit 100 


with mini as 
(
select p2.symbol, min(p2.close) 
from prices p2
 --and symbol ='HBAN'
group by p2.symbol 
)
select p2.symbol , p2."close" - mini.min(p2.close)  as mi2
from prices p2 
inner join mini 
on p2.symbol = m.symbol
where p2.symbol ='HBAN' and p2."date" = '2010-01-04'

---мини кейс 
select case when gender=1 then 'women' else 'men'
end as gender_1
,case when (current_date-date_of_birth::DATe)/365::float<18 then 'before 18'
else 'after 18' end as gruppa
,count(*) as count_gruppa
from users
where 
lower((case when gender=1 then left(split_part(full_name,' ',2),-1)
 else split_part(full_name,' ',2) end))  like '%а%'
group by  gender_1,gruppa
order by gender_1,gruppa

select Now()
>>2024-04-14 11:40:28.181 +0300