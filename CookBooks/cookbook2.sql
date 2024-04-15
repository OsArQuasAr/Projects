/*TEXT — символьная строка без указания длины, может содержать неограниченное количество символов.
CHARACTER VARYING или VARCHAR(n) — символьная строка переменной длины, в которой можно хранить n символов.
SMALLINT — целое число со знаком, которое находится в диапазоне от -32 768 до 32 767. Используется для хранения небольших чисел в целях экономии памяти.
INT — целое число в диапазоне от -2 147 483 648 до 2 147 483 647. В редких случаях, когда необходимо хранить число больше, чем позволяет INT, используют BIGINT.
DOUBLE PRECISION — дробное число, содержащее до 15 знака после запятой.
NUMERIC(p, s) — дробное число, состоящее из p цифр, из которых s цифр находятся после запятой. Ограничение по размеру — до 131 072 цифр перед десятичным разделителем и до 16 383 — после. Например, столбец с типом данных NUMERIC(3, 1)  будет содержать значения, округленные до одного десятичного знака, от -99.9 до 99.9 включительно.
BOOLEAN, или флажок, — логический тип данных, который имеет два возможных значения: true (истина) и false (ложь).
DATE — дата. Стандарт SQL рекомендует хранить ее в формате YYYY-MM-DD. Так, 22 ноября 2022 года нужно записать в виде 2022-11-22.
TIMESTAMP — дата и время. Стандарт SQL рекомендует хранить timestamp в формате YYYY-MM-DD HH:MI:SS. То есть 14 часов 21 минута 5 секунд 22 ноября 2022 года рекомендуется записать в виде 2022-11-22 14:21:05.*/
--
/*NUMERIC(2, -3), число будет округляться до ближайшей тысячи, и в нём можно будет хранить значения от -99000 до 99000 включительно.
NUMERIC(3, 5) , будет округлять значения до пяти знаков после точки и сможет хранить значения от -0.00999 до 0.00999 включительно.*/
CREATE DATABASE 

create table courier_student74 (
  courier_id int
, courier_auto varchar(10)
, courier_firstname varchar(50)
, start_date date
, rate numeric(7,2)
, fired bool)
PARTITION by RANGE(start_date);  --столбец по которому необходимо выполнять партиционирование

													----- создадим партиции для таблицы case_data_raw
create table client_transactions_before_2017 partition of case_data_raw
for values from ('1900-01-01') to ('2017-01-01');
create table client_transactions_2017 partition of case_data_raw
for values from ('2017-01-01') to ('2018-01-01');
create table client_transactions_2018 partition of case_data_raw
for values from ('2018-01-01') to ('2019-01-01');
													----- какой столбец или столбцы будут являться ключом
alter table courier_student74
add constraint courier_student74_pk 
primary key (courier_id)
													-----переименовать таблицу
ALTER TABLE prices_new_student744
rename to prices_new_student74
													-----добавить столбец 
ALTER TABLE prices_new_student74
add column  in_portfolio boolean													
													-----вставить данные
insert into courier_student74 (courier_id, courier_auto)
values (5,'АУ569М750')
						----OR----
insert into courier_student74
values (5,'АУ569М750')

insert into courier_student74 (courier_id, courier_auto)
values (6,'ВС783К790'),(7,'МК951Д790')

insert into courier_student74 (courier_firstname, start_date, rate, fired)
values ('Иван','2019-01-01', 400, False), ('Игорь','2019-07-01', 150, False), ('Кристина','2019-03-09', 300, False)
													-----обновить данные
update courier_student74
set (courier_firstname, start_date, rate, fired) =
('Иван','2019-01-01', 400, False)
where courier_id = 5

update courier_student74
set fired = True
where courier_id =  5

update order_student1
set courier_id = -1
where courier_id =  5
													-----удаление строку
delete from courier_student74
where сourier_id is NULL
													-----удалит все строки из таблицы
truncate courier_student74
										-----OR-----
delete from courier_student74
													-----удалить таблицу из базы данных безвозвратно
drop table сourier_student74


													-----Создадим индекс для таблицы
create unique index tel_number_index
on abonent_student74 (tel_number)
include (cite_name, address)
													-----Создадим индекс для каждой из партиций
create unique index client_transactions_before_2017_i on client_transactions_before_2017 (transaction_number);
create unique index client_transactions_2017_i on client_transactions_2017  (transaction_number);
create unique index client_transactions_2018_i on client_transactions_2018  (transaction_number);
create unique index client_transactions_2019_i on client_transactions_2019  (transaction_number);
													-----Обновление индекса
reindex index tel_number_index
													-----обновить сразу все индексы таблицы
reindex table abonent_student74													
													-----удалим ранее созданный индекс
drop index tel_number_index													
													-----измерения времени выполнения запросов
explain analyze
select * 
from имя_таблицы													
													
													-----Реализуем транзакцию
begin;
update prices_new set volume = volume - 100
where date = '2024-05-05' and symbol = 'RUSS';
update prices_new set volume = volume + 100
where date = '2024-05-04' and symbol = 'RUSS';
commit;													
													-----Настроим доступ к данным
grant all privileges on prices_new to "education-changellenge";
revoke all privileges on prices_new from "monitor"



SELECT  avg(current_date - date_of_birth::date)::float/365 age
from users
--group by 



SELECT count(DATE_TRUNC('year', date_of_birth::date)) as year_, COUNT(*)
from users
group by  year_
order by COUNT(*)

select date_trunc('year',date_of_birth::DATE) as year_,count(*)
FROM users
GROUP BY year_
orDER BY count(*) DESC													

select *, date_of_birth::date +INTERVAL '18 year' as y18
FROM users

SELECT count(gender) as сколько, 
case when gender = '1' then 'Ж'
else 'М'
end as гендр
from users
GROUP by gender

SELECT 
  case WHEN gender=1 THEN 'F' ELSE 'M'  END as gender_text
     ,count(*) as cnt
FROM users
GROUP BY gender

select case when (current_date-date_of_birth::DATe)/365::float<18 then 'before 18'
when (current_date-date_of_birth::DATe)/365::float>18 then 'after 18'
end as gruppa
,count(*)
from users
group by gruppa

--неверно
SELECT  
case when (current_date - date_of_birth::date)> INTERVAL '18 year' then '+18'
else '-18'
end as legal
from users

--объединение
SELECT
	 case when gender = 1 THEN 'Ж' ELSE 'М'END as gender_letter	
    ,case when (CURRENT_DATE - date_of_birth::DATE)/365 >= 18 THEN '+'
 ELSE '-' END as over_18
    ,COUNT(*)
from users
GROUP by gender_letter, over_18
ORDER BY gender_letter

----задание 4
WITH sal_rank AS (
SELECT 
    empno 
  , RANK() OVER(ORDER BY salary DESC) rnk
FROM 
  salaries
)
SELECT 
  empno
FROM
  sal_rank
WHERE 
  rnk = 1