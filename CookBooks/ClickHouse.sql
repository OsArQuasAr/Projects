							--cell_towers, radio не нужно в ковычках
SELECT * FROM cell_towers
WHERE radio = 'UMTS'

SELECT array(array(1, 3)) AS x

CREATE TABLE имя_базы данных.имя_таблицы 
(имя_столбца тип_столбца_1, имя столбца_2 тип_столбца_2) 
ENGINE = MergeTree() --Движок MergeTree
ORDER BY (имя_столбца_для_сортировки)

							--Если таблица уже существует и указано IF NOT EXISTS, то запрос ничего не делает.
CREATE TABLE IF NOT EXISTS имя_базы данных.имя_таблицы (имя_столбца тип_столбца_1, имя столбца_2 тип_столбца_2)
							--отобразить информацию о таблице
DESCRIBE TABLE hits_100m_obfuscated
							--добавить столбец
ALTER TABLE datasets.hits_100m_obfuscated 
ADD COLUMN IF NOT EXISTS 'CLID' UInt3
							--удалить столбец
ALTER TABLE datasets.hits_100m_obfuscated 
DROP COLUMN  'CLID'
							--Удалим таблицу
DROP TABLE datasets.hits_100m_obfuscated
							--Вставим новые данные в таблицу
INSERT INTO datasets.hits_100m_obfuscated ('WatchID',  'JavaEnable', 'Title', 'GoodEvent') 
VALUES (900024523, 1, 'клик по кнопке',1)
							--отобразить только уникальные строки
SELECT DISTINCT * 
FROM имя_таблицы
							-----без учета пустых значений: NOT empty(destination)
SELECT destination, COUNT(destination) FROM opensky
WHERE NOT empty(destination)
GROUP BY destination
ORDER BY COUNT(destination) DESC

							-----функция разделяет массив Array в строки по числу элементов массива;
arrayJoin(NER)
							-----проверяет наличие элементов в массиве. 
has(имя_массива, 'имя_нужного_элемента')
							-----длина в ячейке или количество элементов в массиве
length(NER)


join, union, group by, агрегирующие функции --работают идентично