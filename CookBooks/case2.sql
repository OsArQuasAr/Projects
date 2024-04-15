with minmax as (
select p.symbol  , min(p.low) as mi, max(p.high) as ma
from prices p 
group by p.symbol 
having min(p.low)<30 and max(p.high)>200
)
, excon as (--external constraint
select p.symbol , count(p."date") as co , sum(p.volume) as su
from prices p 
inner join minmax mi
on p.symbol = mi.symbol
group by p.symbol 
having count(p."date") >504 and sum(p.volume)>5000000
)
, daidev as (--daily deviation
select p.symbol ,p."date" , ex.co , ex.su, p."close" - lag(p."close") over(partition by p.symbol ) as diff
from prices p  
inner join excon ex
on p.symbol = ex.symbol
-- having p."close" - lag(p."close")over(partition by p.symbol) >0
)
, daidev2 as( 
select d.symbol,s."security" , avg(diff) 
from daidev d
inner join securities s
on d.symbol = s.symbol
where d.diff>0
group by d.symbol,s."security"
)
select *--, p."close" - lag(p."close") over(partition by p.symbol) as diff
from daidev2

/*from excon exx
where exx.symbol in (select da.symbol from daidev da where da.diff>0)*/

