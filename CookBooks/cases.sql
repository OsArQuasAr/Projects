select  s.security 
,case 
	when s.sector in ('Health Care') then 'Health Care'
	when lower(s.sub_industry) like '%insurance%'then 'Insurance'
end as category
,s.sector 
,s.sub_industry 
,split_part(s.adress,',',1) as city
from securities s 
where split_part(s.adress,',',1)  = 'New York' 
and s.sector in ('Health Care','Financials')
and (lower(s.sub_industry) like '%insurance%' or s.sector = 'Health Care')
order by s.sector asc,s.sub_industry asc,s.security asc