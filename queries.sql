-- Basic stats
select min(time), max(time), count(*), avg(score), max(score) 
from hacker_news.items 
where type='story' and dead!=false;

-- Basic score histogram
select score, count(*)
from hacker_news.items and dead!=false
where type='story'
group by score
order by score;

-- Score histogram by year
select extract(year from time) as year, score, count(*)
from hacker_news.items
where type='story' and dead!=false
group by extract(year from time), score
order by year, score;

-- Number of users by year
select extract(year from time) as year, count(distinct "by") as users from hacker_news.items
group by extract(year from time);

-- Number of users by month
select extract(month FROM time) as month, extract(year from time) as year, count(distinct "by") as users from hacker_news.items
group by extract(month FROM time), extract(year from time)
order by year, month;

-- Pearson correlation between hour and score
select corr((extract(hour from time)), score) from hacker_news.items where type='story' and dead!=false;

-- Pearson correlation between raw timestamp and score
select corr((extract(epoch from time)), score) from hacker_news.items where type='story' and dead!=false;