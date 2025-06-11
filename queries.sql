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




select sum(length(text)) from hacker_news.items where text is not null
and cardinality(kids) > 2 and dead is null;

select text from hacker_news.items where length(text) > 2048
and cardinality(kids) > 4 and dead is null;

select sum(length(title)) from hacker_news.items where title is not null and dead is null
and score > 2

select max(length(title)),  min(length(title)), avg(length(title))
from hacker_news.items where title is not null and dead is null and parent is null and type='story';

select id, "by", time, text, url, score, title
from hacker_news.items
where dead is null and type='story';

select percentile_cont(0.5) within group (order by score) from hacker_news.items where title is not null
and dead is null;

select count(distinct "by") from hacker_news.items
where "by" not in (select id from hacker_news.users);

select "by", created, karma, cardinality(submitted) as length_submitted,
count(*) as story_count,
max(score) as max_score, min(score) as min_score, avg(score) as mean_score,
max(descendants) as max_descendants, min(descendants) as min_descendants, avg(descendants) as mean_descendants
from hacker_news.items i left join hacker_news.users u
on i."by" = u.id
where dead is null and type='story'
group by "by", created, karma, cardinality(submitted)
limit 10;


