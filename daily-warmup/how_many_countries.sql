# Using the world database: How many different countries are in each region?
use world;
select * from country;

SELECT Region, COUNT(Code) 
FROM country 
GROUP BY Region
ORDER BY Region