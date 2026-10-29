USE paper_bottle;
SELECT COUNT(*) AS total_users FROM users;
SELECT COUNT(*) AS total_bottles FROM bottles;
SELECT gender, COUNT(*) AS cnt FROM users GROUP BY gender;
