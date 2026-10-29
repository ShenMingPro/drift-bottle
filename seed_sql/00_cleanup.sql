USE paper_bottle;
SET FOREIGN_KEY_CHECKS = 0;
TRUNCATE TABLE messages;
TRUNCATE TABLE conversations;
DELETE FROM pick_commits;
TRUNCATE TABLE pick_records;
TRUNCATE TABLE bottles;
TRUNCATE TABLE users;
SET FOREIGN_KEY_CHECKS = 1;

INSERT INTO users (id, username, password, nickname, gender, age, hobbies, contact_type, contact_value, province, city, university, write_chances, pick_chances) VALUES
(1, '123456789123', 'a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3', '神明', '男', 20, '游戏,动漫', 'QQ号', '281736495', '上海市', '', '上海东海职业技术学院', 1, 3),
(2, '186149915920', 'a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3', '刘硕', '男', 21, '编程,科技', 'QQ号', '394857612', '河南省', '新乡市', '河南师范大学', 1, 3);
