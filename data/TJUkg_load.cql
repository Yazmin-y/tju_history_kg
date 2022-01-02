LOAD CSV WITH HEADERS FROM "file:///unit.csv" AS line with line
create (:unit{name:line.name,introduction:line.introduction,contact:line.contact,time:line.time});
 
LOAD CSV WITH HEADERS FROM "file:///event.csv" AS line with line
create 
(:event{name:line.name,time:line.time,introduction:line.introduction,image:line.image,related_school:line.school});

LOAD CSV WITH HEADERS FROM "file:///headmaster.csv" AS line with line
create (:headmaster{name:line.name,birthyear:line.birthyear,deathyear:line.deathyear,title:line.title,school:line.school,start_time:line.start_time,end_time:line.end_time,introduction:line.introduction,image:line.image});

LOAD CSV WITH HEADERS FROM "file:///school.csv" AS line with line
create (:school{name:line.name,start_time:line.starttime,end_time:line.endtime,location:line.place,department:line.department,type:line.type});
 
LOAD CSV WITH HEADERS FROM "file:///constitution.csv" AS line with line
create (:constitution{name:line.name,introduction:line.introduction,website:line.website,time:line.time });

LOAD CSV WITH HEADERS FROM "file:///lead_relation.csv" AS line with line
match (from:headmaster{name:line.name}),(to:school{name:line.school})
merge (from)-[r:担任校长]->(to);

LOAD CSV WITH HEADERS FROM "file:///constitution_school.csv" AS line with line
match (from: school{name:line.school}),(to:constitution{name:line.name})
merge (from)-[r:下设学院]->(to);

LOAD CSV WITH HEADERS FROM "file:///unit_school.csv" AS line with line
match (from: school{name:line.school}),(to:unit{name:line.name})
merge (from)-[r:下属机构]->(to);

LOAD CSV WITH HEADERS FROM "file:///precursor.csv" AS line with line
match (from: school{name:line.new}),(to:school{name:line.old})
merge (from)-[r:前身]->(to);

LOAD CSV WITH HEADERS FROM "file:///event_school.csv" AS line with line
match (from: event{name: line.name}),(to:school{name:line.school})
merge (from)-[r:涉及]->(to);

LOAD CSV WITH HEADERS FROM "file:///majorInfo.csv" AS line with line
create 
(:majorInfo{name:line.department,time:line.time,introduction:line.introduction,address:line.address});

LOAD CSV WITH HEADERS FROM "file:///teacherInfoCIC.csv" AS line with line
create (:teacherInfoCIC{name:line.name,title:line.title,department:line.department,course:line.course,instructor:line.instructor,email:line.email,field:line.field,direction:line.direction,homepage:line.homepage,imageURL:line.imageURL });

LOAD CSV WITH HEADERS FROM "file:///teacherInfoMechanics.csv" AS line with line
create (:teacherInfoMechanics{name:line.name,title:line.title,department:line.department,course:line.course,instructor:line.instructor,email:line.email,field:line.field,direction:line.direction,homepage:line.homepage,imageURL:line.imageURL });
 
LOAD CSV WITH HEADERS FROM "file:///teacherInfoElec.csv" AS line with line
create (:teacherInfoElec{name:line.name,title:line.title,department:line.department,course:line.course,instructor:line.instructor,email:line.email,direction:line.direction,homepage:line.homepage,imageURL:line.imageURL });
 
LOAD CSV WITH HEADERS FROM "file:///teacherInfoMicro.csv" AS line with line
create (:teacherInfoMicro{name:line.name,title:line.title,department:line.department,course:line.course,instructor:line.instructor,email:line.email,field:line.field,homepage:line.homepage,imageURL:line.imageURL });

LOAD CSV WITH HEADERS FROM "file:///teacherInfoEngineering.csv" AS line with line
create (:teacherInfoEngineering{name:line.name,title:line.title,department:line.department,course:line.course,instructor:line.instructor,homepage:line.homepage});

LOAD CSV WITH HEADERS FROM "file:///electrical.csv" AS line with line
match (from:teacherInfoElec{name:line.name}),(to:majorInfo{name:line.department})
merge (from)-[r:从事于]->(to);

LOAD CSV WITH HEADERS FROM "file:///computer.csv" AS line with line
match (from: teacherInfoCIC{name:line.name}),(to:majorInfo{name:line.department})
merge (from)-[r:从事于]->(to);

LOAD CSV WITH HEADERS FROM "file:///mechanics.csv" AS line with line
match (from: teacherInfoMechanics{name:line.name}),(to:majorInfo{name:line.department})
merge (from)-[r:从事于]->(to);
 
LOAD CSV WITH HEADERS FROM "file:///majorRelation.csv" AS line with line
match (from:majorInfo{name:line.department}),(to:constitution{name:line.academy})
merge (from)-[r:属于]->(to);

LOAD CSV WITH HEADERS FROM "file:///Engineering.csv" AS line with line
match (from: teacherInfoEngineering{name: line.name}),(to:majorInfo{name:line.department})
merge (from)-[r:从事于]->(to);

LOAD CSV WITH HEADERS FROM "file:///Microelectronics.csv" AS line with line
match (from: teacherInfoMicro{name: line.name}),(to:college{name:line.academy})
merge (from)-[r:从事于]->(to);
 

LOAD CSV WITH HEADERS FROM "file:///constitution_built.csv" AS line with line
match (from:constitution{name:line.name}),(to:constitution{name:line.new_object})
merge (from)-[r:成立{time:line.time}]->(to);
 
LOAD CSV WITH HEADERS FROM "file:///constitution_changename.csv" AS line with line
match (from: constitution{name:line.name}),(to:constitution{name:line.new_object})
merge (from)-[r:改名{time:line.time}]->(to);


 
LOAD CSV WITH HEADERS FROM "file:///constitution_changeunit.csv" AS line with line
match (from: constitution{name:line.original_object}),(to:constitution{name:line.name})
merge (from)-[r:调出{time:line.time}]->(to);

LOAD CSV WITH HEADERS FROM "file:///constitution_changeunit.csv" AS line with line
match (from:constitution{name:line.name}),(to:constitution{name:line.new_object})
merge (from)-[r:调入{time:line.time}]->(to);
 
LOAD CSV WITH HEADERS FROM "file:///constitution_changeschool.csv" AS line with line
match (from:constitution{name:line.name}),(to:school{name:line.new_object})
merge (from)-[r:调入]->(to);

LOAD CSV WITH HEADERS FROM "file:///constitution_changeschool.csv" AS line with line
match (from:school{name:line.original_object}),(to:constitution{name:line.name})
merge (from)-[r:调出]->(to);

