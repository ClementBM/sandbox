# Introduction

url
https://naat.up.railway.app/

## Nocodb

### Docker
```shell
docker build -t nocodbtest .

docker run -d --name nocodb -p 8080:8080 nocodbtest:latest
docker run -it --name nocodb -v "$(pwd)"/nocodb/:/usr/app/data/ -p 8080:8080 nocodbtest:latest

docker run -it --name nocodbtest -p 8080:8080 nocodbtest:latest
docker run -it --name nocodbtest -p 8080:8080 nocodb_climate_litigation:latest

docker stop nocodb && docker rm nocodb
docker image remove nocodbtest:latest
```


```shell
# Create an Image From a Container
docker commit <id_container>
# Tag the Image
docker tag 9565323927bf nocodb_climate_litigation
```

### Postgres dump

```shell
pg_dump -h <host> -d <database> -U <user> -p <port> -W -F t > latest.dump
```

pg_dump -h 127.0.0.1 -d naatdb -U postgres -W -F t > naat-26-01-2024.dump

docker exec -it d6723b6ee61d pg_dump -U postgres -F t -d root_db > naat-railway-03-02-2024.dump


pg_dump -U postgres -h containers-us-west-15.railway.app -p 5802 -W -F t railway > mydatabasebackup

### Setup Railway CLI
```shell
curl -fsSL https://railway.app/install.sh | sh
railway connect
railway link 690374dd-f5ef-4361-b5c3-3e968fa16d72

```

### Postgres restore
```shell
pg_restore -U postgres -h 127.0.0.1 -p 8080 -W -F t -d railway latest.dump
```

pg_restore -U postgres -h roundhouse.proxy.rlwy.net -p 22052 -W -F t -d naad_db ./naat-03-02-2024.dump

pg_restore -U postgres -h 127.0.0.1 -d naatdb -1 latest.dump

psql -h 127.0.0.1 -U postgres

OR

```shell
psql -h monorail.proxy.rlwy.net -U postgres -p 32000 -d railway
```

docker exec -it 63dead7906e8 pg_restore -U postgres -d naatdb -1 ./dumps/naat-26-01-2024.dump

docker exec -it 63dead7906e8 psql -U postgres

https://www.timescale.com/learn/postgres-cheat-sheet/tables

## TO CSV
```psql
\copy (SELECT * FROM "nc_jb59___Agent")                        TO agent.csv TO WITH CSV;
\copy (SELECT * FROM "nc_jb59___Agent_Type")                   TO agent_type.csv WITH CSV;
\copy (SELECT * FROM "nc_jb59___Appeal_Type")                  TO appeal_type.csv WITH CSV;
\copy (SELECT * FROM "nc_jb59___Case")                         TO case.csv WITH CSV;
\copy (SELECT * FROM "nc_jb59___Case_Complainant")             TO case_complainant.csv WITH CSV;
\copy (SELECT * FROM "nc_jb59___Case_Complainant_Recipient")   TO case_complainant_recipient.csv WITH CSV;
\copy (SELECT * FROM "nc_jb59___Case_Status")                  TO case_status.csv WITH CSV;
\copy (SELECT * FROM "nc_jb59___Decision")                     TO decision.csv WITH CSV;
\copy (SELECT * FROM "nc_jb59___Decision_Ground")              TO decision_ground.csv WITH CSV;
\copy (SELECT * FROM "nc_jb59___Decision_Resource")            TO decision_resource.csv WITH CSV;
\copy (SELECT * FROM "nc_jb59___Decision_Status")              TO decision_status.csv WITH CSV;
\copy (SELECT * FROM "nc_jb59___Ground")                       TO ground.csv WITH CSV;
\copy (SELECT * FROM "nc_jb59___Ground_Type")                  TO ground_type.csv WITH CSV;
\copy (SELECT * FROM "nc_jb59___Jurisdiction")                 TO jurisdiction.csv WITH CSV;
\copy (SELECT * FROM "nc_jb59___Resource")                     TO resource.csv WITH CSV;
\copy (SELECT * FROM "nc_jb59___Resource_Type")                TO resource_type.csv WITH CSV;
```

# Google Apps Script

[Mastering npm modules in google apps script (medium)](https://medium.com/geekculture/the-ultimate-guide-to-npm-modules-in-google-apps-script-a84545c3f57c)
[Write Google Apps Script Locally & Deploy with Clasp (medium)](https://medium.com/geekculture/how-to-write-google-apps-script-code-locally-in-vs-code-and-deploy-it-with-clasp-9a4273e2d018)


## Developer Resources
[nocodb rest apis](https://docs.nocodb.com/developer-resources/rest-apis/)
[nocodb sdk](https://docs.nocodb.com/developer-resources/sdk/)


# PostGre

```sql
SELECT con.*
       FROM pg_catalog.pg_constraint con
            INNER JOIN pg_catalog.pg_class rel
                       ON rel.oid = con.conrelid
            INNER JOIN pg_catalog.pg_namespace nsp
                       ON nsp.oid = connamespace
       WHERE nsp.nspname = '<schema name>'
             AND rel.relname = '<table name>';
```


```sql
create or replace function raiseException() returns void language plpgsql volatile as $$ begin   raise exception 'Cannot delete row.'; end$$;   

CREATE RULE shoe_del_protect AS ON DELETE TO shoe DO INSTEAD NOTHING;
CREATE or replace RULE prevent_deletes AS ON DELETE TO shoe DO INSTEAD select raiseException();

CREATE OR REPLACE RULE check_many_to_many as
  on delete to <your_table>
  where old.id = <your_id>
  do instead nothing;

CREATE FUNCTION check_many_to_many_agent_complainant()
RETURNS trigger
LANGUAGE plpgsql
AS $function$
BEGIN
    IF (SELECT Count(*) FROM Legal_Case_Complainant_Recipient WHERE Agent_Id = OLD.Agent_Id) > 0
    THEN
        RAISE EXCEPTION 'Cannot delete !!!';
    END IF;
END;
$function$;


CREATE TRIGGER check_many_to_many_agent_complainant BEFORE DELETE ON Agent
    FOR EACH ROW EXECUTE PROCEDURE check_many_to_many_agent_complainant();
```