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

# ADS Impact

|      | Case Categories                                                                               |
| ---: | :-------------------------------------------------------------------------------------------- |
|    0 | Advisory Opinions                                                                             |
|    1 | Suits against corporations, individuals                                                       |
|    2 | Suits against corporations, individuals>Corporations                                          |
|    3 | Suits against corporations, individuals>Corporations>Carbon credits                           |
|    4 | Suits against corporations, individuals>Corporations>Climate damage                           |
|    5 | Suits against corporations, individuals>Corporations>Disclosures                              |
|    6 | Suits against corporations, individuals>Corporations>Environmental assessment and permitting  |
|    7 | Suits against corporations, individuals>Corporations>Financing and investment                 |
|    8 | Suits against corporations, individuals>Corporations>GHG emissions reduction                  |
|    9 | Suits against corporations, individuals>Corporations>Just Transition                          |
|   10 | Suits against corporations, individuals>Corporations>Misleading advertising                   |
|   11 | Suits against corporations, individuals>Corporations>Pollution                                |
|   12 | Suits against corporations, individuals>Others                                                |
|   13 | Suits against corporations, individuals>Protesters                                            |
|   14 | Suits against governments                                                                     |
|   15 | Suits against governments>Access to information                                               |
|   16 | Suits against governments>Energy and power                                                    |
|   17 | Suits against governments>Environmental Crimes                                                |
|   18 | Suits against governments>Environmental assessment and permitting                             |
|   19 | Suits against governments>Environmental assessment and permitting>Climate adaptation          |
|   20 | Suits against governments>Environmental assessment and permitting>Natural resource extraction |
|   21 | Suits against governments>Environmental assessment and permitting>Other projects              |
|   22 | Suits against governments>Environmental assessment and permitting>Renewable projects          |
|   23 | Suits against governments>Environmental assessment and permitting>Utilities                   |
|   24 | Suits against governments>Failure to adapt                                                    |
|   25 | Suits against governments>GHG emissions reduction and trading                                 |
|   26 | Suits against governments>GHG emissions reduction and trading>EU ETS                          |
|   27 | Suits against governments>GHG emissions reduction and trading>Kyoto Protocol                  |
|   28 | Suits against governments>GHG emissions reduction and trading>Other                           |
|   29 | Suits against governments>Human Rights                                                        |
|   30 | Suits against governments>Human Rights>Climate migration                                      |
|   31 | Suits against governments>Human Rights>Indigenous Groups                                      |
|   32 | Suits against governments>Human Rights>Other                                                  |
|   33 | Suits against governments>Human Rights>Right to a healthy environment                         |
|   34 | Suits against governments>Human Rights>Women                                                  |
|   35 | Suits against governments>Human Rights>Youth/Children                                         |
|   36 | Suits against governments>Just transition                                                     |
|   37 | Suits against governments>Protecting biodiveristy and ecosystems                              |
|   38 | Suits against governments>Public Trust                                                        |
|   39 | Suits against governments>Trade and Investment                                                |
|   40 | Suits against governments>Trade and Investment>Climate-justified measures                     |
|   41 | Suits against governments>Trade and Investment>Environmental permitting                       |
|   42 | Suits against governments>Trade and Investment>Rollback of climate-justified measures         |


|      | Principal Laws                                                                                               |
| ---: | :----------------------------------------------------------------------------------------------------------- |
|    5 | Aarhus Convention on Access to Environmental Information                                                     |
|    6 | Act Against Unfair Competition>Article 5a                                                                    |
|    7 | African Charter on Human and Peoples’ Rights                                                                 |
|    8 | Agreement on Subsidies and Countervailing Measures (SCM Agreement)                                           |
|    9 | Agreement on Trade-Related Investment Measures (TRIMs Agreement)                                             |
|   10 | American Convention on Human Rights                                                                          |
|   11 | American Convention on Human Rights>San Salvador Protocol                                                    |
|   12 | American Declaration of the Rights and Duties of Man                                                         |
|   25 | Atlantic Forest Law                                                                                          |
|  245 | Competition and Consumer Act 2010                                                                            |
|  249 | Declaration of Rio de Janeiro of 1992;                                                                       |
|  251 | East African Community Treaty                                                                                |
|  252 | Economic and Cultural Rights                                                                                 |
|  256 | Energy Charter Treaty                                                                                        |
|  257 | Environmental Crimes Law                                                                                     |
|  258 | Escazú Agreement                                                                                             |
|  260 | European Convention on Human Rights                                                                          |
|  261 | European Social Charter of 1961                                                                              |
|  262 | European Union                                                                                               |
|  263 | European Union>Commission Delegated Regulation (EU) 2022/1214                                                |
|  264 | European Union>Commission Delegated Regulation 2021/2139                                                     |
|  265 | European Union>Effort Sharing Regulation 2018/842                                                            |
|  266 | European Union>European Code of Good Administrative Behavior                                                 |
|  267 | European Union>Primary Law>Aarhus Convention                                                                 |
|  268 | European Union>Primary Law>Charter of Fundamental Rights of the EU                                           |
|  269 | European Union>Primary Law>EU Charter on Human Rights                                                        |
|  270 | European Union>Primary Law>Treaty on the Functioning of the European Union                                   |
|  271 | European Union>Primary Law>Treaty on the Functioning of the European Union>Article 11                        |
|  272 | European Union>Primary Law>Treaty on the Functioning of the European Union>Article 13(2)                     |
|  273 | European Union>Primary Law>Treaty on the Functioning of the European Union>Article 192                       |
|  274 | European Union>Primary Law>Treaty on the Functioning of the European Union>Article 218(2) to (4)             |
|  275 | European Union>Primary Law>Treaty on the Functioning of the European Union>Article 263                       |
|  276 | European Union>Primary Law>Treaty on the Functioning of the European Union>Article 295                       |
|  277 | European Union>Primary Law>Treaty on the Functioning of the European Union>Article 340                       |
|  278 | European Union>Regulation (EU) 2020/852                                                                      |
|  279 | European Union>Secondary Law                                                                                 |
|  280 | European Union>Secondary Law>Decision 377/2013/EU                                                            |
|  281 | European Union>Secondary Law>Directives                                                                      |
|  282 | European Union>Secondary Law>Directives>1991/676/EEC                                                         |
|  283 | European Union>Secondary Law>Directives>2000/60/EC                                                           |
|  284 | European Union>Secondary Law>Directives>2001/42/EC                                                           |
|  285 | European Union>Secondary Law>Directives>2001/77/EC                                                           |
|  286 | European Union>Secondary Law>Directives>2003/4/EC                                                            |
|  287 | European Union>Secondary Law>Directives>2003/54/EC                                                           |
|  288 | European Union>Secondary Law>Directives>2003/87/EC                                                           |
|  289 | European Union>Secondary Law>Directives>2007/46/EC                                                           |
|  290 | European Union>Secondary Law>Directives>2008/101/EC                                                          |
|  291 | European Union>Secondary Law>Directives>2008/50/EC                                                           |
|  292 | European Union>Secondary Law>Directives>2009/28/EC                                                           |
|  293 | European Union>Secondary Law>Directives>2009/29/EC                                                           |
|  294 | European Union>Secondary Law>Directives>2009/30/EC                                                           |
|  295 | European Union>Secondary Law>Directives>2011/92/EU                                                           |
|  296 | European Union>Secondary Law>Directives>2018 Revised Renewable Energy Directive                              |
|  297 | European Union>Secondary Law>Directives>98/70/EC                                                             |
|  298 | European Union>Secondary Law>Directives>Directive 2014/52/EU - The Environmental Impact Assessment Directive |
|  299 | European Union>Secondary Law>Directives>Directive 2018/2001                                                  |
|  300 | European Union>Secondary Law>Directives>Energy Efficiency Directive                                          |
|  301 | European Union>Secondary Law>Directives>Sharing Decision (406/2009/EC)                                       |
|  302 | European Union>Secondary Law>Directives>¬ 2005/29/EC Unfair Commercial Practices Directive                   |
|  303 | European Union>Secondary Law>Regulations                                                                     |
|  304 | European Union>Secondary Law>Regulations>Commission Decision 2011/278/EU                                     |
|  305 | European Union>Secondary Law>Regulations>Commission Delegated Regulation 2022/1214                           |
|  306 | European Union>Secondary Law>Regulations>Commission Regulation (EU) 2016/646                                 |
|  307 | European Union>Secondary Law>Regulations>Commission Regulation 601/2012                                      |
|  308 | European Union>Secondary Law>Regulations>Council Regulation (EC) 73/2009                                     |
|  309 | European Union>Secondary Law>Regulations>Council Regulation 1049/2001                                        |
|  310 | European Union>Secondary Law>Regulations>Council Regulation 1367/2006                                        |
|  311 | European Union>Secondary Law>Regulations>EU Regulation (EU) 2017/1129                                        |
|  312 | European Union>Secondary Law>Regulations>Greenhouse Gas Emissions Trading Scheme Regulations 2012            |
|  313 | European Union>Secondary Law>Regulations>Regulation (EU) 2018/1999                                           |
|  314 | European Union>Secondary Law>Regulations>Regulation 2019/807                                                 |
|  315 | European Union>Secondary Law>Regulations>Regulation 2021/1119                                                |
|  316 | European Union>Secondary Law>Regulations>Regulation 715/2007                                                 |
|  317 | European Union>Secondary Law>Regulations>Taxonomy Regulation 2020/852                                        |
|  326 | Forest Code (Law 12.651/2012)                                                                                |
|  349 | Free Trade Agreement between Colombia and Canada                                                             |
|  350 | General Agreement on Tariffs and Trade 1994                                                                  |
|  373 | ICESCR                                                                                                       |
|  374 | ILO Indigenous and Tribal Peoples Convention 169                                                             |
|  389 | International Court of Justice>Statute of the ICJ                                                            |
|  390 | International Covenant on Civil and Political Rights                                                         |
|  391 | International Covenant on Social                                                                             |
|  392 | International Human Rights Law                                                                               |
|  411 | Kyoto Protocol Compliance Tribunals>Kyoto Protocol                                                           |
|  449 | North American Free Trade Agreement                                                                          |
|  451 | OECD Guidelines for Multinational Enterprises                                                                |
|  461 | Paris Agreement (enacted by Decree 9073/2017)                                                                |
|  462 | Paris Agreement (enacted by Federal Decree 9073/2017)                                                        |
|  473 | Rome Statute                                                                                                 |
|  474 | Social and Cultural Rights                                                                                   |
|  499 | Technical Barriers to Trade Agreement (TBT Agreement)                                                        |
|  512 | UNFCCC                                                                                                       |
|  513 | UNFCCC>Kyoto Protocol                                                                                        |
|  514 | UNFCCC>National Action Plan on Climate Change                                                                |
|  515 | UNFCCC>Paris Agreement                                                                                       |
|  516 | UNGA Resolution 64/92                                                                                        |
|  517 | UNHCR's Guiding Principles on Business and Human Rights                                                      |
|  567 | United Nations Convention on the Law of the Sea                                                              |
|  568 | United Nations Convention on the Rights of the Child                                                         |
|  569 | United Nations Declaration on Indigenous Peoples Rights                                                      |
|  570 | Universal Declaration of Human Rights                                                                        |
