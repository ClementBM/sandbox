
/*
#############
Final_Decision Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "Final_Decision_Id_seq";

CREATE TABLE Final_Decision (
    Final_Decision_Id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"Final_Decision_Id_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_Final_Decision_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "Final_Decision_Id_seq" owned by Final_Decision.Final_Decision_Id;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Final_Decision_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Final_Decision_updated_at" 
BEFORE UPDATE
ON Final_Decision FOR EACH ROW
EXECUTE FUNCTION "function_Final_Decision_updated_at"();


/*
#############
Jurisdiction Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "Jurisdiction_Id_seq";

CREATE TABLE Jurisdiction (
    Jurisdiction_Id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"Jurisdiction_Id_seq"'::regclass),
    Name TEXT NOT NULL,

    Geolocation TEXT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_Jurisdiction_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "Jurisdiction_Id_seq" owned by Jurisdiction.Jurisdiction_Id;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Jurisdiction_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Jurisdiction_updated_at" 
BEFORE UPDATE
ON Jurisdiction FOR EACH ROW
EXECUTE FUNCTION "function_Jurisdiction_updated_at"();


/*
#############
Ground_Type Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "Ground_Type_Id_seq";

CREATE TABLE Ground_Type (
    Ground_Type_Id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"Ground_Type_Id_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_Ground_Type_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "Ground_Type_Id_seq" owned by Ground_Type.Ground_Type_Id;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Ground_Type_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Ground_Type_updated_at" 
BEFORE UPDATE
ON Ground_Type FOR EACH ROW
EXECUTE FUNCTION "function_Ground_Type_updated_at"();

/*
#############
Ground Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "Ground_Id_seq";

CREATE TABLE Ground (
    Ground_Id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"Ground_Id_seq"'::regclass),
    Name TEXT NOT NULL,

    Ground_Type_Id INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Foreign Keys
    FOREIGN KEY(Ground_Type_Id) REFERENCES Ground_Type(Ground_Type_Id),

    CONSTRAINT UX_Ground_NameType UNIQUE (Name, Ground_Type_Id)
);

-- AUTO INCREMENT
ALTER SEQUENCE "Ground_Id_seq" owned by Ground.Ground_Id;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Ground_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Ground_updated_at" 
BEFORE UPDATE
ON Ground FOR EACH ROW
EXECUTE FUNCTION "function_Ground_updated_at"();

/*
#############
Case_Status Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "Case_Status_Id_seq";

CREATE TABLE Case_Status (
    Case_Status_Id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"Case_Status_Id_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_Case_Status_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "Case_Status_Id_seq" owned by Case_Status.Case_Status_Id;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Case_Status_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Case_Status_updated_at" 
BEFORE UPDATE
ON Case_Status FOR EACH ROW
EXECUTE FUNCTION "function_Case_Status_updated_at"();


/*
#############
Agent_Type Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "Agent_Type_Id_seq";

CREATE TABLE Agent_Type (
    Agent_Type_Id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"Agent_Type_Id_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_Agent_Type_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "Agent_Type_Id_seq" owned by Agent_Type.Agent_Type_Id;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Agent_Type_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Agent_Type_updated_at" 
BEFORE UPDATE
ON Agent_Type FOR EACH ROW
EXECUTE FUNCTION "function_Agent_Type_updated_at"();

/*
#############
Agent Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "Agent_Id_seq";

CREATE TABLE Agent (
    Agent_Id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"Agent_Id_seq"'::regclass),
    Name TEXT NOT NULL,
    Agent_Url VARCHAR(255) NULL,

    Agent_Type_Id INT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(Agent_Type_Id) REFERENCES Agent_Type(Agent_Type_Id),

    CONSTRAINT UX_Agent_NameType UNIQUE (Name, Agent_Type_Id)
);

-- AUTO INCREMENT
ALTER SEQUENCE "Agent_Id_seq" owned by Agent.Agent_Id;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Agent_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Agent_updated_at" 
BEFORE UPDATE
ON Agent FOR EACH ROW
EXECUTE FUNCTION "function_Agent_updated_at"();


/*
#############
Appeal_Type Table
#############
*/


-- AUTO INCREMENT
CREATE SEQUENCE "Appeal_Type_Id_seq";

CREATE TABLE Appeal_Type (
    Appeal_Type_Id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"Appeal_Type_Id_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_AppealType_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "Appeal_Type_Id_seq" owned by Appeal_Type.Appeal_Type_Id;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Appeal_Type_updated_at"()
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Appeal_Type_updated_at" 
BEFORE UPDATE
ON Appeal_Type FOR EACH ROW
EXECUTE FUNCTION "function_Appeal_Type_updated_at"();

/*
#############
Legal_Case Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "Legal_Case_Id_seq";

CREATE TABLE Legal_Case (
    Legal_Case_Id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"Legal_Case_Id_seq"'::regclass),
    Title TEXT NOT NULL,
    Abstract TEXT NULL,

    Google_Drive_Folder TEXT NULL,

    Introduction_Date DATE NULL,

    Final_Decision_Date DATE NULL,
    Final_Decision_Comment TEXT NULL,

    Case_Status_Id INT NULL,
    Jurisdiction_Id INT NULL,
    Final_Decision_Id INT NULL,
    Appeal_Type_Id INT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(Case_Status_Id) REFERENCES Case_Status(Case_Status_Id),
    FOREIGN KEY(Jurisdiction_Id) REFERENCES Jurisdiction(Jurisdiction_Id),
    FOREIGN KEY(Final_Decision_Id) REFERENCES Final_Decision(Final_Decision_Id),
    FOREIGN KEY(Appeal_Type_Id) REFERENCES Appeal_Type(Appeal_Type_Id),

    CONSTRAINT UX_Legal_Case_Title UNIQUE (Title)
);

-- AUTO INCREMENT
ALTER SEQUENCE "Legal_Case_Id_seq" owned by Legal_Case.Legal_Case_Id;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Legal_Case_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Legal_Case_updated_at" 
BEFORE UPDATE
ON Legal_Case FOR EACH ROW
EXECUTE FUNCTION "function_Legal_Case_updated_at"();


/*
#############
Resource_Type Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "Resource_Type_Id_seq";

CREATE TABLE Resource_Type (
    Resource_Type_Id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"Resource_Type_Id_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_Resource_Type_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "Resource_Type_Id_seq" owned by Resource_Type.Resource_Type_Id;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Resource_Type_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Resource_Type_updated_at" 
BEFORE UPDATE
ON Resource_Type FOR EACH ROW
EXECUTE FUNCTION "function_Resource_Type_updated_at"();

/*
#############
Resource Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "Resource_Id_seq";

CREATE TABLE Resource (
    Resource_Id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"Resource_Id_seq"'::regclass),
    
    Name TEXT NOT NULL,
    Url TEXT,
    
    Resource_Type_Id INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(Resource_Type_Id) REFERENCES Resource_Type(Resource_Type_Id),

    CONSTRAINT UX_Resource_NameType UNIQUE (Name, Resource_Type_Id)
);

-- AUTO INCREMENT
ALTER SEQUENCE "Resource_Id_seq" owned by Resource.Resource_Id;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Resource_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Resource_updated_at" 
BEFORE UPDATE
ON Resource FOR EACH ROW
EXECUTE FUNCTION "function_Resource_updated_at"();


/*
#############
Legal_Case_Resource Table
#############
*/

-- AUTO INCREMENT

CREATE TABLE Legal_Case_Resource (
    Resource_Id INT NOT NULL,
    Legal_Case_Id INT NOT NULL,

    Comment TEXT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Foreign Keys
    FOREIGN KEY(Resource_Id) REFERENCES Resource(Resource_Id),
    FOREIGN KEY(Legal_Case_Id) REFERENCES Legal_Case(Legal_Case_Id),

    PRIMARY KEY(Resource_Id, Legal_Case_Id)
);

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Legal_Case_Resource_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Legal_Case_Resource_updated_at" 
BEFORE UPDATE
ON Legal_Case_Resource FOR EACH ROW
EXECUTE FUNCTION "function_Legal_Case_Resource_updated_at"();


/*
#############
Agent_Party Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "Agent_Party_Id_seq";

CREATE TABLE Agent_Party (
    Agent_Party_Id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"Agent_Party_Id_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_Agent_Party_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "Agent_Party_Id_seq" owned by Agent_Party.Agent_Party_Id;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Agent_Party_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Agent_Party_updated_at" 
BEFORE UPDATE
ON Agent_Party FOR EACH ROW
EXECUTE FUNCTION "function_Agent_Party_updated_at"();


/*
#############
Legal_Case_Agent Table
#############
*/

CREATE TABLE Legal_Case_Agent (
    Agent_Id INT NOT NULL,
    Legal_Case_Id INT NOT NULL,
    Agent_Party_Id INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(Agent_Id) REFERENCES Agent(Agent_Id),
    FOREIGN KEY(Legal_Case_Id) REFERENCES Legal_Case(Legal_Case_Id),
    FOREIGN KEY(Agent_Party_Id) REFERENCES Agent_Party(Agent_Party_Id),

    PRIMARY KEY(Agent_Id, Legal_Case_Id, Agent_Party_Id)
);

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Legal_Case_Agent_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Legal_Case_Agent_updated_at" 
BEFORE UPDATE
ON Legal_Case_Agent FOR EACH ROW
EXECUTE FUNCTION "function_Legal_Case_Agent_updated_at"();

/*
#############
Legal_Case_Ground Table
#############
*/

CREATE TABLE Legal_Case_Ground (
    Ground_Id INT NOT NULL,
    Legal_Case_Id INT NOT NULL,

    Comment TEXT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(Ground_Id) REFERENCES Ground(Ground_Id),
    FOREIGN KEY(Legal_Case_Id) REFERENCES Legal_Case(Legal_Case_Id),

    PRIMARY KEY(Ground_Id, Legal_Case_Id)
);

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Legal_Case_Ground_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Legal_Case_Ground_updated_at" 
BEFORE UPDATE
ON Legal_Case_Ground FOR EACH ROW
EXECUTE FUNCTION "function_Legal_Case_Ground_updated_at"();


/*
#############
ADD CONSTANT VALUES
#############
*/

/*
Agent_Type
Entreprise, Etat, Organisation publique, Association, Particulier
*/
INSERT INTO Agent_Type (Name) VALUES ('Entreprise');
INSERT INTO Agent_Type (Name) VALUES ('Etat');
INSERT INTO Agent_Type (Name) VALUES ('Organisation publique');
INSERT INTO Agent_Type (Name) VALUES ('Association');
INSERT INTO Agent_Type (Name) VALUES ('Particulier');

/*
Ground_Type
Public trust, Tort law, Droits humains, Normes environnementales, Droit civil
*/
INSERT INTO Ground_Type (Name) VALUES ('Public trust');
INSERT INTO Ground_Type (Name) VALUES ('Tort law');
INSERT INTO Ground_Type (Name) VALUES ('Droits humains');
INSERT INTO Ground_Type (Name) VALUES ('Normes environnementales');
INSERT INTO Ground_Type (Name) VALUES ('Droit civil');

/*
Appeal_Type
Climat, Environnement, Climat/Environnement
*/
INSERT INTO Appeal_Type (Name) VALUES ('Climat');
INSERT INTO Appeal_Type (Name) VALUES ('Environnement');
INSERT INTO Appeal_Type (Name) VALUES ('Climat/Environnement');

/*
Resource_Type
Décision, Ouvrage/Article, Newsletter
*/
INSERT INTO Resource_Type (Name) VALUES ('Décision');
INSERT INTO Resource_Type (Name) VALUES ('Ouvrage/Article');
INSERT INTO Resource_Type (Name) VALUES ('Newsletter');

/*
Final_Decision
Demandes accueillies, Rejet, Ne statue pas sur le fond, Demande partiellement accueillies
*/
INSERT INTO Final_Decision (Name) VALUES ('Demandes accueillies');
INSERT INTO Final_Decision (Name) VALUES ('Rejet');
INSERT INTO Final_Decision (Name) VALUES ('Ne statue pas sur le fond');
INSERT INTO Final_Decision (Name) VALUES ('Demande partiellement accueillies');

/*
Case_Status
En cours, Finie, En appel
*/
INSERT INTO Case_Status (Name) VALUES ('En cours');
INSERT INTO Case_Status (Name) VALUES ('Finie');
INSERT INTO Case_Status (Name) VALUES ('En appel');

/*
AgentParty
Partie demanderesse, Partie défenderesse, Tiers
*/
INSERT INTO Agent_Party (Name) VALUES ('Partie demanderesse');
INSERT INTO Agent_Party (Name) VALUES ('Partie défenderesse');
INSERT INTO Agent_Party (Name) VALUES ('Tiers');

/*
#############
RENAME TABLES
#############
*/

/* PREFIX_CODE varchar := 'nc_ucsq___'; */

ALTER TABLE Final_Decision RENAME TO            "nc_ucsq___Final_Decision";
ALTER TABLE Jurisdiction RENAME TO              "nc_ucsq___Jurisdiction";
ALTER TABLE Ground_Type RENAME TO               "nc_ucsq___Ground_Type";
ALTER TABLE Ground RENAME TO                    "nc_ucsq___Ground";
ALTER TABLE Case_Status RENAME TO               "nc_ucsq___Case_Status";
ALTER TABLE Agent_Type RENAME TO                "nc_ucsq___Agent_Type";
ALTER TABLE Agent RENAME TO                     "nc_ucsq___Agent";
ALTER TABLE Agent_Party RENAME TO               "nc_ucsq___Agent_Party";
ALTER TABLE Appeal_Type RENAME TO               "nc_ucsq___Appeal_Type";
ALTER TABLE Legal_Case RENAME TO                "nc_ucsq___Legal_Case";
ALTER TABLE Resource_Type RENAME TO             "nc_ucsq___Resource_Type";
ALTER TABLE Resource RENAME TO                  "nc_ucsq___Resource";
ALTER TABLE Legal_Case_Resource RENAME TO       "nc_ucsq___Legal_Case_Resource";
ALTER TABLE Legal_Case_Agent RENAME TO          "nc_ucsq___Legal_Case_Agent";
ALTER TABLE Legal_Case_Ground RENAME TO         "nc_ucsq___Legal_Case_Ground";


/* DROP TABLES */

DROP TABLE "nc_ucsq___LegalCaseGround";
DROP TABLE "nc_ucsq___LegalCaseAgent";
DROP TABLE "nc_ucsq___LegalCaseResource";
DROP TABLE "nc_ucsq___Resource";
DROP TABLE "nc_ucsq___ResourceType";
DROP TABLE "nc_ucsq___LegalCase";
DROP TABLE "nc_ucsq___AppealType";
DROP TABLE "nc_ucsq___Agent";
DROP TABLE "nc_ucsq___AgentParty";
DROP TABLE "nc_ucsq___AgentType";
DROP TABLE "nc_ucsq___CaseStatus";
DROP TABLE "nc_ucsq___Ground";
DROP TABLE "nc_ucsq___GroundType";
DROP TABLE "nc_ucsq___Jurisdiction";
DROP TABLE "nc_ucsq___FinalDecision";
