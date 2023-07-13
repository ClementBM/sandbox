
/*
#############
FinalDecision Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "FinalDecisionId_seq";

CREATE TABLE FinalDecision (
    FinalDecisionId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"FinalDecisionId_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_FinalDecision_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "FinalDecisionId_seq" owned by FinalDecision.FinalDecisionId;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_FinalDecision_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_FinalDecision_updated_at" 
BEFORE UPDATE
ON FinalDecision FOR EACH ROW
EXECUTE FUNCTION "function_FinalDecision_updated_at"();


/*
#############
Jurisdiction Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "JurisdictionId_seq";

CREATE TABLE Jurisdiction (
    JurisdictionId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"JurisdictionId_seq"'::regclass),
    Name TEXT NOT NULL,

    Geolocation TEXT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_Jurisdiction_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "JurisdictionId_seq" owned by Jurisdiction.JurisdictionId;

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
GroundType Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "GroundTypeId_seq";

CREATE TABLE GroundType (
    GroundTypeId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"GroundTypeId_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_GroundType_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "GroundTypeId_seq" owned by GroundType.GroundTypeId;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_GroundType_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_GroundType_updated_at" 
BEFORE UPDATE
ON GroundType FOR EACH ROW
EXECUTE FUNCTION "function_GroundType_updated_at"();

/*
#############
Ground Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "GroundId_seq";

CREATE TABLE Ground (
    GroundId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"GroundId_seq"'::regclass),
    Name TEXT NOT NULL,

    GroundTypeId INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Foreign Keys
    FOREIGN KEY(GroundTypeId) REFERENCES GroundType(GroundTypeId),

    CONSTRAINT UX_Ground_NameType UNIQUE (Name, GroundTypeId)
);

-- AUTO INCREMENT
ALTER SEQUENCE "GroundId_seq" owned by Ground.GroundId;

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
CaseStatus Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "CaseStatusId_seq";

CREATE TABLE CaseStatus (
    CaseStatusId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"CaseStatusId_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_CaseStatus_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "CaseStatusId_seq" owned by CaseStatus.CaseStatusId;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_CaseStatus_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_CaseStatus_updated_at" 
BEFORE UPDATE
ON CaseStatus FOR EACH ROW
EXECUTE FUNCTION "function_CaseStatus_updated_at"();


/*
#############
AgentType Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "AgentTypeId_seq";

CREATE TABLE AgentType (
    AgentTypeId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"AgentTypeId_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_AgentType_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "AgentTypeId_seq" owned by AgentType.AgentTypeId;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_AgentType_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_AgentType_updated_at" 
BEFORE UPDATE
ON AgentType FOR EACH ROW
EXECUTE FUNCTION "function_AgentType_updated_at"();

/*
#############
Agent Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "AgentId_seq";

CREATE TABLE Agent (
    AgentId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"AgentId_seq"'::regclass),
    Name TEXT NOT NULL,
    AgentUrl VARCHAR(255) NULL,

    AgentTypeId INT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(AgentTypeId) REFERENCES AgentType(AgentTypeId),

    CONSTRAINT UX_Agent_NameType UNIQUE (Name, AgentTypeId)
);

-- AUTO INCREMENT
ALTER SEQUENCE "AgentId_seq" owned by Agent.AgentId;

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
AppealType Table
#############
*/


-- AUTO INCREMENT
CREATE SEQUENCE "AppealTypeId_seq";

CREATE TABLE AppealType (
    AppealTypeId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"AppealTypeId_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_AppealType_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "AppealTypeId_seq" owned by AppealType.AppealTypeId;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_AppealType_updated_at"()
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_AppealType_updated_at" 
BEFORE UPDATE
ON AppealType FOR EACH ROW
EXECUTE FUNCTION "function_AppealType_updated_at"();

/*
#############
LegalCase Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "LegalCaseId_seq";

CREATE TABLE LegalCase (
    LegalCaseId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"LegalCaseId_seq"'::regclass),
    Title TEXT NOT NULL,
    Abstract TEXT NULL,

    GoogleDriveFolder TEXT NULL,

    IntroductionDate DATE NULL,

    FinalDecisionDate DATE NULL,
    FinalDecisionComment TEXT NULL,

    CaseStatusId INT NULL,
    JurisdictionId INT NULL,
    FinalDecisionId INT NULL,
    AppealTypeId INT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(CaseStatusId) REFERENCES CaseStatus(CaseStatusId),
    FOREIGN KEY(JurisdictionId) REFERENCES Jurisdiction(JurisdictionId),
    FOREIGN KEY(FinalDecisionId) REFERENCES FinalDecision(FinalDecisionId),
    FOREIGN KEY(AppealTypeId) REFERENCES AppealType(AppealTypeId),

    CONSTRAINT UX_LegalCase_Title UNIQUE (Title)
);

-- AUTO INCREMENT
ALTER SEQUENCE "LegalCaseId_seq" owned by LegalCase.LegalCaseId;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_LegalCase_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_LegalCase_updated_at" 
BEFORE UPDATE
ON LegalCase FOR EACH ROW
EXECUTE FUNCTION "function_LegalCase_updated_at"();


/*
#############
ResourceType Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "ResourceTypeId_seq";

CREATE TABLE ResourceType (
    ResourceTypeId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"ResourceTypeId_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_ResourceType_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "ResourceTypeId_seq" owned by ResourceType.ResourceTypeId;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_ResourceType_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_ResourceType_updated_at" 
BEFORE UPDATE
ON ResourceType FOR EACH ROW
EXECUTE FUNCTION "function_ResourceType_updated_at"();

/*
#############
Resource Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "ResourceId_seq";

CREATE TABLE Resource (
    ResourceId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"ResourceId_seq"'::regclass),
    
    Name TEXT NOT NULL,
    Url TEXT,
    
    ResourceTypeId INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(ResourceTypeId) REFERENCES ResourceType(ResourceTypeId),

    CONSTRAINT UX_Resource_NameType UNIQUE (Name, ResourceTypeId)
);

-- AUTO INCREMENT
ALTER SEQUENCE "ResourceId_seq" owned by Resource.ResourceId;

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
LegalCaseResource Table
#############
*/

-- AUTO INCREMENT

CREATE TABLE LegalCaseResource (
    ResourceId INT NOT NULL,
    LegalCaseId INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Foreign Keys
    FOREIGN KEY(ResourceId) REFERENCES Resource(ResourceId),
    FOREIGN KEY(LegalCaseId) REFERENCES LegalCase(LegalCaseId),

    PRIMARY KEY(ResourceId, LegalCaseId)
);

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_LegalCaseResource_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_LegalCaseResource_updated_at" 
BEFORE UPDATE
ON LegalCaseResource FOR EACH ROW
EXECUTE FUNCTION "function_LegalCaseResource_updated_at"();


/*
#############
LegalCaseComplainant Table
#############
*/

CREATE TABLE LegalCaseComplainant (
    AgentId INT NOT NULL,
    LegalCaseId INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(AgentId) REFERENCES Agent(AgentId),
    FOREIGN KEY(LegalCaseId) REFERENCES LegalCase(LegalCaseId),

    PRIMARY KEY(AgentId, LegalCaseId)
);

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_LegalCaseComplainant_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_LegalCaseComplainant_updated_at" 
BEFORE UPDATE
ON LegalCaseComplainant FOR EACH ROW
EXECUTE FUNCTION "function_LegalCaseComplainant_updated_at"();

/*
#############
LegalCaseComplainantRecipient Table
#############
*/

CREATE TABLE LegalCaseComplainantRecipient (
    AgentId INT NOT NULL,
    LegalCaseId INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(AgentId) REFERENCES Agent(AgentId),
    FOREIGN KEY(LegalCaseId) REFERENCES LegalCase(LegalCaseId),

    PRIMARY KEY(AgentId, LegalCaseId)
);

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_LegalCaseComplainantRecipient_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_LegalCaseComplainantRecipient_updated_at" 
BEFORE UPDATE
ON LegalCaseComplainantRecipient FOR EACH ROW
EXECUTE FUNCTION "function_LegalCaseComplainantRecipient_updated_at"();

/*
#############
LegalCaseGround Table
#############
*/

CREATE TABLE LegalCaseGround (
    GroundId INT NOT NULL,
    LegalCaseId INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(GroundId) REFERENCES Ground(GroundId),
    FOREIGN KEY(LegalCaseId) REFERENCES LegalCase(LegalCaseId),

    PRIMARY KEY(GroundId, LegalCaseId)
);

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_LegalCaseGround_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_LegalCaseGround_updated_at" 
BEFORE UPDATE
ON LegalCaseGround FOR EACH ROW
EXECUTE FUNCTION "function_LegalCaseGround_updated_at"();


/*
#############
ADD CONSTANT VALUES
#############
*/

/*
AgentType
Entreprise, Etat, Organisation publique, Association, Particulier
*/
INSERT INTO AgentType (Name) VALUES ('Entreprise');
INSERT INTO AgentType (Name) VALUES ('Etat');
INSERT INTO AgentType (Name) VALUES ('Organisation publique');
INSERT INTO AgentType (Name) VALUES ('Association');
INSERT INTO AgentType (Name) VALUES ('Particulier');

/*
GroundType
Public trust, Tort law, Droits humains, Normes environnementales, Droit civil
*/
INSERT INTO GroundType (Name) VALUES ('Public trust');
INSERT INTO GroundType (Name) VALUES ('Tort law');
INSERT INTO GroundType (Name) VALUES ('Droits humains');
INSERT INTO GroundType (Name) VALUES ('Normes environnementales');
INSERT INTO GroundType (Name) VALUES ('Droit civil');

/*
AppealType
Climat, Environnement, Climat/Environnement
*/
INSERT INTO AppealType (Name) VALUES ('Climat');
INSERT INTO AppealType (Name) VALUES ('Environnement');
INSERT INTO AppealType (Name) VALUES ('Climat/Environnement');

/*
ResourceType
Décision, Ouvrage/Article, Newsletter
*/
INSERT INTO ResourceType (Name) VALUES ('Décision');
INSERT INTO ResourceType (Name) VALUES ('Ouvrage/Article');
INSERT INTO ResourceType (Name) VALUES ('Newsletter');

/*
FinalDecision
Demandes accueillies, Rejet, Ne statue pas sur le fond, Demande partiellement accueillies
*/
INSERT INTO FinalDecision (Name) VALUES ('Demandes accueillies');
INSERT INTO FinalDecision (Name) VALUES ('Rejet');
INSERT INTO FinalDecision (Name) VALUES ('Ne statue pas sur le fond');
INSERT INTO FinalDecision (Name) VALUES ('Demande partiellement accueillies');

/*
CaseStatus
En cours, Finie, En appel
*/
INSERT INTO CaseStatus (Name) VALUES ('En cours');
INSERT INTO CaseStatus (Name) VALUES ('Finie');
INSERT INTO CaseStatus (Name) VALUES ('En appel');

/*
#############
RENAME TABLES
#############
*/


/* PREFIX_CODE varchar := 'nc_0rga___'; */

ALTER TABLE FinalDecision RENAME TO                 "nc_0rga___FinalDecision";
ALTER TABLE Jurisdiction RENAME TO                  "nc_0rga___Jurisdiction";
ALTER TABLE GroundType RENAME TO                    "nc_0rga___GroundType";
ALTER TABLE Ground RENAME TO                        "nc_0rga___Ground";
ALTER TABLE CaseStatus RENAME TO                    "nc_0rga___CaseStatus";
ALTER TABLE AgentType RENAME TO                     "nc_0rga___AgentType";
ALTER TABLE Agent RENAME TO                         "nc_0rga___Agent";
ALTER TABLE AppealType RENAME TO                    "nc_0rga___AppealType";
ALTER TABLE LegalCase RENAME TO                     "nc_0rga___LegalCase";
ALTER TABLE ResourceType RENAME TO                  "nc_0rga___ResourceType";
ALTER TABLE Resource RENAME TO                      "nc_0rga___Resource";
ALTER TABLE LegalCaseResource RENAME TO             "nc_0rga___LegalCaseResource";
ALTER TABLE LegalCaseComplainant RENAME TO          "nc_0rga___LegalCaseComplainant";
ALTER TABLE LegalCaseComplainantRecipient RENAME TO "nc_0rga___LegalCaseComplainantRecipient";
ALTER TABLE LegalCaseGround RENAME TO               "nc_0rga___LegalCaseGround";


/* DROP TABLES */

DROP TABLE "nc_0rga___LegalCaseGround";
DROP TABLE "nc_0rga___LegalCaseComplainantRecipient";
DROP TABLE "nc_0rga___LegalCaseComplainant";
DROP TABLE "nc_0rga___LegalCaseResource";
DROP TABLE "nc_0rga___Resource";
DROP TABLE "nc_0rga___ResourceType";
DROP TABLE "nc_0rga___LegalCase";
DROP TABLE "nc_0rga___AppealType";
DROP TABLE "nc_0rga___Agent";
DROP TABLE "nc_0rga___AgentType";
DROP TABLE "nc_0rga___CaseStatus";
DROP TABLE "nc_0rga___Ground";
DROP TABLE "nc_0rga___GroundType";
DROP TABLE "nc_0rga___Jurisdiction";
DROP TABLE "nc_0rga___FinalDecision";
