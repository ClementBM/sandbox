
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
    Code VARCHAR(255) NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_FinalDecision_Code UNIQUE (Code),
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
    Code VARCHAR(255) NOT NULL,

    Geolocation TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_Jurisdiction_Code UNIQUE (Code),
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
    Code VARCHAR(255) NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_GroundType_Code UNIQUE (Code),
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
    Code VARCHAR(255) NOT NULL,

    GroundTypeId INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Foreign Keys
    FOREIGN KEY(GroundTypeId) REFERENCES GroundType(GroundTypeId),

    CONSTRAINT UX_Ground_Code UNIQUE (Code),
    CONSTRAINT UX_Ground_Name UNIQUE (Name)
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
LegalCaseStatus Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "LegalCaseStatusId_seq";

CREATE TABLE LegalCaseStatus (
    LegalCaseStatusId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"LegalCaseStatusId_seq"'::regclass),
    Name TEXT NOT NULL,
    Code VARCHAR(255) NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_LegalCaseStatus_Code UNIQUE (Code),
    CONSTRAINT UX_LegalCaseStatus_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "LegalCaseStatusId_seq" owned by LegalCaseStatus.LegalCaseStatusId;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_LegalCaseStatus_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_LegalCaseStatus_updated_at" 
BEFORE UPDATE
ON LegalCaseStatus FOR EACH ROW
EXECUTE FUNCTION "function_LegalCaseStatus_updated_at"();


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
    Code VARCHAR(255) NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_AgentType_Code UNIQUE (Code),
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
    Code VARCHAR(255) NOT NULL,

    AgentTypeId INT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(AgentTypeId) REFERENCES AgentType(AgentTypeId),

    CONSTRAINT UX_Agent_Code UNIQUE (Code),
    CONSTRAINT UX_Agent_Name UNIQUE (Name)
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
LegalCaseAgentType Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "LegalCaseAgentTypeId_seq";

CREATE TABLE LegalCaseAgentType (
    LegalCaseAgentTypeId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"LegalCaseAgentTypeId_seq"'::regclass),
    Name TEXT NOT NULL,
    Code VARCHAR(255) NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_LegalCaseAgentType_Code UNIQUE (Code),
    CONSTRAINT UX_LegalCaseAgentType_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "LegalCaseAgentTypeId_seq" owned by LegalCaseAgentType.LegalCaseAgentTypeId;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_LegalCaseAgentType_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_LegalCaseAgentType_updated_at" 
BEFORE UPDATE
ON LegalCaseAgentType FOR EACH ROW
EXECUTE FUNCTION "function_LegalCaseAgentType_updated_at"();


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
    Code VARCHAR(255) NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_AppealType_Code UNIQUE (Code),
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

    LegalCaseStatusId INT NULL,
    JurisdictionId INT NULL,
    FinalDecisionId INT NULL,
    AppealTypeId INT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(LegalCaseStatusId) REFERENCES LegalCaseStatus(LegalCaseStatusId),
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
    Code VARCHAR(255) NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_ResourceType_Code UNIQUE (Code),
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
LegalCaseResource Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "LegalCaseResourceId_seq";

CREATE TABLE LegalCaseResource (
    LegalCaseResourceId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"LegalCaseResourceId_seq"'::regclass),
    Name TEXT NOT NULL,
    Code VARCHAR(255) NOT NULL,

    ResourceTypeId INT NOT NULL,
    LegalCaseId INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Foreign Keys
    FOREIGN KEY(ResourceTypeId) REFERENCES ResourceType(ResourceTypeId),
    FOREIGN KEY(LegalCaseId) REFERENCES LegalCase(LegalCaseId),

    CONSTRAINT UX_LegalCaseResource_Code UNIQUE (Code, LegalCaseId),
    CONSTRAINT UX_LegalCaseResource_Name UNIQUE (Name, LegalCaseId)
);

-- AUTO INCREMENT
ALTER SEQUENCE "LegalCaseResourceId_seq" owned by LegalCaseResource.LegalCaseResourceId;

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
LegalCaseAgent Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "LegalCaseAgentId_seq";

CREATE TABLE LegalCaseAgent (
    LegalCaseAgentId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"LegalCaseAgentId_seq"'::regclass),

    AgentId INT NOT NULL,
    LegalCaseId INT NOT NULL,
    LegalCaseAgentTypeId INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(AgentId) REFERENCES Agent(AgentId),
    FOREIGN KEY(LegalCaseId) REFERENCES LegalCase(LegalCaseId),
    FOREIGN KEY(LegalCaseAgentTypeId) REFERENCES LegalCaseAgentType(LegalCaseAgentTypeId),

    CONSTRAINT UX_LegalCaseAgent_AgentLegalCaseType UNIQUE (AgentId, LegalCaseId, LegalCaseAgentTypeId)
);

-- AUTO INCREMENT
ALTER SEQUENCE "LegalCaseAgentId_seq" owned by LegalCaseAgent.LegalCaseAgentId;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_LegalCaseAgent_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_LegalCaseAgent_updated_at" 
BEFORE UPDATE
ON LegalCaseAgent FOR EACH ROW
EXECUTE FUNCTION "function_LegalCaseAgent_updated_at"();


/*
#############
LegalCaseGround Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "LegalCaseGroundId_seq";

CREATE TABLE LegalCaseGround (
    LegalCaseGroundId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"LegalCaseGroundId_seq"'::regclass),

    GroundId INT NOT NULL,
    LegalCaseId INT NOT NULL,

    Comment VARCHAR(255) NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(GroundId) REFERENCES Ground(GroundId),
    FOREIGN KEY(LegalCaseId) REFERENCES LegalCase(LegalCaseId),

    CONSTRAINT UX_LegalCaseGround_LegalCaseGround UNIQUE (GroundId, LegalCaseId)
);

-- AUTO INCREMENT
ALTER SEQUENCE "LegalCaseGroundId_seq" owned by LegalCaseGround.LegalCaseGroundId;

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
RENAME TABLES
#############
*/

ALTER TABLE FinalDecision RENAME TO "nc_pghc___FinalDecision";
ALTER TABLE Jurisdiction RENAME TO "nc_pghc___Jurisdiction";
ALTER TABLE GroundType RENAME TO "nc_pghc___GroundType";
ALTER TABLE Ground RENAME TO "nc_pghc___Ground";
ALTER TABLE LegalCaseStatus RENAME TO "nc_pghc___LegalCaseStatus";
ALTER TABLE AgentType RENAME TO "nc_pghc___AgentType";
ALTER TABLE Agent RENAME TO "nc_pghc___Agent";
ALTER TABLE LegalCaseAgentType RENAME TO "nc_pghc___LegalCaseAgentType";
ALTER TABLE AppealType RENAME TO "nc_pghc___AppealType";
ALTER TABLE LegalCase RENAME TO "nc_pghc___LegalCase";
ALTER TABLE ResourceType RENAME TO "nc_pghc___ResourceType";
ALTER TABLE LegalCaseResource RENAME TO "nc_pghc___LegalCaseResource";
ALTER TABLE LegalCaseAgent RENAME TO "nc_pghc___LegalCaseAgent";
ALTER TABLE LegalCaseGround RENAME TO "nc_pghc___LegalCaseGround";


-- ALTER TABLE "nc_pghc__LegalCase" ALTER COLUMN Abstract DROP NOT NULL;
-- ALTER TABLE "nc_pghc__LegalCase" ALTER COLUMN LegalCaseStatusId DROP NOT NULL;
-- ALTER TABLE "nc_pghc__LegalCase" ALTER COLUMN JurisdictionId DROP NOT NULL;



-- TODO:
-- LegalCaseResource
ALTER TABLE "nc_pghc__LegalCaseResource" DROP CONSTRAINT UX_LegalCaseResource_Code UNIQUE (col_name);
ALTER TABLE "nc_pghc__LegalCaseResource" DROP CONSTRAINT UX_LegalCaseResource_Name UNIQUE (col_name);

ALTER TABLE "nc_pghc__LegalCaseResource" ADD CONSTRAINT UX_LegalCaseResource_Code UNIQUE (Code, LegalCaseId)
ALTER TABLE "nc_pghc__LegalCaseResource" ADD CONSTRAINT UX_LegalCaseResource_Name UNIQUE (Name, LegalCaseId)


-- TODO: ajouter les champs calculés sur les entités Many to Many

-- TODO: entités éditables sur les vues non figées

-- TODO: ne pas pouvoir ajouter des tables pour les utilisateurs éditeurs

-- TODO: Jurisdiction, add url