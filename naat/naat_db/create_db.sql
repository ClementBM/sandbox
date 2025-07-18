
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
AgentParty Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "AgentPartyId_seq";

CREATE TABLE AgentParty (
    AgentPartyId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"AgentPartyId_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_AgentParty_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "AgentPartyId_seq" owned by AgentParty.AgentPartyId;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_AgentParty_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_AgentParty_updated_at" 
BEFORE UPDATE
ON AgentParty FOR EACH ROW
EXECUTE FUNCTION "function_AgentParty_updated_at"();


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
CREATE SEQUENCE "LegalCaseResourceId_seq";

CREATE TABLE LegalCaseResource (
    LegalCaseResourceId INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"LegalCaseResourceId_seq"'::regclass),

    ResourceId INT NOT NULL,
    LegalCaseId INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Foreign Keys
    FOREIGN KEY(ResourceId) REFERENCES Resource(ResourceId),
    FOREIGN KEY(LegalCaseId) REFERENCES LegalCase(LegalCaseId),

    CONSTRAINT UX_LegalCaseResource_ResourceLegal UNIQUE (ResourceId, LegalCaseId),
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
    AgentPartyId INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(AgentId) REFERENCES Agent(AgentId),
    FOREIGN KEY(LegalCaseId) REFERENCES LegalCase(LegalCaseId),
    FOREIGN KEY(AgentPartyId) REFERENCES AgentParty(AgentPartyId),

    CONSTRAINT UX_LegalCaseAgent_AgentLegalCaseType UNIQUE (AgentId, LegalCaseId, AgentPartyId)
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
AgentParty
Partie demanderesse, Partie défenderesse, Tiers
*/
INSERT INTO AgentParty (Name) VALUES ('Partie demanderesse');
INSERT INTO AgentParty (Name) VALUES ('Partie défenderesse');
INSERT INTO AgentParty (Name) VALUES ('Tiers');

/*
#############
RENAME TABLES
#############
*/

ALTER TABLE FinalDecision RENAME TO "nc_pghc___FinalDecision";
ALTER TABLE Jurisdiction RENAME TO "nc_pghc___Jurisdiction";
ALTER TABLE GroundType RENAME TO "nc_pghc___GroundType";
ALTER TABLE Ground RENAME TO "nc_pghc___Ground";
ALTER TABLE CaseStatus RENAME TO "nc_pghc___CaseStatus";
ALTER TABLE AgentType RENAME TO "nc_pghc___AgentType";
ALTER TABLE Agent RENAME TO "nc_pghc___Agent";
ALTER TABLE AgentParty RENAME TO "nc_pghc___AgentParty";
ALTER TABLE AppealType RENAME TO "nc_pghc___AppealType";
ALTER TABLE LegalCase RENAME TO "nc_pghc___LegalCase";
ALTER TABLE ResourceType RENAME TO "nc_pghc___ResourceType";
ALTER TABLE Resource RENAME TO "nc_pghc___Resource";
ALTER TABLE LegalCaseResource RENAME TO "nc_pghc___LegalCaseResource";
ALTER TABLE LegalCaseAgent RENAME TO "nc_pghc___LegalCaseAgent";
ALTER TABLE LegalCaseGround RENAME TO "nc_pghc___LegalCaseGround";

-- ALTER TABLE "nc_pghc__LegalCase" ALTER COLUMN Abstract DROP NOT NULL;
-- ALTER TABLE "nc_pghc__LegalCase" ALTER COLUMN CaseStatusId DROP NOT NULL;
-- ALTER TABLE "nc_pghc__LegalCase" ALTER COLUMN JurisdictionId DROP NOT NULL;

-- ALTER TABLE "nc_pghc__LegalCaseResource" DROP CONSTRAINT UX_LegalCaseResource_Code;
-- ALTER TABLE "nc_pghc__LegalCaseResource" DROP CONSTRAINT UX_LegalCaseResource_Name;

-- ALTER TABLE "nc_pghc__LegalCaseResource" ADD CONSTRAINT UX_LegalCaseResource_Code UNIQUE (Code, LegalCaseId);
-- ALTER TABLE "nc_pghc__LegalCaseResource" ADD CONSTRAINT UX_LegalCaseResource_Name UNIQUE (Name, LegalCaseId);

-- TODO: entités éditables sur les vues non figées
-- TODO: ne pas pouvoir ajouter des tables pour les utilisateurs éditeurs

-- TODO: ajout conseil de la partie demanderesse ?
-- TODO: ajout dans AgentType: Cabinet Avocat ?
