
/*
#############
Decision_Status Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "Decision_Status_Id_seq";

CREATE TABLE Decision_Status (
    Decision_Status_Id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"Decision_Status_Id_seq"'::regclass),
    Name TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_Decision_Status_Name UNIQUE (Name)
);

-- AUTO INCREMENT
ALTER SEQUENCE "Decision_Status_Id_seq" owned by Decision_Status.Decision_Status_Id;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Decision_Status_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Decision_Status_updated_at" 
BEFORE UPDATE
ON Decision_Status FOR EACH ROW
EXECUTE FUNCTION "function_Decision_Status_updated_at"();


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
    Jurisdiction_Url TEXT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT UX_Jurisdiction_NameLocation UNIQUE (Name, Geolocation)
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

    FOREIGN KEY(Ground_Type_Id) REFERENCES Ground_Type(Ground_Type_Id) ON DELETE RESTRICT,

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

    Description TEXT NULL,

    Agent_Type_Id INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(Agent_Type_Id) REFERENCES Agent_Type(Agent_Type_Id) ON DELETE RESTRICT,

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
    Facts_Digest TEXT NULL,
    Facts_Date DATE NULL,

    Google_Drive_Folder TEXT NULL,

    Case_Status_Id INT NULL,
    Appeal_Type_Id INT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(Case_Status_Id) REFERENCES Case_Status(Case_Status_Id) ON DELETE RESTRICT,
    FOREIGN KEY(Appeal_Type_Id) REFERENCES Appeal_Type(Appeal_Type_Id) ON DELETE RESTRICT,

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
Legal_Procedure Table
#############
*/

-- AUTO INCREMENT
CREATE SEQUENCE "Legal_Procedure_Id_seq";

CREATE TABLE Legal_Procedure (
    Legal_Procedure_Id INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('"Legal_Procedure_Id_seq"'::regclass),
    
    Abstract TEXT NULL,

    Introduction_Date DATE NULL,
    Decision_Date DATE NULL,
    
    Solution TEXT NULL,
    Comment TEXT NULL,
    
    Legal_Case_Id INT NOT NULL,
    Decision_Status_Id INT NULL,
    Jurisdiction_Id INT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(Legal_Case_Id) REFERENCES Legal_Case(Legal_Case_Id) ON DELETE RESTRICT,
    FOREIGN KEY(Decision_Status_Id) REFERENCES Decision_Status(Decision_Status_Id) ON DELETE RESTRICT,
    FOREIGN KEY(Jurisdiction_Id) REFERENCES Jurisdiction(Jurisdiction_Id) ON DELETE RESTRICT,

    CONSTRAINT UX_Legal_Procedure_TitleLegalCase UNIQUE (Legal_Case_Id, Abstract)
);

-- AUTO INCREMENT
ALTER SEQUENCE "Legal_Procedure_Id_seq" owned by Legal_Procedure.Legal_Procedure_Id;

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Legal_Procedure_updated_at"()
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Legal_Procedure_updated_at" 
BEFORE UPDATE
ON Legal_Procedure FOR EACH ROW
EXECUTE FUNCTION "function_Legal_Procedure_updated_at"();


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

    FOREIGN KEY(Resource_Type_Id) REFERENCES Resource_Type(Resource_Type_Id) ON DELETE RESTRICT,

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
Legal_Procedure_Resource Table
#############
*/

-- AUTO INCREMENT

CREATE TABLE Legal_Procedure_Resource (
    Resource_Id INT NOT NULL,
    Legal_Procedure_Id INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Foreign Keys
    FOREIGN KEY(Resource_Id) REFERENCES Resource(Resource_Id) ON DELETE RESTRICT,
    FOREIGN KEY(Legal_Procedure_Id) REFERENCES Legal_Procedure(Legal_Procedure_Id) ON DELETE RESTRICT,

    PRIMARY KEY(Resource_Id, Legal_Procedure_Id)
);

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Legal_Procedure_Resource_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Legal_Procedure_Resource_updated_at" 
BEFORE UPDATE
ON Legal_Procedure_Resource FOR EACH ROW
EXECUTE FUNCTION "function_Legal_Procedure_Resource_updated_at"();


/*
#############
Legal_Case_Complainant Table
#############
*/

CREATE TABLE Legal_Case_Complainant (
    Agent_Id INT NOT NULL,
    Legal_Case_Id INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(Agent_Id) REFERENCES Agent(Agent_Id) ON DELETE RESTRICT,
    FOREIGN KEY(Legal_Case_Id) REFERENCES Legal_Case(Legal_Case_Id) ON DELETE RESTRICT,

    PRIMARY KEY(Agent_Id, Legal_Case_Id)
);

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Legal_Case_Complainant_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Legal_Case_Complainant_updated_at" 
BEFORE UPDATE
ON Legal_Case_Complainant FOR EACH ROW
EXECUTE FUNCTION "function_Legal_Case_Complainant_updated_at"();

/*
#############
Legal_Case_Complainant_Recipient Table
#############
*/

CREATE TABLE Legal_Case_Complainant_Recipient (
    Agent_Id INT NOT NULL,
    Legal_Case_Id INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(Agent_Id) REFERENCES Agent(Agent_Id) ON DELETE RESTRICT,
    FOREIGN KEY(Legal_Case_Id) REFERENCES Legal_Case(Legal_Case_Id) ON DELETE RESTRICT,

    PRIMARY KEY(Agent_Id, Legal_Case_Id)
);

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Legal_Case_Complainant_Recipient_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Legal_Case_Complainant_Recipient_updated_at" 
BEFORE UPDATE
ON Legal_Case_Complainant_Recipient FOR EACH ROW
EXECUTE FUNCTION "function_Legal_Case_Complainant_Recipient_updated_at"();


/*
#############
Legal_Procedure_Ground Table
#############
*/

CREATE TABLE Legal_Procedure_Ground (
    Ground_Id INT NOT NULL,
    Legal_Procedure_Id INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY(Ground_Id) REFERENCES Ground(Ground_Id) ON DELETE RESTRICT,
    FOREIGN KEY(Legal_Procedure_Id) REFERENCES Legal_Procedure(Legal_Procedure_Id) ON DELETE RESTRICT,

    PRIMARY KEY(Ground_Id, Legal_Procedure_Id)
);

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION "function_Legal_Procedure_Ground_updated_at"() 
RETURNS trigger
LANGUAGE plpgsql
AS $function$
begin
    NEW."updated_at" = NOW();
    RETURN NEW;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER "trigger_Legal_Procedure_Ground_updated_at" 
BEFORE UPDATE
ON Legal_Procedure_Ground FOR EACH ROW
EXECUTE FUNCTION "function_Legal_Procedure_Ground_updated_at"();


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
Decision_Status
Demandes accueillies, Rejet, Ne statue pas sur le fond, Demande partiellement accueillies
*/
INSERT INTO Decision_Status (Name) VALUES ('Demandes accueillies');
INSERT INTO Decision_Status (Name) VALUES ('Rejet');
INSERT INTO Decision_Status (Name) VALUES ('Ne statue pas sur le fond');
INSERT INTO Decision_Status (Name) VALUES ('Demande partiellement accueillies');

/*
Case_Status
En cours, Finie, En appel
*/
INSERT INTO Case_Status (Name) VALUES ('En cours');
INSERT INTO Case_Status (Name) VALUES ('Finie');
INSERT INTO Case_Status (Name) VALUES ('En appel');

/*
Agent

INSERT INTO Agent (Name, Agent_Type_Id) VALUES ('', (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = ''));
*/

INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Gloucester Resources Limited', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Entreprise'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Scott Gilmore et autres (Hausfeld LLP)', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Entreprise'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Minister for Planning', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Etat'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Ramin Pejan et autres (Earthjustice)', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Entreprise'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Argentine', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Etat'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Brésil', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Etat'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('France', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Etat'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Allemagne', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Etat'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Turquie', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Etat'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Alaska Wilderness League', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Association'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Defenders of Wildlife', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Association'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Northern Alaska Environmental Center', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Association'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Sierra Club', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Association'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('The Wilderness Society', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Association'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Center for Biological Diversity', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Association'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Friends of the Earth', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Association'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Greenpeace', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Association'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('ConocoPhillips Ltd', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Entreprise'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('William Tsama', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Particulier'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Saúl Luciano Lliuya', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Particulier'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('RWE', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Entreprise'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Tribue Standing Rock Sioux', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Association'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Tribue Cheyenne River Sioux', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Association'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Corps des ingénieurs de l''armée états-unienne (US Army Corp of Engineers, USACE)', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Organisation publique'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Chiara Sacchi', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Particulier'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('État ougandais', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Etat'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Autorités locales ougandaises', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Organisation publique'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Greenpeace Espana', 'https://es.greenpeace.org/es/', (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Association'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Ecologistas en Acción', 'https://www.ecologistasenaccion.org/', (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Association'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('Oxfam Intermón', 'https://www.oxfamintermon.org/es', (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Association'));
INSERT INTO Agent (Name, Agent_Url, Agent_Type_Id) VALUES ('État espagnol', null, (SELECT Agent_Type_Id FROM Agent_Type WHERE Name = 'Etat'));


/*
Resource

INSERT INTO Resource (Name, Resource_Type_Id) VALUES ('', (SELECT Resource_Type_Id FROM Resource_Type WHERE Name = ''));
*/


INSERT INTO Resource (Name, Url, Resource_Type_Id) VALUES ('12', 'https://notreaffaireatous.org/numero-12-de-la-newsletter-des-affaires-climatiques-ecocide-loccasion-manquee/', (SELECT Resource_Type_Id FROM Resource_Type WHERE Name = 'Newsletter'));
INSERT INTO Resource (Name, Url, Resource_Type_Id) VALUES ('16', 'https://notreaffaireatous.org/numero-16-de-la-newsletter-des-affaires-climatiques/', (SELECT Resource_Type_Id FROM Resource_Type WHERE Name = 'Newsletter'));
INSERT INTO Resource (Name, Url, Resource_Type_Id) VALUES ('Case No. 3:20-cv-00290-SLG, Case No. 3:20-cv-00308-SLG', 'https://s3.documentcloud.org/documents/21045581/210818-willow-order.pdf ', (SELECT Resource_Type_Id FROM Resource_Type WHERE Name = 'Décision'));
INSERT INTO Resource (Name, Url, Resource_Type_Id) VALUES ('Tsama William et alii v. Attorney General of Uganda, mémoire en demande, 12 octobre 2020', 'nan', (SELECT Resource_Type_Id FROM Resource_Type WHERE Name = 'Ouvrage/Article'));
INSERT INTO Resource (Name, Url, Resource_Type_Id) VALUES ('W., FRANK, C. BALS, J. GRIMM, "The Case of Huaraz: First Climate Lawsuit on Loss and Damage Against an Energy Company Before German Courts" in. R. MECHLER, L. BOUWER, T. SCHINKO, S. SURMINSKI, J. LINNEROOTH-BAYER (éd.) Loss and Damage from Climate Change. Climate Risk Management, Policy and Governance. Springer, Cham, Suisse, 2018. https://doi.org/10.1007/978-3-319-72026-5_20.', 'https://link.springer.com/chapter/10.1007/978-3-319-72026-5_20#citeas', (SELECT Resource_Type_Id FROM Resource_Type WHERE Name = 'Ouvrage/Article'));
INSERT INTO Resource (Name, Url, Resource_Type_Id) VALUES ('Décision US District Court for DC, “Standing Rock Sioux Tribe v. USACE”, 6 juillet 2020', 'https://earthjustice.org/sites/default/files/files/standing_rock_sioux_tribe_v._army_corps_of_engineers.pdf ', (SELECT Resource_Type_Id FROM Resource_Type WHERE Name = 'Décision'));
INSERT INTO Resource (Name, Url, Resource_Type_Id) VALUES ('9', 'https://notreaffaireatous.org/numero-9-de-la-newsletter-des-affaires-climatiques-droit-a-un-environnement-sain/', (SELECT Resource_Type_Id FROM Resource_Type WHERE Name = 'Newsletter'));
INSERT INTO Resource (Name, Url, Resource_Type_Id) VALUES ('Article "Gloucester Resources (“Rocky Hill”) case", Environmental Law Australia', 'http://envlaw.com.au/gloucester-resources-case/', (SELECT Resource_Type_Id FROM Resource_Type WHERE Name = 'Ouvrage/Article'));
INSERT INTO Resource (Name, Url, Resource_Type_Id) VALUES ('8', 'https://notreaffaireatous.org/numero-8-de-la-newsletter-des-affaires-climatiques/', (SELECT Resource_Type_Id FROM Resource_Type WHERE Name = 'Newsletter'));
INSERT INTO Resource (Name, Url, Resource_Type_Id) VALUES ('14', 'https://notreaffaireatous.org/numero-14-de-la-newsletter-des-affaires-climatiques-la-proposition-de-directive-europeenne-sur-le-devoir-de-vigilance-des-entreprises/', (SELECT Resource_Type_Id FROM Resource_Type WHERE Name = 'Newsletter'));


/*
Ground

INSERT INTO Ground (Name, Ground_Type_Id) VALUES ('', (SELECT Ground_Type_Id FROM Ground_Type WHERE Name = ''));
*/


INSERT INTO Ground (Name, Ground_Type_Id) VALUES ('Convention internationale des droits de l’enfant du 20 novembre 1989', (SELECT Ground_Type_Id FROM Ground_Type WHERE Name = 'Droits humains'));
INSERT INTO Ground (Name, Ground_Type_Id) VALUES ('Loi états-unienne sur la Politique Nationale sur l’Environnement ( National Environment Policy Act –NEPA)', (SELECT Ground_Type_Id FROM Ground_Type WHERE Name = 'Normes environnementales'));
INSERT INTO Ground (Name, Ground_Type_Id) VALUES ('Loi états-unienne sur la Propreté des Eaux (Clean Water Act –CWA)', (SELECT Ground_Type_Id FROM Ground_Type WHERE Name = 'Normes environnementales'));
INSERT INTO Ground (Name, Ground_Type_Id) VALUES ('Loi états-unienne sur la Protection des Mammifères Marins (Marine Mammals Protection Act -MMPA)', (SELECT Ground_Type_Id FROM Ground_Type WHERE Name = 'Normes environnementales'));
INSERT INTO Ground (Name, Ground_Type_Id) VALUES ('Loi états-unienne sur la protection des Espèces en Danger(Endangered Species Act – ESA)', (SELECT Ground_Type_Id FROM Ground_Type WHERE Name = 'Normes environnementales'));
INSERT INTO Ground (Name, Ground_Type_Id) VALUES ('Constitution Ougandaise', (SELECT Ground_Type_Id FROM Ground_Type WHERE Name = 'Public trust'));
INSERT INTO Ground (Name, Ground_Type_Id) VALUES ('Droit à la vie de l’article 22§1 de la Constitution Ougandaise', (SELECT Ground_Type_Id FROM Ground_Type WHERE Name = 'Droits humains'));
INSERT INTO Ground (Name, Ground_Type_Id) VALUES ('Article 1004 du Code civil allemand', (SELECT Ground_Type_Id FROM Ground_Type WHERE Name = 'Droit civil'));
INSERT INTO Ground (Name, Ground_Type_Id) VALUES ('Accord de Paris, 2015', (SELECT Ground_Type_Id FROM Ground_Type WHERE Name = 'Normes environnementales'));
INSERT INTO Ground (Name, Ground_Type_Id) VALUES ('Règlement (UE) n° 2018/1999 du 11 décembre 2018 sur la gouvernance de l''Union de l''énergie et de l''action pour le climat', (SELECT Ground_Type_Id FROM Ground_Type WHERE Name = 'Normes environnementales'));



/*

#############
RENAME TABLES
#############

*/

/* PREFIX_CODE varchar := 'nc_1j7o___'; */

ALTER TABLE Decision_Status RENAME TO                   "nc_1j7o___Decision_Status";
ALTER TABLE Legal_Procedure RENAME TO                   "nc_1j7o___Legal_Procedure";
ALTER TABLE Jurisdiction RENAME TO                      "nc_1j7o___Jurisdiction";
ALTER TABLE Ground_Type RENAME TO                       "nc_1j7o___Ground_Type";
ALTER TABLE Ground RENAME TO                            "nc_1j7o___Ground";
ALTER TABLE Case_Status RENAME TO                       "nc_1j7o___Case_Status";
ALTER TABLE Agent_Type RENAME TO                        "nc_1j7o___Agent_Type";
ALTER TABLE Agent RENAME TO                             "nc_1j7o___Agent";
ALTER TABLE Appeal_Type RENAME TO                       "nc_1j7o___Appeal_Type";
ALTER TABLE Legal_Case RENAME TO                        "nc_1j7o___Legal_Case";
ALTER TABLE Resource_Type RENAME TO                     "nc_1j7o___Resource_Type";
ALTER TABLE Resource RENAME TO                          "nc_1j7o___Resource";
ALTER TABLE Legal_Procedure_Resource RENAME TO          "nc_1j7o___Legal_Procedure_Resource";
ALTER TABLE Legal_Case_Complainant RENAME TO            "nc_1j7o___Legal_Case_Complainant";
ALTER TABLE Legal_Case_Complainant_Recipient RENAME TO  "nc_1j7o___Legal_Case_Complainant_Recipient";
ALTER TABLE Legal_Procedure_Ground RENAME TO            "nc_1j7o___Legal_Procedure_Ground";


/*
delete from "nc_05yt___Legal_Case_Complainant" 
where "agent_id" = (select "agent_id" from "nc_05yt___Agent" where "agent_id" = $1 limit $2) 
    and "legal_case_id" = (select "legal_case_id" from "nc_05yt___Legal_Case" where "legal_case_id" = $3 limit $4) - Cannot delete !!!
*/

/*
Delete failed: delete from "nc_05yt___Agent" where "agent_id" = $1 - control reached end of trigger procedure without RETURN
*/

ALTER TABLE                   "nc_05yt___Decision_Status" RENAME TO Decision_Status                 ;
ALTER TABLE                   "nc_05yt___Legal_Procedure" RENAME TO Legal_Procedure                 ;
ALTER TABLE                      "nc_05yt___Jurisdiction" RENAME TO Jurisdiction                    ;
ALTER TABLE                       "nc_05yt___Ground_Type" RENAME TO Ground_Type                     ;
ALTER TABLE                            "nc_05yt___Ground" RENAME TO Ground                          ;
ALTER TABLE                       "nc_05yt___Case_Status" RENAME TO Case_Status                     ;
ALTER TABLE                        "nc_05yt___Agent_Type" RENAME TO Agent_Type                      ;
ALTER TABLE                             "nc_05yt___Agent" RENAME TO Agent                           ;
ALTER TABLE                       "nc_05yt___Appeal_Type" RENAME TO Appeal_Type                     ;
ALTER TABLE                        "nc_05yt___Legal_Case" RENAME TO Legal_Case                      ;
ALTER TABLE                     "nc_05yt___Resource_Type" RENAME TO Resource_Type                   ;
ALTER TABLE                          "nc_05yt___Resource" RENAME TO Resource                        ;
ALTER TABLE          "nc_05yt___Legal_Procedure_Resource" RENAME TO Legal_Procedure_Resource        ;
ALTER TABLE            "nc_05yt___Legal_Case_Complainant" RENAME TO Legal_Case_Complainant          ;
ALTER TABLE  "nc_05yt___Legal_Case_Complainant_Recipient" RENAME TO Legal_Case_Complainant_Recipient;
ALTER TABLE            "nc_05yt___Legal_Procedure_Ground" RENAME TO Legal_Procedure_Ground          ;





/*
Many-to-Many On Delete Constraints
*/

-- CREATE FUNCTION
CREATE OR REPLACE FUNCTION function_many_to_many_agent_temp_blocking()
RETURNS trigger
LANGUAGE plpgsql
AS $function$
BEGIN
    -- RAISE EXCEPTION 'Cannot delete !!!';
    RETURN OLD;
END;
$function$;

-- CREATE TRIGGER
CREATE TRIGGER trigger_many_to_many_agent_complainant
BEFORE DELETE
ON "Legal_Case_Complainant"
FOR EACH ROW
EXECUTE FUNCTION function_many_to_many_agent_temp_blocking();

-- CREATE TRIGGER
CREATE TRIGGER trigger_many_to_many_agent_complainant_recipient
BEFORE DELETE
ON "Legal_Case_Complainant_Recipient"
FOR EACH ROW
EXECUTE FUNCTION function_many_to_many_agent_temp_blocking();

-- CREATE TRIGGER
CREATE TRIGGER trigger_many_to_many_resource
BEFORE DELETE
ON "Legal_Procedure_Resource"
FOR EACH ROW
EXECUTE FUNCTION function_many_to_many_agent_temp_blocking();

-- CREATE TRIGGER
CREATE TRIGGER trigger_many_to_many_ground
BEFORE DELETE
ON "Legal_Procedure_Ground"
FOR EACH ROW
EXECUTE FUNCTION function_many_to_many_agent_temp_blocking();

-- DROP TRIGGER WHEN NEEDED
DROP TRIGGER trigger_many_to_many_agent_complainant ON "Agent";
DROP FUNCTION function_many_to_many_agent_temp_blocking;
